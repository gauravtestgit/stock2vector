"""
Query a trained Stock2Vec model for cosine similarity.

Usage:
    python query_model.py                          # interactive mode
    python query_model.py --ticker NVDA            # top 10 similar to NVDA
    python query_model.py --ticker NVDA --top 5    # top 5 similar to NVDA
    python query_model.py --compare NVDA AMD       # compare two stocks
    python query_model.py --matrix                 # full similarity matrix
    python query_model.py --model data/models/sample/current  # use a different model

    OOV stocks (not in vocab) are estimated automatically using co-movement data.
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from config.settings import settings
from src.implementations.training.persistence import NumpyPersistence
from src.implementations.embeddings.metrics import CosineMetric
from src.implementations.embeddings.oov import CorrelationOOV
from src.implementations.pipeline.sources import YFinancePriceSource
from src.implementations.pipeline.returns import LogReturnsProcessor, MarketNeutralReturnsProcessor
from src.implementations.pipeline.cache import ParquetCache


def load_model(model_path):
    persistence = NumpyPersistence()
    embeddings, vocab, metadata = persistence.load(model_path)
    t2i = vocab
    i2t = {int(i): t for t, i in vocab.items()}
    return embeddings, t2i, i2t, metadata


def get_vocab_returns(metadata):
    """Load cached vocab returns for OOV estimation (market-neutral if configured)."""
    run_name = metadata.get("run_name", "")
    cache = ParquetCache(settings.raw_dir)
    prices = cache.read(run_name)
    if prices is None or prices.empty:
        return None
    if settings.returns_mode == "market_neutral":
        returns_proc = MarketNeutralReturnsProcessor(settings.benchmark_ticker)
    else:
        returns_proc = LogReturnsProcessor()
    return returns_proc.compute(prices)


def estimate_oov(ticker, embeddings, t2i, metadata):
    """Estimate embedding for an OOV ticker."""
    vocab_returns = get_vocab_returns(metadata)
    if vocab_returns is None:
        print(f"  Cannot estimate OOV: no cached price data found.")
        return None, None

    # Fetch OOV stock prices for the same period
    start = metadata.get("start", "2026-04-01")
    end = metadata.get("end", "2026-04-13")
    source = YFinancePriceSource(
        batch_size=1, retry_count=2,
        retry_wait_secs=2, rate_limit_secs=0.5
    )
    # Also fetch benchmark for market-neutral
    tickers_to_fetch = [ticker]
    if settings.returns_mode == "market_neutral":
        tickers_to_fetch.append(settings.benchmark_ticker)
    oov_prices = source.fetch(tickers_to_fetch, start=start, end=end)
    if oov_prices.empty or ticker not in oov_prices.columns:
        print(f"  Cannot fetch price data for '{ticker}'.")
        return None, None

    if settings.returns_mode == "market_neutral":
        returns_proc = MarketNeutralReturnsProcessor(settings.benchmark_ticker)
    else:
        returns_proc = LogReturnsProcessor()
    oov_returns = returns_proc.compute(oov_prices)[ticker]

    oov_strategy = CorrelationOOV(
        window_days=settings.correlation_window_days,
        min_correlation=0.5,
        step_days=settings.correlation_step_days,
    )
    result = oov_strategy.estimate(
        ticker=ticker,
        new_returns=oov_returns,
        embeddings=embeddings,
        t2i=t2i,
        vocab_returns=vocab_returns,
    )
    if result is None:
        print(f"  No co-movement found for '{ticker}' — cannot estimate.")
        return None, None
    return result


def print_model_info(t2i, metadata):
    print(f"\n  Model: {metadata.get('run_name', 'unknown')}")
    print(f"  Vocab: {len(t2i)} stocks")
    print(f"  Embed dim: {metadata.get('embed_dim', '?')}")
    print(f"  Epochs: {metadata.get('epochs', '?')}")
    print(f"  Final loss: {metadata.get('final_loss', '?'):.4f}" if isinstance(metadata.get('final_loss'), float) else "")
    print(f"  Date range: {metadata.get('start', '?')} → {metadata.get('end', '?')}")
    print(f"  Training pairs: {metadata.get('pair_count', '?')}")


def query_similar(ticker, metric, t2i, i2t, embeddings, top_n=10, metadata=None):
    if ticker in t2i:
        similar = metric.most_similar(ticker, t2i, i2t, embeddings, top_n=top_n)
        print(f"\n  Top {top_n} most similar to {ticker} [IN VOCAB]:")
    else:
        # OOV estimation
        print(f"\n  '{ticker}' not in vocab — estimating via OOV...")
        result = estimate_oov(ticker, embeddings, t2i, metadata)
        if result is None or result[0] is None:
            return
        oov_embedding, oov_meta = result
        print(f"  OOV confidence: {oov_meta.confidence}")
        print(f"  Co-movement days: {oov_meta.co_movement_days}")
        print(f"  Top co-movers: {oov_meta.top_5_comovers}")
        if oov_meta.confidence == "low":
            print(f"  ⚠ LOW CONFIDENCE: no strong correlation with vocab stocks.")
            print(f"    Results below are unreliable — this stock may not fit this model's universe.")

        # Find similar using OOV embedding
        similar = []
        for t, idx in t2i.items():
            sim = metric.compute(oov_embedding, embeddings[idx])
            similar.append((t, sim))
        similar.sort(key=lambda x: -x[1])
        similar = similar[:top_n]
        print(f"\n  Top {top_n} most similar to {ticker} [OOV ESTIMATED]:")

    for rank, (t, s) in enumerate(similar, 1):
        bar = "█" * int(max(0, s) * 30)
        print(f"    {rank:2d}. {t:8s}  {s:+.4f}  {bar}")


def compare_stocks(ticker_a, ticker_b, metric, t2i, embeddings, metadata=None):
    emb_a = _get_embedding(ticker_a, embeddings, t2i, metadata, metric)
    emb_b = _get_embedding(ticker_b, embeddings, t2i, metadata, metric)
    if emb_a is None or emb_b is None:
        return
    sim = metric.compute(emb_a, emb_b)
    label_a = f"{ticker_a}" + (" [OOV]" if ticker_a not in t2i else "")
    label_b = f"{ticker_b}" + (" [OOV]" if ticker_b not in t2i else "")
    print(f"\n  {label_a} vs {label_b}: {sim:+.4f}")
    if sim > 0.8:
        print("  → Very similar behavior (strong co-movement)")
    elif sim > 0.5:
        print("  → Moderate similarity")
    elif sim > 0.2:
        print("  → Weak similarity")
    elif sim > -0.2:
        print("  → No meaningful relationship")
    else:
        print("  → Tend to move in opposite directions")


def _get_embedding(ticker, embeddings, t2i, metadata, metric):
    """Get embedding for a ticker, using OOV if needed."""
    if ticker in t2i:
        return embeddings[t2i[ticker]]
    print(f"  '{ticker}' not in vocab — estimating via OOV...")
    result = estimate_oov(ticker, embeddings, t2i, metadata)
    if result is None or result[0] is None:
        return None
    oov_embedding, oov_meta = result
    print(f"  OOV confidence: {oov_meta.confidence}")
    return oov_embedding


def print_matrix(metric, t2i, i2t, embeddings):
    tickers = sorted(t2i.keys())
    if len(tickers) > 30:
        print(f"\n  Matrix too large ({len(tickers)} stocks). Showing first 30.")
        tickers = tickers[:30]

    matrix = metric.compute_matrix(embeddings)

    print(f"\n  {'':8s}", end="")
    for t in tickers:
        print(f"{t:>8s}", end="")
    print()
    for t in tickers:
        i = t2i[t]
        print(f"  {t:8s}", end="")
        for t2 in tickers:
            j = t2i[t2]
            print(f"{matrix[i][j]:>8.3f}", end="")
        print()


def interactive_mode(metric, t2i, i2t, embeddings, metadata):
    print("\n  Interactive mode. Commands:")
    print("    <TICKER>          — top 10 similar stocks (OOV auto-estimated)")
    print("    <TICKER> <N>      — top N similar stocks")
    print("    <TICKER> <TICKER> — compare two stocks")
    print("    list              — show all tickers")
    print("    matrix            — similarity matrix")
    print("    quit              — exit")

    while True:
        try:
            user_input = input("\n  > ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in ("QUIT", "EXIT", "Q"):
            break
        if user_input == "LIST":
            tickers = sorted(t2i.keys())
            for i, t in enumerate(tickers):
                print(f"    {t:8s}", end="")
                if (i + 1) % 8 == 0:
                    print()
            print()
            continue
        if user_input == "MATRIX":
            print_matrix(metric, t2i, i2t, embeddings)
            continue

        parts = user_input.split()
        if len(parts) == 1:
            query_similar(parts[0], metric, t2i, i2t, embeddings, top_n=10, metadata=metadata)
        elif len(parts) == 2 and parts[1].isdigit():
            query_similar(parts[0], metric, t2i, i2t, embeddings, top_n=int(parts[1]), metadata=metadata)
        elif len(parts) == 2:
            compare_stocks(parts[0], parts[1], metric, t2i, embeddings, metadata=metadata)
        else:
            print("  Unknown command. Try a ticker name, two tickers, or 'list'.")


def main():
    parser = argparse.ArgumentParser(description="Query a trained Stock2Vec model")
    parser.add_argument("--model", default="data/models/nasdaq_100/current",
                        help="Path to model directory (default: data/models/nasdaq_100/current)")
    parser.add_argument("--ticker", help="Query similar stocks for this ticker")
    parser.add_argument("--top", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--compare", nargs=2, metavar=("TICKER_A", "TICKER_B"),
                        help="Compare two stocks")
    parser.add_argument("--matrix", action="store_true", help="Print similarity matrix")
    args = parser.parse_args()

    print("=" * 60)
    print("Stock2Vec Model Query")
    print("=" * 60)

    embeddings, t2i, i2t, metadata = load_model(args.model)
    metric = CosineMetric()
    print_model_info(t2i, metadata)

    if args.ticker:
        query_similar(args.ticker, metric, t2i, i2t, embeddings, top_n=args.top, metadata=metadata)
    elif args.compare:
        compare_stocks(args.compare[0], args.compare[1], metric, t2i, embeddings, metadata=metadata)
    elif args.matrix:
        print_matrix(metric, t2i, i2t, embeddings)
    else:
        interactive_mode(metric, t2i, i2t, embeddings, metadata)


if __name__ == "__main__":
    main()
