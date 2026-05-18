"""
Full end-to-end Stock2Vec flow using spec-compliant pipeline:
  UniverseManager → IncrementalDownloader → StandardCleaner
  → LogReturnsProcessor → VolatilityThresholdStrategy → CoMovementPairGenerator
  → Trainer → NumpyPersistence → CosineMetric queries

All parameters driven by config/settings.py and config/hyperparams.yaml.

Usage:
    python run_full_flow.py                                    # reads from config/run.yaml
    python run_full_flow.py --run-name my_model --stock-file nasdaq_100.csv --start 2024-04-01 --end 2026-05-10
    python run_full_flow.py --run-name my_model --stock-file nasdaq_100.csv --anchors market_etfs,sector_etfs
"""
import sys
import os
import logging
import argparse
sys.path.insert(0, os.path.dirname(__file__))

import yaml

from config.settings import settings
from src.universe.universe import UniverseManager
from src.implementations.pipeline.sources import YFinancePriceSource
from src.implementations.pipeline.cache import ParquetCache
from src.implementations.pipeline.downloader import IncrementalDownloader
from src.implementations.pipeline.cleaner import StandardCleaner
from src.implementations.pipeline.returns import LogReturnsProcessor, MarketNeutralReturnsProcessor
from src.implementations.pipeline.threshold import VolatilityThresholdStrategy
from src.implementations.pipeline.pairs import CoMovementPairGenerator
from src.implementations.pipeline.correlation_pairs import CorrelationPairGenerator
from src.implementations.training.word2vec import Word2VecModel
from src.implementations.training.trainer import Trainer
from src.implementations.training.clipping import GradientClipper
from src.implementations.training.lr_strategy import FixedLRStrategy, ScanLRStrategy
from src.implementations.training.stopping import PatienceEarlyStopper
from src.implementations.training.observers import (
    LossCsvObserver,
    StatusFileObserver,
    CheckpointObserver,
    DashboardNotifierObserver,
)
from src.implementations.training.persistence import NumpyPersistence
from src.implementations.embeddings.metrics import CosineMetric

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_run_config(args):
    """Load run config from CLI args, falling back to config/run.yaml."""
    run_yaml_path = os.path.join(os.path.dirname(__file__), "config", "run.yaml")

    # Defaults from run.yaml if it exists
    defaults = {}
    if os.path.exists(run_yaml_path):
        with open(run_yaml_path) as f:
            defaults = yaml.safe_load(f) or {}

    # CLI overrides run.yaml
    run_name = args.run_name or defaults.get("run_name", "default_run")
    stock_files = ([args.stock_file] if args.stock_file
                   else defaults.get("stock_files", ["nasdaq_100.csv"]))
    anchor_groups = (args.anchors.split(",") if args.anchors
                     else defaults.get("anchor_groups", ["market_etfs", "sector_etfs"]))
    start = args.start or defaults.get("start", "2024-04-01")
    end = args.end or defaults.get("end", "2026-05-10")

    return run_name, stock_files, anchor_groups, start, end


def main():
    parser = argparse.ArgumentParser(description="Run full Stock2Vec training pipeline")
    parser.add_argument("--run-name", help="Name for this run")
    parser.add_argument("--stock-file", help="Stock file (e.g., nasdaq_100.csv)")
    parser.add_argument("--anchors", help="Comma-separated anchor groups (e.g., market_etfs,sector_etfs)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--quiet", action="store_true", help="Skip verbose output (steps 7-9)")
    args = parser.parse_args()

    RUN_NAME, STOCK_FILES, ANCHOR_GROUPS, START, END = load_run_config(args)

    # ── 1. Build universe ────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Building universe")
    print("=" * 60)

    manager = UniverseManager()
    print(f"  Available stock files:  {manager.list_stock_files()}")
    print(f"  Available anchor groups: {manager.list_anchor_groups()}")
    print(f"  Selected stock files:   {STOCK_FILES}")
    print(f"  Selected anchor groups: {ANCHOR_GROUPS}")

    tickers = manager.build_universe(STOCK_FILES, ANCHOR_GROUPS)
    print(f"  Universe: {len(tickers)} tickers")

    print(f"  Settings:")
    print(f"    embed_dim:  {settings.embed_dim}")
    print(f"    epochs:     {settings.epochs}")
    print(f"    lr:         {settings.lr}")
    print(f"    clip:       {settings.gradient_clip}")
    print(f"    patience:   {settings.early_stop_patience}")
    print(f"    seed:       {settings.random_seed}")

    # ── 2. Download prices ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Downloading prices (incremental, cached)")
    print("=" * 60)

    source = YFinancePriceSource(
        batch_size=settings.download_batch_size,
        retry_count=settings.download_retry_count,
        retry_wait_secs=settings.download_retry_wait_secs,
        rate_limit_secs=settings.download_rate_limit_secs,
    )
    cache = ParquetCache(settings.raw_dir)
    downloader = IncrementalDownloader(source, cache)

    prices = downloader.download(
        market=RUN_NAME,
        tickers=tickers,
        start=START,
        end=END,
        force_refresh=True,
    )

    print(f"  Prices shape: {prices.shape}")
    print(f"  Date range:   {prices.index.min()} → {prices.index.max()}")
    print(f"  Tickers:      {list(prices.columns)}")

    # ── 3. Clean data ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Cleaning data")
    print("=" * 60)

    anomaly_log = os.path.join(settings.quality_dir, "anomaly_log.csv")
    cleaner = StandardCleaner(
        min_history_pct=settings.min_history_pct,
        min_price=settings.min_price,
        anomaly_log_path=anomaly_log,
    )

    cleaned = cleaner.clean(prices)
    dropped = set(prices.columns) - set(cleaned.columns)

    print(f"  Before: {len(prices.columns)} tickers")
    print(f"  After:  {len(cleaned.columns)} tickers")
    if dropped:
        print(f"  Dropped: {dropped}")

    # ── 4. Compute returns, threshold, pairs ─────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Computing returns, threshold, and training pairs")
    print("=" * 60)

    if settings.returns_mode == "market_neutral":
        # Ensure benchmark is in the universe
        benchmark = settings.benchmark_ticker
        if benchmark not in tickers:
            tickers.append(benchmark)
            tickers = sorted(tickers)
            print(f"  Added benchmark '{benchmark}' to universe")
        returns_processor = MarketNeutralReturnsProcessor(benchmark)
        print(f"  Returns mode: market_neutral (benchmark={benchmark})")
    else:
        returns_processor = LogReturnsProcessor()
        print(f"  Returns mode: log")

    threshold_strategy = VolatilityThresholdStrategy(settings.threshold_multiplier)

    if settings.pair_mode == "correlation":
        pair_generator = CorrelationPairGenerator(
            window_days=settings.correlation_window_days,
            min_correlation=settings.correlation_min,
            step_days=settings.correlation_step_days,
            max_pairs_per_window=settings.max_pairs_per_day,
            seed=settings.random_seed,
        )
        print(f"  Pair mode: correlation (window={settings.correlation_window_days}, min_corr={settings.correlation_min}, step={settings.correlation_step_days})")
    else:
        pair_generator = CoMovementPairGenerator(
            max_pairs_per_day=settings.max_pairs_per_day,
            seed=settings.random_seed,
        )
        print(f"  Pair mode: comovement")

    returns = returns_processor.compute(cleaned)
    threshold = threshold_strategy.compute(returns)

    vocab_tickers = sorted(cleaned.columns.tolist())
    t2i = {t: i for i, t in enumerate(vocab_tickers)}
    i2t = {i: t for i, t in enumerate(vocab_tickers)}

    all_pairs = pair_generator.generate(returns, t2i, threshold)

    print(f"  Returns shape: {returns.shape}")
    print(f"  Threshold:     {threshold:.6f}")
    print(f"  Vocab size:    {len(t2i)}")
    print(f"  Training pairs: {len(all_pairs)}")
    print(f"  Tickers: {vocab_tickers}")

    if not all_pairs:
        print("\n  No training pairs generated. Try a wider date range.")
        return

    # ── 5. Train model ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Training Word2Vec model")
    print("=" * 60)

    model_output_dir = os.path.join(settings.models_dir, RUN_NAME)

    model = Word2VecModel(mode=settings.model_mode, num_negatives=settings.num_negatives)
    model.initialise(
        vocab_size=len(t2i),
        embed_dim=settings.embed_dim,
        seed=settings.random_seed,
    )

    print(f"  Model mode: {settings.model_mode}")
    if settings.model_mode == "negative_sampling":
        print(f"  Num negatives: {settings.num_negatives}")

    if settings.lr == "auto":
        lr_strategy = ScanLRStrategy(
            scan_values=settings.lr_scan_values,
            scan_epochs=settings.lr_scan_epochs,
        )
    else:
        lr_strategy = FixedLRStrategy(float(settings.lr))

    trainer = Trainer(
        model=model,
        lr_strategy=lr_strategy,
        gradient_clipper=GradientClipper(settings.gradient_clip),
        early_stopper=PatienceEarlyStopper(settings.early_stop_patience),
        observers=[
            LossCsvObserver(model_output_dir),
            StatusFileObserver(model_output_dir, market=RUN_NAME, total_epochs=settings.epochs),
            CheckpointObserver(model_output_dir, checkpoint_every=settings.checkpoint_every_epochs),
            DashboardNotifierObserver(model_output_dir),
        ],
        persistence=NumpyPersistence(),
    )

    loss_history = trainer.train(
        all_pairs,
        epochs=settings.epochs,
        checkpoint_every=settings.checkpoint_every_epochs,
    )

    print(f"\n  First epoch loss: {loss_history[0]:.4f}")
    print(f"  Final epoch loss: {loss_history[-1]:.4f}")
    print(f"  Epochs trained:   {len(loss_history)}")

    # ── 6. Save model ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Saving model")
    print("=" * 60)

    persistence = NumpyPersistence()
    metadata = {
        "run_name": RUN_NAME,
        "stock_files": STOCK_FILES,
        "anchor_groups": ANCHOR_GROUPS,
        "embed_dim": settings.embed_dim,
        "epochs": len(loss_history),
        "final_loss": float(loss_history[-1]),
        "pair_count": len(all_pairs),
        "threshold": threshold,
        "vocab_size": len(t2i),
        "start": START,
        "end": END,
        "seed": settings.random_seed,
        "gradient_clip": settings.gradient_clip,
    }
    current_dir = os.path.join(model_output_dir, "current")
    persistence.save(model, t2i, metadata, current_dir)

    print(f"  Saved to: {current_dir}")

    # ── 7. Load and verify ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Load model and verify")
    print("=" * 60)

    loaded_emb, loaded_vocab, loaded_meta = persistence.load(current_dir)
    print(f"  Embeddings shape: {loaded_emb.shape}")
    print(f"  Vocab size:       {len(loaded_vocab)}")
    print(f"  Final loss:       {loaded_meta['final_loss']:.4f}")

    # ── 8. Cosine similarity queries ─────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: Cosine similarity queries")
    print("=" * 60)

    metric = CosineMetric()
    embeddings = model.get_embeddings()

    print("\n  Most similar stocks:")
    for ticker in vocab_tickers:
        similar = metric.most_similar(ticker, t2i, i2t, embeddings, top_n=3)
        result_str = "  ".join([f"{t}({s:.3f})" for t, s in similar])
        print(f"    {ticker:6s} → {result_str}")

    # Similarity matrix
    print(f"\n  Similarity matrix:")
    sim_matrix = metric.compute_matrix(embeddings)

    print(f"  {'':8s}", end="")
    for t in vocab_tickers:
        print(f"{t:>8s}", end="")
    print()
    for i, t in enumerate(vocab_tickers):
        print(f"  {t:8s}", end="")
        for j in range(len(vocab_tickers)):
            print(f"{sim_matrix[i][j]:>8.3f}", end="")
        print()

    # Specific stock query
    print(f"\n  Query: AAPL top 5")
    similar = metric.most_similar("AAPL", t2i, i2t, embeddings, top_n=5)
    for rank, (t, s) in enumerate(similar, 1):
        print(f"    {rank}. {t:6s}  similarity: {s:.4f}")

    # OOV query
    print(f"\n  Query: AMD (out of vocab)")
    similar = metric.most_similar("AMD", t2i, i2t, embeddings, top_n=5)
    if not similar:
        print(f"    AMD not in vocab — OOV estimation not yet implemented")

    # ── 9. List output files ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9: Output files")
    print("=" * 60)

    for root, dirs, files in os.walk(settings.data_dir):
        for f in files:
            filepath = os.path.join(root, f)
            size = os.path.getsize(filepath)
            print(f"  {filepath} ({size:,} bytes)")


if __name__ == "__main__":
    main()
