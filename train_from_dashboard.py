"""Training script launched by the dashboard.

Accepts CLI args for run configuration, runs the full pipeline,
and writes training_status.json for the dashboard to poll.
"""
import sys
import os
import argparse
import logging
sys.path.insert(0, os.path.dirname(__file__))

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
from src.implementations.training.lr_strategy import FixedLRStrategy
from src.implementations.training.stopping import PatienceEarlyStopper
from src.implementations.training.observers import (
    LossCsvObserver, StatusFileObserver, CheckpointObserver, DashboardNotifierObserver,
)
from src.implementations.training.persistence import NumpyPersistence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stock-file", required=True)
    parser.add_argument("--anchor-groups", default="")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--min-corr", type=float, default=0.85)
    args = parser.parse_args()

    run_name = args.run_name
    stock_files = [args.stock_file]
    anchor_groups = [g for g in args.anchor_groups.split(",") if g]
    model_output_dir = os.path.join(settings.models_dir, run_name)

    # 1. Universe
    manager = UniverseManager()
    tickers = manager.build_universe(stock_files, anchor_groups)
    logger.info("Universe: %d tickers", len(tickers))

    # 2. Download
    source = YFinancePriceSource(
        batch_size=settings.download_batch_size,
        retry_count=settings.download_retry_count,
        retry_wait_secs=settings.download_retry_wait_secs,
        rate_limit_secs=settings.download_rate_limit_secs,
    )
    cache = ParquetCache(settings.raw_dir)
    downloader = IncrementalDownloader(source, cache)
    prices = downloader.download(market=run_name, tickers=tickers,
                                 start=args.start, end=args.end, force_refresh=True)
    logger.info("Prices: %s", prices.shape)

    # 3. Clean
    cleaner = StandardCleaner(min_history_pct=settings.min_history_pct, min_price=settings.min_price)
    cleaned = cleaner.clean(prices)
    logger.info("Cleaned: %d tickers", len(cleaned.columns))

    # 4. Returns + pairs
    if settings.returns_mode == "market_neutral":
        returns_proc = MarketNeutralReturnsProcessor(settings.benchmark_ticker)
    else:
        returns_proc = LogReturnsProcessor()

    returns = returns_proc.compute(cleaned)
    threshold_strategy = VolatilityThresholdStrategy(settings.threshold_multiplier)
    threshold = threshold_strategy.compute(returns)

    vocab_tickers = sorted(cleaned.columns.tolist())
    t2i = {t: i for i, t in enumerate(vocab_tickers)}

    pair_generator = CorrelationPairGenerator(
        window_days=settings.correlation_window_days,
        min_correlation=args.min_corr,
        step_days=settings.correlation_step_days,
        max_pairs_per_window=settings.max_pairs_per_day,
        seed=settings.random_seed,
    )
    all_pairs = pair_generator.generate(returns, t2i, threshold)
    logger.info("Pairs: %d", len(all_pairs))

    if not all_pairs:
        logger.error("No pairs generated")
        return

    # 5. Train
    model = Word2VecModel(mode=settings.model_mode, num_negatives=settings.num_negatives)
    model.initialise(vocab_size=len(t2i), embed_dim=settings.embed_dim, seed=settings.random_seed)

    trainer = Trainer(
        model=model,
        lr_strategy=FixedLRStrategy(args.lr),
        gradient_clipper=GradientClipper(settings.gradient_clip),
        early_stopper=PatienceEarlyStopper(settings.early_stop_patience),
        observers=[
            LossCsvObserver(model_output_dir),
            StatusFileObserver(model_output_dir, market=run_name, total_epochs=args.epochs),
            CheckpointObserver(model_output_dir, checkpoint_every=settings.checkpoint_every_epochs),
            DashboardNotifierObserver(model_output_dir),
        ],
        persistence=NumpyPersistence(),
    )

    loss_history = trainer.train(all_pairs, epochs=args.epochs,
                                 checkpoint_every=settings.checkpoint_every_epochs)

    # 6. Save
    persistence = NumpyPersistence()
    meta = {
        "run_name": run_name,
        "stock_files": stock_files,
        "anchor_groups": anchor_groups,
        "embed_dim": settings.embed_dim,
        "epochs": len(loss_history),
        "final_loss": float(loss_history[-1]),
        "pair_count": len(all_pairs),
        "threshold": threshold,
        "vocab_size": len(t2i),
        "start": args.start,
        "end": args.end,
        "seed": settings.random_seed,
        "gradient_clip": settings.gradient_clip,
    }
    current_dir = os.path.join(model_output_dir, "current")
    persistence.save(model, t2i, meta, current_dir)
    logger.info("Model saved to %s", current_dir)


if __name__ == "__main__":
    main()
