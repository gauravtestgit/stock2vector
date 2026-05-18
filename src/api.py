"""Stock2Vec API layer.

Provides a clean interface for the dashboard and any future consumers.
Handles training (async via threading), querying, and model management.
"""
import os
import json
import threading
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import settings
from src.universe.universe import UniverseManager
from src.implementations.pipeline.sources import YFinancePriceSource
from src.implementations.pipeline.cache import ParquetCache
from src.implementations.pipeline.downloader import IncrementalDownloader
from src.implementations.pipeline.cleaner import StandardCleaner
from src.implementations.pipeline.returns import LogReturnsProcessor, MarketNeutralReturnsProcessor
from src.implementations.pipeline.threshold import VolatilityThresholdStrategy
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
from src.implementations.embeddings.metrics import CosineMetric
from src.implementations.embeddings.oov import CorrelationOOV

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    run_name: str
    stock_file: str
    anchor_groups: List[str] = field(default_factory=lambda: ["market_etfs", "sector_etfs"])
    start: str = "2024-04-01"
    end: str = "2026-05-10"
    epochs: int = 200
    lr: float = 0.01
    min_corr: float = 0.85


@dataclass
class TrainStatus:
    status: str = "idle"  # idle, running, complete, failed
    epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    final_loss: float = 0.0
    error: str = ""


class Stock2VecAPI:
    """Single API class for all Stock2Vec operations."""

    def __init__(self):
        self._persistence = NumpyPersistence()
        self._metric = CosineMetric()
        self._training_thread: Optional[threading.Thread] = None
        self._train_status = TrainStatus()

    # ── Model Management ─────────────────────────────────────

    def list_models(self) -> List[dict]:
        """List all available trained models."""
        models_dir = settings.models_dir
        options = []
        if os.path.exists(models_dir):
            for name in sorted(os.listdir(models_dir)):
                current_path = os.path.join(models_dir, name, "current")
                if os.path.exists(os.path.join(current_path, "W1.npy")):
                    try:
                        _, _, meta = self._persistence.load(current_path)
                        options.append({
                            "name": name,
                            "path": current_path,
                            "vocab_size": meta.get("vocab_size", "?"),
                            "final_loss": meta.get("final_loss", 0),
                            "epochs": meta.get("epochs", "?"),
                            "start": meta.get("start", "?"),
                            "end": meta.get("end", "?"),
                        })
                    except Exception:
                        options.append({"name": name, "path": current_path})
        return options

    def load_model(self, model_path: str):
        """Load a model and return embeddings, vocab, metadata."""
        return self._persistence.load(model_path)

    # ── Querying ─────────────────────────────────────────────

    def most_similar(self, ticker: str, embeddings: np.ndarray,
                     t2i: dict, i2t: dict, top_n: int = 10) -> List[Tuple[str, float]]:
        """Find most similar stocks to a ticker."""
        return self._metric.most_similar(ticker, t2i, i2t, embeddings, top_n=top_n)

    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return self._metric.compute(a, b)

    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        return self._metric.compute_matrix(embeddings)

    def estimate_oov(self, ticker: str, embeddings: np.ndarray, t2i: dict,
                     metadata: dict) -> Optional[Tuple[np.ndarray, object]]:
        """Estimate OOV embedding using correlation method."""
        try:
            run_name = metadata.get("run_name", "")
            cache = ParquetCache(settings.raw_dir)
            cached_prices = cache.read(run_name)
            if cached_prices is None or cached_prices.empty:
                return None

            if settings.returns_mode == "market_neutral":
                returns_proc = MarketNeutralReturnsProcessor(settings.benchmark_ticker)
            else:
                returns_proc = LogReturnsProcessor()
            vocab_returns = returns_proc.compute(cached_prices)

            start = metadata.get("start", "2024-04-01")
            end = metadata.get("end", "2026-05-10")
            source = YFinancePriceSource(batch_size=1, retry_count=2,
                                         retry_wait_secs=2, rate_limit_secs=0.5)
            tickers_to_fetch = [ticker]
            if settings.returns_mode == "market_neutral":
                tickers_to_fetch.append(settings.benchmark_ticker)
            oov_prices = source.fetch(tickers_to_fetch, start=start, end=end)
            if oov_prices.empty or ticker not in oov_prices.columns:
                return None

            oov_returns = returns_proc.compute(oov_prices)[ticker]

            strategy = CorrelationOOV(
                window_days=settings.correlation_window_days,
                min_correlation=0.5,
                step_days=settings.correlation_step_days,
            )
            return strategy.estimate(ticker, oov_returns, embeddings, t2i, vocab_returns)
        except Exception as e:
            logger.error("OOV estimation failed for %s: %s", ticker, e)
            return None

    # ── Training ─────────────────────────────────────────────

    def list_stock_files(self) -> List[str]:
        return UniverseManager().list_stock_files()

    def list_anchor_groups(self) -> List[str]:
        return UniverseManager().list_anchor_groups()

    def get_train_status(self, run_name: str) -> TrainStatus:
        """Get training status from file."""
        status_path = os.path.join(settings.models_dir, run_name, "training_status.json")
        if os.path.exists(status_path):
            try:
                with open(status_path) as f:
                    data = json.load(f)
                return TrainStatus(
                    status=data.get("status", "unknown"),
                    epoch=data.get("current_epoch", 0),
                    total_epochs=data.get("total_epochs", 0),
                    current_loss=data.get("current_loss", 0),
                    final_loss=data.get("final_loss", 0),
                    error=data.get("error", ""),
                )
            except Exception:
                pass
        return self._train_status

    def get_loss_history(self, run_name: str) -> Optional[pd.DataFrame]:
        """Read loss history CSV for a run."""
        loss_path = os.path.join(settings.models_dir, run_name, "loss_history.csv")
        if os.path.exists(loss_path):
            try:
                return pd.read_csv(loss_path)
            except Exception:
                pass
        return None

    @property
    def is_training(self) -> bool:
        return self._training_thread is not None and self._training_thread.is_alive()

    def start_training(self, config: TrainConfig) -> bool:
        """Start training in a background thread. Returns False if already training."""
        if self.is_training:
            return False

        self._train_status = TrainStatus(status="running", total_epochs=config.epochs)
        self._training_thread = threading.Thread(
            target=self._run_training, args=(config,), daemon=True
        )
        self._training_thread.start()
        return True

    def _run_training(self, config: TrainConfig):
        """Execute the full training pipeline."""
        try:
            # 1. Universe
            manager = UniverseManager()
            tickers = manager.build_universe([config.stock_file], config.anchor_groups)

            # 2. Download
            source = YFinancePriceSource(
                batch_size=settings.download_batch_size,
                retry_count=settings.download_retry_count,
                retry_wait_secs=settings.download_retry_wait_secs,
                rate_limit_secs=settings.download_rate_limit_secs,
            )
            cache = ParquetCache(settings.raw_dir)
            downloader = IncrementalDownloader(source, cache)
            prices = downloader.download(
                market=config.run_name, tickers=tickers,
                start=config.start, end=config.end, force_refresh=True,
            )

            # 3. Clean
            cleaner = StandardCleaner(
                min_history_pct=settings.min_history_pct,
                min_price=settings.min_price,
            )
            cleaned = cleaner.clean(prices)

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
                min_correlation=config.min_corr,
                step_days=settings.correlation_step_days,
                max_pairs_per_window=settings.max_pairs_per_day,
                seed=settings.random_seed,
            )
            all_pairs = pair_generator.generate(returns, t2i, threshold)

            if not all_pairs:
                self._train_status = TrainStatus(status="failed", error="No pairs generated")
                return

            # 5. Train
            model_output_dir = os.path.join(settings.models_dir, config.run_name)
            model = Word2VecModel(mode=settings.model_mode, num_negatives=settings.num_negatives)
            model.initialise(vocab_size=len(t2i), embed_dim=settings.embed_dim,
                             seed=settings.random_seed)

            trainer = Trainer(
                model=model,
                lr_strategy=FixedLRStrategy(config.lr),
                gradient_clipper=GradientClipper(settings.gradient_clip),
                early_stopper=PatienceEarlyStopper(settings.early_stop_patience),
                observers=[
                    LossCsvObserver(model_output_dir),
                    StatusFileObserver(model_output_dir, market=config.run_name,
                                       total_epochs=config.epochs),
                    CheckpointObserver(model_output_dir,
                                       checkpoint_every=settings.checkpoint_every_epochs),
                    DashboardNotifierObserver(model_output_dir),
                ],
                persistence=self._persistence,
            )

            loss_history = trainer.train(all_pairs, epochs=config.epochs,
                                          checkpoint_every=settings.checkpoint_every_epochs)

            # 6. Save
            meta = {
                "run_name": config.run_name,
                "stock_files": [config.stock_file],
                "anchor_groups": config.anchor_groups,
                "embed_dim": settings.embed_dim,
                "epochs": len(loss_history),
                "final_loss": float(loss_history[-1]),
                "pair_count": len(all_pairs),
                "threshold": threshold,
                "vocab_size": len(t2i),
                "start": config.start,
                "end": config.end,
                "seed": settings.random_seed,
                "gradient_clip": settings.gradient_clip,
            }
            current_dir = os.path.join(model_output_dir, "current")
            self._persistence.save(model, t2i, meta, current_dir)

            self._train_status = TrainStatus(
                status="complete", final_loss=float(loss_history[-1]),
                epoch=len(loss_history), total_epochs=config.epochs,
            )
            logger.info("Training complete: %s (loss=%.4f)", config.run_name, loss_history[-1])

        except Exception as e:
            self._train_status = TrainStatus(status="failed", error=str(e))
            logger.error("Training failed: %s", e, exc_info=True)


# Module-level singleton
api = Stock2VecAPI()
