"""Application settings loaded from environment variables.

All src/ files import from config.settings.
Zero hardcoded paths or values anywhere in src/.
"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration loaded from environment variables and hyperparams.yaml."""

    def __init__(self):
        # Environment
        self.env = os.getenv("STOCK2VEC_ENV", "development")
        self.data_dir = os.getenv("STOCK2VEC_DATA_DIR", "data")
        self.log_level = os.getenv("STOCK2VEC_LOG_LEVEL", "DEBUG")
        self.port = int(os.getenv("STOCK2VEC_PORT", "8050"))
        self.aws_region = os.getenv("AWS_REGION", "ap-southeast-2")
        self.s3_backup_bucket = os.getenv("S3_BACKUP_BUCKET", "")

        # Derived paths
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.models_dir = os.path.join(self.data_dir, "models")
        self.quality_dir = os.path.join(self.data_dir, "quality")
        self.logs_dir = os.path.join(self.data_dir, "logs")

        # Load hyperparameters from YAML
        hyperparams_path = Path(__file__).parent / "hyperparams.yaml"
        if hyperparams_path.exists():
            with open(hyperparams_path, "r") as f:
                hp = yaml.safe_load(f)
        else:
            hp = {}

        # Training hyperparameters
        self.embed_dim = hp.get("embed_dim", 32)
        self.epochs = hp.get("epochs", 300)
        self.lr = hp.get("lr", "auto")
        self.lr_scan_values = hp.get("lr_scan_values", [0.1, 0.01, 0.001, 0.0001])
        self.lr_scan_epochs = hp.get("lr_scan_epochs", 50)
        self.max_pairs_per_day = hp.get("max_pairs_per_day", 10000)
        self.gradient_clip = hp.get("gradient_clip", 1.0)
        self.threshold_multiplier = hp.get("threshold_multiplier", 0.5)
        self.training_start = hp.get("training_start", "2019-01-01")
        self.validation_year = hp.get("validation_year", "2019")
        self.random_seed = hp.get("random_seed", 42)
        self.finetune_window_days = hp.get("finetune_window_days", 90)
        self.finetune_lr_multiplier = hp.get("finetune_lr_multiplier", 0.1)
        self.checkpoint_every_epochs = hp.get("checkpoint_every_epochs", 50)
        self.early_stop_patience = hp.get("early_stop_patience", 20)

        # Model mode
        self.model_mode = hp.get("model_mode", "negative_sampling")
        self.num_negatives = hp.get("num_negatives", 10)

        # Returns mode
        self.returns_mode = hp.get("returns_mode", "market_neutral")
        self.benchmark_ticker = hp.get("benchmark_ticker", "SPY")

        # Pair generation mode
        self.pair_mode = hp.get("pair_mode", "correlation")
        self.correlation_window_days = hp.get("correlation_window_days", 30)
        self.correlation_min = hp.get("correlation_min", 0.7)
        self.correlation_step_days = hp.get("correlation_step_days", 5)

        # Data pipeline
        self.min_history_pct = hp.get("min_history_pct", 0.80)
        self.min_price = hp.get("min_price", 1.00)
        self.download_batch_size = hp.get("download_batch_size", 100)
        self.download_retry_count = hp.get("download_retry_count", 3)
        self.download_retry_wait_secs = hp.get("download_retry_wait_secs", 5)
        self.download_rate_limit_secs = hp.get("download_rate_limit_secs", 1)

        # OOV
        self.oov_high_confidence_days = hp.get("oov_high_confidence_days", 60)
        self.oov_medium_confidence_days = hp.get("oov_medium_confidence_days", 20)

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def host(self) -> str:
        return "0.0.0.0" if self.is_production else "127.0.0.1"


# Module-level singleton
settings = Settings()
