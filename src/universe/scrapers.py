"""Ticker list loaders from local config files.

Loads S&P 500, NASDAQ 100, ASX 200 from CSV files and NZX from universe.yaml.
Supports both CSV (Symbol column) and YAML (tickers key) formats.
"""
import logging
from typing import List

import pandas as pd
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class TickerLoader:
    """Loads stock ticker lists from local CSV and YAML files."""

    def __init__(self, universe_config_path: str = None, vocab_dir: str = None):
        if universe_config_path is None:
            universe_config_path = str(Path(__file__).parent.parent.parent / "config" / "universe.yaml")
        if vocab_dir is None:
            vocab_dir = str(Path(__file__).parent.parent.parent / "config" / "vocabulary" / "stocks")

        with open(universe_config_path, "r") as f:
            self._config = yaml.safe_load(f)
        self._vocab_dir = Path(vocab_dir)

    def _load_csv_tickers(self, filename: str) -> List[str]:
        """Load tickers from the Symbol column of a CSV file."""
        path = self._vocab_dir / filename
        if not path.exists():
            logger.warning("File not found: %s", path)
            return []
        df = pd.read_csv(path)
        if "Symbol" not in df.columns:
            logger.warning("No 'Symbol' column in %s", path)
            return []
        tickers = df["Symbol"].dropna().astype(str).str.strip().tolist()
        logger.info("Loaded %d tickers from %s", len(tickers), filename)
        return tickers

    def _load_yaml_tickers(self, filename: str) -> List[str]:
        """Load tickers from a YAML file in the vocab directory."""
        path = self._vocab_dir / filename
        if not path.exists():
            logger.warning("File not found: %s", path)
            return []
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        tickers = data.get("tickers", [])
        logger.info("Loaded %d tickers from %s", len(tickers), filename)
        return tickers

    def load_sp500(self) -> List[str]:
        """Load S&P 500 tickers from sp_500.csv."""
        return self._load_csv_tickers("sp_500.csv")

    def load_nasdaq100(self) -> List[str]:
        """Load NASDAQ 100 tickers from nasdaq_100.csv."""
        return self._load_csv_tickers("nasdaq_100.csv")

    def load_asx200(self) -> List[str]:
        """Load ASX 200 tickers from asx_200.csv."""
        return self._load_csv_tickers("asx_200.csv")

    def load_nzx(self) -> List[str]:
        """Load hardcoded NZX stock list from universe.yaml."""
        return list(self._config.get("nzx_stocks", []))
