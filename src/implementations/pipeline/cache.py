"""Parquet-based cache for price data."""
import logging
import os
from datetime import date
from typing import Optional

import pandas as pd

from ...interfaces.data import ICache

logger = logging.getLogger(__name__)


class ParquetCache(ICache):
    """Caches DataFrames as Parquet files.

    Key maps to filepath under the base directory.
    e.g., key="us" -> {base_dir}/us/prices.parquet
    """

    def __init__(self, base_dir: str):
        self._base_dir = base_dir

    def _path(self, key: str) -> str:
        return os.path.join(self._base_dir, key, "prices.parquet")

    def read(self, key: str) -> Optional[pd.DataFrame]:
        """Read cached Parquet file. Returns None if not found."""
        path = self._path(key)
        if not os.path.exists(path):
            logger.info("Cache miss: %s", path)
            return None
        logger.info("Cache hit: %s", path)
        return pd.read_parquet(path)

    def write(self, key: str, df: pd.DataFrame) -> None:
        """Write DataFrame to Parquet cache."""
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)
        logger.info("Cache written: %s (%d rows, %d columns)", path, len(df), len(df.columns))

    def exists(self, key: str) -> bool:
        """Check if cache file exists."""
        return os.path.exists(self._path(key))

    def get_last_date(self, key: str) -> Optional[date]:
        """Get the last date in cached data."""
        df = self.read(key)
        if df is None or df.empty:
            return None
        last = df.index.max()
        if hasattr(last, "date"):
            return last.date()
        return last

    def delete(self, key: str) -> None:
        """Delete cached Parquet file."""
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)
            logger.info("Cache deleted: %s", path)
