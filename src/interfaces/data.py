from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import pandas as pd


class IPriceSource(ABC):
    """Interface for fetching OHLCV price data."""

    @abstractmethod
    def fetch(self, tickers: list, start: str, end: str) -> pd.DataFrame:
        """Fetch close prices. Returns DataFrame with tickers as columns, date as index.

        Missing tickers silently omitted, not raised.
        """
        pass


class ICache(ABC):
    """Interface for caching price data."""

    @abstractmethod
    def read(self, key: str) -> Optional[pd.DataFrame]:
        """Read cached data. Returns None if not found."""
        pass

    @abstractmethod
    def write(self, key: str, df: pd.DataFrame) -> None:
        """Write data to cache."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if cache key exists."""
        pass

    @abstractmethod
    def get_last_date(self, key: str) -> Optional[date]:
        """Get the last date in cached data. Returns None if no cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete cached data for a key."""
        pass


class IDataCleaner(ABC):
    """Interface for cleaning price data."""

    @abstractmethod
    def clean(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Returns cleaned prices. Never raises on individual ticker issues.

        Logs warnings for anomalies.
        """
        pass
