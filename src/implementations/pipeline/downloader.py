"""Incremental downloader — fetches only new data since last cached date."""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict

import pandas as pd

from ...interfaces.data import IPriceSource, ICache

logger = logging.getLogger(__name__)


class IncrementalDownloader:
    """Downloads price data incrementally.

    On first run: fetches full history from start date.
    On subsequent runs: fetches only since get_last_date().
    Appends new data to existing cache.
    force_refresh=True: deletes cache and re-downloads everything.

    Depends on IPriceSource and ICache via constructor.
    Does not know whether source is yfinance or anything else.
    """

    def __init__(self, price_source: IPriceSource, cache: ICache):
        self._source = price_source
        self._cache = cache

    def download(self, market: str, tickers: list, start: str,
                 end: str = None, force_refresh: bool = False) -> pd.DataFrame:
        """Download prices, using cache when available.

        Args:
            market: cache key (e.g., "us", "nzx", "asx", "global")
            tickers: list of ticker symbols
            start: start date string (used on first download or force_refresh)
            end: end date string (default: today)
            force_refresh: if True, delete cache and re-download everything

        Returns:
            DataFrame with tickers as columns, date as index.
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        if force_refresh:
            logger.info("Force refresh: deleting cache for '%s'", market)
            self._cache.delete(market)

        last_date = self._cache.get_last_date(market)

        if last_date is None:
            logger.info("No cache for '%s', fetching full history from %s", market, start)
            fetch_start = start
        else:
            fetch_start = str(last_date)
            logger.info("Cache found for '%s', fetching from %s", market, fetch_start)

        new_data = self._source.fetch(tickers, fetch_start, end)

        if new_data.empty:
            logger.warning("No new data fetched for '%s'", market)
            cached = self._cache.read(market)
            return cached if cached is not None else pd.DataFrame()

        existing = self._cache.read(market)
        if existing is not None and not existing.empty:
            # Add any new tickers not in existing cache
            for col in new_data.columns:
                if col not in existing.columns:
                    existing[col] = pd.NA

            # Append new rows, drop duplicates by index
            combined = pd.concat([existing, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = new_data

        self._cache.write(market, combined)
        self._write_meta(market, combined, tickers, new_data)

        logger.info("Download complete for '%s': %d rows, %d tickers",
                     market, len(combined), len(combined.columns))
        return combined

    def _write_meta(self, market: str, combined: pd.DataFrame,
                    requested: list, fetched: pd.DataFrame) -> None:
        """Write download_meta.json after each run."""
        meta_dir = os.path.join(self._cache._base_dir, market)
        os.makedirs(meta_dir, exist_ok=True)
        meta_path = os.path.join(meta_dir, "download_meta.json")

        failed = [t for t in requested if t not in combined.columns]
        meta = {
            "last_downloaded_date": str(combined.index.max()) if not combined.empty else None,
            "ticker_count": len(combined.columns),
            "failed_tickers": failed,
            "failed_count": len(failed),
            "total_rows": len(combined),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Download meta written: %s", meta_path)
