"""YFinance price source with batching, retry, and rate limiting."""
import logging
import time
from typing import List

import pandas as pd
import yfinance as yf

from ...interfaces.data import IPriceSource

logger = logging.getLogger(__name__)


class YFinancePriceSource(IPriceSource):
    """Fetches close prices from Yahoo Finance.

    Handles batching, retry, and rate limiting.
    Never raises on individual ticker failure — logs and continues.
    """

    def __init__(self, batch_size: int, retry_count: int,
                 retry_wait_secs: int, rate_limit_secs: float):
        self._batch_size = batch_size
        self._retry_count = retry_count
        self._retry_wait_secs = retry_wait_secs
        self._rate_limit_secs = rate_limit_secs

    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """Fetch close prices for tickers in batches.

        Returns DataFrame with tickers as columns, date as index.
        Failed tickers silently omitted.
        """
        all_prices = {}
        failed = []

        batches = [tickers[i:i + self._batch_size]
                   for i in range(0, len(tickers), self._batch_size)]

        for batch_idx, batch in enumerate(batches):
            logger.info("Fetching batch %d/%d (%d tickers)",
                        batch_idx + 1, len(batches), len(batch))

            for ticker in batch:
                prices = self._fetch_single(ticker, start, end)
                if prices is not None and len(prices) > 1:
                    all_prices[ticker] = prices
                else:
                    failed.append(ticker)

            if batch_idx < len(batches) - 1:
                time.sleep(self._rate_limit_secs)

        if failed:
            logger.warning("Failed to fetch %d tickers: %s", len(failed), failed[:20])

        logger.info("Fetched %d/%d tickers successfully", len(all_prices), len(tickers))
        return pd.DataFrame(all_prices)

    def _fetch_single(self, ticker: str, start: str, end: str) -> pd.Series:
        """Fetch a single ticker with retry logic."""
        for attempt in range(self._retry_count):
            try:
                df = yf.download(
                    ticker, start=start, end=end,
                    interval="1d", progress=False, auto_adjust=True
                )
                if df.empty or len(df) <= 1:
                    return None

                if isinstance(df.columns, pd.MultiIndex):
                    return df["Close"][ticker]
                return df["Close"]

            except Exception as e:
                logger.warning("Attempt %d/%d failed for %s: %s",
                               attempt + 1, self._retry_count, ticker, e)
                if attempt < self._retry_count - 1:
                    time.sleep(self._retry_wait_secs)

        logger.error("All %d attempts failed for %s", self._retry_count, ticker)
        return None
