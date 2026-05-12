"""Standard data cleaner for price data."""
import csv
import logging
import os

import pandas as pd

from ...interfaces.data import IDataCleaner

logger = logging.getLogger(__name__)


class StandardCleaner(IDataCleaner):
    """Cleans price data per spec requirements.

    - Forward fill maximum N consecutive missing days
    - Drop ticker if fewer than min_history_pct trading days have data
    - Drop ticker if average price below min_price
    - Drop ticker if entire series is NaN
    - Log anomalies (single day absolute return > anomaly_threshold) to CSV
    - Does NOT remove anomalies — user decides via dashboard
    """

    def __init__(self, min_history_pct: float, min_price: float,
                 max_ffill_days: int = 5, anomaly_threshold: float = 0.25,
                 anomaly_log_path: str = None):
        self._min_history_pct = min_history_pct
        self._min_price = min_price
        self._max_ffill_days = max_ffill_days
        self._anomaly_threshold = anomaly_threshold
        self._anomaly_log_path = anomaly_log_path

    def clean(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Clean price data. Returns cleaned DataFrame."""
        if prices.empty:
            return prices

        original_count = len(prices.columns)
        cleaned = prices.copy()

        # Forward fill (max N consecutive days)
        cleaned = cleaned.ffill(limit=self._max_ffill_days)

        # Drop tickers with entire NaN series
        all_nan = cleaned.columns[cleaned.isna().all()]
        if len(all_nan) > 0:
            logger.warning("Dropping %d tickers with all NaN: %s", len(all_nan), list(all_nan)[:10])
            cleaned = cleaned.drop(columns=all_nan)

        # Drop tickers with insufficient history
        total_days = len(cleaned)
        if total_days > 0:
            valid_pct = cleaned.notna().sum() / total_days
            insufficient = valid_pct[valid_pct < self._min_history_pct].index.tolist()
            if insufficient:
                logger.warning("Dropping %d tickers with < %.0f%% history: %s",
                               len(insufficient), self._min_history_pct * 100, insufficient[:10])
                cleaned = cleaned.drop(columns=insufficient)

        # Drop tickers with average price below minimum
        if not cleaned.empty:
            avg_prices = cleaned.mean()
            low_price = avg_prices[avg_prices < self._min_price].index.tolist()
            if low_price:
                logger.warning("Dropping %d tickers with avg price < %.2f: %s",
                               len(low_price), self._min_price, low_price[:10])
                cleaned = cleaned.drop(columns=low_price)

        # Log anomalies (don't remove them)
        if not cleaned.empty:
            self._log_anomalies(cleaned)

        # Drop remaining NaN rows
        cleaned = cleaned.dropna(how="all")

        logger.info("Cleaning complete: %d -> %d tickers, %d rows",
                     original_count, len(cleaned.columns), len(cleaned))
        return cleaned

    def _log_anomalies(self, prices: pd.DataFrame) -> None:
        """Detect and log single-day anomalies (absolute return > threshold)."""
        if self._anomaly_log_path is None:
            return

        returns = prices.pct_change().abs()
        anomalies = []

        for ticker in returns.columns:
            for date_idx, value in returns[ticker].items():
                if pd.notna(value) and value > self._anomaly_threshold:
                    anomalies.append({
                        "date": str(date_idx),
                        "ticker": ticker,
                        "return": f"{value:.4f}",
                        "price": f"{prices.loc[date_idx, ticker]:.2f}" if pd.notna(prices.loc[date_idx, ticker]) else "NaN"
                    })

        if anomalies:
            os.makedirs(os.path.dirname(self._anomaly_log_path), exist_ok=True)
            write_header = not os.path.exists(self._anomaly_log_path)
            with open(self._anomaly_log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["date", "ticker", "return", "price"])
                if write_header:
                    writer.writeheader()
                writer.writerows(anomalies)
            logger.info("Logged %d anomalies to %s", len(anomalies), self._anomaly_log_path)
