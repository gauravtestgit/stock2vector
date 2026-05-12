import numpy as np
import pandas as pd
from ...interfaces.pipeline import IReturnsProcessor


class LogReturnsProcessor(IReturnsProcessor):
    """Computes daily log returns from price data."""

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Returns log returns: ln(price_t / price_t-1), drops first row."""
        return np.log(prices / prices.shift(1)).dropna()


class MarketNeutralReturnsProcessor(IReturnsProcessor):
    """Computes market-neutral log returns by subtracting a benchmark (e.g., SPY).

    For each day: adjusted_return = stock_return - benchmark_return
    This removes market-wide moves so only stock-specific signal remains.
    """

    def __init__(self, benchmark_ticker: str = "SPY"):
        self._benchmark = benchmark_ticker

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Returns market-neutral log returns.

        If benchmark is in the DataFrame, subtracts its return from all stocks.
        If benchmark is not present, falls back to regular log returns.
        The benchmark column is kept in the output (with zero return after subtraction).
        """
        log_returns = np.log(prices / prices.shift(1)).dropna()

        if self._benchmark in log_returns.columns:
            benchmark_returns = log_returns[self._benchmark]
            adjusted = log_returns.subtract(benchmark_returns, axis=0)
            # Keep benchmark in output with its original returns
            # so it can still be used in pair generation
            adjusted[self._benchmark] = benchmark_returns
            return adjusted

        # Fallback: no benchmark available
        return log_returns


class WeeklyResampleProcessor(IReturnsProcessor):
    """Decorator that resamples to weekly prices before computing returns.

    Used for Global model only. Wraps another IReturnsProcessor.
    """

    def __init__(self, wrapped: IReturnsProcessor):
        self._wrapped = wrapped

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Resamples prices to weekly (Friday close) then delegates to wrapped processor."""
        weekly_prices = prices.resample("W-FRI").last().dropna(how="all")
        return self._wrapped.compute(weekly_prices)
