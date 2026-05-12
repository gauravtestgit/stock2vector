import pandas as pd
from ...interfaces.pipeline import IThresholdStrategy


class FixedThresholdStrategy(IThresholdStrategy):
    """Returns a fixed threshold value. Used for testing or manual override."""

    def __init__(self, threshold: float = 0.005):
        self._threshold = threshold

    def compute(self, returns: pd.DataFrame) -> float:
        """Returns the fixed threshold value, ignoring returns data."""
        return self._threshold


class VolatilityThresholdStrategy(IThresholdStrategy):
    """Computes threshold as multiplier * mean(std per stock).

    Adapts to current market volatility.
    """

    def __init__(self, multiplier: float = 0.5):
        self._multiplier = multiplier

    def compute(self, returns: pd.DataFrame) -> float:
        """Returns multiplier * mean of per-stock standard deviations."""
        return self._multiplier * returns.std().mean()
