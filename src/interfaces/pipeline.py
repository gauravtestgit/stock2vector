from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Tuple


class IReturnsProcessor(ABC):
    """Interface for computing returns from price data."""

    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Returns log returns DataFrame same shape minus first row.

        Preconditions: prices is a DataFrame with tickers as columns, date as index.
        Postconditions: returns DataFrame with same columns, one fewer row.
        """
        pass


class IThresholdStrategy(ABC):
    """Interface for computing co-movement threshold."""

    @abstractmethod
    def compute(self, returns: pd.DataFrame) -> float:
        """Returns co-movement threshold as positive float.

        Preconditions: returns is a DataFrame of daily/weekly returns.
        Postconditions: returns a positive float threshold value.
        """
        pass


class IPairGenerator(ABC):
    """Interface for generating training pairs from returns data."""

    @abstractmethod
    def generate(self, returns: pd.DataFrame, t2i: Dict[str, int], threshold: float) -> List[Tuple[int, int]]:
        """Returns list of (center_idx, target_idx) integer tuples.

        All pairs are bidirectional. Capped per day.
        Preconditions: returns DataFrame, t2i mapping, positive threshold.
        Postconditions: list of (int, int) tuples, bidirectional.
        """
        pass
