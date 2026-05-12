from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class OOVMetadata:
    """Metadata about an OOV estimation."""
    method: str
    co_movement_days: int
    confidence: str
    top_5_comovers: List[Tuple[str, float]] = field(default_factory=list)
    data_days_used: int = 0


class ISimilarityMetric(ABC):
    """Interface for computing similarity between embedding vectors."""

    @abstractmethod
    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between two vectors. Returns float in [-1, 1]."""
        pass

    @abstractmethod
    def compute_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity matrix for all embeddings.

        Returns (V x V) matrix where V is vocab size.
        """
        pass

    @abstractmethod
    def most_similar(self, ticker: str, t2i: dict, i2t: dict,
                     embeddings: np.ndarray, top_n: int) -> List[Tuple[str, float]]:
        """Find top_n most similar stocks to ticker.

        Returns list of (ticker, score) tuples sorted by similarity descending.
        """
        pass


class IOOVStrategy(ABC):
    """Interface for estimating embeddings for out-of-vocabulary stocks."""

    @abstractmethod
    def estimate(self, ticker: str, new_returns: pd.Series,
                 embeddings: np.ndarray, t2i: Dict[str, int],
                 vocab_returns: pd.DataFrame,
                 threshold: float) -> Optional[Tuple[np.ndarray, OOVMetadata]]:
        """Estimate embedding for an OOV ticker.

        Args:
            ticker: the OOV ticker symbol
            new_returns: daily log returns for the OOV stock
            embeddings: trained W1 embedding table
            t2i: vocab ticker → index mapping
            vocab_returns: daily log returns for all vocab stocks
            threshold: co-movement threshold

        Returns:
            (embedding_vector, metadata) or None if insufficient data.
        """
        pass
