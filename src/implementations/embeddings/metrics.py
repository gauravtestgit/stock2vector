import numpy as np
from typing import Dict, List, Tuple
from ...interfaces.embeddings import ISimilarityMetric


class CosineMetric(ISimilarityMetric):
    """Cosine similarity metric for embedding vectors."""

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 if either vector has zero norm.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def compute_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute full pairwise cosine similarity matrix (vectorised).

        Returns (V x V) matrix. Diagonal is 1.0.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalised = embeddings / norms
        return normalised @ normalised.T

    def most_similar(self, ticker: str, t2i: Dict[str, int], i2t: Dict[int, str],
                     embeddings: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
        """Find top_n most similar stocks to ticker.

        Returns empty list if ticker not in vocab.
        """
        if ticker not in t2i:
            return []

        idx = t2i[ticker]
        vec = embeddings[idx]
        scores = []

        for i in range(len(embeddings)):
            if i == idx:
                continue
            sim = self.compute(vec, embeddings[i])
            scores.append((i2t[i], sim))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_n]
