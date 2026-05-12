import numpy as np
from typing import Tuple
from ...interfaces.model import IEmbeddingModel


class Word2VecModel(IEmbeddingModel):
    """Skip-gram Word2Vec model for stock embeddings. Pure numpy.

    Supports two modes:
        - "softmax": full softmax over entire vocab (original, slow for large V)
        - "negative_sampling": binary classification with K negative samples (scales to large V)

    W1 is the embedding table (kept after training).
    W2 is the output projection (discarded after training).
    Loss is NOT computed here — computed in Trainer for monitoring only.
    """

    def __init__(self, mode: str = "negative_sampling", num_negatives: int = 10):
        """
        Args:
            mode: "softmax" or "negative_sampling"
            num_negatives: number of negative samples per pair (only used in negative_sampling mode)
        """
        self._W1 = None
        self._W2 = None
        self._vocab_size = 0
        self._embed_dim = 0
        self._mode = mode
        self._num_negatives = num_negatives
        self._rng = None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e = np.exp(x - x.max())
        return e / e.sum()

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def initialise(self, vocab_size: int, embed_dim: int, seed: int = 42) -> None:
        """Randomly initialise W1 and W2 weight matrices."""
        self._vocab_size = vocab_size
        self._embed_dim = embed_dim
        self._rng = np.random.default_rng(seed)
        self._W1 = self._rng.standard_normal((vocab_size, embed_dim)) * 0.1
        self._W2 = self._rng.standard_normal((embed_dim, vocab_size)) * 0.1

    def forward(self, center_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass. Returns (h, scores, probs).

        In negative_sampling mode, probs is still computed via softmax for loss monitoring,
        but is NOT used in the backward pass.
        """
        h = self._W1[center_idx]
        scores = h @ self._W2
        probs = self._softmax(scores)
        return h, scores, probs

    def backward(self, h: np.ndarray, probs: np.ndarray, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Backward pass. Returns (dW2, dh).

        In softmax mode: CRITICAL uses probs[target_idx] -= 1, NEVER = -1.
        In negative_sampling mode: uses sigmoid gradients on target + K negatives.
        """
        if self._mode == "softmax":
            return self._backward_softmax(h, probs, target_idx)
        else:
            return self._backward_negative_sampling(h, target_idx)

    def _backward_softmax(self, h: np.ndarray, probs: np.ndarray, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Full softmax backward pass."""
        dscores = probs.copy()
        dscores[target_idx] -= 1
        dW2 = np.outer(h, dscores)
        dh = self._W2 @ dscores
        return dW2, dh

    def _backward_negative_sampling(self, h: np.ndarray, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Negative sampling backward pass.

        For the positive pair (center, target):
            score = h · W2[:, target]
            gradient pushes sigmoid(score) toward 1

        For each negative sample k:
            score_neg = h · W2[:, k]
            gradient pushes sigmoid(score_neg) toward 0
        """
        # Sample negatives (excluding target)
        negatives = self._sample_negatives(target_idx)

        # Initialize gradients
        dW2 = np.zeros_like(self._W2)
        dh = np.zeros(self._embed_dim)

        # Positive sample: target
        score_pos = h @ self._W2[:, target_idx]
        sig_pos = self._sigmoid(score_pos)
        grad_pos = sig_pos - 1  # push toward 1

        dW2[:, target_idx] = grad_pos * h
        dh += grad_pos * self._W2[:, target_idx]

        # Negative samples
        for neg_idx in negatives:
            score_neg = h @ self._W2[:, neg_idx]
            sig_neg = self._sigmoid(score_neg)
            grad_neg = sig_neg  # push toward 0

            dW2[:, neg_idx] = grad_neg * h
            dh += grad_neg * self._W2[:, neg_idx]

        return dW2, dh

    def _sample_negatives(self, target_idx: int) -> np.ndarray:
        """Sample K negative indices, excluding the target."""
        negatives = []
        while len(negatives) < self._num_negatives:
            candidate = self._rng.integers(0, self._vocab_size)
            if candidate != target_idx:
                negatives.append(candidate)
        return negatives

    def update(self, center_idx: int, dW2: np.ndarray, dh: np.ndarray, lr: float) -> None:
        """Apply gradient update. Clipping handled externally by IGradientClipper."""
        self._W2 -= lr * dW2
        self._W1[center_idx] -= lr * dh

    def get_embeddings(self) -> np.ndarray:
        """Returns a copy of W1 (the embedding table)."""
        return self._W1.copy()
