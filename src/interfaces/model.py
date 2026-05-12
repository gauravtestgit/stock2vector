from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List

import numpy as np


class IEmbeddingModel(ABC):
    """Interface for embedding models (Word2Vec skip-gram)."""

    @abstractmethod
    def initialise(self, vocab_size: int, embed_dim: int, seed: int) -> None:
        """Randomly initialise weight matrices."""
        pass

    @abstractmethod
    def forward(self, center_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass. Returns (h, scores, probs)."""
        pass

    @abstractmethod
    def backward(self, h: np.ndarray, probs: np.ndarray, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Backward pass. Returns (dW2, dh).

        CRITICAL: uses probs[target_idx] -= 1, NEVER = -1.
        """
        pass

    @abstractmethod
    def update(self, center_idx: int, dW2: np.ndarray, dh: np.ndarray, lr: float) -> None:
        """Apply gradient update to weights."""
        pass

    @abstractmethod
    def get_embeddings(self) -> np.ndarray:
        """Returns embedding table (W1 copy)."""
        pass


class ITrainingObserver(ABC):
    """Interface for training side-effect observers."""

    @abstractmethod
    def on_epoch_start(self, epoch: int) -> None:
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        pass

    @abstractmethod
    def on_training_complete(self, final_loss: float) -> None:
        pass

    @abstractmethod
    def on_checkpoint(self, epoch: int, W1: np.ndarray) -> None:
        pass

    @abstractmethod
    def on_error(self, error: Exception) -> None:
        pass


class ILRStrategy(ABC):
    """Interface for learning rate selection."""

    @abstractmethod
    def select(self, model: IEmbeddingModel, pairs: List[Tuple[int, int]]) -> float:
        """Select optimal learning rate. Returns float."""
        pass


class IGradientClipper(ABC):
    """Interface for gradient clipping."""

    @abstractmethod
    def clip(self, gradient: np.ndarray) -> np.ndarray:
        """Returns clipped gradient array."""
        pass


class IEarlyStopper(ABC):
    """Interface for early stopping."""

    @abstractmethod
    def should_stop(self, loss_history: List[float]) -> bool:
        """Returns True if training should stop."""
        pass


class IModelPersistence(ABC):
    """Interface for saving and loading models."""

    @abstractmethod
    def save(self, model: IEmbeddingModel, vocab: dict, metadata: dict, path: Path) -> None:
        """Save model weights, vocab, and metadata to path."""
        pass

    @abstractmethod
    def load(self, path: Path) -> Tuple[np.ndarray, dict, dict]:
        """Load model. Returns (embeddings, vocab, metadata)."""
        pass
