import logging
import numpy as np
from typing import List, Tuple
from ...interfaces.model import ILRStrategy, IEmbeddingModel

logger = logging.getLogger(__name__)


class FixedLRStrategy(ILRStrategy):
    """Returns a fixed learning rate."""

    def __init__(self, lr: float):
        self._lr = lr

    def select(self, model: IEmbeddingModel, pairs: List[Tuple[int, int]]) -> float:
        """Returns the fixed learning rate, ignoring model and pairs."""
        return self._lr


class ScanLRStrategy(ILRStrategy):
    """Tests multiple learning rates and selects the best one.

    Runs a short training loop for each candidate LR.
    Selects the largest LR where final loss < starting loss.
    """

    def __init__(self, scan_values: List[float], scan_epochs: int):
        self._scan_values = scan_values
        self._scan_epochs = scan_epochs

    def select(self, model: IEmbeddingModel, pairs: List[Tuple[int, int]]) -> float:
        """Scan candidate LRs and return the best one."""
        best_lr = self._scan_values[-1]  # smallest as fallback

        # Save original weights to restore after each scan
        original_W1 = model.get_embeddings()
        original_W2 = model._W2.copy()

        for lr in self._scan_values:
            # Reset weights
            model._W1[:] = original_W1
            model._W2[:] = original_W2

            # Compute starting loss
            start_loss = self._compute_avg_loss(model, pairs)

            # Train for scan_epochs
            for _ in range(self._scan_epochs):
                indices = np.random.permutation(len(pairs))
                for idx in indices:
                    center_idx, target_idx = pairs[idx]
                    h, _, probs = model.forward(center_idx)
                    dW2, dh = model.backward(h, probs, target_idx)
                    model.update(center_idx, dW2, dh, lr)

            final_loss = self._compute_avg_loss(model, pairs)

            if final_loss < start_loss:
                best_lr = lr
                logger.info("LR scan: lr=%.6f start_loss=%.4f final_loss=%.4f SELECTED", lr, start_loss, final_loss)
                break
            else:
                logger.info("LR scan: lr=%.6f start_loss=%.4f final_loss=%.4f skipped", lr, start_loss, final_loss)

        # Restore original weights
        model._W1[:] = original_W1
        model._W2[:] = original_W2

        logger.info("Selected learning rate: %.6f", best_lr)
        return best_lr

    @staticmethod
    def _compute_avg_loss(model: IEmbeddingModel, pairs: List[Tuple[int, int]]) -> float:
        """Compute average loss over all pairs."""
        total_loss = 0.0
        for center_idx, target_idx in pairs:
            _, _, probs = model.forward(center_idx)
            total_loss += -np.log(probs[target_idx] + 1e-9)
        return total_loss / len(pairs)
