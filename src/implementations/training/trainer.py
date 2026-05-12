import logging
from typing import List, Tuple

import numpy as np

from ...interfaces.model import (
    IEmbeddingModel,
    ILRStrategy,
    IGradientClipper,
    IEarlyStopper,
    ITrainingObserver,
    IModelPersistence,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the training loop. Depends on interfaces only.

    Does no math — delegates to IEmbeddingModel.
    Does no file IO — delegates to observers and persistence.
    Loss is computed for monitoring only, never used in weight updates.
    """

    def __init__(
        self,
        model: IEmbeddingModel,
        lr_strategy: ILRStrategy,
        gradient_clipper: IGradientClipper,
        early_stopper: IEarlyStopper,
        observers: List[ITrainingObserver],
        persistence: IModelPersistence,
    ):
        self._model = model
        self._lr_strategy = lr_strategy
        self._gradient_clipper = gradient_clipper
        self._early_stopper = early_stopper
        self._observers = observers
        self._persistence = persistence

    def train(self, pairs: List[Tuple[int, int]], epochs: int, checkpoint_every: int = 50) -> List[float]:
        """Run the training loop.

        Returns loss history (avg loss per epoch).
        """
        try:
            lr = self._lr_strategy.select(self._model, pairs)
            logger.info("Training started: %d pairs, %d epochs, lr=%.6f", len(pairs), epochs, lr)

            # Detect if model uses negative sampling
            is_ns = hasattr(self._model, '_mode') and self._model._mode == "negative_sampling"
            if is_ns:
                logger.info("Using negative sampling loss for monitoring")

            loss_history = []
            log_every = max(1, epochs // 20)

            for epoch in range(epochs):
                self._notify_epoch_start(epoch)

                indices = np.random.permutation(len(pairs))
                total_loss = 0.0

                for idx in indices:
                    center_idx, target_idx = pairs[idx]

                    # Forward
                    h, scores, probs = self._model.forward(center_idx)

                    # Loss (monitoring only — never used in update)
                    if is_ns:
                        # Negative sampling loss: -log(sigmoid(pos_score))
                        pos_score = h @ self._model._W2[:, target_idx]
                        sig = 1.0 / (1.0 + np.exp(-np.clip(pos_score, -10, 10)))
                        loss = -np.log(sig + 1e-9)
                    else:
                        loss = -np.log(probs[target_idx] + 1e-9)
                    total_loss += loss

                    # Backward
                    dW2, dh = self._model.backward(h, probs, target_idx)

                    # Clip
                    dW2 = self._gradient_clipper.clip(dW2)
                    dh = self._gradient_clipper.clip(dh)

                    # Update
                    self._model.update(center_idx, dW2, dh, lr)

                avg_loss = total_loss / len(pairs)
                loss_history.append(avg_loss)

                self._notify_epoch_end(epoch, avg_loss)

                # Progress logging
                if epoch % log_every == 0 or epoch == epochs - 1:
                    logger.info("Epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg_loss)

                # Checkpoint
                if epoch % checkpoint_every == 0:
                    self._notify_checkpoint(epoch, self._model.get_embeddings())

                # Early stopping
                if self._early_stopper.should_stop(loss_history):
                    logger.info("Early stopping at epoch %d", epoch)
                    break

            final_loss = loss_history[-1] if loss_history else 0.0
            self._notify_training_complete(final_loss)
            logger.info("Training complete: final_loss=%.6f", final_loss)

            return loss_history

        except Exception as e:
            self._notify_error(e)
            logger.error("Training failed: %s", e)
            raise

    def _notify_epoch_start(self, epoch: int) -> None:
        for observer in self._observers:
            observer.on_epoch_start(epoch)

    def _notify_epoch_end(self, epoch: int, avg_loss: float) -> None:
        for observer in self._observers:
            observer.on_epoch_end(epoch, avg_loss)

    def _notify_checkpoint(self, epoch: int, W1: np.ndarray) -> None:
        for observer in self._observers:
            observer.on_checkpoint(epoch, W1)

    def _notify_training_complete(self, final_loss: float) -> None:
        for observer in self._observers:
            observer.on_training_complete(final_loss)

    def _notify_error(self, error: Exception) -> None:
        for observer in self._observers:
            observer.on_error(error)
