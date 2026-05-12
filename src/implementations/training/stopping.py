from typing import List
from ...interfaces.model import IEarlyStopper


class PatienceEarlyStopper(IEarlyStopper):
    """Stops training if loss increases for consecutive epochs."""

    def __init__(self, patience: int):
        self._patience = patience

    def should_stop(self, loss_history: List[float]) -> bool:
        """Returns True if loss increased for patience consecutive epochs."""
        if len(loss_history) <= self._patience:
            return False
        recent = loss_history[-self._patience:]
        return all(recent[i] >= recent[i - 1] for i in range(1, len(recent)))
