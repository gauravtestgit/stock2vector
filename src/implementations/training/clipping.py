import numpy as np
from ...interfaces.model import IGradientClipper


class GradientClipper(IGradientClipper):
    """Clips gradient values to prevent exploding gradients."""

    def __init__(self, clip_value: float):
        self._clip_value = clip_value

    def clip(self, gradient: np.ndarray) -> np.ndarray:
        """Returns gradient clipped to [-clip_value, clip_value]."""
        return np.clip(gradient, -self._clip_value, self._clip_value)
