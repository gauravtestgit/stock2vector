"""Factory for OOV estimation strategies.

Adding a new strategy means adding a new entry to the dict — zero existing files touched.
"""
import logging
from typing import Dict, Type

from ...interfaces.embeddings import IOOVStrategy
from .oov import WeightedAverageOOV

logger = logging.getLogger(__name__)


class OOVStrategyFactory:
    """Maps strategy names to IOOVStrategy implementations."""

    def __init__(self):
        self._strategies: Dict[str, Type[IOOVStrategy]] = {
            "weighted_average": WeightedAverageOOV,
        }

    def create(self, method: str = "weighted_average", **kwargs) -> IOOVStrategy:
        """Create an OOV strategy by name.

        Args:
            method: strategy name (default: "weighted_average")
            **kwargs: passed to strategy constructor

        Returns:
            IOOVStrategy instance
        """
        if method not in self._strategies:
            available = list(self._strategies.keys())
            logger.error("Unknown OOV method '%s'. Available: %s", method, available)
            raise ValueError(f"Unknown OOV method '{method}'. Available: {available}")

        return self._strategies[method](**kwargs)

    def register(self, name: str, strategy_class: Type[IOOVStrategy]) -> None:
        """Register a new OOV strategy."""
        self._strategies[name] = strategy_class

    def list_methods(self):
        """List available OOV methods."""
        return list(self._strategies.keys())
