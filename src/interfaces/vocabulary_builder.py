from abc import ABC, abstractmethod
from typing import List, Dict


class IVocabularyBuilder(ABC):
    """Interface for vocabulary building components."""

    @abstractmethod
    def build_vocabulary(self, source: str) -> List:
        """Build vocabulary list from a config source."""
        pass

    @abstractmethod
    def build_training_pairs(self, source: str, start: str, end: str, output_dir: str) -> Dict:
        """Build training pairs from price co-movement data.

        Preconditions: source is a valid config path, start/end are date strings.
        Postconditions: returns dict with pairs_file, vocab_file, t2i, i2t, failed, pair_count.
        """
        pass
