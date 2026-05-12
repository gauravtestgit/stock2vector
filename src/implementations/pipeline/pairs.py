import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from ...interfaces.pipeline import IPairGenerator


class CoMovementPairGenerator(IPairGenerator):
    """Generates bidirectional training pairs from co-moving stocks.

    Stocks moving in the same direction (up or down) beyond the threshold
    on the same day are paired. Capped at max_pairs_per_day.
    """

    def __init__(self, max_pairs_per_day: int = 10000, seed: int = 42):
        self._max_pairs_per_day = max_pairs_per_day
        self._seed = seed

    def generate(self, returns: pd.DataFrame, t2i: Dict[str, int], threshold: float) -> List[Tuple[int, int]]:
        """Returns list of (center_idx, target_idx) bidirectional pairs."""
        all_pairs = []

        for date, row in returns.iterrows():
            up = [t for t in row.index if row[t] > threshold and t in t2i and not np.isnan(row[t])]
            down = [t for t in row.index if row[t] < -threshold and t in t2i and not np.isnan(row[t])]

            day_pairs = []
            for group in [up, down]:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        day_pairs.append((t2i[group[i]], t2i[group[j]]))
                        day_pairs.append((t2i[group[j]], t2i[group[i]]))

            if len(day_pairs) > self._max_pairs_per_day:
                rng = random.Random(self._seed)
                day_pairs = rng.sample(day_pairs, self._max_pairs_per_day)

            all_pairs.extend(day_pairs)

        return all_pairs
