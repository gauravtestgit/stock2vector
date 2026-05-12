"""Correlation-based pair generator.

Instead of pairing stocks that move in the same direction on a single day,
this pairs stocks that have high rolling correlation over a window period.

Stocks with correlation > threshold are paired. Stocks with low correlation
are never paired. This captures sustained relationships, not single-day coincidences.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ...interfaces.pipeline import IPairGenerator

logger = logging.getLogger(__name__)


class CorrelationPairGenerator(IPairGenerator):
    """Generates training pairs from rolling correlation windows.

    For each window:
      1. Compute pairwise correlation matrix over window_days
      2. Pair stocks with correlation > threshold (bidirectional)
      3. Stocks with correlation < threshold are NOT paired

    This produces pairs where frequency reflects how consistently
    two stocks are correlated over time — not just single-day moves.
    """

    def __init__(self, window_days: int = 30, min_correlation: float = 0.7,
                 step_days: int = 5, max_pairs_per_window: int = 5000,
                 seed: int = 42):
        """
        Args:
            window_days: rolling window size for correlation computation
            min_correlation: minimum correlation to generate a pair
            step_days: how many days to advance between windows
            max_pairs_per_window: cap pairs per window (random sample if exceeded)
            seed: random seed for sampling
        """
        self._window_days = window_days
        self._min_correlation = min_correlation
        self._step_days = step_days
        self._max_pairs_per_window = max_pairs_per_window
        self._seed = seed

    def generate(self, returns: pd.DataFrame, t2i: Dict[str, int],
                 threshold: float) -> List[Tuple[int, int]]:
        """Generate pairs from rolling correlations.

        Note: the threshold parameter from IThresholdStrategy is ignored —
        this generator uses min_correlation instead. The threshold parameter
        is kept for interface compatibility.
        """
        import random
        rng = random.Random(self._seed)

        all_pairs = []
        total_days = len(returns)
        tickers = [t for t in returns.columns if t in t2i]

        if len(tickers) < 2:
            logger.warning("Need at least 2 tickers for correlation pairs")
            return []

        # Slide window across the returns
        window_count = 0
        for start in range(0, total_days - self._window_days, self._step_days):
            end = start + self._window_days
            window = returns.iloc[start:end][tickers]

            # Compute correlation matrix for this window
            corr_matrix = window.corr()

            # Generate pairs for stocks with correlation > min_correlation
            window_pairs = []
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    corr = corr_matrix.iloc[i, j]
                    if not np.isnan(corr) and corr > self._min_correlation:
                        idx_i = t2i[tickers[i]]
                        idx_j = t2i[tickers[j]]
                        # Bidirectional
                        window_pairs.append((idx_i, idx_j))
                        window_pairs.append((idx_j, idx_i))

            # Cap per window
            if len(window_pairs) > self._max_pairs_per_window:
                window_pairs = rng.sample(window_pairs, self._max_pairs_per_window)

            all_pairs.extend(window_pairs)
            window_count += 1

        unique_pairs = len(set(all_pairs))
        max_possible = len(tickers) * (len(tickers) - 1)
        coverage = unique_pairs / max_possible * 100 if max_possible > 0 else 0

        logger.info("Correlation pairs: %d total, %d unique, %d windows, "
                    "coverage=%.1f%%, min_corr=%.2f, window=%d days",
                    len(all_pairs), unique_pairs, window_count,
                    coverage, self._min_correlation, self._window_days)

        return all_pairs
