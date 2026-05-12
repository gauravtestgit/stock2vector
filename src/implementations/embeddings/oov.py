"""OOV estimation strategies.

CorrelationOOV: estimates embedding using rolling correlation with vocab stocks.
WeightedAverageOOV: legacy threshold-based co-movement approach.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...interfaces.embeddings import IOOVStrategy, OOVMetadata

logger = logging.getLogger(__name__)


class CorrelationOOV(IOOVStrategy):
    """Estimates OOV embedding via rolling correlation with vocab stocks.

    Matches the correlation-based pair generation used in training.
    """

    def __init__(self, window_days: int = 30, min_correlation: float = 0.3,
                 step_days: int = 5):
        self._window = window_days
        self._min_corr = min_correlation
        self._step = step_days

    def estimate(self, ticker: str, new_returns: pd.Series,
                 embeddings: np.ndarray, t2i: Dict[str, int],
                 vocab_returns: pd.DataFrame,
                 threshold: float = 0.0) -> Optional[Tuple[np.ndarray, OOVMetadata]]:
        """Estimate embedding using rolling correlation weights."""
        common_dates = new_returns.index.intersection(vocab_returns.index)
        if len(common_dates) < self._window:
            logger.warning("Insufficient data for OOV '%s': %d days < %d window",
                           ticker, len(common_dates), self._window)
            return None

        oov_aligned = new_returns.loc[common_dates].values
        vocab_tickers = [t for t in vocab_returns.columns if t in t2i]
        vocab_aligned = vocab_returns.loc[common_dates, vocab_tickers].values

        # Compute average correlation per vocab stock across rolling windows
        n_days = len(common_dates)
        corr_sums = np.zeros(len(vocab_tickers))
        corr_counts = np.zeros(len(vocab_tickers))

        for start in range(0, n_days - self._window + 1, self._step):
            end = start + self._window
            oov_window = oov_aligned[start:end]

            if np.isnan(oov_window).sum() > self._window * 0.3:
                continue

            for j in range(len(vocab_tickers)):
                vocab_window = vocab_aligned[start:end, j]
                mask = ~(np.isnan(oov_window) | np.isnan(vocab_window))
                if mask.sum() < self._window * 0.5:
                    continue

                o = oov_window[mask]
                v = vocab_window[mask]
                # Pearson correlation
                o_std = o.std()
                v_std = v.std()
                if o_std == 0 or v_std == 0:
                    continue
                corr = np.corrcoef(o, v)[0, 1]
                if not np.isnan(corr):
                    corr_sums[j] += corr
                    corr_counts[j] += 1

        # Average correlation per stock
        valid = corr_counts > 0
        if not valid.any():
            return None

        avg_corr = np.zeros(len(vocab_tickers))
        avg_corr[valid] = corr_sums[valid] / corr_counts[valid]

        # Filter to stocks above min_correlation
        above_min = avg_corr >= self._min_corr
        if not above_min.any():
            # Nothing passes threshold — use top 10 positive correlations but mark low confidence
            top_indices = np.argsort(avg_corr)[-10:]
            above_min = np.zeros(len(vocab_tickers), dtype=bool)
            above_min[top_indices] = True
            above_min &= (avg_corr > 0)
            if not above_min.any():
                return None
            forced_fallback = True
        else:
            forced_fallback = False

        # Weighted average embedding
        weights = avg_corr[above_min]
        embed_dim = embeddings.shape[1]
        weighted_sum = np.zeros(embed_dim)
        total_weight = 0.0

        for idx_local in np.where(above_min)[0]:
            vt = vocab_tickers[idx_local]
            w = avg_corr[idx_local]
            weighted_sum += w * embeddings[t2i[vt]]
            total_weight += w

        estimated = weighted_sum / total_weight

        # Normalise
        norm = np.linalg.norm(estimated)
        if norm > 0:
            estimated = estimated / norm

        # Confidence based on number of windows and whether we hit threshold
        n_windows = int(corr_counts[above_min].mean())
        if forced_fallback:
            confidence = "low"
        elif n_windows >= 20:
            confidence = "high"
        elif n_windows >= 8:
            confidence = "medium"
        else:
            confidence = "low"

        # Top 5 by correlation
        sorted_indices = np.argsort(avg_corr)[::-1][:5]
        top_5 = [(vocab_tickers[i], float(avg_corr[i])) for i in sorted_indices if avg_corr[i] > 0]

        metadata = OOVMetadata(
            method="correlation",
            co_movement_days=int(corr_counts.sum()),
            confidence=confidence,
            top_5_comovers=top_5,
            data_days_used=n_days,
        )

        return estimated, metadata


class WeightedAverageOOV(IOOVStrategy):
    """Legacy: estimates OOV embedding as weighted average of co-movers."""

    def __init__(self, high_confidence_days: int = 60, medium_confidence_days: int = 20):
        self._high_days = high_confidence_days
        self._medium_days = medium_confidence_days

    def estimate(self, ticker: str, new_returns: pd.Series,
                 embeddings: np.ndarray, t2i: Dict[str, int],
                 vocab_returns: pd.DataFrame,
                 threshold: float) -> Optional[Tuple[np.ndarray, OOVMetadata]]:
        common_dates = new_returns.index.intersection(vocab_returns.index)
        if len(common_dates) == 0:
            return None

        oov_aligned = new_returns.loc[common_dates]
        vocab_aligned = vocab_returns.loc[common_dates]

        co_move_counts = {}
        for date in common_dates:
            oov_ret = oov_aligned.loc[date]
            if np.isnan(oov_ret):
                continue
            oov_up = oov_ret > threshold
            oov_down = oov_ret < -threshold
            if not oov_up and not oov_down:
                continue
            for vocab_ticker in vocab_aligned.columns:
                if vocab_ticker not in t2i:
                    continue
                vocab_ret = vocab_aligned.loc[date, vocab_ticker]
                if np.isnan(vocab_ret):
                    continue
                if (oov_up and vocab_ret > threshold) or (oov_down and vocab_ret < -threshold):
                    co_move_counts[vocab_ticker] = co_move_counts.get(vocab_ticker, 0) + 1

        total = sum(co_move_counts.values())
        if total == 0:
            return None

        embed_dim = embeddings.shape[1]
        weighted_sum = np.zeros(embed_dim)
        for vt, count in co_move_counts.items():
            weighted_sum += count * embeddings[t2i[vt]]
        estimated = weighted_sum / total

        norm = np.linalg.norm(estimated)
        if norm > 0:
            estimated = estimated / norm

        unique_days = len([d for d in common_dates
                          if not np.isnan(oov_aligned.loc[d])
                          and (oov_aligned.loc[d] > threshold or oov_aligned.loc[d] < -threshold)])
        if unique_days >= self._high_days:
            confidence = "high"
        elif unique_days >= self._medium_days:
            confidence = "medium"
        else:
            confidence = "low"

        sorted_comovers = sorted(co_move_counts.items(), key=lambda x: -x[1])
        top_5 = [(t, float(c)) for t, c in sorted_comovers[:5]]

        metadata = OOVMetadata(
            method="weighted_average",
            co_movement_days=total,
            confidence=confidence,
            top_5_comovers=top_5,
            data_days_used=len(common_dates),
        )
        return estimated, metadata
