"""Point-in-time primitives: per-key aggregates over rows strictly before a cutoff time.

Every feature for a transaction at time t may only use transactions with time < t (and,
for labels, time + L <= t). PastIndex implements "rows of the same key before a cutoff"
once, with one vectorized searchsorted, so every feature built on it inherits the rule and
transactions sharing the same second never see each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY


class PastIndex:
    """Index of (key, time) pairs answering 'how many / what sum before cutoff, same key'."""

    def __init__(self, key: pd.Series | np.ndarray, time: pd.Series | np.ndarray):
        codes, _ = pd.factorize(np.asarray(key), use_na_sentinel=False)
        time = np.asarray(time, dtype=np.int64)
        if time.min() < 0:
            raise ValueError("times must be non-negative")
        self._span = int(time.max()) + 2  # room for cutoffs up to max time + 1
        self.codes = codes.astype(np.int64)
        self.time = time
        self.order = np.lexsort((time, self.codes))  # sort by key, then time
        self.sorted_composite = self.codes[self.order] * self._span + time[self.order]
        group_sizes = np.bincount(self.codes)
        self.group_start = np.concatenate([[0], np.cumsum(group_sizes)[:-1]])

    def _positions(self, cutoff: np.ndarray) -> np.ndarray:
        """Position in sorted order of the first row of the same key with time >= cutoff."""
        cutoff = np.clip(np.asarray(cutoff, dtype=np.int64), 0, self._span - 1)
        return np.searchsorted(self.sorted_composite, self.codes * self._span + cutoff, side="left")

    def count_before(self, cutoff: np.ndarray) -> np.ndarray:
        """Rows with the same key and time < cutoff."""
        return self._positions(cutoff) - self.group_start[self.codes]

    def sum_before(self, values: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
        """Sum of values (NaN treated as 0) over rows with the same key and time < cutoff.

        Running sums restart for every key, so a result depends only on that key's earlier
        rows, bit for bit (a global running sum would let other keys' rows change the
        floating-point rounding).
        """
        v = np.nan_to_num(np.asarray(values, dtype=np.float64)[self.order], nan=0.0)
        within = pd.Series(v).groupby(self.codes[self.order]).cumsum().to_numpy()
        pos = self._positions(cutoff)
        has_prev = pos > self.group_start[self.codes]
        out = np.zeros(len(pos))
        out[has_prev] = within[pos[has_prev] - 1]
        return out

    def _within(self, values: np.ndarray, how: str) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)[self.order]
        grouped = pd.Series(v).groupby(self.codes[self.order])
        return getattr(grouped, how)().to_numpy()

    def max_before(self, values: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
        """Max of values over rows with the same key and time < cutoff (NaN if none)."""
        running = self._within(values, "cummax")
        pos = self._positions(cutoff)
        has_prev = pos > self.group_start[self.codes]
        out = np.full(len(pos), np.nan)
        out[has_prev] = running[pos[has_prev] - 1]
        return out

    def last_k_sum_count(
        self, values: np.ndarray, cutoff: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum and count of the last k rows with the same key and time < cutoff."""
        within = self._within(np.nan_to_num(np.asarray(values, dtype=np.float64)), "cumsum")
        pos = self._positions(cutoff)
        start = self.group_start[self.codes]
        lo = np.maximum(pos - k, start)
        count = pos - lo
        upper = np.where(pos > start, within[np.maximum(pos - 1, 0)], 0.0)
        lower = np.where(lo > start, within[np.maximum(lo - 1, 0)], 0.0)
        return upper - lower, count

    def value_before(self, values: np.ndarray, cutoff: np.ndarray, fill=None) -> np.ndarray:
        """Value of the latest row with the same key and time < cutoff (fill if none)."""
        v = np.asarray(values, dtype=object)[self.order]
        pos = self._positions(cutoff)
        has_prev = pos > self.group_start[self.codes]
        out = np.full(len(pos), fill, dtype=object)
        out[has_prev] = v[pos[has_prev] - 1]
        return out

    def last_time_before(self, cutoff: np.ndarray) -> np.ndarray:
        """Time of the latest row with the same key and time < cutoff (NaN if none)."""
        pos = self._positions(cutoff)
        has_prev = pos > self.group_start[self.codes]
        out = np.full(len(pos), np.nan)
        out[has_prev] = self.time[self.order][pos[has_prev] - 1]
        return out


def label_cutoff(time: np.ndarray, delay_days: int) -> np.ndarray:
    """Exclusive cutoff for labels known at time t: its_time + L <= t and its_time < t."""
    time = np.asarray(time, dtype=np.int64)
    return np.minimum(time - delay_days * SECONDS_PER_DAY + 1, time)


class FrequencyEncoder:
    """Value counts fitted on the training period only; unseen values map to 0."""

    def fit(self, values: pd.Series) -> FrequencyEncoder:
        self.counts_ = values.astype(object).value_counts(dropna=False)
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        mapped = values.astype(object).map(self.counts_)
        return mapped.fillna(0).to_numpy(dtype=np.float32)
