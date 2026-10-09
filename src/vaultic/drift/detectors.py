"""Phase 10.1 label-based detectors on the error stream of matured labels: ADWIN and
Page-Hinkley, implemented here (no streaming library in the pinned environment).

ADWIN (Bifet & Gavalda 2007): keeps a window of recent values and drops its older part
whenever two sub-windows have means that differ by more than the paper's practical cut
    eps_cut = sqrt(2 / m * var_W * ln(2 / d')) + 2 / (3 m) * ln(2 / d'),
    m = 1 / (1/n0 + 1/n1),  d' = delta / ln(n)
(variance-based; the plain Hoeffding form is too conservative for error rates of a few %).
This is the exact, unbucketed variant with a capped window and a check every `clock`
updates; enough for daily error streams.

Page-Hinkley (Page 1954): cumulative deviation from the running mean minus a tolerance delta;
alarm when it rises more than lambda above its minimum (detects an increase in error).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class ADWIN:
    def __init__(self, delta: float = 0.002, clock: int = 32, max_window: int = 20_000,
                 min_side: int = 30):  # fmt: skip
        self.delta, self.clock, self.max_window, self.min_side = delta, clock, max_window, min_side
        self.reset()

    def reset(self) -> None:
        self.window: deque[float] = deque()
        self.n_seen = 0

    def update(self, x: float) -> bool:
        """Add one value; True if a change was detected (the older part is then dropped)."""
        self.window.append(float(x))
        if len(self.window) > self.max_window:
            self.window.popleft()
        self.n_seen += 1
        if self.n_seen % self.clock or len(self.window) < 2 * self.min_side:
            return False
        return self._check()

    def _check(self) -> bool:
        v = np.fromiter(self.window, dtype=float)
        n = len(v)
        csum = np.cumsum(v)
        n0 = np.arange(self.min_side, n - self.min_side + 1)
        n1 = n - n0
        mean0 = csum[n0 - 1] / n0
        mean1 = (csum[-1] - csum[n0 - 1]) / n1
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        log_term = np.log(2.0 * np.log(n) / self.delta)
        eps = np.sqrt(2.0 / m * v.var() * log_term) + 2.0 / (3.0 * m) * log_term
        cut = np.flatnonzero(np.abs(mean0 - mean1) > eps)
        if not len(cut):
            return False
        drop = int(n0[cut[-1]])  # keep the most recent sub-window that differs
        for _ in range(drop):
            self.window.popleft()
        return True

    @property
    def mean(self) -> float:
        return float(np.mean(self.window)) if self.window else math.nan


class PageHinkley:
    def __init__(self, delta: float = 0.005, threshold: float = 50.0, min_instances: int = 30):
        self.delta, self.threshold, self.min_instances = delta, threshold, min_instances
        self.reset()

    def reset(self) -> None:
        self.n, self.mean, self.cum, self.min_cum = 0, 0.0, 0.0, 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        if self.n >= self.min_instances and self.cum - self.min_cum > self.threshold:
            self.reset()
            return True
        return False


def alarms(detector, stream) -> list[int]:
    """Indices of the stream at which the detector fired."""
    return [i for i, x in enumerate(stream) if detector.update(x)]
