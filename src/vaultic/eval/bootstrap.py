"""Bootstrap confidence intervals and paired comparisons on the test set."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


def _resample_indices(n: int, n_boot: int, seed: int, y: np.ndarray) -> list[np.ndarray]:
    """Resamples that contain both classes (ranking metrics need both)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_boot:
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n:
            out.append(idx)
    return out


def seed_mean_ci(
    y: np.ndarray,
    scores_per_seed: Sequence[np.ndarray],
    metric: Metric,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """95% CI of the seed-averaged metric: each resample scores every seed on the same rows."""
    y = np.asarray(y)
    stats = [
        np.mean([metric(y[idx], s[idx]) for s in scores_per_seed])
        for idx in _resample_indices(len(y), n_boot, seed, y)
    ]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap(
    y: np.ndarray,
    scores_a: Sequence[np.ndarray],
    scores_b: Sequence[np.ndarray],
    metric: Metric,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Difference A - B of the seed-averaged metric on identical resamples.

    p_value is the two-sided bootstrap p-value for 'no difference'.
    """
    y = np.asarray(y)
    point = np.mean([metric(y, s) for s in scores_a]) - np.mean([metric(y, s) for s in scores_b])
    diffs = np.array(
        [
            np.mean([metric(y[idx], s[idx]) for s in scores_a])
            - np.mean([metric(y[idx], s[idx]) for s in scores_b])
            for idx in _resample_indices(len(y), n_boot, seed, y)
        ]
    )
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(p, 1.0))}


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()
