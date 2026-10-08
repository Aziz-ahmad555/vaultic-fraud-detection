"""Bootstrap confidence intervals and paired comparisons on the test set.

Speed (research/decisions.md D48): resample indices are drawn in the main thread exactly as
before (same generator, same order, same rejection of one-class resamples), and the metric of
each resample is computed by a pool of worker threads; results are collected in resample order
and aggregated exactly as before. The CIs are therefore bit-identical to the original
single-threaded code, which is kept below as `seed_mean_ci_reference` /
`paired_bootstrap_reference` and checked against in tests/test_bootstrap_parallel.py.
Threads (not processes) because metrics may be lambdas; NumPy's sort releases the GIL.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]

CHUNK = 25  # resamples handed to a worker at a time (bounds memory: CHUNK x n indices)


def _resample_indices(n: int, n_boot: int, seed: int, y: np.ndarray) -> list[np.ndarray]:
    """Resamples that contain both classes (ranking metrics need both)."""
    return list(_iter_resample_indices(n, n_boot, seed, y))


def _iter_resample_indices(n: int, n_boot: int, seed: int, y: np.ndarray) -> Iterator[np.ndarray]:
    """The same resamples as _resample_indices, in the same order, one at a time."""
    rng = np.random.default_rng(seed)
    made = 0
    while made < n_boot:
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n:
            made += 1
            yield idx


def _n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        n_jobs = int(os.environ.get("VAULTIC_BOOTSTRAP_JOBS", "0")) or (os.cpu_count() or 1)
    return max(1, int(n_jobs))


def _progress(label: str | None, done: int, total: int, started: float, last: list[float]) -> None:
    """One line about every 10% (and at the end), to stderr."""
    if label is None:
        return
    step = max(1, total // 10)
    if done == total or done // step > last[0]:
        last[0] = done // step
        print(
            f"bootstrap {label}: {done}/{total} resamples, {time.perf_counter() - started:.0f} s",
            file=sys.stderr,
            flush=True,
        )


def _map_resamples(
    stat: Callable[[np.ndarray], float],
    y: np.ndarray,
    n_boot: int,
    seed: int,
    n_jobs: int | None,
    label: str | None,
) -> np.ndarray:
    """stat(idx) for every resample, in resample order, computed by a thread pool."""
    jobs = _n_jobs(n_jobs)
    started = time.perf_counter()
    last = [-1.0]
    out: list[float] = []
    indices = _iter_resample_indices(len(y), n_boot, seed, y)
    if jobs == 1:
        for idx in indices:
            out.append(stat(idx))
            _progress(label, len(out), n_boot, started, last)
        return np.asarray(out)

    def run_chunk(chunk: list[np.ndarray]) -> list[float]:
        return [stat(idx) for idx in chunk]

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = []
        chunk: list[np.ndarray] = []
        for idx in indices:
            chunk.append(idx)
            if len(chunk) == CHUNK:
                pending.append(pool.submit(run_chunk, chunk))
                chunk = []
            # keep at most 2 chunks per worker in flight (bounded memory), in order
            while len(pending) > 2 * jobs:
                out.extend(pending.pop(0).result())
                _progress(label, len(out), n_boot, started, last)
        if chunk:
            pending.append(pool.submit(run_chunk, chunk))
        for fut in pending:
            out.extend(fut.result())
            _progress(label, len(out), n_boot, started, last)
    return np.asarray(out)


def seed_mean_ci(
    y: np.ndarray,
    scores_per_seed: Sequence[np.ndarray],
    metric: Metric,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    n_jobs: int | None = None,
    label: str | None = None,
) -> tuple[float, float]:
    """95% CI of the seed-averaged metric: each resample scores every seed on the same rows."""
    y = np.asarray(y)
    stats = _map_resamples(
        lambda idx: np.mean([metric(y[idx], s[idx]) for s in scores_per_seed]),
        y,
        n_boot,
        seed,
        n_jobs,
        label,
    )
    lo, hi = np.quantile(stats.tolist(), [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap(
    y: np.ndarray,
    scores_a: Sequence[np.ndarray],
    scores_b: Sequence[np.ndarray],
    metric: Metric,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    n_jobs: int | None = None,
    label: str | None = None,
) -> dict[str, float]:
    """Difference A - B of the seed-averaged metric on identical resamples.

    p_value is the two-sided bootstrap p-value for 'no difference'.
    """
    y = np.asarray(y)
    point = np.mean([metric(y, s) for s in scores_a]) - np.mean([metric(y, s) for s in scores_b])
    diffs = _map_resamples(
        lambda idx: np.mean([metric(y[idx], s[idx]) for s in scores_a])
        - np.mean([metric(y[idx], s[idx]) for s in scores_b]),
        y,
        n_boot,
        seed,
        n_jobs,
        label,
    )
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diff": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
    }


# ---- reference implementations (the original single-threaded code, kept verbatim) ----------


def seed_mean_ci_reference(
    y: np.ndarray,
    scores_per_seed: Sequence[np.ndarray],
    metric: Metric,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Original seed_mean_ci (before D48); used to check the parallel version is identical."""
    y = np.asarray(y)
    stats = [
        np.mean([metric(y[idx], s[idx]) for s in scores_per_seed])
        for idx in _resample_indices(len(y), n_boot, seed, y)
    ]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_reference(
    y: np.ndarray,
    scores_a: Sequence[np.ndarray],
    scores_b: Sequence[np.ndarray],
    metric: Metric,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Original paired_bootstrap (before D48)."""
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
    return {
        "diff": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
    }


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
