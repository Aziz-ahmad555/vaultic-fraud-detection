"""Similar earlier cases (D224 (1)), a read-only product-layer helper: the nearest transactions
strictly before a time with the same frozen E12 action, by the calibrated view scores.

Distance: root mean squared difference over the views both transactions have, plus 0.5 per view
that only one of them has (missing views stay missing, rule 11: never filled). Rows sharing no view
are never similar. Similar by scores, NOT by investigation outcome; labels are left to the caller,
which applies the label-delay curtain."""

from __future__ import annotations

import numpy as np
import pandas as pd

VIEWS = ("tabular", "behavioral", "temporal", "graph", "anomaly")
MISMATCH_PENALTY = 0.5


def similar_cases(frame: pd.DataFrame, actions: np.ndarray, position: int, before: float,
                  k: int = 3) -> list[tuple[int, float]]:  # fmt: skip
    """(row position, distance) of the k nearest rows with TransactionDT < before and the same
    action as row `position`, nearest first (ties by time order)."""
    t = frame["TransactionDT"].to_numpy(dtype=float)
    acts = np.asarray(actions, dtype=object)
    cand = np.flatnonzero((t < before) & (acts == acts[position]))
    cand = cand[cand != position]
    if len(cand) == 0:
        return []
    scores = frame[[f"p_{v}" for v in VIEWS]].to_numpy(dtype=float)
    q, c = scores[position], scores[cand]
    qa, ca = np.isfinite(q), np.isfinite(c)
    both = ca & qa
    n_both = both.sum(axis=1)
    sq = np.where(both, (c - np.where(qa, q, 0.0)) ** 2, 0.0).sum(axis=1)
    dist = np.sqrt(sq / np.maximum(n_both, 1)) + MISMATCH_PENALTY * (ca != qa).sum(axis=1)
    dist[n_both == 0] = np.inf
    order = np.argsort(dist, kind="stable")[:k]
    return [(int(cand[i]), float(dist[i])) for i in order if np.isfinite(dist[i])]
