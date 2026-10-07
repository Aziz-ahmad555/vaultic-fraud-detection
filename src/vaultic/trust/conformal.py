"""Phase 8.2 conformal prediction: split and class-conditional (Mondrian).

Nonconformity of label c for a transaction with fraud probability p: 1 - p(c), where
p(fraud) = p and p(legit) = 1 - p. A prediction set holds every label whose nonconformity is
at most the calibrated threshold: {legit}, {fraud}, {legit, fraud} ("uncertain": goes to
review) or, rarely, {} ("empty"). Split conformal uses one threshold from all calibration
rows; Mondrian uses one threshold per class, so the rare fraud class gets its own coverage
guarantee. Thresholds come from a held-out calibration slice of validation.

Not yet implemented: adaptive conformal inference for the streaming setting (roadmap 8.2).
"""

from __future__ import annotations

import math

import numpy as np

LABELS = ("legit", "fraud")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score; +inf if that exceeds n."""
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n or n == 0 else float(scores[k - 1])


def _label_scores(p: np.ndarray) -> np.ndarray:
    """(n, 2) nonconformity for legit and fraud."""
    p = np.asarray(p, dtype=np.float64)
    return np.column_stack([p, 1 - p])  # 1 - p(legit) = p ; 1 - p(fraud) = 1 - p


class SplitConformal:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, y, p) -> SplitConformal:
        y = np.asarray(y).astype(int)
        scores = _label_scores(p)[np.arange(len(y)), y]
        self.thresholds_ = np.full(2, conformal_quantile(scores, self.alpha))
        return self

    def predict_sets(self, p) -> np.ndarray:
        """(n, 2) booleans: is legit / fraud in the set."""
        return _label_scores(p) <= self.thresholds_[None, :]


class MondrianConformal(SplitConformal):
    """One threshold per class, from that class's calibration rows only."""

    def fit(self, y, p) -> MondrianConformal:
        y = np.asarray(y).astype(int)
        scores = _label_scores(p)
        self.thresholds_ = np.array(
            [conformal_quantile(scores[y == c, c], self.alpha) for c in (0, 1)]
        )
        return self


def set_labels(sets: np.ndarray) -> np.ndarray:
    """'legit', 'fraud', 'uncertain' (both) or 'empty' per row."""
    out = np.full(len(sets), "empty", dtype=object)
    out[sets[:, 0] & ~sets[:, 1]] = "legit"
    out[~sets[:, 0] & sets[:, 1]] = "fraud"
    out[sets[:, 0] & sets[:, 1]] = "uncertain"
    return out


def coverage(y, sets: np.ndarray) -> dict[str, float]:
    """Share of rows whose true label is in the set: overall, per class, and set mix."""
    y = np.asarray(y).astype(int)
    covered = sets[np.arange(len(y)), y]
    labels = set_labels(sets)
    return {
        "coverage": float(covered.mean()),
        "coverage_legit": float(covered[y == 0].mean()) if (y == 0).any() else np.nan,
        "coverage_fraud": float(covered[y == 1].mean()) if (y == 1).any() else np.nan,
        "share_uncertain": float((labels == "uncertain").mean()),
        "share_empty": float((labels == "empty").mean()),
    }
