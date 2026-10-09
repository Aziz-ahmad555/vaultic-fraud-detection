"""Phase 8.1 calibration: Platt and isotonic, chosen by out-of-fold ECE on validation.

The roadmap says: fit isotonic and Platt on validation and keep the one with lower ECE.
Comparing them on the same rows they were fitted on would favour isotonic (it can memorise),
so `choose_calibrator` compares out-of-fold ECE over contiguous (time-ordered) folds of the
validation period, then fits the winner on all of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from vaultic.eval.metrics import ECE_BINS, brier, ece

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Logistic regression on the logit of the score (unregularised)."""

    name = "platt"

    def fit(self, p, y) -> PlattCalibrator:
        self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p)[:, None], y)
        return self

    def predict(self, p) -> np.ndarray:
        return self.model_.predict_proba(_logit(p)[:, None])[:, 1]


class IsotonicCalibrator:
    """Monotone step function; scores outside the fitted range are clipped."""

    name = "isotonic"

    def fit(self, p, y) -> IsotonicCalibrator:
        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.model_.fit(np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, p) -> np.ndarray:
        return self.model_.predict(np.asarray(p, dtype=np.float64))


CALIBRATORS = {"platt": PlattCalibrator, "isotonic": IsotonicCalibrator}


def choose_calibrator(y, p, folds: int = 5) -> tuple[object, dict[str, float]]:
    """Out-of-fold ECE per method on contiguous folds; returns (winner fitted on all, ECEs)."""
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    bounds = np.linspace(0, len(y), folds + 1).astype(int)
    scores = {}
    for name, cls in CALIBRATORS.items():
        out = np.empty(len(y))
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            train = np.r_[0:lo, hi : len(y)]
            out[lo:hi] = cls().fit(p[train], y[train]).predict(p[lo:hi])
        scores[name] = ece(y, out)
    best = min(scores, key=scores.get)
    return CALIBRATORS[best]().fit(p, y), scores


def reliability_table(y, p, bins: int = ECE_BINS) -> pd.DataFrame:
    """Data for a reliability diagram: equal-width bins on [0, 1]."""
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=np.float64), 0, 1)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append(
            {
                "bin": b,
                "lower": b / bins,
                "upper": (b + 1) / bins,
                "count": int(m.sum()),
                "mean_predicted": float(p[m].mean()) if m.any() else np.nan,
                "observed_rate": float(y[m].mean()) if m.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_report(y, p_before, p_after) -> dict[str, float]:
    return {
        "brier_before": brier(np.asarray(y), np.asarray(p_before)),
        "brier_after": brier(np.asarray(y), np.asarray(p_after)),
        "ece_before": ece(np.asarray(y), np.asarray(p_before)),
        "ece_after": ece(np.asarray(y), np.asarray(p_after)),
    }
