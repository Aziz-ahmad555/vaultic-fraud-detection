"""Phase 8.2 conformal prediction: split and class-conditional (Mondrian).

Nonconformity of label c for a transaction with fraud probability p: 1 - p(c), where
p(fraud) = p and p(legit) = 1 - p. A prediction set holds every label whose nonconformity is
at most the calibrated threshold: {legit}, {fraud}, {legit, fraud} ("uncertain": goes to
review) or, rarely, {} ("empty"). Split conformal uses one threshold from all calibration
rows; Mondrian uses one threshold per class, so the rare fraud class gets its own coverage
guarantee. Thresholds come from a held-out calibration slice of validation.

AdaptiveConformal adds the streaming version (roadmap 8.2): the error level adapts per batch.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

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


# ---- adaptive conformal inference (roadmap 8.2, streaming) ----------------------------------


def threshold_at(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold for any error level: alpha <= 0 admits everything (+inf),
    and a level asking for fewer than one calibration score admits nothing (-inf)."""
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(scores)
    if n == 0 or alpha <= 0:
        return float("inf")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    if k < 1:
        return float("-inf")
    return float(scores[k - 1])


class AdaptiveConformal:
    """Adaptive conformal inference (Gibbs and Candes, 2021) over a time-ordered stream.

    The stream is processed in batches (e.g. days). Each batch gets sets from the calibration
    scores at the current error level alpha_t. Once a batch's labels are known, alpha moves
    towards the target:  alpha_{t+1} = alpha_t + gamma * (alpha - err_t),  where err_t is the
    batch's miscoverage. Under-coverage lowers alpha_t (wider sets) and over-coverage raises it.
    With `delay` > 0, a batch's error is applied `delay` batches later (labels arrive late, as
    in Vaultic's label-delay rule). `mondrian=True` keeps one alpha per class.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.05, mondrian: bool = False,
                 delay: int = 0):  # fmt: skip
        self.alpha, self.gamma, self.mondrian, self.delay = alpha, gamma, mondrian, delay

    def fit(self, y, p) -> AdaptiveConformal:
        y = np.asarray(y).astype(int)
        scores = _label_scores(p)
        if self.mondrian:
            self.cal_scores_ = [scores[y == c, c] for c in (0, 1)]
        else:
            shared = scores[np.arange(len(y)), y]
            self.cal_scores_ = [shared, shared]
        return self

    def run(self, batch, y, p) -> tuple[np.ndarray, list[dict]]:
        """Sets for every row (rows must be in time order) and the alpha used per batch."""
        batch, y = np.asarray(batch), np.asarray(y).astype(int)
        scores = _label_scores(p)
        sets = np.zeros((len(y), 2), dtype=bool)
        alphas = np.full(2, float(self.alpha))
        pending: list[np.ndarray] = []  # per-batch errors waiting for their labels
        trace = []
        for b in pd.unique(batch):
            rows = np.flatnonzero(batch == b)
            thresholds = np.array([threshold_at(self.cal_scores_[c], alphas[c]) for c in (0, 1)])
            sets[rows] = scores[rows] <= thresholds[None, :]
            covered = sets[rows, y[rows]]
            if self.mondrian:
                err = [np.mean(~covered[y[rows] == c]) if (y[rows] == c).any() else np.nan
                       for c in (0, 1)]  # fmt: skip
            else:
                err = [np.mean(~covered)] * 2
            trace.append({"batch": b, "alpha_legit": alphas[0], "alpha_fraud": alphas[1],
                          "coverage": float(covered.mean())})  # fmt: skip
            pending.append(np.array(err, dtype=float))
            if len(pending) > self.delay:
                known = pending.pop(0)
                for c in (0, 1):
                    if not np.isnan(known[c]):
                        alphas[c] += self.gamma * (self.alpha - known[c])
        return sets, trace


def coverage_by_block(day, y, sets: np.ndarray, block_days: int = 30) -> pd.DataFrame:
    """Coverage per block of `block_days` days (counted from the first day), e.g. per month."""
    day, y = np.asarray(day), np.asarray(y).astype(int)
    block = (day - day.min()) // block_days
    rows = []
    for b in np.unique(block):
        m = block == b
        cov = coverage(y[m], sets[m])
        rows.append({"block": int(b), "first_day": int(day[m].min()),
                     "last_day": int(day[m].max()), "rows": int(m.sum()), **cov})  # fmt: skip
    return pd.DataFrame(rows)
