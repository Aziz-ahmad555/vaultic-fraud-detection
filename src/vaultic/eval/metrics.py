"""Evaluation metrics. PR-AUC is primary; accuracy is never reported.

Thresholds are always chosen on validation (choose_* functions) and then applied to test.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

C_FP = 10.0  # cost of a false positive (stated assumption, roadmap Phase 13)
C_REVIEW = 5.0  # cost of one analyst review
ECE_BINS = 15


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def recall_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float) -> float:
    """Highest recall reachable with false-positive rate <= fpr.

    Every distinct score is a possible threshold; tied scores are taken or left together (a
    threshold cannot split a tie). drop_intermediate=False is required (D49): with the
    default, roc_curve drops collinear ROC points, which can drop the best point inside the
    FPR budget and understate the recall.
    """
    fprs, tprs, _ = roc_curve(y, s, drop_intermediate=False)
    ok = fprs <= fpr
    return float(tprs[ok].max()) if ok.any() else 0.0


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    """Precision of the k highest scores. If a tie straddles position k, the tied rows count
    with their expected share (as under random tie-breaking), so the result never depends on
    row order (D49)."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    k = min(k, len(s))
    if k <= 0:
        return 0.0
    kth = np.sort(s)[::-1][k - 1]
    above = s > kth
    tied = s == kth
    n_above = int(above.sum())
    share = (k - n_above) / int(tied.sum())
    return float((y[above].sum() + share * y[tied].sum()) / k)


def f1_at(y: np.ndarray, s: np.ndarray, threshold: float) -> float:
    pred = s >= threshold
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def brier(y: np.ndarray, s: np.ndarray) -> float:
    return float(np.mean((np.asarray(s, dtype=float) - y) ** 2))


def ece(y: np.ndarray, s: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    s = np.clip(np.asarray(s, dtype=float), 0, 1)
    idx = np.minimum((s * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(s[m].mean() - np.mean(y[m]))
    return float(total)


def cost_at(
    y: np.ndarray,
    s: np.ndarray,
    amount: np.ndarray,
    threshold: float,
    c_fp: float = C_FP,
    c_rev: float = C_REVIEW,
) -> float:
    """Missed fraud amount + c_fp per false positive + c_rev per flagged transaction."""
    flagged = s >= threshold
    missed = float(np.sum(amount[(y == 1) & ~flagged]))
    fp = float(np.sum(flagged & (y == 0)))
    return missed + c_fp * fp + c_rev * float(flagged.sum())


def _candidate_thresholds(s: np.ndarray, n: int = 200) -> np.ndarray:
    return np.unique(np.quantile(s, np.linspace(0.5, 1.0, n)))


def choose_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    cands = _candidate_thresholds(s)
    return float(cands[int(np.argmax([f1_at(y, s, t) for t in cands]))])


def choose_cost_threshold(
    y: np.ndarray, s: np.ndarray, amount: np.ndarray, c_fp: float = C_FP, c_rev: float = C_REVIEW
) -> float:
    cands = _candidate_thresholds(s)
    return float(cands[int(np.argmin([cost_at(y, s, amount, t, c_fp, c_rev) for t in cands]))])


# Threshold-free metrics, bootstrapped for confidence intervals.
RANKING_METRICS = {
    "pr_auc": pr_auc,
    "roc_auc": roc_auc,
    "recall_at_1pct_fpr": lambda y, s: recall_at_fpr(y, s, 0.01),
    "recall_at_5pct_fpr": lambda y, s: recall_at_fpr(y, s, 0.05),
}
