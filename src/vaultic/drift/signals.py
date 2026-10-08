"""Phase 10.1 label-free drift signals: feature PSI, KS with Holm correction, score drift and
disagreement drift. All compare a reference window against a current window; none needs
labels, so they work while labels are still delayed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from vaultic.eval.bootstrap import holm

PSI_ALERT = 0.2
EPS = 1e-4


def psi(reference, current, bins: int = 10) -> float:
    """Population stability index sum (c - r) ln(c / r) over bins.

    Numeric: bins are reference quantiles (ties merged), with missing values as an extra bin.
    Non-numeric: one bin per reference category plus "other" and "missing".
    Empty bins are floored at 1e-4 so the index stays finite."""
    r, c = pd.Series(reference), pd.Series(current)
    if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(c):
        rv = r.dropna().to_numpy(dtype=float)
        edges = np.unique(np.quantile(rv, np.linspace(0, 1, bins + 1))) if len(rv) else np.array([])
        inner = edges[1:-1]

        def shares(s: pd.Series) -> np.ndarray:
            v = s.to_numpy(dtype=float)
            idx = np.where(np.isnan(v), len(inner) + 1, np.searchsorted(inner, v, side="right"))
            return np.bincount(idx, minlength=len(inner) + 2) / max(len(v), 1)

    else:
        cats = pd.Index(r.dropna().astype(str).unique())

        def shares(s: pd.Series) -> np.ndarray:
            text = s.astype(object)
            missing = text.isna().to_numpy()
            pos = cats.get_indexer(text.where(~text.isna(), "").astype(str))
            idx = np.where(missing, len(cats) + 1, np.where(pos >= 0, pos, len(cats)))
            return np.bincount(idx, minlength=len(cats) + 2) / max(len(s), 1)

    e, a = np.maximum(shares(r), EPS), np.maximum(shares(c), EPS)
    return float(((a - e) * np.log(a / e)).sum())


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, psi_alert: float = PSI_ALERT,
                  alpha: float = 0.05) -> pd.DataFrame:  # fmt: skip
    """Per feature: PSI, KS statistic and p-value (numeric columns), Holm-adjusted p-value over
    all KS tests, and the two alerts (PSI > psi_alert; Holm-adjusted p < alpha)."""
    rows = []
    for col in reference.columns:
        row = {"feature": col, "psi": psi(reference[col], current[col]), "ks": np.nan,
               "p_value": np.nan}  # fmt: skip
        if pd.api.types.is_numeric_dtype(reference[col]):
            r, c = reference[col].dropna(), current[col].dropna()
            if len(r) and len(c):
                test = ks_2samp(r, c)
                row.update(ks=float(test.statistic), p_value=float(test.pvalue))
        rows.append(row)
    table = pd.DataFrame(rows)
    tested = table["p_value"].notna()
    table["p_holm"] = np.nan
    if tested.any():
        table.loc[tested, "p_holm"] = holm(table.loc[tested, "p_value"].tolist())
    table["psi_alert"] = table["psi"] > psi_alert
    table["ks_alert"] = table["p_holm"] < alpha
    return table


def score_drift(reference_scores, current_scores) -> float:
    """PSI of the fused score distribution."""
    return psi(np.asarray(reference_scores, dtype=float), np.asarray(current_scores, dtype=float))


class DisagreementMonitor:
    """Rise in mean view disagreement (and in the share of uncertain conformal sets): alarm when
    a current window's mean exceeds the reference daily mean by k reference daily std."""

    def __init__(self, k: float = 3.0):
        self.k = k

    def fit(self, reference_daily_means) -> DisagreementMonitor:
        v = np.asarray(reference_daily_means, dtype=float)
        self.mean_, self.std_ = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
        return self

    def z(self, current_mean: float) -> float:
        return (current_mean - self.mean_) / max(self.std_, 1e-12)

    def alarm(self, current_mean: float) -> bool:
        return self.z(current_mean) > self.k
