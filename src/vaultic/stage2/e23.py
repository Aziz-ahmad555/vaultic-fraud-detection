"""E23: subgroup performance gaps (D121 section 3; details D158-).

No training: the stored final-run test scores of B5, F0, F4 and MVAF (EXP-200-final) and the raw
columns of the test rows. IEEE-CIS has no demographic data, so this is a performance-gap check,
not a fairness audit.

Groups (roadmap list only): ProductCD (W, C, R, H, S); DeviceType (desktop, mobile, missing);
purchaser email domain (the 5 most frequent in the TRAINING period, other, missing); cold start
(no earlier transaction of the customer) vs history.

Per method and group: rows, frauds, prevalence; PR-AUC; recall at the method's GLOBAL 1%-FPR
threshold (the threshold reaching the highest recall with FPR <= 1% on all test rows, applied
inside the group); false-positive rate at the method's frozen cost threshold (chosen on
calibrate_fused in the final run, Table 1). Every interval comes from the same 1,000 resamples
of all test rows (seed 0), each group's metric computed on the group's rows inside a resample,
so groups and methods are paired.

Gaps per attribute and metric: max - min and max / min across the groups with >= MIN_FRAUDS
frauds (groups below are reported, flagged unreliable, and left out of the gaps; D161), with
intervals. MVAF - B5 per group, paired, PR-AUC, two-sided p, Holm across the groups of each
attribute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from vaultic.eval.metrics import pr_auc
from vaultic.fusion.final_eval import _ci, _p_two_sided, boot_indices, holm_nan

METHODS = ("B5", "F0", "F4", "MVAF")
MIN_FRAUDS = 30
METRICS = ("pr_auc", "recall_at_global_1pct_fpr", "fpr_at_cost_threshold")
PRODUCTS = ("W", "C", "R", "H", "S")
DEVICES = ("desktop", "mobile")
TOP_EMAILS = 5


def global_fpr_threshold(y, s, fpr: float = 0.01) -> float:
    """The score threshold (flag s >= t) with the highest recall at FPR <= fpr on all rows."""
    fprs, tprs, thr = roc_curve(y, s, drop_intermediate=False)
    ok = np.flatnonzero(fprs <= fpr)
    best = ok[np.argmax(tprs[ok])]
    return float(thr[best])


def attribute_groups(raw: pd.DataFrame, train_emails: pd.Series, hist_n_past) -> dict:
    """attribute -> array of group labels per test row."""
    prod = raw["ProductCD"].astype(object).where(raw["ProductCD"].isin(PRODUCTS), "other")
    dev = raw["DeviceType"].astype(object)
    device = np.where(dev.isna(), "missing", np.where(dev.isin(DEVICES), dev, "other"))
    top = train_emails.dropna().astype(str).value_counts().index[:TOP_EMAILS].tolist()
    email = raw["P_emaildomain"].astype(object)
    mail = np.where(email.isna(), "missing", np.where(email.astype(str).isin(top), email, "other"))
    cold = np.where(np.nan_to_num(np.asarray(hist_n_past, float)) > 0, "history", "cold start")
    return {"ProductCD": prod.to_numpy(dtype=object), "DeviceType": device.astype(object),
            "P_emaildomain": mail.astype(object), "cold start": cold.astype(object)}  # fmt: skip


def _group_metrics(y, s, flagged_cost, flagged_1pct) -> dict:
    out = {"pr_auc": np.nan, "recall_at_global_1pct_fpr": np.nan, "fpr_at_cost_threshold": np.nan}
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos and n_neg:
        out["pr_auc"] = pr_auc(y, s)
    if n_pos:
        out["recall_at_global_1pct_fpr"] = float(flagged_1pct[y == 1].mean())
    if n_neg:
        out["fpr_at_cost_threshold"] = float(flagged_cost[y == 0].mean())
    return out


def e23_tables(y, scores: dict, groups: dict, cost_thresholds: dict, n_boot: int = 1000,
               seed: int = 0) -> dict[str, pd.DataFrame]:  # fmt: skip
    """groups table (per attribute, group, method), gaps table and MVAF - B5 table."""
    y = np.asarray(y)
    idx = boot_indices(len(y), n_boot, seed)
    t1 = {m: global_fpr_threshold(y, scores[m]) for m in scores}
    flag_cost = {m: np.asarray(scores[m]) >= cost_thresholds[m] for m in scores}
    flag_1 = {m: np.asarray(scores[m]) >= t1[m] for m in scores}
    rows, gaps, diffs = [], [], []
    for attr, labels in groups.items():
        labels = np.asarray(labels, dtype=object)
        names = sorted(pd.unique(labels), key=str)
        point = {}
        boot = {}  # (group, method) -> {metric: [values]}
        for g in names:
            gm = labels == g
            frauds = int(y[gm].sum())
            for m in scores:
                s = np.asarray(scores[m])
                point[g, m] = _group_metrics(y[gm], s[gm], flag_cost[m][gm], flag_1[m][gm])
                vals = {k: [] for k in METRICS}
                for ix in idx:
                    sel = ix[gm[ix]]
                    r = _group_metrics(y[sel], s[sel], flag_cost[m][sel], flag_1[m][sel])
                    for k in METRICS:
                        vals[k].append(r[k])
                boot[g, m] = vals
                row = {"attribute": attr, "group": g, "method": m, "rows": int(gm.sum()),
                       "frauds": frauds, "prevalence": float(y[gm].mean()),
                       "reliable": frauds >= MIN_FRAUDS}  # fmt: skip
                for k in METRICS:
                    lo, hi = _ci(vals[k])
                    row.update({k: point[g, m][k], f"{k}_ci_low": lo, f"{k}_ci_high": hi})
                rows.append(row)
        reliable = [g for g in names if int(y[labels == g].sum()) >= MIN_FRAUDS]
        for m in scores:
            for k in METRICS:
                vals = np.array([point[g, m][k] for g in reliable], float)
                bmat = np.array([boot[g, m][k] for g in reliable], float)  # (groups, n_boot)
                entry = {"attribute": attr, "method": m, "metric": k, "groups": len(reliable)}
                if len(reliable) >= 2 and np.isfinite(vals).all():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        bdiff = np.nanmax(bmat, 0) - np.nanmin(bmat, 0)
                        bratio = np.nanmax(bmat, 0) / np.nanmin(bmat, 0)
                    entry.update(max_minus_min=float(vals.max() - vals.min()),
                                 max_over_min=float(vals.max() / vals.min()) if vals.min() > 0 else np.inf,
                                 best=reliable[int(np.argmax(vals))], worst=reliable[int(np.argmin(vals))])  # fmt: skip
                    entry["max_minus_min_ci_low"], entry["max_minus_min_ci_high"] = _ci(bdiff)
                    entry["max_over_min_ci_low"], entry["max_over_min_ci_high"] = _ci(bratio)
                gaps.append(entry)
        if "MVAF" in scores and "B5" in scores:
            ps = []
            for g in names:
                d = np.array(boot[g, "MVAF"]["pr_auc"], float) - np.array(boot[g, "B5"]["pr_auc"])
                lo, hi = _ci(d)
                ps.append(_p_two_sided(d))
                diffs.append({"attribute": attr, "group": g,
                              "diff": point[g, "MVAF"]["pr_auc"] - point[g, "B5"]["pr_auc"],
                              "ci_low": lo, "ci_high": hi, "p_value": ps[-1],
                              "reliable": int(y[labels == g].sum()) >= MIN_FRAUDS})  # fmt: skip
            for d, p in zip(diffs[-len(names) :], holm_nan(ps), strict=True):
                d["p_holm"] = p
    return {"groups": pd.DataFrame(rows), "gaps": pd.DataFrame(gaps),
            "mvaf_minus_b5": pd.DataFrame(diffs),
            "thresholds": {"global_1pct_fpr": t1, "cost": dict(cost_thresholds)}}  # fmt: skip
