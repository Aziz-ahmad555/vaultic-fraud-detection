"""Statistics of the pre-registered final experiments E10 and E11 (D101, D102).

Pure functions over score arrays, so they are tested on synthetic data and used unchanged by
fusion/final_run.py. Every comparison is paired: all methods are evaluated on the same
bootstrap resamples of the same rows (1,000 by default). Two-sided p-values; Holm correction
within each table; effect sizes from the per-seed values (eval/stats.effect_sizes).

E10  per method: PR-AUC, recall@1%FPR and total cost at a threshold chosen on the calibration
     rows (never on the evaluation rows); MVAF minus every other method for each metric;
     subgroups; the H2 decision rule.
E11  PR-AUC drop of each method when a view is removed for every row (and "tabular only");
     MVAF's drop minus F3's / F4's drop, Holm across the five single-view conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaultic.eval.bootstrap import holm
from vaultic.eval.metrics import cost_at, pr_auc, recall_at_fpr
from vaultic.eval.stats import effect_sizes

N_BOOT = 1000
VIEWS = ("tabular", "behavioral", "temporal", "graph", "anomaly")


def boot_indices(n: int, n_boot: int = N_BOOT, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def _p_two_sided(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(min(p, 1.0))


def _ok(y: np.ndarray) -> bool:
    return 0 < y.sum() < len(y)


def metric_fns(amount: np.ndarray, thresholds: dict[str, float]):
    """name -> fn(method, y, s, idx) for the three E10 metrics (cost uses the method's threshold
    chosen on the calibration rows and the resampled amounts)."""
    return {
        "pr_auc": lambda m, y, s, idx: pr_auc(y, s),
        "recall_at_1pct_fpr": lambda m, y, s, idx: recall_at_fpr(y, s, 0.01),
        "cost": lambda m, y, s, idx: cost_at(y, s, amount[idx], thresholds[m]),
    }


def e10_table(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    per_seed: dict[str, list[np.ndarray]],
    amount: np.ndarray,
    thresholds: dict[str, float],
    reference: str = "MVAF",
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(methods, comparisons): each method's metrics with 95% CIs; reference minus each other
    method per metric with CI, p and Holm-adjusted p (one Holm family per metric), and effect
    sizes (from per-seed values where a method has several seeds)."""
    y = np.asarray(y).astype(int)
    fns = metric_fns(np.asarray(amount, dtype=float), thresholds)
    idx = boot_indices(len(y), n_boot, seed)
    boots = {m: {k: [] for k in fns} for m in scores}
    for i in idx:
        yy = y[i]
        if not _ok(yy):
            continue
        for m, s in scores.items():
            for k, f in fns.items():
                boots[m][k].append(f(m, yy, s[i], i))
    all_rows = np.arange(len(y))
    methods = []
    for m, s in scores.items():
        row = {"method": m}
        for k, f in fns.items():
            b = np.asarray(boots[m][k])
            row[k] = f(m, y, s, all_rows)
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = np.quantile(b, [0.025, 0.975])
        seeds = per_seed.get(m)
        row["pr_auc_per_seed"] = [float(pr_auc(y, x)) for x in seeds] if seeds else None
        methods.append(row)
    comps = []
    for k in fns:
        fam = []
        for m in scores:
            if m == reference:
                continue
            d = np.asarray(boots[reference][k]) - np.asarray(boots[m][k])
            entry = {"metric": k, "comparison": f"{reference} - {m}",
                     "diff": fns[k](reference, y, scores[reference], all_rows)
                     - fns[k](m, y, scores[m], all_rows),
                     "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
                     "p_value": _p_two_sided(d)}  # fmt: skip
            if k == "pr_auc":
                a, b = per_seed.get(reference), per_seed.get(m)
                if a and b and len(a) == len(b) and len(a) > 1:
                    es = effect_sizes([pr_auc(y, x) for x in a], [pr_auc(y, x) for x in b])
                    entry.update({"relative_pct": es["relative_pct"],
                                  "cohens_d_paired": es["cohens_d_paired"]})  # fmt: skip
                else:
                    base = fns[k](m, y, scores[m], all_rows)
                    entry.update({"relative_pct": entry["diff"] / base * 100 if base else np.nan,
                                  "cohens_d_paired": np.nan})  # fmt: skip
            fam.append(entry)
        adj = holm([e["p_value"] for e in fam])
        for e, p in zip(fam, adj, strict=True):
            e["p_holm"] = p
        comps.extend(fam)
    return pd.DataFrame(methods), pd.DataFrame(comps)


def subgroup_masks(context_hist: np.ndarray, has_identity: np.ndarray,
                   view_masks: np.ndarray) -> dict[str, np.ndarray]:  # fmt: skip
    """E10 subgroups (D101): cold start vs history, has_identity, graph view available, and
    the number of available views (1-2, 3-4, 5). view_masks: (n, 5) in VIEWS order."""
    n_views = view_masks.sum(axis=1)
    graph = view_masks[:, VIEWS.index("graph")].astype(bool)
    return {
        "cold start": context_hist == 0, "with history": context_hist > 0,
        "has_identity yes": has_identity == 1, "has_identity no": has_identity == 0,
        "graph view available": graph, "graph view missing": ~graph,
        "1-2 views": (n_views >= 1) & (n_views <= 2), "3-4 views": (n_views >= 3) & (n_views <= 4),
        "5 views": n_views == 5,
    }  # fmt: skip


def paired_pr_auc(y, a, b, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    d = [pr_auc(y[i], a[i]) - pr_auc(y[i], b[i]) for i in boot_indices(len(y), n_boot, seed)
         if _ok(y[i])]  # fmt: skip
    return {"diff": pr_auc(y, a) - pr_auc(y, b), "ci_low": float(np.quantile(d, 0.025)),
            "ci_high": float(np.quantile(d, 0.975)), "p_value": _p_two_sided(np.asarray(d))}  # fmt: skip


def e10_subgroups(y, scores: dict[str, np.ndarray], masks: dict[str, np.ndarray],
                  reference: str = "MVAF", versus=("F3", "F4"), n_boot: int = N_BOOT) -> pd.DataFrame:  # fmt: skip
    rows = []
    y = np.asarray(y).astype(int)
    for name, m in masks.items():
        row = {"subgroup": name, "rows": int(m.sum()), "frauds": int(y[m].sum())}
        if _ok(y[m]):
            row.update({k: pr_auc(y[m], s[m]) for k, s in scores.items()})
            for v in versus:
                r = paired_pr_auc(y[m], scores[reference][m], scores[v][m], n_boot)
                row[f"{reference} - {v}"] = (r["diff"], r["ci_low"], r["ci_high"], r["p_value"])
        rows.append(row)
    return pd.DataFrame(rows)


def h2_verdict(comparisons: pd.DataFrame, subgroups: pd.DataFrame, alpha: float = 0.05) -> dict:
    """D101/D102: H2 is supported only if MVAF beats F3 AND F4 overall (PR-AUC, positive and
    Holm-significant) AND its margin over each on rows with missing views (1-4 views) is larger
    than on rows with all 5 views."""
    pr = comparisons[comparisons["metric"] == "pr_auc"].set_index("comparison")
    overall = {v: bool(pr.loc[f"MVAF - {v}", "diff"] > 0 and pr.loc[f"MVAF - {v}", "p_holm"] < alpha)
               for v in ("F3", "F4")}  # fmt: skip
    sub = subgroups.set_index("subgroup")

    def margin(group, v):
        val = sub.loc[group].get(f"MVAF - {v}") if group in sub.index else None
        return val[0] if isinstance(val, tuple) else np.nan

    larger = {}
    for v in ("F3", "F4"):
        missing = [margin(g, v) for g in ("1-2 views", "3-4 views")]
        missing = [m for m in missing if np.isfinite(m)]
        full = margin("5 views", v)
        larger[v] = bool(missing and np.isfinite(full) and min(missing) > full)
    supported = all(overall.values()) and all(larger.values())
    return {"beats_overall": overall, "larger_margin_on_missing_views": larger,
            "supported": supported}  # fmt: skip


def e11_drops(y, full: dict[str, np.ndarray], removed: dict[str, dict[str, np.ndarray]],
              reference: str = "MVAF", versus=("F3", "F4"), n_boot: int = N_BOOT,
              seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """(drops, comparisons). removed[condition][method] = scores with that view removed for all
    rows. Drop = PR-AUC(full) - PR-AUC(removed), with a bootstrap CI. Comparisons: reference's
    drop minus each `versus` method's drop, paired on the same resamples; Holm across the
    single-view conditions (one family per `versus` method); 'tabular only' is reported but
    not in the Holm family."""
    y = np.asarray(y).astype(int)
    idx = [i for i in boot_indices(len(y), n_boot, seed) if _ok(y[i])]
    drops, boot_drop = [], {}
    for cond, by_method in removed.items():
        for m, s in by_method.items():
            point = pr_auc(y, full[m]) - pr_auc(y, s)
            b = np.array([pr_auc(y[i], full[m][i]) - pr_auc(y[i], s[i]) for i in idx])
            boot_drop[(cond, m)] = b
            drops.append({"condition": cond, "method": m, "pr_auc_full": pr_auc(y, full[m]),
                          "pr_auc_removed": pr_auc(y, s), "drop": point,
                          "ci_low": float(np.quantile(b, 0.025)),
                          "ci_high": float(np.quantile(b, 0.975))})  # fmt: skip
    comps = []
    for v in versus:
        fam = []
        for cond, by_method in removed.items():
            if reference not in by_method or v not in by_method:
                continue
            d = boot_drop[(cond, reference)] - boot_drop[(cond, v)]
            point = ((pr_auc(y, full[reference]) - pr_auc(y, by_method[reference]))
                     - (pr_auc(y, full[v]) - pr_auc(y, by_method[v])))  # fmt: skip
            fam.append({"condition": cond, "comparison": f"drop {reference} - drop {v}",
                        "diff": point, "ci_low": float(np.quantile(d, 0.025)),
                        "ci_high": float(np.quantile(d, 0.975)), "p_value": _p_two_sided(d)})  # fmt: skip
        single = [e for e in fam if e["condition"] != "tabular only"]
        adj = holm([e["p_value"] for e in single]) if single else []
        for e, p in zip(single, adj, strict=True):
            e["p_holm"] = p
        for e in fam:
            e.setdefault("p_holm", np.nan)
        comps.extend(fam)
    return pd.DataFrame(drops), pd.DataFrame(comps)
