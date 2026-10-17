"""The ablation ladder E3, E4, E5, E8 (D121 section 2): cumulative views fused by F4.

  rung 1  tabular
  rung 2  + behavioral        (E3)
  rung 3  + temporal          (E4)
  rung 4  + graph             (E5)
  rung 5  + anomaly, all five (E8) = E10's F4, reused (stored final-run scores, not re-run)
  refs    MVAF (E10) and B5 (Table 1), reused

Rungs 1-4 use the D95 pipeline unchanged (cross-fitted views, calibrate_views, gate rows): F4
(LightGBM stacking on the rung's calibrated views, missing left NaN, plus the E10 context) is
tuned per rung on the inner split with E10's F4 budget (50 Optuna trials, TPE seed 0;
`tune` mode, validation only) and fitted with seeds 0-4 on the gate rows (fit + tune).

Comparisons: each rung minus the previous one (four steps) for test PR-AUC (primary), recall at
1% FPR and total cost (threshold chosen on calibrate_fused), paired on the same rows and
resamples as E10 (boot_indices(n, 1000, seed 0), one-class resamples skipped as in E10), Holm
across the four steps per metric (H1). MVAF - rung 5 and B5 - rung 5 are reported alongside
(they equal E10's comparisons; not a new family). Subgroups as E10, descriptive.

Bounded-count sensitivity (D117-D119): the context's ctx_hist_n_past replaced by
log1p(min(count, gate-row max)) for every rung (rungs 1-4 with their tuned settings, rung 5
refitted with E10's F4 settings, not re-tuned); reported separately, not in the H1 family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaultic.eval.metrics import pr_auc
from vaultic.eval.stats import effect_sizes
from vaultic.fusion.final_eval import (
    _ci,
    _ok,
    _p_two_sided,
    boot_indices,
    holm_nan,
    metric_fns,
)

RUNGS = {
    1: ("tabular",),
    2: ("tabular", "behavioral"),
    3: ("tabular", "behavioral", "temporal"),
    4: ("tabular", "behavioral", "temporal", "graph"),
    5: ("tabular", "behavioral", "temporal", "graph", "anomaly"),
}
NEW_RUNGS = (1, 2, 3, 4)  # rung 5 is E10's F4
STEP_IDS = {2: "E3 (+behavioral)", 3: "E4 (+temporal)", 4: "E5 (+graph)", 5: "E8 (+anomaly)"}
TRIALS = 50


def rung_name(k: int) -> str:
    return f"rung {k}"


def view_index(k: int) -> list[int]:
    from vaultic.views.orchestrate import VIEWS

    return [VIEWS.index(v) for v in RUNGS[k]]


def subset_rows(rows, k: int, context=None):
    """FusionSplit Rows restricted to rung k's views (and optionally another context)."""
    from vaultic.fusion.data import Rows

    return Rows(ids=rows.ids, views=rows.views[:, view_index(k)],
                context=rows.context if context is None else context, y=rows.y, day=rows.day)  # fmt: skip


def bounded_context(context: np.ndarray, gate_max: float) -> np.ndarray:
    """D119's bounded history count: log1p(min(count, gate-row max)) in ctx_hist_n_past."""
    from vaultic.views.orchestrate import CONTEXT

    out = np.array(context, dtype=float, copy=True)
    j = CONTEXT.index("ctx_hist_n_past")
    out[:, j] = np.log1p(np.minimum(out[:, j], gate_max))
    return out


def tune_rungs(split, n_trials: int = TRIALS, seed: int = 0, log=print) -> dict:
    """F4 tuned per rung 1-4 on the inner split (fit -> tune rows), E10's budget."""
    from vaultic.fusion.data import FusionSplit
    from vaultic.fusion.tune_fusion import tune_method

    out = {}
    for k in NEW_RUNGS:
        sub = FusionSplit(fit=subset_rows(split.fit, k), tune=subset_rows(split.tune, k),
                          calibrate=subset_rows(split.calibrate, k), test=subset_rows(split.test, k))  # fmt: skip
        out[str(k)] = {"views": list(RUNGS[k]), **tune_method("F4", sub, n_trials, seed)}
        log(f"rung {k}: tune PR-AUC {out[str(k)]['tune_pr_auc']:.4f} {out[str(k)]['params']}")
    return out


def fit_rung_scores(both, cal, ev, params: dict, k: int, seeds, context=None) -> tuple:
    """F4 on rung k's views with seeds; 5-seed mean on calibration / evaluation rows and the
    per-seed evaluation scores. context: optional (both, cal, ev) replacement contexts."""
    from vaultic.fusion.tune_fusion import fit_seeds

    ctx = context or (None, None, None)
    rows = [subset_rows(r, k, c) for r, c in zip((both, cal, ev), ctx, strict=True)]
    models = fit_seeds("F4", params, rows[0], seeds)
    s_cal = np.mean([m.predict_proba(rows[1].views, rows[1].context) for m in models], axis=0)
    preds = [m.predict_proba(rows[2].views, rows[2].context) for m in models]
    return s_cal, np.mean(preds, axis=0), preds


def ladder_comparisons(y, scores: dict, per_seed: dict, amount, thresholds: dict,
                       n_boot: int = 1000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """(steps, references). steps: rung k - rung k-1 for k = 2..5 per metric, Holm across the
    four steps per metric (H1); references: MVAF - rung 5 and B5 - rung 5 (not a family)."""
    y = np.asarray(y).astype(int)
    fns = metric_fns(np.asarray(amount, dtype=float), thresholds)
    boots = {m: {k: [] for k in fns} for m in scores}
    for i in boot_indices(len(y), n_boot, seed):  # E10's resamples and skipping rule
        if not _ok(y[i]):
            continue
        for m, s in scores.items():
            for k, f in fns.items():
                boots[m][k].append(f(m, y[i], s[i], i))
    rows_all = np.arange(len(y))

    def compare(a, b, k):
        d = np.asarray(boots[a][k]) - np.asarray(boots[b][k])
        point = fns[k](a, y, scores[a], rows_all) - fns[k](b, y, scores[b], rows_all)
        lo, hi = _ci(d)
        e = {"metric": k, "comparison": f"{a} - {b}", "diff": point, "ci_low": lo, "ci_high": hi,
             "p_value": _p_two_sided(d)}  # fmt: skip
        if k == "pr_auc":
            base = fns[k](b, y, scores[b], rows_all)
            e["relative_pct"] = point / base * 100 if base else np.nan
            sa, sb = per_seed.get(a), per_seed.get(b)
            e["cohens_d_paired"] = np.nan
            if sa and sb and len(sa) == len(sb) > 1:
                e["cohens_d_paired"] = effect_sizes([pr_auc(y, x) for x in sa],
                                                    [pr_auc(y, x) for x in sb])["cohens_d_paired"]  # fmt: skip
        return e

    steps = []
    for k in fns:
        fam = [{"step": STEP_IDS[r], **compare(rung_name(r), rung_name(r - 1), k)}
               for r in range(2, 6)]  # fmt: skip
        for e, p in zip(fam, holm_nan([e["p_value"] for e in fam]), strict=True):
            e["p_holm"] = p
            e["h1_supported"] = bool(k == "pr_auc" and e["diff"] > 0 and p < 0.05)
        steps.extend(fam)
    refs = [compare(r, rung_name(5), k) for k in fns for r in ("MVAF", "B5") if r in scores]
    return pd.DataFrame(steps), pd.DataFrame(refs)


def rung_metrics(y, scores: dict, amount, thresholds: dict, n_boot: int = 1000,
                 seed: int = 0) -> pd.DataFrame:  # fmt: skip
    """Each rung's / reference's metrics with 95% CIs (E10's resamples)."""
    y = np.asarray(y).astype(int)
    fns = metric_fns(np.asarray(amount, dtype=float), thresholds)
    idx = [i for i in boot_indices(len(y), n_boot, seed) if _ok(y[i])]
    rows = []
    for m, s in scores.items():
        row = {"method": m}
        for k, f in fns.items():
            row[k] = f(m, y, s, np.arange(len(y)))
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = _ci([f(m, y[i], s[i], i) for i in idx])
        rows.append(row)
    return pd.DataFrame(rows)


def rung_subgroups(y, scores: dict, masks: dict) -> pd.DataFrame:
    """Descriptive PR-AUC per E10 subgroup and rung (no tests)."""
    y = np.asarray(y).astype(int)
    rows = []
    for name, m in masks.items():
        row = {"subgroup": name, "rows": int(m.sum()), "frauds": int(y[m].sum())}
        if _ok(y[m]):
            row.update({k: pr_auc(y[m], s[m]) for k, s in scores.items()})
        rows.append(row)
    return pd.DataFrame(rows)
