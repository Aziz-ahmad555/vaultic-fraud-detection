"""POST-HOC / EXPLORATORY: why F3 collapsed on the test period in the EXP-200-final run (D117).

Run:  python -m vaultic.fusion.f3_diagnosis            rebuild the final view table (cached)
      python -m vaultic.fusion.f3_diagnosis --reuse    reuse the cached inputs

Not a pre-registered experiment and not part of the final results: it reads the test period
after the final run, changes no reported number and writes only research/f3_collapse_diagnosis.md
and research/tables/f3_diagnosis.json. It rebuilds the final run's fusion inputs with the same
frozen pipeline (build_table(final=True), calibrate_views, fusion_split), refits F3 exactly as
the final run did (D95-tuned C, seeds 0-4, fit + tune rows) and first checks that it reproduces
the stored F3 test scores of run 20261011-002400-202994. Then:

  1. inputs  F3's standardised coefficient for each input (the five view logits, missing -> 0,
             and the five context columns) and the shift of each input between the gate rows
             (fit + tune, where F3 was fitted) and the test rows: PSI (10 gate-quantile bins),
             KS statistic, missing share, mean shift in the units F3 sees (standardised by the
             gate rows) and coefficient x shift; overall, on rows with history and on 5-view rows
  2. swaps   F3's test PR-AUC with one input at a time mapped onto its gate-row distribution
             (quantile mapping of the available test values: ranks kept, marginal shift removed;
             missingness unchanged), overall and per subgroup; also all views / all context
  3. time    F3 / MVAF test PR-AUC and the history count per 7-day block
  4. E5      the graph view's own shift, for the link to the E5 graph-feature drift (D99/D100)
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from vaultic.paths import INTERIM_DIR, REPO_ROOT, RESEARCH_DIR

CACHE = INTERIM_DIR / "f3_diagnosis_inputs.npz"
FINAL_RUN = REPO_ROOT / "experiments/runs/EXP-200-final/20261011-002400-202994"
OUT_MD = RESEARCH_DIR / "f3_collapse_diagnosis.md"
OUT_JSON = RESEARCH_DIR / "tables" / "f3_diagnosis.json"
VIEWS = ("tabular", "behavioral", "temporal", "graph", "anomaly")
CONTEXT = ("ctx_log_amount", "ctx_hist_n_past", "ctx_has_identity", "ctx_product", "ctx_hour")
INPUTS = tuple(f"view_{v}" for v in VIEWS) + CONTEXT
SEEDS = (0, 1, 2, 3, 4)


def build_inputs(reuse: bool) -> dict[str, np.ndarray]:
    if reuse and CACHE.exists():
        return dict(np.load(CACHE))
    from vaultic.fusion.data import fusion_split
    from vaultic.fusion.dev_compare import build_table
    from vaultic.fusion.tune_fusion import concat_rows
    from vaultic.views.orchestrate import calibrate_views

    table = build_table(False, d95=True, final=True)
    table, _ = calibrate_views(table)
    split = fusion_split(table)
    gate = concat_rows(split.fit, split.tune)
    out = {}
    for name, rows in (("gate", gate), ("cal", split.calibrate), ("test", split.test)):
        for f in ("ids", "views", "context", "y", "day"):
            out[f"{name}_{f}"] = np.asarray(getattr(rows, f))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def fit_f3(d: dict) -> list:
    from vaultic.fusion.tune_fusion import fit_seeds

    tuned = json.loads((RESEARCH_DIR / "tables" / "fusion_d95.json").read_text("utf-8"))["tuned"]

    class _Rows:
        views, context, y = d["gate_views"], d["gate_context"], d["gate_y"]

    return fit_seeds("F3", tuned["F3"]["params"], _Rows, SEEDS)


def f3_scores(models, views, context) -> np.ndarray:
    return np.mean([m.predict_proba(views, context) for m in models], axis=0)


def input_matrix(views, context) -> np.ndarray:
    """What F3 sees, before standardising: view logits (missing -> 0) and the context."""
    from vaultic.fusion.baselines import _stack

    return _stack(views, context, fill=0.0)


def psi(ref: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    ref, new = ref[np.isfinite(ref)], new[np.isfinite(new)]
    if len(ref) < 2 or len(new) < 2:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    n = np.histogram(new, edges)[0] / len(new) + 1e-4
    return float(np.sum((n - r) * np.log(n / r)))


def ks(ref: np.ndarray, new: np.ndarray) -> float:
    from scipy.stats import ks_2samp

    ref, new = ref[np.isfinite(ref)], new[np.isfinite(new)]
    return float(ks_2samp(ref, new).statistic) if len(ref) and len(new) else float("nan")


def raw_input(views, context, j: int) -> np.ndarray:
    """Input j on its natural scale: calibrated view probability (NaN if missing) or context."""
    return views[:, j] if j < len(VIEWS) else context[:, j - len(VIEWS)]


def subgroup_masks(views, context) -> dict[str, np.ndarray]:
    hist = context[:, CONTEXT.index("ctx_hist_n_past")]
    n_views = (~np.isnan(views)).sum(axis=1)
    return {"all rows": np.ones(len(hist), bool), "cold start": hist == 0,
            "with history": hist > 0, "1-2 views": (n_views >= 1) & (n_views <= 2),
            "3-4 views": (n_views >= 3) & (n_views <= 4), "5 views": n_views == 5}  # fmt: skip


def shift_table(models, d) -> list[dict]:
    """Per input: coefficient (mean over seeds, standardised scale) and gate -> test shift."""
    coefs = np.array([m.model_[-1].coef_[0] for m in models])  # (seeds, inputs)
    scaler_mean = np.mean([m.model_[0].mean_ for m in models], axis=0)
    scaler_scale = np.mean([m.model_[0].scale_ for m in models], axis=0)
    xg = input_matrix(d["gate_views"], d["gate_context"])
    xt = input_matrix(d["test_views"], d["test_context"])
    mg = subgroup_masks(d["gate_views"], d["gate_context"])
    mt = subgroup_masks(d["test_views"], d["test_context"])
    rows = []
    for j, name in enumerate(INPUTS):
        row = {
            "input": name,
            "coef": float(coefs[:, j].mean()),
            "coef_sd": float(coefs[:, j].std()),
        }
        for group in ("all rows", "with history", "5 views"):
            g, t = mg[group], mt[group]
            rg = raw_input(d["gate_views"][g], d["gate_context"][g], j)
            rt = raw_input(d["test_views"][t], d["test_context"][t], j)
            z = (xt[t, j].mean() - xg[g, j].mean()) / scaler_scale[j]
            row[group] = {
                "psi": psi(rg, rt), "ks": ks(rg, rt),
                "missing_gate": float(np.mean(np.isnan(rg))), "missing_test": float(np.mean(np.isnan(rt))),
                "mean_gate": float(np.nanmean(rg)) if np.isfinite(rg).any() else float("nan"),
                "mean_test": float(np.nanmean(rt)) if np.isfinite(rt).any() else float("nan"),
                "max_gate": float(np.nanmax(rg)) if np.isfinite(rg).any() else float("nan"),
                "max_test": float(np.nanmax(rt)) if np.isfinite(rt).any() else float("nan"),
                "z_shift": float(z), "coef_x_shift": float(coefs[:, j].mean() * z),
            }  # fmt: skip
        row["scaler_mean"], row["scaler_scale"] = float(scaler_mean[j]), float(scaler_scale[j])
        rows.append(row)
    return rows


def quantile_map(test_vals: np.ndarray, gate_vals: np.ndarray) -> np.ndarray:
    """Available test values mapped onto the gate-row distribution by rank (NaN stays NaN)."""
    from scipy.stats import rankdata

    out = test_vals.copy()
    ok = np.isfinite(test_vals)
    ref = gate_vals[np.isfinite(gate_vals)]
    if ok.sum() and len(ref):
        q = (rankdata(test_vals[ok]) - 0.5) / ok.sum()
        out[ok] = np.quantile(ref, q)
    return out


def _pr_auc(y, s) -> float:
    """PR-AUC, or NaN for a subgroup without both classes."""
    from vaultic.eval.metrics import pr_auc

    return float(pr_auc(y, s)) if 0 < np.sum(y) < len(y) else float("nan")


def swap_table(models, d) -> list[dict]:
    y = d["test_y"]
    masks = subgroup_masks(d["test_views"], d["test_context"])

    def scores_to_row(label, views, context):
        s = f3_scores(models, views, context)
        return {"swap": label, **{g: _pr_auc(y[m], s[m]) for g, m in masks.items()}}

    rows = [scores_to_row("none (as in the final run)", d["test_views"], d["test_context"])]
    for j, name in enumerate(INPUTS):
        v, c = d["test_views"].copy(), d["test_context"].copy()
        if j < len(VIEWS):
            v[:, j] = quantile_map(v[:, j], d["gate_views"][:, j])
        else:
            k = j - len(VIEWS)
            c[:, k] = quantile_map(c[:, k], d["gate_context"][:, k])
        rows.append(scores_to_row(name, v, c))
    v = d["test_views"].copy()
    for j in range(len(VIEWS)):
        v[:, j] = quantile_map(v[:, j], d["gate_views"][:, j])
    rows.append(scores_to_row("all five views", v, d["test_context"]))
    c = d["test_context"].copy()
    for k in range(len(CONTEXT)):
        c[:, k] = quantile_map(c[:, k], d["gate_context"][:, k])
    rows.append(scores_to_row("all five context columns", d["test_views"], c))
    return rows


def time_table(models, d) -> list[dict]:
    from vaultic.eval.metrics import pr_auc

    stored = pd.read_parquet(FINAL_RUN / "predictions.parquet").set_index("TransactionID")
    mvaf = stored["score_MVAF"].reindex(d["test_ids"]).to_numpy()
    s = f3_scores(models, d["test_views"], d["test_context"])
    y, day = d["test_y"], d["test_day"]
    hist = d["test_context"][:, CONTEXT.index("ctx_hist_n_past")]
    hist_gate = d["gate_context"][:, CONTEXT.index("ctx_hist_n_past")]
    rows = []
    start = int(day.min())
    for first in range(start, int(day.max()) + 1, 7):
        m = (day >= first) & (day < first + 7)
        rows.append({"days": f"{first}-{min(first + 6, int(day.max()))}", "rows": int(m.sum()),
                     "frauds": int(y[m].sum()), "F3": float(pr_auc(y[m], s[m])),
                     "MVAF": float(pr_auc(y[m], mvaf[m])),
                     "hist_mean": float(hist[m].mean()),
                     "share_hist_above_gate_max": float(np.mean(hist[m] > hist_gate.max()))})  # fmt: skip
    return rows


def main() -> None:
    from vaultic.eval.metrics import pr_auc

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="reuse the cached fusion inputs")
    args = parser.parse_args()
    d = build_inputs(args.reuse)
    models = fit_f3(d)

    # reproduction check against the stored final-run F3 scores
    stored = pd.read_parquet(FINAL_RUN / "predictions.parquet").set_index("TransactionID")
    ref = stored["score_F3"].reindex(d["test_ids"]).to_numpy()
    s = f3_scores(models, d["test_views"], d["test_context"])
    ok = np.isfinite(ref)
    repro = {"rows": int(len(s)), "matched": int(ok.sum()),
             "max_abs_diff": float(np.max(np.abs(ref[ok] - s[ok]))),
             "corr": float(np.corrcoef(ref[ok], s[ok])[0, 1]),
             "pr_auc_refit": float(pr_auc(d["test_y"], s)),
             "pr_auc_stored": float(pr_auc(d["test_y"][ok], ref[ok]))}  # fmt: skip
    print("reproduction:", repro, flush=True)
    if repro["max_abs_diff"] > 1e-6:
        raise RuntimeError("the refitted F3 does not reproduce the stored final-run scores")

    cal_s = f3_scores(models, d["cal_views"], d["cal_context"])
    gate_s = f3_scores(models, d["gate_views"], d["gate_context"])
    levels = {"gate rows (fit + tune; in-sample)": float(pr_auc(d["gate_y"], gate_s)),
              "calibrate_fused (147-150)": float(pr_auc(d["cal_y"], cal_s)),
              "test (151-182)": repro["pr_auc_refit"]}  # fmt: skip
    result = {"generated": date.today().isoformat(), "post_hoc": True, "reproduction": repro,
              "f3_pr_auc_by_period": levels, "inputs": shift_table(models, d),
              "swaps": swap_table(models, d), "time": time_table(models, d)}  # fmt: skip
    OUT_JSON.write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_md(result)
    print(f"wrote {OUT_JSON} and {OUT_MD}")


def _f(x, nd=4):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_md(r: dict) -> None:
    rp = r["reproduction"]
    lines = [
        "# F3 test-period collapse: diagnosis (POST-HOC / EXPLORATORY)", "",
        "**POST-HOC / EXPLORATORY.** Not pre-registered and not part of the Paper 2 final results "
        "(D101). It was run after the EXP-200-final final run, reads the test period, and changes "
        "no reported number: research/tables/final_e10.md - final_e12.md stand as generated. "
        f"Generated {r['generated']} by `python -m vaultic.fusion.f3_diagnosis` (D117).", "",
        "## Reproduction", "",
        f"The diagnosis rebuilds the final run's fusion inputs with the same frozen pipeline and "
        f"refits F3 (D95-tuned C, seeds 0-4, fit + tune rows). Against the stored F3 test scores of "
        f"run `20261011-002400-202994`: {rp['matched']} of {rp['rows']} rows matched, max |diff| "
        f"{rp['max_abs_diff']:.2e}, correlation {rp['corr']:.6f}; test PR-AUC {rp['pr_auc_refit']:.4f} "
        f"(stored {rp['pr_auc_stored']:.4f}).", "",
        "F3 PR-AUC by period: " + "; ".join(f"{k} {v:.4f}" for k, v in r["f3_pr_auc_by_period"].items())
        + ".", "",
        "## 1. Coefficients and input shift (gate rows -> test rows)", "",
        "Coefficient: F3's logistic-regression weight on the standardised input (mean over the 5 "
        "seeds). PSI / KS on the input's natural scale (calibrated view probability on rows where "
        "the view is available; context as is). z shift: change in the input's mean in the units F3 "
        "sees (gate-row standard deviations; a missing view enters as logit 0). coef x z: the "
        "resulting average change in F3's logit.", "",
    ]  # fmt: skip
    for group in ("all rows", "with history", "5 views"):
        lines += [f"### {group}", "",
                  "| input | coef | PSI | KS | missing gate -> test | mean gate -> test | max gate -> test | z shift | coef x z |",
                  "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for row in r["inputs"]:
            g = row[group]
            lines.append(
                f"| {row['input']} | {row['coef']:+.3f} | {_f(g['psi'], 3)} | {_f(g['ks'], 3)} | "
                f"{g['missing_gate']:.3f} -> {g['missing_test']:.3f} | {_f(g['mean_gate'], 3)} -> "
                f"{_f(g['mean_test'], 3)} | {_f(g['max_gate'], 1)} -> {_f(g['max_test'], 1)} | "
                f"{g['z_shift']:+.2f} | {g['coef_x_shift']:+.2f} |"
            )
        lines.append("")
    groups = ["all rows", "cold start", "with history", "1-2 views", "3-4 views", "5 views"]
    lines += ["## 2. One input at a time mapped onto its gate-row distribution", "",
              "F3's test PR-AUC when the available test values of one input are mapped by rank onto "
              "that input's gate-row distribution (ranks kept, marginal shift removed, missingness "
              "unchanged). An input whose swap restores PR-AUC is the one whose shift drives the "
              "collapse.", "",
              "| swapped input | " + " | ".join(groups) + " |",
              "|---|" + "---|" * len(groups)]  # fmt: skip
    for row in r["swaps"]:
        lines.append(f"| {row['swap']} | " + " | ".join(_f(row[g]) for g in groups) + " |")
    lines += ["", "## 3. Over time (7-day blocks of the test period)", "",
              "| days | rows | frauds | F3 | MVAF | mean ctx_hist_n_past | share above the gate-row max |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for row in r["time"]:
        lines.append(f"| {row['days']} | {row['rows']} | {row['frauds']} | {row['F3']:.4f} | "
                     f"{row['MVAF']:.4f} | {row['hist_mean']:.1f} | "
                     f"{row['share_hist_above_gate_max']:.3f} |")  # fmt: skip
    lines += [
        "",
        "## 4. Interpretation (added by hand, not generated)",
        "",
        "_To be written from the tables above._",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
