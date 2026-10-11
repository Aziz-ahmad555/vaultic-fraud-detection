"""POST-HOC / EXPLORATORY: E10 with label-mature gate folds (D141; disclosure D140).

Run:  python -m vaultic.fusion.gate_maturity_sensitivity     (heavy: rebuilds the view table)

In the D95 cross-fitted plan each gate fold's view models train on every label of their training
window, including labels a 30-day label delay would not yet have released when the fold's
predicted block starts (D140). This rebuilds the view table of the final run with
label_maturity_days = 30 on the three cross-fitted gate folds only; the fixed fold is unchanged,
so the test-period view scores must equal the pre-registered ones (checked against the cached
final-run inputs; it stops otherwise). MVAF and F1-F7 are then refitted with their D95 settings
(fusion_d95.json, not re-tuned), seeds 0-4, on the new gate rows (fit + tune), and E10's test
PR-AUC is reported per method with a bootstrap CI and the paired difference to the
pre-registered EXP-200-final score of the same method (same rows, same 1,000 resamples). B5 and F0
do not use the gate and are unchanged. Reads the test period only through the post-hoc guard
(D144). No pre-registered number changes. Writes research/gate_maturity_sensitivity.md and
research/tables/gate_maturity_sensitivity.json.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from vaultic.paths import RESEARCH_DIR

OUT_MD = RESEARCH_DIR / "gate_maturity_sensitivity.md"
OUT_JSON = RESEARCH_DIR / "tables" / "gate_maturity_sensitivity.json"
GATES = ("MVAF", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
MATURITY_DAYS = 30
SEEDS = (0, 1, 2, 3, 4)
N_BOOT = 1000


def main() -> None:
    from vaultic.eval.metrics import pr_auc
    from vaultic.eval.posthoc_guard import open_test_data, require_final_on_record
    from vaultic.fusion.baselines import make_fusion
    from vaultic.fusion.data import fusion_split
    from vaultic.fusion.dev_compare import build_table
    from vaultic.fusion.f3_diagnosis import CACHE
    from vaultic.fusion.final_eval import boot_indices
    from vaultic.fusion.final_run import _params
    from vaultic.fusion.tune_fusion import concat_rows, fit_seeds
    from vaultic.views.orchestrate import calibrate_views

    run = require_final_on_record()  # D144
    cached = dict(np.load(open_test_data(CACHE)))
    table = build_table(False, d95=True, final=True, gate_label_maturity_days=MATURITY_DAYS)
    table, _ = calibrate_views(table)
    split = fusion_split(table)
    both, test = concat_rows(split.fit, split.tune), split.test
    if not np.array_equal(test.ids, cached["test_ids"]):
        raise RuntimeError("the rebuilt test rows differ from the final run's")
    diff = float(np.nanmax(np.abs(test.views - cached["test_views"])))
    same_missing = bool(np.array_equal(np.isnan(test.views), np.isnan(cached["test_views"])))
    print(f"test view scores vs the final run: max |diff| {diff:.2e}, same missingness {same_missing}",
          flush=True)  # fmt: skip
    if diff > 1e-9 or not same_missing:
        raise RuntimeError(
            "the fixed fold changed: test-period view scores differ from the final run"
        )

    tuned = json.loads((RESEARCH_DIR / "tables" / "fusion_d95.json").read_text("utf-8"))["tuned"]
    stored = pd.read_parquet(run / "predictions.parquet").set_index("TransactionID")
    y = test.y
    idx = [i for i in boot_indices(len(y), N_BOOT, 0) if 0 < y[i].sum() < len(i)]
    old_gate = (len(cached["gate_y"]), int(cached["gate_y"].sum()))
    rows = []
    for name in GATES:
        if name in tuned:
            models = fit_seeds(name, _params(tuned[name]["params"]), both, SEEDS)
        else:
            models = [
                make_fusion(name, seed=s).fit(both.views, both.y, both.context) for s in SEEDS
            ]
        new = np.mean([m.predict_proba(test.views, test.context) for m in models], axis=0)
        old = stored[f"score_{name}"].reindex(test.ids).to_numpy()
        b_new = np.array([pr_auc(y[i], new[i]) for i in idx])
        b_old = np.array([pr_auc(y[i], old[i]) for i in idx])
        d = b_new - b_old
        rows.append({"method": name, "pr_auc_preregistered": float(pr_auc(y, old)),
                     "pr_auc_mature_gate": float(pr_auc(y, new)),
                     "ci_low": float(np.quantile(b_new, 0.025)), "ci_high": float(np.quantile(b_new, 0.975)),
                     "diff": float(pr_auc(y, new) - pr_auc(y, old)),
                     "diff_ci_low": float(np.quantile(d, 0.025)), "diff_ci_high": float(np.quantile(d, 0.975)),
                     "p_value": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})  # fmt: skip
        print(name, rows[-1], flush=True)
    for name in ("B5", "F0"):
        s = stored[f"score_{name}"].reindex(test.ids).to_numpy()
        rows.append({"method": name, "pr_auc_preregistered": float(pr_auc(y, s)),
                     "pr_auc_mature_gate": None, "note": "does not use the gate; unchanged"})  # fmt: skip
    result = {"generated": date.today().isoformat(), "post_hoc": True, "maturity_days": MATURITY_DAYS,
              "final_run": run.name, "test_rows": int(len(y)), "test_frauds": int(y.sum()),
              "gate_rows_preregistered": old_gate, "gate_rows_mature": (len(both.y), int(both.y.sum())),
              "test_view_scores_max_abs_diff": diff, "methods": rows}  # fmt: skip
    OUT_JSON.write_text(json.dumps(result, indent=1), encoding="utf-8")
    from vaultic.reports.render_saved import keep_hand_section

    keep_hand_section(OUT_MD, lambda: write_md(result))
    print(f"wrote {OUT_MD} and {OUT_JSON}")


def write_md(r: dict) -> None:
    g0, g1 = r["gate_rows_preregistered"], r["gate_rows_mature"]
    lines = ["# E10 with label-mature gate folds (POST-HOC / EXPLORATORY)", "",
             "**POST-HOC / EXPLORATORY (D141; disclosure D140).** Not pre-registered; every E10 number in "
             "research/tables/final_e10.md stands as reported. "
             f"Generated {r['generated']} by `python -m vaultic.fusion.gate_maturity_sensitivity`.", "",
             f"The three cross-fitted gate folds are rebuilt with {r['maturity_days']}-day label maturity "
             "(each fold's view models train only on labels released by the start of its predicted block); "
             "the first fold (1-30 -> 31-60) has no released labels under maturity and is left out (D147); "
             "the fixed fold is unchanged, so the test-period view scores equal the final run's (max |diff| "
             f"{r['test_view_scores_max_abs_diff']:.1e}). Gate rows (fit + tune): {g0[0]:,} ({g0[1]:,} frauds) "
             f"pre-registered, {g1[0]:,} ({g1[1]:,} frauds) with mature folds. The gate methods are refitted "
             "with their D95 settings (not re-tuned), seeds 0-4. Test period of final run "
             f"`{r['final_run']}`: {r['test_rows']:,} rows, {r['test_frauds']:,} frauds; 1,000 paired bootstrap "
             "resamples; p unadjusted (exploratory).", "",
             "| method | test PR-AUC, pre-registered | test PR-AUC, mature gate folds [95% CI] | difference [95% CI] | p |",
             "|---|---|---|---|---|"]  # fmt: skip
    for m in r["methods"]:
        if m["pr_auc_mature_gate"] is None:
            lines.append(f"| {m['method']} | {m['pr_auc_preregistered']:.4f} | {m['note']} | | |")
        else:
            lines.append(f"| {m['method']} | {m['pr_auc_preregistered']:.4f} | {m['pr_auc_mature_gate']:.4f} "
                         f"[{m['ci_low']:.4f}, {m['ci_high']:.4f}] | {m['diff']:+.4f} "
                         f"[{m['diff_ci_low']:+.4f}, {m['diff_ci_high']:+.4f}] | {m['p_value']:.3f} |")  # fmt: skip
    lines += ["", "## Reading (added by hand, not generated)", "", "_To be written._", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
