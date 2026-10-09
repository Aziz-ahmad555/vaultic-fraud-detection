"""The ONE pre-registered evaluation of MVAF after the D95 changes (validation only, no test rows).

Run:  python -m vaultic.fusion.d95_compare --b5-run <B5 run> --f0-run <F0 inner-split run> [--smoke]

Pre-registered in research/decisions.md D95 (before any of this ran):
  1. view predictions cross-fitted out of time (views/plan.crossfit_plan: 1-30 -> 31-60,
     1-60 -> 61-90, 1-90 -> 91-120, and the fixed fold 1-120 -> validation); gate training = all
     predicted training-period rows (minus each fold's own calibration days) + days 128-143;
  2. every view with its own tuned hyperparameters (EXP-V-<view>.yaml, Optuna 50 trials, trials
     scored on validation days 128-143 only);
  3. every trainable fusion method (MVAF, F3-F7) tuned with the same Optuna budget on the inner
     time split (fusion/tune_fusion.py), then refitted with 5 seeds on fit + tune rows; F1 and F2
     fitted with 5 seeds; F0 re-tuned on 128-143 only (its run's validation predictions);
  4. ONE evaluation on the calibrate_fused days 147-150, reported whatever it shows; the script
     refuses to run again once research/tables/fusion_d95.md exists (no further MVAF changes on
     this slice; the next look is the --final test run).

Scores per method are the mean of the 5 seeds' predictions (B5 and F0: their runs' 5-seed
mean); per-seed PR-AUC mean and std are reported too. Paired bootstrap of MVAF minus each method
and the subgroups of fusion_dev.md. B5 was tuned on the whole validation period (frozen Phase 2
model), so it is favoured on these days.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from vaultic.eval.metrics import pr_auc
from vaultic.paths import INTERIM_DIR, RESEARCH_DIR

ORDER = ("MVAF", "F0", "B5", "F1", "F2", "F3", "F4", "F5", "F6", "F7")


def gate_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Gate-training rows and frauds per fold (D95 asks for them)."""
    gate = table[table["role"] == "gate_train"]
    out = gate.groupby("fold", sort=False)["label"].agg(rows="size", frauds="sum").reset_index()
    total = pd.DataFrame([{"fold": "all", "rows": len(gate), "frauds": int(gate["label"].sum())}])
    return pd.concat([out, total], ignore_index=True)


def main() -> None:
    from vaultic.fusion.baselines import make_fusion
    from vaultic.fusion.data import fusion_split
    from vaultic.fusion.dev_compare import build_table, evaluate, external_scores, subgroup_table
    from vaultic.fusion.tune_fusion import SEEDS, TRAINABLE, concat_rows, fit_seeds, tune_method
    from vaultic.views.orchestrate import VIEWS, calibrate_views

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="1 in 10 uids, 3 trials, 2 seeds")
    parser.add_argument("--b5-run", type=Path, required=True)
    parser.add_argument("--f0-run", type=Path, required=True)
    args = parser.parse_args()
    suffix = "-smoke" if args.smoke else ""
    out = RESEARCH_DIR / "tables" / f"fusion_d95{suffix}.md"
    if out.exists() and not args.smoke:
        raise RuntimeError(f"{out} exists: the pre-registered evaluation runs once (D95)")
    trials, seeds = (3, SEEDS[:2]) if args.smoke else (50, SEEDS)
    started = time.perf_counter()

    table = build_table(args.smoke, d95=True)
    if (table["role"] == "test").any() or (table.get("split", pd.Series()) == "test").any():
        raise RuntimeError("a test-period row reached the development view table")
    table, cal_info = calibrate_views(table)
    table.to_parquet(INTERIM_DIR / f"view_table_d95{suffix}.parquet", index=False)
    counts = gate_counts(table)
    split = fusion_split(table)
    rows, both = split.calibrate, concat_rows(split.fit, split.tune)

    tuned, per_seed, scores = {}, {}, {}
    for name in ("F1", "F2", *TRAINABLE):
        if name in TRAINABLE:
            tuned[name] = tune_method(name, split, n_trials=trials)
            models = fit_seeds(name, tuned[name]["params"], both, seeds)
        else:
            models = [
                make_fusion(name, seed=s).fit(both.views, both.y, both.context) for s in seeds
            ]
        preds = [m.predict_proba(rows.views, rows.context) for m in models]
        per_seed[name] = [pr_auc(rows.y, p) for p in preds]
        scores[name] = np.mean(preds, axis=0)
    for name, run in (("B5", args.b5_run), ("F0", args.f0_run)):
        scores[name] = external_scores(run, rows.ids)
    result = evaluate({}, rows, external=scores)
    sub = subgroup_table(scores, rows)

    cal = table[table["role"] == "calibrate_fused"]
    views = []
    for v in VIEWS:
        m = cal[f"m_{v}"].to_numpy() == 1
        y = cal["label"].to_numpy()[m]
        ok = m.any() and 0 < y.sum() < m.sum()
        views.append({"view": v, "available": float(m.mean()),
                      "pr_auc": pr_auc(y, cal[f"p_{v}"].to_numpy()[m]) if ok else float("nan")})  # fmt: skip

    def fmt(x, d=4):
        return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"

    res = result.set_index("method")
    lines = [
        "# MVAF after the pre-registered changes (D95): the one evaluation on days 147-150"
        + (" — SMOKE (not a result)" if args.smoke else ""),
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.fusion.d95_compare`. "
        "**Validation only; no test rows.** Plan, tuning and this single evaluation were "
        "pre-registered in D95 before any of it ran. "
        f"Evaluation rows: {len(rows)} ({int(rows.y.sum())} frauds), never used to fit or tune "
        "anything here. B5 (frozen Phase 2 model) was tuned on the whole validation period, so it "
        "is favoured on these days.",
        "",
        "## Gate-training rows (cross-fitted, D95)",
        "",
        "| fold | rows | frauds |",
        "|---|---|---|",
        *[f"| {r['fold']} | {r['rows']} | {r['frauds']} |" for r in counts.to_dict("records")],
        "",
        "## Views on the evaluation rows",
        "",
        "| view | available | PR-AUC (available rows) |",
        "|---|---|---|",
        *[f"| {v['view']} | {v['available']:.1%} | {fmt(v['pr_auc'])} |" for v in views],
        "",
        "## Fused and single models",
        "",
        "| method | PR-AUC (5-seed mean score) | 95% CI | per-seed mean ± std | MVAF − method | "
        "95% CI | p | tuned on the inner split |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in [m for m in ORDER if m in res.index]:
        r = res.loc[name]
        seed = per_seed.get(name)
        seed_txt = f"{np.mean(seed):.4f} ± {np.std(seed):.4f}" if seed else "(run's 5 seeds)"
        diff = (
            ""
            if pd.isna(r.get("MVAF - method", np.nan))
            else (
                f"{r['MVAF - method']:+.4f} | [{r['d_low']:+.4f}, {r['d_high']:+.4f}] | {r['p']:.3f}"
            )
        )
        if not diff:
            diff = " | | "
        tune_txt = (f"{tuned[name]['trials']} trials, tune PR-AUC {tuned[name]['tune_pr_auc']:.4f}"
                    if name in tuned else ("F0: Optuna on 128-143" if name == "F0" else "—"))  # fmt: skip
        lines.append(f"| {name} | {r['PR-AUC']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | "
                     f"{seed_txt} | {diff} | {tune_txt} |")  # fmt: skip
    methods = [m for m in ORDER if m in sub.columns]
    versus = [c for c in sub.columns if c.startswith("MVAF - ")]
    lines += [
        "",
        "## Subgroups",
        "",
        "| subgroup | rows | frauds | " + " | ".join(methods) + " | " + " | ".join(versus) + " |",
        "|---|---|---|" + "---|" * (len(methods) + len(versus)),
    ]
    for r in sub.to_dict("records"):
        vals = [fmt(r.get(m)) for m in methods]
        diffs = ["" if not isinstance(r.get(v), tuple) else
                 f"{r[v][0]:+.4f} [{r[v][1]:+.4f}, {r[v][2]:+.4f}]" for v in versus]  # fmt: skip
        lines.append(f"| {r['subgroup']} | {r['rows']} | {r['frauds']} | " + " | ".join(vals)
                     + " | " + " | ".join(diffs) + " |")  # fmt: skip
    lines += ["", f"Runtime {time.perf_counter() - started:.0f} s.", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({
        "gate_counts": counts.to_dict("records"), "views": views,
        "fusion": result.to_dict("records"), "per_seed": per_seed, "tuned": tuned,
        "subgroups": sub.to_dict("records"),
        "calibrators": {f"{f}/{v}": e for (f, v), e in cal_info.items()},
    }, indent=1, default=str), encoding="utf-8")  # fmt: skip
    print(f"wrote {out}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
