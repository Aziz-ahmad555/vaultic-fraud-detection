"""Emerging-fraud experiment over every held-out group and the Phase 6 exit gate (D69, D94).

Run:  python -m vaultic.reports.emerging configs    (write the per-group configs)
      python -m vaultic.reports.emerging report     (after the runs; validation only)

Protocol (as D69 for ProductCD R): for each group (every ProductCD, and every card4 type with
frauds in validation) the group's training-period frauds are removed (`train_drop_fraud`) and
two development runs are made with B5's frozen hyperparameters: B5 (EXP-009-holdout-<g>) and
B5 + anomaly scores (EXP-108-holdout-<g>). Each is scored on the group's validation rows.

The anomaly view's label-free score is the mean of the rank-normalised global Isolation Forest
and autoencoder scores (views/anomaly_build.py; fitted on legit rows only, so a hold-out does not
change them; the per-customer IF covers almost no rows and is left out). It "beats random" for a
group when the lower bound of its PR-AUC's 95% bootstrap CI (1,000 resamples) on the group's
scored validation rows lies above the group's fraud rate there (the PR-AUC of a random score).

Phase 6 exit gate (roadmap): the anomaly view detects held-out fraud better than random for at
least half the groups; if not, the view is kept as a fusion input only. Measured on validation
here (development; the test period is touched only by --final runs).

Writes research/tables/emerging_fraud.md (+ .json).
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from vaultic.eval.metrics import pr_auc
from vaultic.paths import CONFIG_DIR, RESEARCH_DIR, RUNS_DIR

GROUPS = [("ProductCD", v) for v in ("C", "H", "R", "S", "W")] + [
    ("card4", v) for v in ("american express", "discover", "mastercard", "visa")
]
N_BOOT = 1000


def slug(column: str, value: str) -> str:
    """Config suffix; ProductCD R keeps the name of the first run (D69)."""
    if (column, value) == ("ProductCD", "R"):
        return "R"
    return f"{column}-{value.replace(' ', '-')}"


def write_configs() -> list[str]:
    written = []
    for column, value in GROUPS:
        g = slug(column, value)
        for base, parent, what in (("EXP-009", "EXP-009.yaml", "B5"),
                                   ("EXP-108", "EXP-108-frozen.yaml", "B5 + anomaly")):  # fmt: skip
            path = CONFIG_DIR / f"{base}-holdout-{g}.yaml"
            if path.exists():
                continue
            path.write_text(
                f"# Emerging-fraud experiment (validation only, D69/D94): {what} with frozen B5\n"
                f"# hyperparameters, trained without the training-period frauds of {column} "
                f"= {value}.\n"
                f"extends: {parent}\n"
                f"id: {base}-holdout-{g}\n"
                f'question: "{what} without {column} = {value} training frauds '
                f'(emerging fraud, development)"\n'
                f'train_drop_fraud: {{{column}: "{value}"}}\n',
                encoding="utf-8",
            )
            written.append(path.name)
    return written


def beats_random(y: np.ndarray, score: np.ndarray, seed: int = 0) -> dict:
    """PR-AUC of a score, its 95% bootstrap CI, the base rate, and CI-low > base rate."""
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots.append(pr_auc(y[i], score[i]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    base = float(y.mean())
    return {"pr_auc": pr_auc(y, score), "ci_low": float(lo), "ci_high": float(hi),
            "base_rate": base, "beats_random": bool(lo > base)}  # fmt: skip


def _run_pr_auc(experiment: str, ids: np.ndarray) -> float:
    """Mean over seeds of the run's validation PR-AUC on the given rows (latest run)."""
    runs = sorted(d for d in (RUNS_DIR / experiment).iterdir() if (d / "metrics.json").exists())
    preds = pd.read_parquet(runs[-1] / "predictions.parquet")
    preds = preds[preds["split"] == "validation"].set_index("TransactionID").loc[ids]
    y = preds["label"].to_numpy()
    seeds = [c for c in preds.columns if c.startswith("score_seed")]
    return float(np.mean([pr_auc(y, preds[c].to_numpy()) for c in seeds]))


def report() -> str:
    from vaultic.data.load import load_merged
    from vaultic.data.splits import load_splits
    from vaultic.paths import FEATURES_DIR, MERGED_PATH

    splits = load_splits()
    df = load_merged(MERGED_PATH, columns=["TransactionID", "day", "isFraud", "ProductCD", "card4"])
    anomaly = pd.read_parquet(FEATURES_DIR / f"anomaly_{splits.uid_variant}.parquet")
    df = df.merge(anomaly, on="TransactionID", how="left")
    val = df[splits.validation.contains(df["day"].to_numpy())]
    view = val[["anomaly_if", "anomaly_ae"]].mean(axis=1, skipna=False)
    rows = []
    for column, value in GROUPS:
        g = val[(val[column].astype(str) == value).to_numpy()]
        scored = view.loc[g.index].notna().to_numpy()
        y = g["isFraud"].to_numpy()
        a = beats_random(y[scored], view.loc[g.index].to_numpy()[scored])
        ids = g["TransactionID"].to_numpy()
        rows.append({
            "group": f"{column} = {value}", "rows": len(g), "frauds": int(y.sum()),
            "base_rate": float(y.mean()), **{f"anomaly_{k}": v for k, v in a.items()},
            "B5_full": _run_pr_auc("EXP-009", ids),
            "B5_without": _run_pr_auc(f"EXP-009-holdout-{slug(column, value)}", ids),
            "B5_anomaly_without": _run_pr_auc(f"EXP-108-holdout-{slug(column, value)}", ids),
        })  # fmt: skip
    table = pd.DataFrame(rows)
    passed = int(table["anomaly_beats_random"].sum())
    gate = passed >= int(np.ceil(len(table) / 2))
    lines = [
        "# Emerging fraud: every held-out group, and the Phase 6 exit gate",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.reports.emerging report`. "
        "**Validation period only** (development). Each group's training frauds removed; B5 "
        "frozen hyperparameters; PR-AUC on the group's validation rows (mean over 5 seeds). "
        "Anomaly view = mean of the rank-normalised Isolation Forest and autoencoder scores "
        "(label-free, fitted on legit rows only, unaffected by the hold-out); 'beats random' = "
        "the lower end of its PR-AUC's 95% bootstrap CI lies above the group's fraud rate (D94).",
        "",
        "| held-out group | validation frauds | base rate | anomaly-alone PR-AUC [95% CI] | "
        "beats random | B5 (saw the group's fraud) | B5 without group | B5 + anomaly without group |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {r['group']} | {r['frauds']} | {r['base_rate']:.4f} | {r['anomaly_pr_auc']:.4f} "
          f"[{r['anomaly_ci_low']:.4f}, {r['anomaly_ci_high']:.4f}] | "
          f"{'yes' if r['anomaly_beats_random'] else 'no'} | {r['B5_full']:.4f} | "
          f"{r['B5_without']:.4f} | {r['B5_anomaly_without']:.4f} |" for r in rows],  # fmt: skip
        "",
        f"**Phase 6 exit gate:** the anomaly view beats random for {passed} of {len(table)} "
        f"groups (needed: at least half). **{'PASSED' if gate else 'NOT PASSED'}**"
        + ("" if gate else " — per the roadmap, the anomaly view is kept as a fusion input only.")
        + " (Validation, development; not a test-period result.)",
        "",
    ]
    out = RESEARCH_DIR / "tables" / "emerging_fraud.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"groups": rows, "beats_random": passed, "gate_passed": gate}, indent=1),
        encoding="utf-8",
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["configs", "report"])
    args = parser.parse_args()
    if args.step == "configs":
        print("wrote", write_configs())
    else:
        print(report())


if __name__ == "__main__":
    main()
