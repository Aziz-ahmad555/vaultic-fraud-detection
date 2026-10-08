"""Table 1: baselines B1-B6, built from harness runs only.

Run:  python -m vaultic.reports.table1 --mode final         (test metrics, for papers)
      python -m vaultic.reports.table1 --mode development   (validation metrics)

Writes research/tables/table1_<mode>.md and .csv. Each row uses the latest run of its
experiment in that mode, or the run named by the row's `run:` key (e.g. the original B1 shown
next to its re-run); rows without such a run are shown as "not run".

Recall@1%FPR, Recall@5%FPR and Precision@500 are recomputed per seed from each run's saved
predictions with the current metric code (D49 fixed their tie handling); every other metric
comes from the run's metrics.json (D61).
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vaultic.paths import CONFIG_DIR, RESEARCH_DIR, RUNS_DIR

SPEC_PATH = CONFIG_DIR / "table1.yaml"
TABLES_DIR = RESEARCH_DIR / "tables"
METRIC_COLUMNS = [
    ("pr_auc", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
    ("recall_at_1pct_fpr", "Recall@1%FPR"),
    ("recall_at_5pct_fpr", "Recall@5%FPR"),
    ("precision_at_500", "Precision@500"),
    ("brier", "Brier"),
    ("ece", "ECE"),
]
FINAL_ONLY = [("f1_at_val_threshold", "F1 (val threshold)"), ("cost_at_val_threshold", "Cost ($)")]
RECOMPUTED = ("recall_at_1pct_fpr", "recall_at_5pct_fpr", "precision_at_500")


def recompute_from_predictions(run_dir: Path, period: str) -> dict[str, dict]:
    """Per-seed values of the D49-affected metrics from predictions.parquet (mean and std)."""
    from vaultic.eval.metrics import precision_at_k, recall_at_fpr

    funcs = {
        "recall_at_1pct_fpr": lambda y, s: recall_at_fpr(y, s, 0.01),
        "recall_at_5pct_fpr": lambda y, s: recall_at_fpr(y, s, 0.05),
        "precision_at_500": lambda y, s: precision_at_k(y, s, 500),
    }
    path = run_dir / "predictions.parquet"
    if not path.exists():  # nothing saved to recompute from: keep the stored values
        return {}
    preds = pd.read_parquet(path)
    part = preds[preds["split"] == period]
    y = part["label"].to_numpy()
    seeds = [c for c in part.columns if c.startswith("score_seed")]
    out = {}
    for key, f in funcs.items():
        values = [f(y, part[c].to_numpy()) for c in seeds]
        out[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return out


def latest_run(experiment: str, mode: str, runs_dir: Path = RUNS_DIR) -> Path | None:
    root = runs_dir / experiment
    if not root.exists():
        return None
    for run_dir in sorted(root.iterdir(), reverse=True):
        metrics = run_dir / "metrics.json"
        if metrics.exists() and json.loads(metrics.read_text(encoding="utf-8")).get("mode") == mode:
            return run_dir
    return None


def _fmt(entry: dict, ci: bool) -> str:
    text = f"{entry['mean']:.4f} ± {entry['std']:.4f}"
    if ci and "ci_low" in entry:
        text += f" [{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]"
    return text


def build_table(spec: dict, mode: str, runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    period = "test" if mode == "final" else "validation"
    metric_cols = METRIC_COLUMNS + (FINAL_ONLY if mode == "final" else [])
    rows = []
    for row in spec["rows"]:
        out = {
            "Baseline": row["baseline"],
            "Model": row["model"],
            "Features": row["features"],
            "Tuning": row["tuning"],
            "Experiment": row["experiment"],
        }
        pinned = row.get("run")
        run_dir = (
            (runs_dir / row["experiment"] / pinned)
            if pinned
            else latest_run(row["experiment"], mode, runs_dir)
        )
        if pinned and not (run_dir / "metrics.json").exists():
            raise FileNotFoundError(f"pinned run {pinned} of {row['experiment']} not found")
        if run_dir is None:
            out.update({label: "not run" for _, label in metric_cols})
            out.update({"Seeds": "", "Training s/seed": "", "Inference ms/1k": "", "Run": ""})
        else:
            result = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
            metrics = {**result[period], **recompute_from_predictions(run_dir, period)}
            for key, label in metric_cols:
                out[label] = _fmt(metrics[key], ci=(key == "pr_auc")) if key in metrics else ""
            timing = info.get("per_seed_timing", [])
            out["Seeds"] = len(result["seeds"])
            out["Training s/seed"] = (
                f"{np.mean([t['train_seconds'] for t in timing]):.0f}" if timing else ""
            )
            out["Inference ms/1k"] = (
                f"{np.mean([t['inference_ms_per_1000'] for t in timing]):.1f}" if timing else ""
            )
            out["Run"] = run_dir.name
        rows.append(out)
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame, mode: str) -> str:
    period = "test period (FINAL runs)" if mode == "final" else "validation period (development)"
    lines = [
        "# Table 1: baselines",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.reports.table1 --mode {mode}` "
        f"from harness runs. Metrics on the **{period}**: mean ± std over seeds; PR-AUC also "
        "shows the 95% bootstrap CI (1,000 resamples). Recall@1%FPR, Recall@5%FPR and "
        "Precision@500 are recomputed from each run's saved predictions with the D49 metric "
        "fixes. Do not edit by hand.",
        "",
        "| " + " | ".join(table.columns) + " |",
        "|" + "---|" * len(table.columns),
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["final", "development"], default="final")
    args = parser.parse_args()
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    table = build_table(spec, args.mode)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / f"table1_{args.mode}.md").write_text(to_markdown(table, args.mode), "utf-8")
    table.to_csv(TABLES_DIR / f"table1_{args.mode}.csv", index=False)
    print(f"wrote research/tables/table1_{args.mode}.md and .csv ({len(table)} rows)")


if __name__ == "__main__":
    main()
