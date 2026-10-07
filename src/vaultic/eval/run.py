"""Phase 1.5 evaluation harness: one YAML config, one command, every reported number.

Run:  python -m vaultic.eval.run experiments/configs/EXP-001.yaml

Trains on the train period (one model per seed), chooses thresholds on validation, scores
the test period, and writes to experiments/runs/<id>/<timestamp>/:
  config.yaml        the exact config used
  metrics.json       test metrics: mean, std, per-seed values and 95% bootstrap CI
                     (deterministic: same config + seeds -> identical file)
  run_info.json      code version (git), data version (file hashes), runtime
  predictions.parquet  TransactionID, TransactionDT, split, label, per-seed and mean score
It also logs to MLflow (experiments/mlruns) when mlflow is installed, and appends one line
to research/experiment_log.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from vaultic.data.load import load_merged
from vaultic.data.splits import SPLITS_PATH, load_splits
from vaultic.eval.bootstrap import seed_mean_ci
from vaultic.eval.metrics import (
    RANKING_METRICS,
    brier,
    choose_cost_threshold,
    choose_f1_threshold,
    cost_at,
    ece,
    f1_at,
    pr_auc,
    precision_at_k,
)
from vaultic.features.pipeline import FEATURES_PATH
from vaultic.features.sets import design_matrix
from vaultic.paths import MERGED_PATH, REPO_ROOT, RESEARCH_DIR, RUNS_DIR
from vaultic.views.tabular import make_model

PRECISION_K = 500
MLRUNS_DIR = REPO_ROOT / "experiments" / "mlruns"


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("id", "question", "model", "features", "seeds"):
        if key not in cfg:
            raise ValueError(f"config is missing '{key}'")
    return cfg


def _git_version() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def evaluate(
    cfg: dict[str, Any], df: pd.DataFrame, base: pd.DataFrame | None, splits
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train per seed, choose thresholds on validation, score test. Pure: no files written."""
    X = design_matrix(df, base, cfg["features"])
    y = df["isFraud"].to_numpy()
    amount = df["TransactionAmt"].to_numpy(dtype=float)
    part = splits.assign(df["day"])
    tr, va, te = (part == p for p in ("train", "validation", "test"))
    if not (tr.any() and va.any() and te.any()):
        raise ValueError("train, validation and test must all be non-empty")

    model_cfg = cfg["model"]
    val_scores, test_scores, per_seed = [], [], []
    for seed in cfg["seeds"]:
        model = make_model(model_cfg["name"], model_cfg.get("params", {}), seed)
        model.fit(X[tr], y[tr])
        s_val = model.predict_proba(X[va])[:, 1]
        s_test = model.predict_proba(X[te])[:, 1]
        t_f1 = choose_f1_threshold(y[va], s_val)
        t_cost = choose_cost_threshold(y[va], s_val, amount[va])
        m = {name: fn(y[te], s_test) for name, fn in RANKING_METRICS.items()}
        m.update(
            {
                f"precision_at_{PRECISION_K}": precision_at_k(y[te], s_test, PRECISION_K),
                "f1_at_val_threshold": f1_at(y[te], s_test, t_f1),
                "brier": brier(y[te], s_test),
                "ece": ece(y[te], s_test),
                "cost_at_val_threshold": cost_at(y[te], s_test, amount[te], t_cost),
                "val_pr_auc": pr_auc(y[va], s_val),
                "threshold_f1": t_f1,
                "threshold_cost": t_cost,
            }
        )
        per_seed.append(m)
        val_scores.append(s_val)
        test_scores.append(s_test)

    boot = cfg.get("bootstrap", {})
    metrics: dict[str, Any] = {}
    for name in per_seed[0]:
        values = [m[name] for m in per_seed]
        entry = {"mean": float(np.mean(values)), "std": float(np.std(values)), "per_seed": values}
        if name in RANKING_METRICS:
            lo, hi = seed_mean_ci(
                y[te], test_scores, RANKING_METRICS[name],
                n_boot=int(boot.get("n", 1000)), seed=int(boot.get("seed", 0)),
            )
            entry.update({"ci_low": lo, "ci_high": hi})
        metrics[name] = entry

    result = {
        "experiment": cfg["id"],
        "question": cfg["question"],
        "rows": {"train": int(tr.sum()), "validation": int(va.sum()), "test": int(te.sum())},
        "test_fraud_rate": float(y[te].mean()),
        "n_features": int(X.shape[1]),
        "seeds": list(cfg["seeds"]),
        "metrics": metrics,
    }

    rows = va | te
    preds = pd.DataFrame(
        {
            "TransactionID": df["TransactionID"].to_numpy()[rows],
            "TransactionDT": df["TransactionDT"].to_numpy()[rows],
            "split": part[rows],
            "label": y[rows],
        }
    )
    for seed, s_val, s_test in zip(cfg["seeds"], val_scores, test_scores):
        col = np.empty(rows.sum())
        col[va[rows]] = s_val
        col[te[rows]] = s_test
        preds[f"score_seed{seed}"] = col
    preds["score"] = preds[[f"score_seed{s}" for s in cfg["seeds"]]].mean(axis=1)
    return result, preds


def _log_mlflow(cfg: dict[str, Any], result: dict[str, Any], out_dir: Path) -> bool:
    try:
        import mlflow
    except ImportError:
        return False
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(cfg["id"])
    with mlflow.start_run(run_name=out_dir.name):
        mlflow.log_params({"model": cfg["model"]["name"], "features": cfg["features"]})
        mlflow.log_metrics({k: v["mean"] for k, v in result["metrics"].items()})
        mlflow.log_artifacts(str(out_dir))
    return True


def run(
    config_path: Path,
    runs_dir: Path = RUNS_DIR,
    experiment_log: Path | None = RESEARCH_DIR / "experiment_log.md",
    data: tuple[pd.DataFrame, pd.DataFrame | None] | None = None,
) -> Path:
    started = time.perf_counter()
    cfg = load_config(config_path)
    splits = load_splits(Path(cfg.get("splits", SPLITS_PATH)))
    if data is None:
        df = load_merged(MERGED_PATH)
        base = pd.read_parquet(FEATURES_PATH) if cfg["features"] == "raw_base" else None
        data_version = {
            p.name: _file_hash(p)
            for p in [MERGED_PATH] + ([FEATURES_PATH] if base is not None else [])
        }
    else:
        df, base = data
        data_version = {"injected": True}

    result, preds = evaluate(cfg, df, base, splits)

    out_dir = runs_dir / cfg["id"] / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_dir.mkdir(parents=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    info = {
        "code_version": _git_version(),
        "data_version": data_version,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    info["mlflow"] = _log_mlflow(cfg, result, out_dir) if data is None else False
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if experiment_log is not None:
        m = result["metrics"]["pr_auc"]
        shown = out_dir.relative_to(REPO_ROOT) if out_dir.is_relative_to(REPO_ROOT) else out_dir
        line = (
            f"| {cfg['id']} | {date.today().isoformat()} | {cfg['question']} | "
            f"PR-AUC {m['mean']:.4f} ± {m['std']:.4f} (95% CI {m['ci_low']:.4f}–{m['ci_high']:.4f}), "
            f"{len(cfg['seeds'])} seeds, run `{shown.as_posix()}` | — |\n"
        )
        with open(experiment_log, "a", encoding="utf-8") as f:
            f.write(line)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    out = run(args.config)
    m = json.loads((out / "metrics.json").read_text())["metrics"]["pr_auc"]
    print(f"{out}: test PR-AUC {m['mean']:.4f} ± {m['std']:.4f} [{m['ci_low']:.4f}, {m['ci_high']:.4f}]")


if __name__ == "__main__":
    main()
