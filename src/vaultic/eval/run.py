"""Phase 1.5 evaluation harness: one YAML config, one command, every reported number.

Run:  python -m vaultic.eval.run experiments/configs/EXP-001.yaml           (development)
      python -m vaultic.eval.run experiments/configs/EXP-001.yaml --final   (final run)

Trains on the train period (one model per seed) and scores the VALIDATION period; thresholds
are chosen on validation. Test rows are not predicted at all unless --final is given; final
runs add test metrics and are logged as FINAL. Writes to experiments/runs/<id>/<timestamp>/:
  config.yaml        the exact config used
  metrics.json       validation (and, if final, test) metrics: mean, std, per-seed values and
                     95% bootstrap CI (deterministic: same config + seeds -> identical file)
  run_info.json      code version (git), data version (file hashes), runtime
  predictions.parquet  TransactionID, TransactionDT, split, label, per-seed and mean score
It also logs to MLflow (experiments/mlflow/mlflow.db) when mlflow is installed, and appends one line
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
from vaultic.data.uid import UID_PATH
from vaultic.eval.bootstrap import seed_mean_ci
from vaultic.eval.metrics import (
    RANKING_METRICS,
    brier,
    choose_cost_threshold,
    choose_f1_threshold,
    cost_at,
    ece,
    f1_at,
    precision_at_k,
)
from vaultic.features.pipeline import features_path
from vaultic.features.sets import NEEDS_BASE, NEEDS_UID, design_matrix
from vaultic.paths import MERGED_PATH, REPO_ROOT, RESEARCH_DIR, RUNS_DIR
from vaultic.views.tabular import effective_device, make_model, resolve_device

PRECISION_K = 500
MLFLOW_DIR = REPO_ROOT / "experiments" / "mlflow"  # git-ignored: mlflow.db + mlartifacts/


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


def _hardware(device: str = "cpu") -> dict[str, Any]:
    import os
    import platform

    info = {"cpu": platform.processor(), "logical_cpus": os.cpu_count(), "os": platform.platform()}
    if device == "cuda":
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        info["gpu"] = gpu.stdout.strip() or "unknown"
    return info


def _dvc_hashes() -> dict[str, str]:
    """md5 of each raw file as recorded by DVC (the repo's data/raw/*.dvc pointer files,
    which travel with the code even when the CSVs live elsewhere, e.g. on Kaggle)."""
    hashes = {}
    for dvc_file in sorted((REPO_ROOT / "data" / "raw").glob("*.dvc")):
        out = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))["outs"][0]
        hashes[out["path"]] = out["md5"]
    return hashes


def _per_seed_metrics(
    y: np.ndarray, s: np.ndarray, amount: np.ndarray, t_f1: float | None, t_cost: float | None
) -> dict[str, float]:
    """Metrics of one seed on one period. Thresholded metrics only when thresholds are given
    (i.e. on test, with thresholds chosen on validation)."""
    m = {name: fn(y, s) for name, fn in RANKING_METRICS.items()}
    m[f"precision_at_{PRECISION_K}"] = precision_at_k(y, s, PRECISION_K)
    m["brier"] = brier(y, s)
    m["ece"] = ece(y, s)
    if t_f1 is not None and t_cost is not None:
        m["f1_at_val_threshold"] = f1_at(y, s, t_f1)
        m["cost_at_val_threshold"] = cost_at(y, s, amount, t_cost)
    return m


def _aggregate(
    per_seed: list[dict[str, float]], y: np.ndarray, scores: list[np.ndarray], boot: dict
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in per_seed[0]:
        values = [m[name] for m in per_seed]
        entry = {"mean": float(np.mean(values)), "std": float(np.std(values)), "per_seed": values}
        if name in RANKING_METRICS:
            lo, hi = seed_mean_ci(
                y,
                scores,
                RANKING_METRICS[name],
                n_boot=int(boot.get("n", 1000)),
                seed=int(boot.get("seed", 0)),
            )
            entry.update({"ci_low": lo, "ci_high": hi})
        out[name] = entry
    return out


def evaluate(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    base: pd.DataFrame | None,
    splits,
    final: bool = False,
    device: str = "cpu",
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, float]]]:
    """Train per seed on train and score validation. Only with final=True are test rows
    predicted and test metrics computed. Pure: no files written."""
    X = design_matrix(df, base, cfg["features"])
    y = df["isFraud"].to_numpy()
    amount = df["TransactionAmt"].to_numpy(dtype=float)
    part = splits.assign(df["day"])
    tr, va, te = (part == p for p in ("train", "validation", "test"))
    if not (tr.any() and va.any()):
        raise ValueError("train and validation must both be non-empty")
    if final and not te.any():
        raise ValueError("a final run needs a non-empty test period")

    model_cfg = cfg["model"]
    val_scores, test_scores, val_seed, test_seed, thresholds = [], [], [], [], []
    timings = []
    for seed in cfg["seeds"]:
        model = make_model(model_cfg["name"], model_cfg.get("params", {}), seed, device=device)
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        t1 = time.perf_counter()
        s_val = model.predict_proba(X[va])[:, 1]
        t2 = time.perf_counter()
        timings.append(
            {"train_seconds": t1 - t0, "inference_ms_per_1000": (t2 - t1) / va.sum() * 1e6}
        )
        t_f1 = choose_f1_threshold(y[va], s_val)
        t_cost = choose_cost_threshold(y[va], s_val, amount[va])
        thresholds.append({"f1": t_f1, "cost": t_cost})
        val_scores.append(s_val)
        val_seed.append(_per_seed_metrics(y[va], s_val, amount[va], None, None))
        if final:
            s_test = model.predict_proba(X[te])[:, 1]
            test_scores.append(s_test)
            test_seed.append(_per_seed_metrics(y[te], s_test, amount[te], t_f1, t_cost))

    boot = cfg.get("bootstrap", {})
    result: dict[str, Any] = {
        "experiment": cfg["id"],
        "question": cfg["question"],
        "mode": "final" if final else "development",
        "device": device,
        "uid_variant": cfg.get("uid_variant", splits.uid_variant),
        "rows": {"train": int(tr.sum()), "validation": int(va.sum())},
        "n_features": int(X.shape[1]),
        "seeds": list(cfg["seeds"]),
        "thresholds_chosen_on_validation": thresholds,
        "validation": _aggregate(val_seed, y[va], val_scores, boot),
    }
    if final:
        result["rows"]["test"] = int(te.sum())
        result["test_fraud_rate"] = float(y[te].mean())
        result["test"] = _aggregate(test_seed, y[te], test_scores, boot)

    rows = va | te if final else va
    preds = pd.DataFrame(
        {
            "TransactionID": df["TransactionID"].to_numpy()[rows],
            "TransactionDT": df["TransactionDT"].to_numpy()[rows],
            "split": part[rows],
            "label": y[rows],
        }
    )
    for i, seed in enumerate(cfg["seeds"]):
        col = np.empty(rows.sum())
        col[va[rows]] = val_scores[i]
        if final:
            col[te[rows]] = test_scores[i]
        preds[f"score_seed{seed}"] = col
    preds["score"] = preds[[f"score_seed{s}" for s in cfg["seeds"]]].mean(axis=1)
    return result, preds, timings


def _log_mlflow(cfg: dict[str, Any], result: dict[str, Any], out_dir: Path) -> bool:
    try:
        import mlflow
    except ImportError:
        return False
    # MLflow 3 refuses the plain-folder store; a local SQLite file is its default backend.
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(cfg["id"]) is None:
        mlflow.create_experiment(
            cfg["id"], artifact_location=(MLFLOW_DIR / "mlartifacts" / cfg["id"]).as_uri()
        )
    mlflow.set_experiment(cfg["id"])
    with mlflow.start_run(run_name=out_dir.name):
        mlflow.log_params(
            {
                "model": cfg["model"]["name"],
                "features": cfg["features"],
                "mode": result["mode"],
                "uid_variant": result["uid_variant"],
            }
        )
        for period in ("validation", "test"):
            if period in result:
                mlflow.log_metrics({f"{period}_{k}": v["mean"] for k, v in result[period].items()})
        mlflow.log_artifacts(str(out_dir))
    return True


def previous_final_runs(experiment: str, runs_dir: Path = RUNS_DIR) -> list[Path]:
    root = runs_dir / experiment
    if not root.exists():
        return []
    return [
        d
        for d in sorted(root.iterdir())
        if (d / "metrics.json").exists()
        and json.loads((d / "metrics.json").read_text(encoding="utf-8")).get("mode") == "final"
    ]


def _guard_final_rerun(
    experiment: str, runs_dir: Path, reason: str | None, decisions_log: Path
) -> None:
    """One --final run per experiment. A re-run needs a reason, which is logged."""
    previous = previous_final_runs(experiment, runs_dir)
    if not previous:
        return
    if not reason:
        raise RuntimeError(
            f"{experiment} already has a --final run ({previous[-1].name}). Final runs are not "
            "repeated after seeing results; if a bug forces a re-run, pass --rerun-reason."
        )
    with open(decisions_log, "a", encoding="utf-8") as f:
        f.write(
            f"| FINAL-RERUN | {date.today().isoformat()} | {experiment} | --final re-run after "
            f"`{previous[-1].name}` | — | {reason} | Logged by the harness |\n"
        )


def load_inputs(cfg: dict[str, Any], splits) -> tuple[pd.DataFrame, pd.DataFrame | None, list]:
    """The merged data plus whatever the feature set needs (base features or uid)."""
    df = load_merged(MERGED_PATH)
    inputs = [MERGED_PATH]
    base = None
    variant = cfg.get("uid_variant", splits.uid_variant)
    if cfg["features"] in NEEDS_BASE:
        path = features_path(variant)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing; run python -m vaultic.features.pipeline --uid-variant {variant}"
            )
        base = pd.read_parquet(path)
        inputs.append(path)
    elif cfg["features"] in NEEDS_UID:
        base = pd.read_parquet(UID_PATH, columns=["TransactionID", variant])
        base = base.rename(columns={variant: "uid"})
        inputs.append(UID_PATH)
    return df, base, inputs


def run(
    config_path: Path,
    runs_dir: Path = RUNS_DIR,
    experiment_log: Path | None = RESEARCH_DIR / "experiment_log.md",
    data: tuple[pd.DataFrame, pd.DataFrame | None] | None = None,
    final: bool = False,
    rerun_reason: str | None = None,
    decisions_log: Path = RESEARCH_DIR / "decisions.md",
    device: str | None = None,
) -> Path:
    started = time.perf_counter()
    cfg = load_config(config_path)
    # the device the model really trains on is part of the run's saved config
    device = effective_device(cfg["model"]["name"], resolve_device(device, cfg.get("device")))
    cfg = {**cfg, "device": device}
    if final:
        _guard_final_rerun(cfg["id"], runs_dir, rerun_reason, decisions_log)
    splits = load_splits(Path(cfg.get("splits", SPLITS_PATH)))
    if data is None:
        df, base, inputs = load_inputs(cfg, splits)
        data_version = {p.name: _file_hash(p) for p in inputs}
        data_version["raw_dvc_md5"] = _dvc_hashes()
    else:
        df, base = data
        data_version = {"injected": True}

    result, preds, timings = evaluate(cfg, df, base, splits, final=final, device=device)

    out_dir = runs_dir / cfg["id"] / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_dir.mkdir(parents=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    info = {
        "code_version": _git_version(),
        "data_version": data_version,
        "runtime_seconds": round(time.perf_counter() - started, 1),
        # timings vary between runs, so they live here and not in metrics.json
        "per_seed_timing": timings,
        "hardware": _hardware(device),
    }
    info["mlflow"] = _log_mlflow(cfg, result, out_dir) if data is None else False
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if experiment_log is not None:
        with open(experiment_log, "a", encoding="utf-8") as f:
            f.write(log_line(cfg, result, out_dir))
    return out_dir


def _fmt(entry: dict[str, float]) -> str:
    return (
        f"{entry['mean']:.4f} ± {entry['std']:.4f} "
        f"(95% CI {entry['ci_low']:.4f}–{entry['ci_high']:.4f})"
    )


def log_line(cfg: dict[str, Any], result: dict[str, Any], out_dir: Path) -> str:
    shown = out_dir.relative_to(REPO_ROOT) if out_dir.is_relative_to(REPO_ROOT) else out_dir
    val = f"val PR-AUC {_fmt(result['validation']['pr_auc'])}"
    if result["mode"] == "final":
        summary = f"**FINAL** test PR-AUC {_fmt(result['test']['pr_auc'])}; {val}"
    else:
        summary = f"{val} (development run)"
    return (
        f"| {cfg['id']} | {date.today().isoformat()} | {cfg['question']} | {summary}, "
        f"{len(cfg['seeds'])} seeds, uid `{result['uid_variant']}`, run `{shown.as_posix()}` | — |\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--final",
        action="store_true",
        help="also predict and score the test period (final runs only; logged as FINAL)",
    )
    parser.add_argument(
        "--rerun-reason",
        default=None,
        help="required to repeat a --final run (e.g. a bug fix); logged in decisions.md",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="override the config / VAULTIC_DEVICE (cuda: XGBoost and LightGBM on the GPU)",
    )
    args = parser.parse_args()
    out = run(args.config, final=args.final, rerun_reason=args.rerun_reason, device=args.device)
    result = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    print(f"{out}\n  validation PR-AUC {_fmt(result['validation']['pr_auc'])}")
    if args.final:
        print(f"  FINAL test PR-AUC {_fmt(result['test']['pr_auc'])}")


if __name__ == "__main__":
    main()
