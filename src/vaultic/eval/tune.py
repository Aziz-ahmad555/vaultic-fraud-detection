"""Optuna tuning of an XGBoost baseline on the VALIDATION period only.

Run:  python -m vaultic.eval.tune --name B5 --features b5 --trials-per-arm 25

Two studies with the same budget and sampler seed: no class weighting, and
scale_pos_weight = (#legit / #fraud) on the training period. Each trial trains on train with
early stopping on validation PR-AUC (aucpr) and is scored by validation PR-AUC; the test
period is never loaded into the model. Studies are stored in
experiments/tuning/<name>/study.db (resumable). Writes a summary to
research/tuning_<name>.md and the tuned config experiments/configs/<config-id>.yaml.
"""

from __future__ import annotations

import argparse
import time
from datetime import date

import numpy as np
import pandas as pd
import yaml

from vaultic.data.load import load_merged
from vaultic.data.splits import load_splits
from vaultic.eval.metrics import pr_auc
from vaultic.features.pipeline import features_path
from vaultic.features.sets import NEEDS_BASE, design_matrix
from vaultic.paths import CONFIG_DIR, MERGED_PATH, REPO_ROOT, RESEARCH_DIR

TUNING_DIR = REPO_ROOT / "experiments" / "tuning"
ARMS = ("none", "scale_pos_weight")
MAX_TREES = 1000
EARLY_STOPPING = 50
REPORT_EVERY = 25  # boosting rounds between pruning checks
STARTUP_TRIALS = 5  # trials that always run to completion before pruning starts


def suggest_params(trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def _pruning_callback(trial):
    """Report validation aucpr every REPORT_EVERY rounds; stop the trial if Optuna prunes it."""
    import optuna
    from xgboost.callback import TrainingCallback

    class Prune(TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if epoch % REPORT_EVERY == 0:
                trial.report(evals_log["validation_0"]["aucpr"][-1], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"pruned at round {epoch}")
            return False

    return Prune()


def fit_trial(params, X_tr, y_tr, X_va, y_va, seed=0, trial=None) -> tuple[float, int]:
    from xgboost import XGBClassifier

    model = XGBClassifier(
        callbacks=[_pruning_callback(trial)] if trial is not None else None,
        n_estimators=MAX_TREES,
        early_stopping_rounds=EARLY_STOPPING,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    score = pr_auc(y_va, model.predict_proba(X_va)[:, 1])  # uses the best iteration
    return score, int(model.best_iteration) + 1


def run_arm(name, arm, X_tr, y_tr, X_va, y_va, trials, storage, seed=0):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=f"{name}-{arm}",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=STARTUP_TRIALS, n_warmup_steps=0),
        load_if_exists=True,
    )
    weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    def objective(trial):
        params = suggest_params(trial)
        if arm == "scale_pos_weight":
            params["scale_pos_weight"] = weight
        started = time.perf_counter()
        try:
            score, n_trees = fit_trial(params, X_tr, y_tr, X_va, y_va, seed, trial)
        finally:
            trial.set_user_attr("seconds", round(time.perf_counter() - started, 1))
        trial.set_user_attr("n_estimators", n_trees)
        print(f"[{name}/{arm}] trial {trial.number}: val PR-AUC {score:.4f}, {n_trees} trees")
        return score

    finished = [t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")]
    remaining = trials - len(finished)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    return study, weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="e.g. B5")
    parser.add_argument("--features", required=True, help="feature set, e.g. b5 or raw")
    parser.add_argument("--trials-per-arm", type=int, default=25)
    parser.add_argument("--config-id", required=True, help="id of the tuned config to write")
    args = parser.parse_args()

    splits = load_splits()
    df = load_merged(MERGED_PATH)
    base = (
        pd.read_parquet(features_path(splits.uid_variant)) if args.features in NEEDS_BASE else None
    )
    X = design_matrix(df, base, args.features)
    y = df["isFraud"].to_numpy()
    part = splits.assign(df["day"])
    tr, va = part == "train", part == "validation"
    X_tr, y_tr, X_va, y_va = X[tr], y[tr], X[va], y[va]
    del df, base, X  # the test period is never handed to the tuner

    out_dir = TUNING_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(out_dir / 'study.db').as_posix()}"

    rows, best = [], None
    for arm in ARMS:
        study, weight = run_arm(
            args.name, arm, X_tr, y_tr, X_va, y_va, args.trials_per_arm, storage
        )
        t = study.best_trial
        finished = [x for x in study.trials if x.state.name in ("COMPLETE", "PRUNED")]
        rows.append(
            {
                "arm": arm,
                "trials": len(finished),
                "pruned": sum(x.state.name == "PRUNED" for x in finished),
                "best val PR-AUC": t.value,
                "trees": t.user_attrs["n_estimators"],
                "minutes": sum(x.user_attrs.get("seconds", 0) for x in finished) / 60,
                "params": t.params,
                "weight": weight if arm == "scale_pos_weight" else None,
            }
        )
        if best is None or t.value > best["best val PR-AUC"]:
            best = rows[-1]

    params = dict(best["params"])
    params["n_estimators"] = best["trees"]
    if best["weight"] is not None:
        params["scale_pos_weight"] = round(best["weight"], 6)
    cfg = {
        "id": args.config_id,
        "question": f"{args.name} tuned (Optuna, {args.trials_per_arm} trials per weighting arm, "
        f"validation only; winning arm: {best['arm']})",
        "model": {"name": "xgboost", "params": {k: _round(v) for k, v in params.items()}},
        "features": args.features,
        "seeds": [0, 1, 2, 3, 4],
        "bootstrap": {"n": 1000, "seed": 0},
    }
    (CONFIG_DIR / f"{args.config_id}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    lines = [
        f"# Tuning {args.name} (feature set `{args.features}`)",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.eval.tune`. Validation period "
        "only; each trial uses early stopping on validation and median pruning (max "
        f"{MAX_TREES} trees, patience {EARLY_STOPPING}). Same sampler seed and budget per arm.",
        "",
        "| weighting | trials | pruned | best val PR-AUC | trees | total minutes |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r['arm']} | {r['trials']} | {r['pruned']} | {r['best val PR-AUC']:.4f} | {r['trees']} | "
            f"{r['minutes']:.0f} |"
            for r in rows
        ],
        "",
        f"Winner: **{best['arm']}**. Tuned config: `experiments/configs/{args.config_id}.yaml`.",
        "These validation scores are optimistic (the same period chose the hyperparameters and "
        "the stopping point); the harness run of the tuned config gives the comparable numbers.",
        "",
        "Best parameters per arm:",
        "",
        *[f"- {r['arm']}: `{ {k: _round(v) for k, v in r['params'].items()} }`" for r in rows],
        "",
    ]
    (RESEARCH_DIR / f"tuning_{args.name}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"winner: {best['arm']} ({best['best val PR-AUC']:.4f}); wrote {args.config_id}.yaml")


def _round(v):
    return round(float(v), 6) if isinstance(v, float | np.floating) else v


if __name__ == "__main__":
    main()
