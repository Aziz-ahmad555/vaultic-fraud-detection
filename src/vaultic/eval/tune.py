"""Hyperparameter tuning of a baseline on the VALIDATION period only.

Run:  python -m vaultic.eval.tune --name B5 --features b5 --trials-per-arm 25 --config-id EXP-009
      python -m vaultic.eval.tune --name B4 --features raw --model lightgbm --config-id EXP-013
      python -m vaultic.eval.tune --name B1 --features raw_lr --model logistic_regression \
          --grid-c 0.001 0.01 0.1 1 10 --config-id EXP-012

Boosting models (xgboost, lightgbm) get Optuna:
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
    """XGBoost search space."""
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


def suggest_params_lightgbm(trial) -> dict:
    """LightGBM search space, mirroring the XGBoost one (leaves instead of depth)."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def _lightgbm_pruning_callback(trial):
    """Same rule as XGBoost: report validation average precision every REPORT_EVERY rounds."""
    import optuna

    def callback(env):
        if env.iteration % REPORT_EVERY == 0:
            value = next(r[2] for r in env.evaluation_result_list if r[1] == "average_precision")
            trial.report(value, env.iteration)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at round {env.iteration}")

    return callback


def fit_trial_lightgbm(params, X_tr, y_tr, X_va, y_va, seed=0, trial=None) -> tuple[float, int]:
    import lightgbm

    callbacks = [lightgbm.early_stopping(EARLY_STOPPING, first_metric_only=True, verbose=False)]
    if trial is not None:
        callbacks.append(_lightgbm_pruning_callback(trial))
    model = lightgbm.LGBMClassifier(
        n_estimators=MAX_TREES,
        metric="average_precision",
        subsample_freq=1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        deterministic=True,
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
    score = pr_auc(y_va, model.predict_proba(X_va)[:, 1])  # uses the best iteration
    return score, int(model.best_iteration_)


MODELS = {
    "xgboost": (lambda trial: suggest_params(trial), lambda *a, **k: fit_trial(*a, **k)),
    "lightgbm": (suggest_params_lightgbm, fit_trial_lightgbm),
}


def grid_logistic_regression(X_tr, y_tr, X_va, y_va, grid_c) -> list[dict]:
    """B1: one model per C (lbfgs is deterministic, so one seed suffices)."""
    from vaultic.views.tabular import make_model

    rows = []
    for c in grid_c:
        started = time.perf_counter()
        model = make_model("logistic_regression", {"C": c, "max_iter": 1000}, seed=0)
        model.fit(X_tr, y_tr)
        score = pr_auc(y_va, model.predict_proba(X_va)[:, 1])
        rows.append({"C": c, "val PR-AUC": score, "seconds": time.perf_counter() - started})
        print(f"[B1 grid] C={c}: val PR-AUC {score:.4f}")
    return rows


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


def run_arm(name, arm, X_tr, y_tr, X_va, y_va, trials, storage, seed=0, model="xgboost"):
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
    suggest, fit = MODELS[model]

    def objective(trial):
        params = suggest(trial)
        if arm == "scale_pos_weight":
            params["scale_pos_weight"] = weight
        started = time.perf_counter()
        try:
            score, n_trees = fit(params, X_tr, y_tr, X_va, y_va, seed, trial)
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
    parser.add_argument("--model", default="xgboost", choices=[*MODELS, "logistic_regression"])
    parser.add_argument("--trials-per-arm", type=int, default=25)
    parser.add_argument("--grid-c", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0])
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

    if args.model == "logistic_regression":
        _tune_logistic_regression(args, X_tr, y_tr, X_va, y_va)
    else:
        _tune_boosting(args, X_tr, y_tr, X_va, y_va)


def _write_config(config_id: str, question: str, model: str, params: dict, features: str) -> None:
    cfg = {
        "id": config_id,
        "question": question,
        "model": {"name": model, "params": {k: _round(v) for k, v in params.items()}},
        "features": features,
        "seeds": [0, 1, 2, 3, 4],
        "bootstrap": {"n": 1000, "seed": 0},
    }
    (CONFIG_DIR / f"{config_id}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )


def _tune_logistic_regression(args, X_tr, y_tr, X_va, y_va) -> None:
    rows = grid_logistic_regression(X_tr, y_tr, X_va, y_va, args.grid_c)
    best = max(rows, key=lambda r: r["val PR-AUC"])
    _write_config(
        args.config_id,
        f"{args.name} logistic regression, C chosen on validation from a grid of "
        f"{len(rows)} values",
        "logistic_regression",
        {"C": best["C"], "max_iter": 1000},
        args.features,
    )
    lines = [
        f"# Tuning {args.name} (feature set `{args.features}`)",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.eval.tune`. Grid over the "
        "inverse regularisation strength C, one fit per value (lbfgs is deterministic), scored "
        "on the validation period only.",
        "",
        "| C | val PR-AUC | seconds |",
        "|---|---|---|",
        *[f"| {r['C']:g} | {r['val PR-AUC']:.4f} | {r['seconds']:.0f} |" for r in rows],
        "",
        f"Chosen: **C = {best['C']:g}**. Config: `experiments/configs/{args.config_id}.yaml`.",
        "",
    ]
    (RESEARCH_DIR / f"tuning_{args.name}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"chose C={best['C']:g} ({best['val PR-AUC']:.4f}); wrote {args.config_id}.yaml")


def _tune_boosting(args, X_tr, y_tr, X_va, y_va) -> None:
    out_dir = TUNING_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(out_dir / 'study.db').as_posix()}"

    rows, best = [], None
    for arm in ARMS:
        study, weight = run_arm(
            args.name, arm, X_tr, y_tr, X_va, y_va, args.trials_per_arm, storage, model=args.model
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
    if args.model == "lightgbm":
        params["subsample_freq"] = 1
    if best["weight"] is not None:
        params["scale_pos_weight"] = round(best["weight"], 6)
    _write_config(
        args.config_id,
        f"{args.name} tuned (Optuna, {args.trials_per_arm} trials per weighting arm, "
        f"validation only; winning arm: {best['arm']})",
        args.model,
        params,
        args.features,
    )

    lines = [
        f"# Tuning {args.name} ({args.model}, feature set `{args.features}`)",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.eval.tune`. Validation period "
        "only; each trial uses early stopping on validation and median pruning (max "
        f"{MAX_TREES} trees, patience {EARLY_STOPPING}). Same sampler seed and budget per arm.",
        "",
        "| weighting | trials | pruned | best val PR-AUC | trees | total minutes |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r['arm']} | {r['trials']} | {r['pruned']} | {r['best val PR-AUC']:.4f} | "
            f"{r['trees']} | {r['minutes']:.0f} |"
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
