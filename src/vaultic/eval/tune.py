"""Hyperparameter tuning of a baseline on the VALIDATION period only.

Run:  python -m vaultic.eval.tune --name B5 --features b5 --trials-per-arm 25 --config-id EXP-009
      python -m vaultic.eval.tune --name B4 --features raw --model lightgbm --config-id EXP-013
      python -m vaultic.eval.tune --name B1 --features raw_lr --model logistic_regression \
          --grid-c 0.001 0.01 0.1 1 10 --config-id EXP-012

Boosting models (xgboost, lightgbm) get Optuna:
Two studies with the same budget and sampler seed: no class weighting, and
scale_pos_weight = (#legit / #fraud) on the training period. Each trial trains on train; every
CHECK_EVERY rounds the validation PR-AUC (scikit-learn average precision) is computed, and it
drives both Optuna's median pruning and early stopping (EARLY_STOPPING rounds without
improvement). The libraries' own per-round validation evaluation is not used: on this data it
cost ~3.3 s per round, 92% of a trial (research/decisions.md D27). The test period is never
loaded into the model. Studies are stored in experiments/tuning/<name>/study.db (resumable).
Writes a summary to research/tuning_<name>.md and the tuned config
experiments/configs/<config-id>.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from vaultic.data.splits import load_splits
from vaultic.eval.metrics import pr_auc
from vaultic.eval.warn_capture import capture_warnings
from vaultic.features.sets import design_matrix
from vaultic.paths import CONFIG_DIR, REPO_ROOT, RESEARCH_DIR
from vaultic.views.tabular import resolve_device

TUNING_DIR = REPO_ROOT / "experiments" / "tuning"
ARMS = ("none", "scale_pos_weight")
MAX_TREES = 2000
EARLY_STOPPING = 100  # rounds without a better validation PR-AUC
CHECK_EVERY = 25  # rounds between validation PR-AUC checks (pruning and early stopping)
STARTUP_TRIALS = 10  # trials per arm that always run to completion before pruning starts
WARMUP_ROUNDS = 200  # no trial is pruned before this many boosting rounds (D30)
LEARNING_RATE = (0.02, 0.3)


def suggest_params(trial) -> dict:
    """XGBoost search space."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", *LEARNING_RATE, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def suggest_params_lightgbm(trial) -> dict:
    """LightGBM search space, mirroring the XGBoost one (leaves instead of depth)."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", *LEARNING_RATE, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def make_pruner():
    """The one pruner every tuned baseline uses (B3, B4, B5): median pruning, but only after
    STARTUP_TRIALS complete trials and never before WARMUP_ROUNDS boosting rounds (reported
    steps are tree counts), so slow, low-learning-rate trials are not cut off early."""
    import optuna

    return optuna.pruners.MedianPruner(
        n_startup_trials=STARTUP_TRIALS, n_warmup_steps=WARMUP_ROUNDS
    )


class ValidationMonitor:
    """Validation PR-AUC every CHECK_EVERY rounds, shared by XGBoost and LightGBM.

    Each check reports to Optuna (which may prune the trial) and tracks the best tree count;
    check() returns True when EARLY_STOPPING rounds passed without improvement.
    """

    def __init__(self, y_va: np.ndarray, trial=None):
        self.y_va = np.asarray(y_va)
        self.trial = trial
        self.best_score = -np.inf
        self.best_trees = 0

    def check(self, trees: int, proba: np.ndarray) -> bool:
        score = pr_auc(self.y_va, proba)
        if score > self.best_score:
            self.best_score, self.best_trees = score, trees
        if self.trial is not None:
            import optuna

            self.trial.report(score, trees)
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"pruned at {trees} trees")
        return trees - self.best_trees >= EARLY_STOPPING


def fit_trial(
    params, X_tr, y_tr, X_va, y_va, seed=0, trial=None, device="cpu"
) -> tuple[float, int]:
    from xgboost import XGBClassifier
    from xgboost.callback import TrainingCallback

    X_check = np.ascontiguousarray(X_va, dtype=np.float32)
    monitor = ValidationMonitor(y_va, trial)

    class Check(TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            trees = epoch + 1
            if trees % CHECK_EVERY:
                return False
            proba = model.inplace_predict(X_check, iteration_range=(0, trees))
            return monitor.check(trees, proba)

    model = XGBClassifier(
        n_estimators=MAX_TREES,
        callbacks=[Check()],
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    model.fit(X_tr, y_tr)
    return float(monitor.best_score), int(monitor.best_trees)


def fit_trial_lightgbm(
    params, X_tr, y_tr, X_va, y_va, seed=0, trial=None, device="cpu"
) -> tuple[float, int]:
    import lightgbm

    X_check = np.ascontiguousarray(X_va, dtype=np.float32)
    monitor = ValidationMonitor(y_va, trial)

    def check(env):
        trees = env.iteration + 1
        if trees % CHECK_EVERY == 0:
            proba = env.model.predict(X_check, num_iteration=trees)
            if monitor.check(trees, proba):
                raise lightgbm.callback.EarlyStopException(env.iteration, [])

    model = lightgbm.LGBMClassifier(
        n_estimators=MAX_TREES,
        subsample_freq=1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        deterministic=True,
        **(
            {"device_type": os.environ.get("VAULTIC_LIGHTGBM_GPU", "gpu")}
            if device == "cuda"
            else {}
        ),
        **params,
    )
    model.fit(X_tr, y_tr, callbacks=[check])
    return float(monitor.best_score), int(monitor.best_trees)


MODELS = {
    "xgboost": (lambda trial: suggest_params(trial), lambda *a, **k: fit_trial(*a, **k)),
    "lightgbm": (suggest_params_lightgbm, fit_trial_lightgbm),
}


# B1 as redefined in D61: clip to training quantiles, then standardise; lbfgs gets enough
# iterations to converge (convergence is recorded per C, never silenced)
LR_FIXED = {"max_iter": 5000, "clip_quantiles": [0.001, 0.999]}
# D62 flat-curve rule for every grid search: take the smallest C (most regularised) whose
# validation PR-AUC is within FLAT_TOL of the grid's highest; extend the grid only when that
# chosen C is the largest value in it (at_upper_edge). No comparison of neighbours.
FLAT_TOL = 0.001


def choose_c(rows: list[dict], tol: float = FLAT_TOL) -> dict:
    """The smallest C whose validation PR-AUC is within `tol` of the best (D62)."""
    best = max(r["val PR-AUC"] for r in rows)
    return min((r for r in rows if r["val PR-AUC"] >= best - tol), key=lambda r: r["C"])


def at_upper_edge(rows: list[dict], chosen: dict) -> bool:
    """True when the chosen C is the largest in the grid: the grid should be extended."""
    return chosen["C"] == max(r["C"] for r in rows)


def grid_logistic_regression(X_tr, y_tr, X_va, y_va, grid_c) -> list[dict]:
    """B1: one model per C (lbfgs is deterministic, so one seed suffices)."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    from vaultic.views.tabular import make_model

    rows = []
    for c in grid_c:
        started = time.perf_counter()
        model = make_model("logistic_regression", {"C": c, **LR_FIXED}, seed=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_tr, y_tr)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        score = pr_auc(y_va, model.predict_proba(X_va)[:, 1])
        n_iter = int(model[-1].n_iter_[0])
        rows.append({"C": c, "val PR-AUC": score, "seconds": time.perf_counter() - started,
                     "iterations": n_iter, "converged": converged})  # fmt: skip
        print(
            f"[B1 grid] C={c}: val PR-AUC {score:.4f}, {n_iter} iterations, converged {converged}"
        )
    return rows


def run_arm(
    name, arm, X_tr, y_tr, X_va, y_va, trials, storage, seed=0, model="xgboost", device="cpu"
):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=f"{name}-{arm}",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=make_pruner(),
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
            score, n_trees = fit(params, X_tr, y_tr, X_va, y_va, seed, trial, device=device)
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
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--grid-c", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--config-id", required=True, help="id of the tuned config to write")
    parser.add_argument(
        "--from-grid",
        type=Path,
        help="logistic regression: re-apply the selection rule to a saved grid (JSON written by "
        "an earlier run) instead of refitting",
    )
    args = parser.parse_args()
    if args.from_grid:
        rows = json.loads(args.from_grid.read_text(encoding="utf-8"))
        _write_lr_results(args.name, args.features, args.config_id, rows)
        return

    splits = load_splits()
    # the same inputs as the harness (base features plus any attached view files, D90)
    from vaultic.eval.run import load_inputs

    df, base, _ = load_inputs({"features": args.features}, splits)
    X = design_matrix(df, base, args.features)
    y = df["isFraud"].to_numpy()
    part = splits.assign(df["day"])
    tr, va = part == "train", part == "validation"
    X_tr, y_tr, X_va, y_va = X[tr], y[tr], X[va], y[va]
    del df, base, X  # the test period is never handed to the tuner

    with capture_warnings() as caught:
        if args.model == "logistic_regression":
            _tune_logistic_regression(args, X_tr, y_tr, X_va, y_va)
        else:
            _tune_boosting(args, X_tr, y_tr, X_va, y_va)
    _append_warnings(RESEARCH_DIR / f"tuning_{args.name}.md", caught)


def _append_warnings(report: Path, caught) -> None:
    """Warnings raised while tuning go into the tuning report (flagged ones in bold)."""
    summary = caught.summary()
    flag = caught.flag_text()
    lines = ["", "## Warnings during tuning", ""]
    if not summary["groups"]:
        lines.append(f"None (apart from {summary['n_ignored']} deprecation/future warnings).")
    else:
        if flag:
            lines += [f"**{flag}**", ""]
        lines += [f"- {g['count']} x {g['category']} at `{g['origin']}`: {g['message']}"
                  for g in summary["groups"]]  # fmt: skip
    with open(report, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(flag or "no flagged warnings while tuning")


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
    _write_lr_results(args.name, args.features, args.config_id, rows)


def _write_lr_results(name: str, features: str, config_id: str, rows: list[dict]) -> None:
    (RESEARCH_DIR / f"tuning_{name}_grid.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8"
    )
    best = choose_c(rows)
    top = max(rows, key=lambda r: r["val PR-AUC"])
    edge = at_upper_edge(rows, best)
    _write_config(
        config_id,
        f"{name} logistic regression, C chosen on validation from a grid of "
        f"{len(rows)} values (D62 flat-curve rule)",
        "logistic_regression",
        {"C": best["C"], **LR_FIXED},
        features,
    )
    lines = [
        f"# Tuning {name} (feature set `{features}`)",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.eval.tune`. Grid over the "
        "inverse regularisation strength C, one fit per value (lbfgs is deterministic), scored "
        "on the validation period only. Model as redefined in D61: median imputation, clipping "
        "to the training 0.1%/99.9% quantiles, standardisation, lbfgs with up to 5000 iterations.",
        "",
        "| C | val PR-AUC | seconds | iterations | converged |",
        "|---|---|---|---|---|",
        *[f"| {r['C']:g} | {r['val PR-AUC']:.4f} | {r['seconds']:.0f} | {r['iterations']} | "
          f"{'yes' if r['converged'] else '**no**'} |" for r in rows],  # fmt: skip
        "",
        f"Highest val PR-AUC: C = {top['C']:g} ({top['val PR-AUC']:.4f}). Selection rule (D62): "
        f"the smallest C within {FLAT_TOL} of the highest.",
        "",
        f"Chosen: **C = {best['C']:g}** ({best['val PR-AUC']:.4f}). "
        f"Config: `experiments/configs/{config_id}.yaml`.",
        *(
            ["", "**The chosen C is at the upper edge of the grid: extend the grid.**"]
            if edge
            else []
        ),
        "",
        f"Grid rows: `research/tuning_{name}_grid.json`.",
        "",
    ]
    (RESEARCH_DIR / f"tuning_{name}.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"chose C={best['C']:g} ({best['val PR-AUC']:.4f}); upper edge: {edge}; wrote {config_id}.yaml"
    )


def _tune_boosting(args, X_tr, y_tr, X_va, y_va) -> None:
    out_dir = TUNING_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(out_dir / 'study.db').as_posix()}"

    rows, best = [], None
    for arm in ARMS:
        study, weight = run_arm(
            args.name,
            arm,
            X_tr,
            y_tr,
            X_va,
            y_va,
            args.trials_per_arm,
            storage,
            model=args.model,
            device=resolve_device(args.device),
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
        "only; validation PR-AUC is checked every "
        f"{CHECK_EVERY} rounds for median pruning (after {STARTUP_TRIALS} complete trials, never "
        f"before {WARMUP_ROUNDS} rounds) and early stopping (max {MAX_TREES} trees, "
        f"stop after {EARLY_STOPPING} rounds without improvement; learning rate "
        f"{LEARNING_RATE[0]}–{LEARNING_RATE[1]}). Same sampler seed and budget per arm. "
        f"Device: {resolve_device(args.device)} (GPU and CPU results can differ slightly).",
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
