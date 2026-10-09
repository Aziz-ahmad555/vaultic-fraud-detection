"""Equal-budget tuning of the trainable fusion methods (D95, D96).

Every trainable method (MVAF, F3, F4, F5, F6, F7) gets the same number of Optuna trials (TPE,
seed 0). A trial fits the method on the FusionSplit's `fit` rows and scores PR-AUC on its `tune`
rows (the latest 20% of gate-training days, all before the calibrate tail). The best settings
are then refitted with 5 seeds on fit + tune rows. F1 (average) and F2 (fixed weights) have no
hyperparameters searched here; F2's weights are fitted as before.
"""

from __future__ import annotations

import numpy as np

from vaultic.eval.metrics import pr_auc

TRIALS = 50
SEEDS = (0, 1, 2, 3, 4)
HIDDEN = {"16": (16,), "32-32": (32, 32), "64-32": (64, 32)}


def _gate_space(trial, dropout: bool = True) -> dict:
    params = {
        "hidden": HIDDEN[trial.suggest_categorical("hidden", list(HIDDEN))],
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 10, 60),
        "l2": trial.suggest_float("l2", 1e-6, 1e-2, log=True),
    }
    if dropout:
        params["dropout"] = trial.suggest_float("dropout", 0.05, 0.5)
    return params


def space(name: str, trial) -> dict:
    """The search space of one method (F6 has no view dropout by definition)."""
    if name in ("MVAF", "F5"):
        return _gate_space(trial)
    if name == "F6":
        return _gate_space(trial, dropout=False)
    if name == "F7":
        return {**_gate_space(trial),
                "mofe_weight": trial.suggest_float("mofe_weight", 0.1, 10.0, log=True)}  # fmt: skip
    if name == "F3":
        return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
    if name == "F4":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 800),
                "num_leaves": trial.suggest_int("num_leaves", 7, 63)}  # fmt: skip
    raise ValueError(f"{name} has no search space")


TRAINABLE = ("MVAF", "F3", "F4", "F5", "F6", "F7")


def params_from(name: str, best: dict) -> dict:
    """Optuna's best_params back into constructor arguments."""
    out = dict(best)
    if "hidden" in out:
        out["hidden"] = HIDDEN[out["hidden"]]
    return out


def tune_method(name: str, split, n_trials: int = TRIALS, seed: int = 0) -> dict:
    """Best settings of one method on the inner split: fit rows -> PR-AUC on tune rows."""
    import optuna

    from vaultic.fusion.baselines import make_fusion

    def objective(trial):
        model = make_fusion(name, seed=seed, **space(name, trial))
        model.fit(split.fit.views, split.fit.y, split.fit.context)
        return pr_auc(split.tune.y, model.predict_proba(split.tune.views, split.tune.context))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))  # fmt: skip
    study.optimize(objective, n_trials=n_trials)
    return {"params": params_from(name, study.best_params), "tune_pr_auc": study.best_value,
            "trials": len(study.trials)}  # fmt: skip


def fit_seeds(name: str, params: dict, rows, seeds=SEEDS) -> list:
    """The method refitted with each seed on `rows` (fit + tune)."""
    from vaultic.fusion.baselines import make_fusion

    return [
        make_fusion(name, seed=s, **params).fit(rows.views, rows.y, rows.context) for s in seeds
    ]


def concat_rows(a, b):
    """Two FusionSplit Rows as one (fit + tune)."""
    from vaultic.fusion.data import Rows

    return Rows(*(np.concatenate([getattr(a, f), getattr(b, f)])
                  for f in ("ids", "views", "context", "y", "day")))  # fmt: skip
