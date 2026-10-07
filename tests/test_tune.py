import numpy as np
import pytest

from vaultic.eval.tune import ARMS, fit_trial, run_arm

optuna = pytest.importorskip("optuna")


def _xy(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = (rng.random(n) < 1 / (1 + np.exp(-(3 * X[:, 0] - 4)))).astype(int)
    return X, y


def test_fit_trial_scores_validation_and_reports_trees():
    X_tr, y_tr = _xy(2000, 0)
    X_va, y_va = _xy(800, 1)
    score, trees = fit_trial({"max_depth": 3, "learning_rate": 0.3}, X_tr, y_tr, X_va, y_va)
    assert 0.2 < score <= 1.0 and trees % 25 == 0 and 25 <= trees <= 2000


def test_arms_are_resumable_and_deterministic(tmp_path):
    X_tr, y_tr = _xy(1500, 0)
    X_va, y_va = _xy(600, 1)
    storage = f"sqlite:///{(tmp_path / 's.db').as_posix()}"
    s1, w = run_arm("T", ARMS[1], X_tr, y_tr, X_va, y_va, trials=2, storage=storage)
    assert len(s1.trials) == 2 and w == pytest.approx((y_tr == 0).sum() / (y_tr == 1).sum())
    # resuming with the same target runs no new trials
    s2, _ = run_arm("T", ARMS[1], X_tr, y_tr, X_va, y_va, trials=2, storage=storage)
    assert len(s2.trials) == 2
    # a fresh study with the same sampler seed proposes the same first parameters
    other = f"sqlite:///{(tmp_path / 'o.db').as_posix()}"
    s3, _ = run_arm("T", ARMS[1], X_tr, y_tr, X_va, y_va, trials=1, storage=other)
    assert s3.trials[0].params == s1.trials[0].params


def test_pruned_trial_stops_training():
    class AlwaysPrune:
        def report(self, value, step):
            self.reported = (value, step)

        def should_prune(self):
            return True

    X_tr, y_tr = _xy(1500, 0)
    X_va, y_va = _xy(600, 1)
    trial = AlwaysPrune()
    with pytest.raises(optuna.TrialPruned):
        fit_trial({"max_depth": 3}, X_tr, y_tr, X_va, y_va, trial=trial)
    assert trial.reported[1] == 25  # pruned at the first check (after 25 trees)


def test_lightgbm_trial_and_pruning():
    from vaultic.eval.tune import fit_trial_lightgbm

    X_tr, y_tr = _xy(2000, 0)
    X_va, y_va = _xy(800, 1)
    score, trees = fit_trial_lightgbm({"num_leaves": 15}, X_tr, y_tr, X_va, y_va)
    assert 0.2 < score <= 1.0 and trees % 25 == 0 and 25 <= trees <= 2000

    class AlwaysPrune:
        def report(self, value, step):
            pass

        def should_prune(self):
            return True

    with pytest.raises(optuna.TrialPruned):
        fit_trial_lightgbm({"num_leaves": 15}, X_tr, y_tr, X_va, y_va, trial=AlwaysPrune())


def test_lightgbm_study_runs(tmp_path):
    X_tr, y_tr = _xy(1500, 0)
    X_va, y_va = _xy(600, 1)
    storage = f"sqlite:///{(tmp_path / 's.db').as_posix()}"
    study, _ = run_arm("L", ARMS[0], X_tr, y_tr, X_va, y_va, 2, storage, model="lightgbm")
    assert len(study.trials) == 2 and "num_leaves" in study.trials[0].params


def test_logistic_regression_grid():
    from vaultic.eval.tune import grid_logistic_regression

    X_tr, y_tr = _xy(1500, 0)
    X_va, y_va = _xy(600, 1)
    rows = grid_logistic_regression(X_tr, y_tr, X_va, y_va, [0.01, 1.0])
    assert [r["C"] for r in rows] == [0.01, 1.0]
    assert all(0 < r["val PR-AUC"] <= 1 for r in rows)


def test_monitor_early_stopping_and_best_score():
    from vaultic.eval.tune import EARLY_STOPPING, ValidationMonitor

    y = np.array([0, 0, 1, 1])
    good, bad = np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.9, 0.8, 0.2, 0.1])
    m = ValidationMonitor(y)
    assert m.check(25, bad) is False
    assert m.check(50, good) is False  # new best at 50 trees
    stops = [m.check(trees, bad) for trees in range(75, 50 + EARLY_STOPPING + 25, 25)]
    assert stops[-1] is True and not any(stops[:-1])  # stops exactly 100 rounds after 50
    assert (m.best_trees, m.best_score) == (50, 1.0)


def test_trial_reports_every_check_and_returns_the_best():
    reported = []

    class Record:
        def report(self, value, step):
            reported.append((step, value))

        def should_prune(self):
            return False

    X_tr, y_tr = _xy(2000, 0)
    X_va, y_va = _xy(800, 1)
    score, trees = fit_trial(
        {"max_depth": 3, "learning_rate": 0.3}, X_tr, y_tr, X_va, y_va, trial=Record()
    )
    steps = [s for s, _ in reported]
    assert steps == list(range(25, steps[-1] + 1, 25))
    assert score == max(v for _, v in reported) and (trees, score) in reported
