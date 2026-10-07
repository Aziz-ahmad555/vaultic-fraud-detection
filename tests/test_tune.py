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
    assert 0.2 < score <= 1.0 and 1 <= trees <= 2000


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
    assert trial.reported[1] == 0  # pruned at the first check
