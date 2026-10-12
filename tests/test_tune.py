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


def _study_with_finished_trials(n_finished, steps=range(25, 401, 25)):
    from vaultic.eval.tune import make_pruner

    study = optuna.create_study(direction="maximize", pruner=make_pruner())
    for _ in range(n_finished):
        t = study.ask()
        for step in steps:
            t.report(0.9, step)
        study.tell(t, 0.9)
    return study


def test_no_pruning_before_200_rounds():
    study = _study_with_finished_trials(10)
    slow = study.ask()  # a slow learner, far below the median at every check
    decisions = {}
    for step in range(25, 401, 25):
        slow.report(0.1, step)
        decisions[step] = slow.should_prune()
    assert not any(v for s, v in decisions.items() if s < 200)
    assert bool(decisions[200])  # Optuna returns numpy.bool_


def test_no_pruning_before_ten_finished_trials():
    study = _study_with_finished_trials(9)
    trial = study.ask()
    for step in range(25, 401, 25):
        trial.report(0.1, step)
        assert not trial.should_prune()


def test_b1_grid_reports_convergence_and_writes_the_redefined_model():
    """D61: the grid records iterations and convergence for every C."""
    from vaultic.eval.tune import LR_FIXED, grid_logistic_regression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    y = (X[:, 0] + rng.normal(0, 1, 600) > 1).astype(int)
    rows = grid_logistic_regression(X[:400], y[:400], X[400:], y[400:], [0.1, 1.0])
    assert [r["C"] for r in rows] == [0.1, 1.0]
    assert all(r["converged"] and 0 < r["iterations"] <= LR_FIXED["max_iter"] for r in rows)
    assert LR_FIXED == {"max_iter": 5000, "clip_quantiles": [0.001, 0.999]}


def test_flat_curve_rule_takes_the_smallest_c_within_tolerance():
    """D62: C = 10 vs C = 100 differ by 0.0001 -> C = 10; a clear winner is still taken."""
    from vaultic.eval.tune import at_upper_edge, choose_c

    def grid(scores):
        return [
            {"C": c, "val PR-AUC": v} for c, v in zip([0.1, 1, 10, 100, 1000], scores, strict=True)
        ]

    flat = grid([0.3556, 0.3678, 0.3703, 0.3704, 0.3700])
    assert choose_c(flat)["C"] == 10 and not at_upper_edge(flat, choose_c(flat))
    assert choose_c(grid([0.30, 0.31, 0.32, 0.33, 0.35]))["C"] == 1000
    assert at_upper_edge(grid([0.30, 0.31, 0.32, 0.33, 0.35]), {"C": 1000})
    within = grid([0.3600, 0.3695, 0.3700, 0.3705, 0.3690])  # 1 is 0.001 below the best
    assert choose_c(within)["C"] == 1


def test_from_grid_reselects_without_refitting(tmp_path, monkeypatch):
    import json
    import sys

    import vaultic.eval.tune as tune

    monkeypatch.setattr(tune, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(tune, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "vaultic.eval.run.load_inputs", lambda *a, **k: pytest.fail("data was loaded")
    )
    rows = [{"C": c, "val PR-AUC": v, "seconds": 1.0, "iterations": 10, "converged": True}
            for c, v in [(1.0, 0.3678), (10.0, 0.3703), (100.0, 0.3704)]]  # fmt: skip
    grid = tmp_path / "g.json"
    grid.write_text(json.dumps(rows))
    argv = ["tune", "--name", "BX", "--features", "raw_lr", "--model", "logistic_regression",
            "--config-id", "EXP-X", "--from-grid", str(grid)]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    tune.main()
    cfg = (tmp_path / "EXP-X.yaml").read_text()
    assert "C: 10.0" in cfg and "clip_quantiles" in cfg
    report = (tmp_path / "tuning_BX.md").read_text(encoding="utf-8")
    assert "Chosen: **C = 10**" in report and "upper edge" not in report


def test_view_availability_rules():
    """D95: view models are tuned on the rows where the view exists."""
    import pandas as pd

    from vaultic.views.definitions import available

    X = pd.DataFrame({"hist_n_past": [0, 2, np.nan], "seq_n_steps": [np.nan, 3, 0],
                      "g_shared_nonhub": [0, 1, np.nan], "anomaly_if": [np.nan, 0.4, 0.9]})  # fmt: skip
    assert available("tabular", X).tolist() == [True, True, True]
    assert available("behavioral", X).tolist() == [False, True, False]
    assert available("temporal", X).tolist() == [False, True, False]
    assert available("graph", X).tolist() == [False, True, False]
    assert available("anomaly", X).tolist() == [False, True, True]
    with pytest.raises(ValueError):
        available("nope", X)


def test_view_tuning_uses_available_rows_and_inner_days(tmp_path, monkeypatch):
    """D95: --view keeps the view's rows, --val-days 128 143 never reaches the calibrate tail."""
    import sys

    import pandas as pd

    import vaultic.eval.tune as tune

    rng = np.random.default_rng(0)
    n = 3000
    day = np.sort(rng.integers(1, 151, n))
    df = pd.DataFrame(
        {"TransactionID": np.arange(n), "day": day, "isFraud": (rng.random(n) < 0.1).astype(int)}
    )
    feats = pd.DataFrame({"TransactionID": df["TransactionID"], "x": rng.normal(size=n),
                          "g_shared_nonhub": rng.integers(0, 2, n).astype(float)})  # fmt: skip
    feats["x"] += df["isFraud"]
    seen = {}

    def fake_boosting(args, X_tr, y_tr, X_va, y_va):
        seen.update(train=len(X_tr), val=len(X_va), cols=list(X_tr.columns))

    monkeypatch.setattr("vaultic.eval.run.load_inputs", lambda cfg, s: (df, feats, []))
    monkeypatch.setattr(tune, "_tune_boosting", fake_boosting)
    monkeypatch.setattr(tune, "RESEARCH_DIR", tmp_path)
    argv = ["tune", "--name", "V-graph", "--features", "graph_only", "--view", "graph",
            "--val-days", "128", "143", "--config-id", "EXP-V-graph"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    tune.main()
    avail = feats["g_shared_nonhub"].to_numpy() > 0
    assert seen["train"] == int(((day <= 120) & avail).sum())
    assert seen["val"] == int(((day >= 128) & (day <= 143) & avail).sum())
    bad = [x if x != "143" else "146" for x in argv]  # would reach the calibrate tail (144+)
    monkeypatch.setattr(sys, "argv", bad)
    with pytest.raises(ValueError, match="calibrate tail"):
        tune.main()
