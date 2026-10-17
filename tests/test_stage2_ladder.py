"""The ablation ladder on synthetic fusion rows (D121 section 2)."""

import numpy as np
import pytest

from vaultic.eval.metrics import pr_auc
from vaultic.fusion.data import FusionSplit, Rows
from vaultic.fusion.final_eval import e10_table
from vaultic.stage2.ladder import (
    RUNGS,
    bounded_context,
    fit_rung_scores,
    ladder_comparisons,
    rung_metrics,
    rung_name,
    subset_rows,
    tune_rungs,
)
from vaultic.views.orchestrate import CONTEXT


def _rows(n, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    views = np.clip(
        0.1 + 0.5 * y[:, None] * rng.random((n, 5)) + 0.2 * rng.random((n, 5)), 0.01, 0.99
    )
    views[rng.random((n, 5)) < 0.2] = np.nan
    views[:, 0] = np.clip(0.2 + 0.3 * y + 0.2 * rng.random(n), 0.01, 0.99)  # tabular always there
    ctx = rng.random((n, len(CONTEXT)))
    ctx[:, CONTEXT.index("ctx_hist_n_past")] = rng.integers(0, 5000, n)
    return Rows(ids=np.arange(n), views=views, context=ctx, y=y, day=np.zeros(n))


def test_rungs_are_cumulative_and_subset_rows_picks_views():
    assert [len(v) for v in RUNGS.values()] == [1, 2, 3, 4, 5]
    assert all(RUNGS[k][:-1] == RUNGS[k - 1] for k in range(2, 6))
    r = _rows(50, 0)
    sub = subset_rows(r, 3)
    assert sub.views.shape == (50, 3)
    np.testing.assert_array_equal(sub.views[:, 2], r.views[:, 2])  # VIEWS order: temporal 3rd


def test_bounded_context_caps_and_logs_the_count_only():
    ctx = np.array([[1.0, 10.0, 0.0, 2.0, 3.0], [1.0, 99999.0, 1.0, 2.0, 3.0]])
    out = bounded_context(ctx, gate_max=1000.0)
    j = CONTEXT.index("ctx_hist_n_past")
    assert out[:, j] == pytest.approx(np.log1p([10.0, 1000.0]))
    np.testing.assert_array_equal(np.delete(out, j, 1), np.delete(ctx, j, 1))


def test_tuning_and_fitting_a_rung():
    pytest.importorskip("optuna")  # the ladder runs in .venv (EXP-210-final); .venv-dice has none
    split = FusionSplit(fit=_rows(600, 1), tune=_rows(300, 2), calibrate=_rows(300, 3),
                        test=_rows(300, 4))  # fmt: skip
    tuned = tune_rungs(split, n_trials=2, log=lambda m: None)
    assert set(tuned) == {"1", "2", "3", "4"} and tuned["2"]["views"] == ["tabular", "behavioral"]
    s_cal, s_ev, per_seed = fit_rung_scores(split.fit, split.calibrate, split.test,
                                            tuned["2"]["params"], 2, seeds=(0, 1))  # fmt: skip
    assert len(per_seed) == 2 and s_ev == pytest.approx(np.mean(per_seed, axis=0))
    assert s_cal.shape == (300,)


def test_step_comparisons_match_e10_resampling():
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.08).astype(int)
    scores = {rung_name(k): y * k * 0.2 + rng.normal(0, 1, n) for k in range(1, 6)}
    scores["MVAF"] = y * 1.2 + rng.normal(0, 1, n)
    amount = rng.lognormal(3, 1, n)
    th = {m: 0.5 for m in scores}
    steps, refs = ladder_comparisons(y, scores, {}, amount, th, n_boot=200)
    pr = steps[steps["metric"] == "pr_auc"]
    assert list(pr["step"]) == [
        "E3 (+behavioral)",
        "E4 (+temporal)",
        "E5 (+graph)",
        "E8 (+anomaly)",
    ]
    first = pr.iloc[0]
    assert first["diff"] == pytest.approx(pr_auc(y, scores["rung 2"]) - pr_auc(y, scores["rung 1"]))
    assert (pr["p_holm"] >= pr["p_value"]).all()
    # MVAF - rung 5 equals E10's comparison on the same resamples
    _, comps = e10_table(y, {"MVAF": scores["MVAF"], "rung 5": scores["rung 5"]}, {}, amount, th,
                         reference="MVAF", n_boot=200)  # fmt: skip
    e10 = comps[comps["metric"] == "pr_auc"].iloc[0]
    ref = refs[(refs["metric"] == "pr_auc") & (refs["comparison"] == "MVAF - rung 5")].iloc[0]
    assert (ref["diff"], ref["ci_low"], ref["ci_high"], ref["p_value"]) == pytest.approx(
        (e10["diff"], e10["ci_low"], e10["ci_high"], e10["p_value"])
    )
    m = rung_metrics(y, scores, amount, th, n_boot=100)
    assert set(m["method"]) == set(scores)
