"""D53: calibrate tail, inner time split and identical fusion rows for every method."""

import numpy as np
import pytest

from vaultic.data.splits import load_splits
from vaultic.fusion.baselines import make_fusion
from vaultic.fusion.data import fit_all, fusion_split
from vaultic.views.orchestrate import calibrate_views, view_table
from vaultic.views.plan import fixed_plan, gate_plan, plan_from_json, plan_to_json, rolling_plan

SPLITS = load_splits()


def test_splits_define_the_calibrate_tail():
    assert (SPLITS.calibrate.first, SPLITS.calibrate.last) == (144, 150)
    assert SPLITS.calibrate.last == SPLITS.validation.last
    # D76: per-view calibrators on the earlier slice, everything fused on the later one
    assert (SPLITS.calibrate_views.first, SPLITS.calibrate_views.last) == (144, 146)
    assert (SPLITS.calibrate_fused.first, SPLITS.calibrate_fused.last) == (147, 150)


def test_calibrate_slices_must_be_consecutive_and_end_validation():
    from dataclasses import replace

    from vaultic.data.splits import DayRange, validate

    with pytest.raises(ValueError, match="consecutive"):
        validate(replace(SPLITS, calibrate_fused=DayRange(148, 150)))  # gap at 147
    with pytest.raises(ValueError, match="consecutive"):
        validate(replace(SPLITS, calibrate_fused=DayRange(147, 149)))  # not the tail
    with pytest.raises(ValueError, match="together"):
        validate(replace(SPLITS, calibrate_fused=None))


def test_plans_mark_the_calibrate_tail_by_row():
    fixed = fixed_plan(SPLITS)[0]
    days = np.arange(128, 151)
    roles = fixed.row_roles(days)
    assert set(days[roles == "calibrate_views"]) == set(range(144, 147))
    assert set(days[roles == "calibrate_fused"]) == set(range(147, 151))
    assert set(days[roles == "gate_train"]) == set(range(128, 144))
    roll = rolling_plan(SPLITS, final=True)
    assert roll[0].row_roles(np.arange(91, 121)).tolist() == ["gate_train"] * 30  # before the tail
    assert (roll[1].row_roles(np.arange(144, 147)) == "calibrate_views").all()
    assert (roll[1].row_roles(np.arange(147, 151)) == "calibrate_fused").all()
    assert (roll[2].row_roles(np.arange(151, 183)) == "test").all()  # test folds never calibrate
    assert plan_from_json(plan_to_json(roll)) == roll


@pytest.fixture(scope="module")
def table():
    from test_view_orchestration import _encoder, _synthetic, _views

    df, features, _ = _synthetic()
    raw = view_table(df, features, _views(), gate_plan(SPLITS), _encoder(df), SPLITS)
    with pytest.raises(ValueError, match="not calibrated"):
        fusion_split(raw)
    return calibrate_views(raw)[0]


def test_view_table_roles_and_fusion_split(table):
    fixed = table["fold"] == "fixed"
    assert set(table.loc[fixed & (table["role"] == "calibrate_views"), "day"]) == set(
        range(144, 147)
    )
    assert set(table.loc[~fixed & (table["role"] == "calibrate_views"), "day"]) == {118, 119, 120}
    assert set(table.loc[table["role"] == "calibrate_fused", "day"]) == set(range(147, 151))
    split = fusion_split(table, tune_fraction=0.2)
    fit, tune, cal = set(split.fit.ids), set(split.tune.ids), set(split.calibrate.ids)
    assert not (fit & tune) and not (fit & cal) and not (tune & cal)  # disjoint
    assert split.fit.day.max() < split.tune.day.min()  # inner split is by time
    # the fused-calibration rows are the later slice only; the per-view slice is in no part
    assert split.tune.day.max() < 144 and split.calibrate.day.min() == 147
    views_slice = set(table.loc[table["role"] == "calibrate_views", "TransactionID"])
    assert not views_slice & (fit | tune | cal)
    gate_days = sorted(set(split.fit.day) | set(split.tune.day))
    # rolling_0's block minus its own calibration slice (118-120) + validation 128-143 (D77)
    assert gate_days[0] == 91 and gate_days[-1] == 143 and not {118, 119, 120} & set(gate_days)
    assert len(split.test) == 0  # development plan


def test_every_fusion_method_gets_exactly_the_same_rows(table):
    split = fusion_split(table)
    seen = {}

    class Spy:
        def __init__(self, name):
            self.name = name

        def fit(self, views, y, context=None, sample_weight=None):
            seen[self.name] = (views.copy(), y.copy(), context.copy())
            return self

    names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "MVAF"]
    fit_all(split, {n: (lambda n=n: Spy(n)) for n in names})
    ref = seen["MVAF"]
    for n in names:
        assert all(
            np.array_equal(a, b, equal_nan=True) for a, b in zip(seen[n], ref, strict=True)
        ), n
    # and the real methods train on it
    fitted = fit_all(split, {"F1": lambda: make_fusion("F1"), "F3": lambda: make_fusion("F3")})
    assert np.isfinite(
        fitted["F3"].predict_proba(split.calibrate.views, split.calibrate.context)
    ).all()


def test_inner_split_needs_two_days(table):
    one_day = table[table["day"] == table["day"].min()]
    one_day.attrs["calibrated"] = True
    with pytest.raises(ValueError, match="two days"):
        fusion_split(one_day)


def test_dev_compare_evaluate_scores_methods_on_given_rows():
    """D71: every method is scored on the same rows; MVAF - method is a paired difference."""
    from types import SimpleNamespace

    from vaultic.fusion.dev_compare import evaluate

    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) < 0.15).astype(int)
    rows = SimpleNamespace(y=y, views=np.column_stack([y + rng.normal(0, 0.4, n)] * 5),
                           context=np.zeros((n, 5)))  # fmt: skip

    class Fixed:
        def __init__(self, noise):
            self.noise = noise

        def predict_proba(self, views, context=None):
            return views[:, 0] + np.random.default_rng(1).normal(0, self.noise, len(views))

    table = evaluate({"MVAF": Fixed(0.1), "F1": Fixed(3.0)}, rows).set_index("method")
    assert table.loc["MVAF", "PR-AUC"] > table.loc["F1", "PR-AUC"]
    assert table.loc["F1", "MVAF - method"] > 0 and table.loc["F1", "d_low"] > 0
    assert np.isnan(table.loc["MVAF"].get("MVAF - method", np.nan))


def test_dev_compare_scores_external_runs_on_the_same_rows(tmp_path):
    """D90: B5 and F0 are scored on exactly the evaluation rows, from their validation preds."""
    import pandas as pd

    from vaultic.fusion.dev_compare import external_scores

    preds = pd.DataFrame({"TransactionID": [1, 2, 3, 4], "split": ["validation"] * 3 + ["test"],
                          "score": [0.1, 0.2, 0.3, 0.9]})  # fmt: skip
    preds.to_parquet(tmp_path / "predictions.parquet")
    assert external_scores(tmp_path, np.array([3, 1])).tolist() == [0.3, 0.1]
    with pytest.raises(ValueError, match="lacks"):
        external_scores(tmp_path, np.array([4]))  # a test row is never used


def test_dev_compare_subgroups_from_context_and_masks():
    from types import SimpleNamespace

    from vaultic.fusion.dev_compare import subgroup_masks, subgroup_table

    rng = np.random.default_rng(0)
    n = 800
    y = (rng.random(n) < 0.2).astype(int)
    ctx = np.zeros((n, 5))
    ctx[:, 1] = rng.integers(0, 3, n)  # ctx_hist_n_past
    ctx[:, 2] = rng.integers(0, 2, n)  # ctx_has_identity
    views = np.full((n, 5), 0.5)
    views[rng.random(n) < 0.5, 3] = np.nan  # graph missing for some rows
    rows = SimpleNamespace(y=y, context=ctx, views=views)
    m = subgroup_masks(rows)
    assert (m["cold start"] | m["with history"]).all() and not (
        m["cold start"] & m["with history"]
    ).any()
    assert np.array_equal(m["graph view missing"], np.isnan(views[:, 3]))
    good = y + rng.normal(0, 0.3, n)
    t = subgroup_table({"MVAF": good, "B5": rng.random(n)}, rows).set_index("subgroup")
    diff, lo, hi, _ = t.loc["with history", "MVAF - B5"]
    assert lo > 0 and t.loc["cold start", "rows"] == int(m["cold start"].sum())
    assert "MVAF - F0" not in t.columns  # F0 not given yet


def test_fusion_tuning_spaces_and_equal_budget(table):
    """D95/D96: every trainable method has a space; tuning uses fit -> tune rows only."""
    pytest.importorskip("optuna")
    from vaultic.fusion.tune_fusion import (
        TRAINABLE,
        concat_rows,
        fit_seeds,
        params_from,
        space,
        tune_method,
    )

    class FakeTrial:
        def suggest_categorical(self, n, c):
            return c[0]

        def suggest_float(self, n, lo, hi, log=False):
            return lo

        def suggest_int(self, n, lo, hi):
            return lo

    for name in TRAINABLE:
        assert space(name, FakeTrial())
    assert "dropout" not in space("F6", FakeTrial())
    assert params_from("MVAF", {"hidden": "32-32"})["hidden"] == (32, 32)
    split = fusion_split(table)
    res = tune_method("F3", split, n_trials=3)
    assert res["trials"] == 3 and "C" in res["params"]
    both = concat_rows(split.fit, split.tune)
    assert len(both) == len(split.fit) + len(split.tune) and both.day.max() < 144
    models = fit_seeds("F3", res["params"], both, seeds=(0, 1))
    assert len(models) == 2


def test_d95_gate_counts_and_single_run_guard(table, tmp_path, monkeypatch):
    import sys

    import vaultic.fusion.d95_compare as d95

    counts = d95.gate_counts(table).set_index("fold")
    gate = table[table["role"] == "gate_train"]
    assert (
        counts.loc["all", "rows"] == len(gate)
        and counts.loc["all", "frauds"] == gate["label"].sum()
    )
    monkeypatch.setattr(d95, "RESEARCH_DIR", tmp_path)
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "fusion_d95.md").write_text("done")
    monkeypatch.setattr(sys, "argv", ["d95", "--b5-run", "x", "--f0-run", "y"])
    with pytest.raises(RuntimeError, match="runs once"):
        d95.main()
