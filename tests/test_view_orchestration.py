"""Roadmap 7.3: out-of-sample view predictions on a small synthetic dataset."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.data.splits import load_splits
from vaultic.features.categories import CategoryEncoder
from vaultic.features.pit import PastIndex
from vaultic.fusion.mvaf import MVAF, view_inputs
from vaultic.views import temporal_step
from vaultic.views.orchestrate import (
    CONTEXT,
    VIEWS,
    SupervisedView,
    calibrate_views,
    has_graph_evidence,
    has_history,
    mvaf_inputs,
    view_table,
)
from vaultic.views.plan import Fold, fixed_plan, plan_from_json, plan_to_json, rolling_plan

SPLITS = load_splits()
ROOT = Path(__file__).resolve().parents[1]
TORCH_PY = ROOT / ".venv-torch" / "Scripts" / "python.exe"


def _synthetic(n=6000, n_uids=150, seed=0):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.integers(SECONDS_PER_DAY, 183 * SECONDS_PER_DAY, n))
    uid = rng.integers(0, n_uids, n).astype(str)
    base = rng.uniform(10, 200, n_uids)[uid.astype(int)]
    fraud = rng.random(n) < 0.06
    amount = base * np.where(fraud, rng.uniform(5, 10, n), rng.uniform(0.7, 1.3, n))
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": time,
            "day": time // SECONDS_PER_DAY,
            "TransactionAmt": amount,
            "ProductCD": rng.choice(["W", "C", "H"], n),
            "isFraud": fraud.astype(int),
            "has_identity": (rng.random(n) < 0.3).astype(int),
        }
    )
    past = PastIndex(uid, time)
    n_past = past.count_before(time)
    mean = past.sum_before(amount, time) / np.maximum(n_past, 1)
    card_seen = PastIndex(rng.integers(0, 2000, n), time).count_before(time)
    features = pd.DataFrame(
        {
            "uid": uid,
            "amt": amount,
            "V1": fraud + rng.normal(0, 1.5, n),
            "hist_n_past": n_past,
            "amt_ratio": np.where(n_past > 0, amount / np.where(n_past > 0, mean, 1), np.nan),
            "g_deg_tx_card": card_seen.astype(float),
            "g_shared_nonhub": np.where(card_seen > 0, rng.poisson(0.8, n), np.nan),
            "g_fraud_rate_card": np.where(card_seen > 0, rng.random(n) * 0.1, np.nan),
        }
    )
    return df, features, pd.Series(uid)


def _views(seed=0):
    params = {"n_estimators": 30, "max_depth": 3}
    return {
        "tabular": SupervisedView(["amt", "V1"], params=params, seed=seed),
        "behavioral": SupervisedView(["amt_ratio", "hist_n_past"], params=params,
                                     available=has_history, seed=seed),  # fmt: skip
        "graph": SupervisedView(["g_shared_nonhub", "g_fraud_rate_card"], params=params,
                                available=has_graph_evidence, seed=seed),  # fmt: skip
    }


def _encoder(df):
    return CategoryEncoder().fit(df[SPLITS.train.contains(df["day"].to_numpy())],
                                 columns=("ProductCD",))  # fmt: skip


# ---- plans -----------------------------------------------------------------------------------


def test_plans_from_splits():
    dev = fixed_plan(SPLITS)
    assert [(f.name, f.train, f.predict, f.role) for f in dev] == [
        ("fixed_validation", (1, 120), (128, 150), "gate_train")
    ]
    assert [f.role for f in fixed_plan(SPLITS, final=True)] == ["gate_train", "test"]
    roll = rolling_plan(SPLITS)
    assert [(f.train, f.predict, f.role) for f in roll] == [
        ((1, 90), (91, 120), "gate_train"),
        ((1, 120), (121, 150), "gate_train"),
    ]
    assert rolling_plan(SPLITS, final=True)[-1].predict == (151, 182)
    assert plan_from_json(plan_to_json(roll)) == roll


def test_plans_refuse_test_days_outside_final_runs():
    from vaultic.views.plan import check_plan

    with pytest.raises(ValueError, match="only final runs"):
        check_plan([Fold("x", (1, 120), (151, 182), "test")], SPLITS, final=False)
    with pytest.raises(ValueError, match="gate-training"):
        check_plan([Fold("x", (1, 120), (151, 182), "gate_train")], SPLITS, final=True)
    with pytest.raises(ValueError, match="after training"):
        Fold("x", (1, 120), (100, 130), "gate_train")


def test_label_maturity_in_folds():
    f = Fold("x", (1, 90), (91, 120), "gate_train", label_maturity_days=30)
    day = np.array([1, 60, 61, 61, 90])
    time = np.array([1, 60, 61, 61, 90]) * SECONDS_PER_DAY + np.array([0, 0, 0, 1, 0])
    # known at the start of day 91 only if time + 30 days <= day 91 00:00, i.e. time <= day 61
    assert f.train_rows(day, time).tolist() == [True, True, True, False, False]


# ---- the table -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dev_table():
    df, features, uid = _synthetic()
    table = view_table(df, features, _views(), fixed_plan(SPLITS), _encoder(df), SPLITS)
    return df, features, table


def test_table_layout_and_missing_views(dev_table):
    df, features, table = dev_table
    expected = ["TransactionID", "TransactionDT", "day", "fold", "role", "split", "label",
                *[f"raw_{v}" for v in VIEWS], *CONTEXT, *[f"m_{v}" for v in VIEWS]]  # fmt: skip
    assert sorted(table.columns) == sorted(expected)
    assert set(table["split"]) == {"validation"} and set(table["role"]) == {
        "gate_train",
        "calibrate",
    }
    assert set(table.loc[table["role"] == "calibrate", "day"]) <= set(range(144, 151))  # D53
    assert table["day"].between(128, 150).all()  # development: no test-period rows at all
    rows = features.loc[SPLITS.validation.contains(df["day"].to_numpy())].reset_index(drop=True)
    assert np.array_equal(np.isnan(table["raw_behavioral"]), ~has_history(rows))
    assert np.array_equal(np.isnan(table["raw_graph"]), ~has_graph_evidence(rows))
    assert table["raw_tabular"].notna().all()
    assert table["raw_temporal"].isna().all() and (table["m_temporal"] == 0).all()  # not given
    assert table["raw_anomaly"].isna().all()
    assert not table.attrs.get("calibrated")  # c_* and disagreement only after calibration (D56)


def test_calibration_uses_only_the_calibrate_tail_and_feeds_confidence(dev_table):
    """D56: p_* are calibrated on calibrate rows only; c_* and disagreement come from them."""
    df, features, table = dev_table
    cal, info = calibrate_views(table)
    assert set(info) == {"tabular", "behavioral", "graph"}  # temporal / anomaly not given
    assert cal["p_temporal"].isna().all() and (cal["c_temporal"] == 0).all()
    mask, _, conf, dis = view_inputs(cal[[f"p_{v}" for v in VIEWS]].to_numpy())
    assert np.array_equal(table[[f"m_{v}" for v in VIEWS]].to_numpy() == 1, mask)
    assert np.array_equal(mask, cal[[f"m_{v}" for v in VIEWS]].to_numpy() == 1)
    assert np.allclose(cal[[f"c_{v}" for v in VIEWS]].to_numpy(), conf)
    assert np.allclose(cal["disagreement"], dis)
    _, _, raw_conf, raw_dis = view_inputs(cal[[f"raw_{v}" for v in VIEWS]].to_numpy())
    assert not np.allclose(raw_dis, dis)  # raw-score disagreement would differ
    # labels outside the calibrate tail never reach the calibrators
    flipped = table.copy()
    outside = (flipped["role"] != "calibrate").to_numpy()
    flipped.loc[outside, "label"] = 1 - flipped.loc[outside, "label"]
    again, _ = calibrate_views(flipped)
    assert np.allclose(again[[f"p_{v}" for v in VIEWS]].to_numpy(), cal[[f"p_{v}" for v in VIEWS]].to_numpy(),
                       equal_nan=True)  # fmt: skip
    with pytest.raises(ValueError, match="calibrate rows"):
        calibrate_views(table[table["role"] != "calibrate"])


def test_table_feeds_mvaf(dev_table):
    _, _, table = dev_table
    with pytest.raises(ValueError, match="not calibrated"):
        mvaf_inputs(table, role="gate_train")
    table, _ = calibrate_views(table)
    views, ctx, y = mvaf_inputs(table, role="gate_train")
    n_gate = int((table["role"] == "gate_train").sum())
    assert views.shape == (n_gate, 5) and ctx.shape == (n_gate, len(CONTEXT))
    p = MVAF(epochs=3).fit(views, y, ctx).predict_proba(views, ctx)
    assert np.isfinite(p).all()


class _Spy:
    """Records what a view was trained on and predicts the fold's mean training label."""

    def __init__(self):
        self.calls = []

    def fit(self, frame, y):
        self.calls.append({"n_train": len(frame), "frame": frame})
        self.rate = y.mean()
        return self

    def predict(self, frame):
        return np.full(len(frame), self.rate)


def test_views_never_see_the_rows_they_predict():
    df, features, _ = _synthetic()
    features = features.assign(
        day=df["day"].to_numpy(), TransactionDT=df["TransactionDT"].to_numpy()
    )
    spy = _Spy()
    plan = rolling_plan(SPLITS, final=True, label_maturity_days=30)
    table = view_table(df, features, {"tabular": spy}, plan, _encoder(df), SPLITS)
    for fold, call in zip(plan, spy.calls, strict=True):
        assert call["frame"]["day"].max() <= fold.train[1] < fold.predict[0]
        assert (call["frame"]["TransactionDT"] + 30 * SECONDS_PER_DAY
                <= fold.predict[0] * SECONDS_PER_DAY).all()  # fmt: skip
        assert set(table.loc[table["fold"] == fold.name, "day"]) <= set(
            range(fold.predict[0], fold.predict[1] + 1)
        )
    assert table.groupby("fold")["role"].first().to_dict() == {
        "rolling_0": "gate_train", "rolling_1": "gate_train", "rolling_2": "test"}  # fmt: skip


def test_development_table_ignores_test_period_labels(dev_table):
    df, features, table = dev_table
    flipped = df.copy()
    test = SPLITS.test.contains(df["day"].to_numpy())
    flipped.loc[test, "isFraud"] = 1 - flipped.loc[test, "isFraud"]
    again = view_table(flipped, features, _views(), fixed_plan(SPLITS), _encoder(df), SPLITS)
    pd.testing.assert_frame_equal(table, again)


def test_external_view_is_joined_and_checked(dev_table):
    df, features, table = dev_table
    plan = fixed_plan(SPLITS)
    ext = table[["TransactionID", "fold"]].assign(p_temporal=0.3)
    ext.loc[ext.index[:10], "p_temporal"] = np.nan  # masked rows stay masked
    out = view_table(df, features, _views(), plan, _encoder(df), SPLITS, {"temporal": ext})
    assert out["raw_temporal"].isna().sum() == 10 and (out["m_temporal"] == 0).sum() == 10
    with pytest.raises(ValueError, match="missing predictions"):
        view_table(df, features, _views(), plan, _encoder(df), SPLITS, {"temporal": ext.iloc[5:]})
    with pytest.raises(ValueError, match="not in the plan"):
        view_table(df, features, _views(), plan, _encoder(df), SPLITS,
                   {"temporal": ext.assign(fold="other")})  # fmt: skip
    with pytest.raises(ValueError, match="distinct"):
        view_table(df, features, {"tabular": _views()["tabular"]}, plan, _encoder(df), SPLITS,
                   {"tabular": ext.rename(columns={"p_temporal": "p_tabular"})})  # fmt: skip


def test_temporal_inputs_never_contain_test_rows(tmp_path):
    df, features, uid = _synthetic()
    out = temporal_step.write_inputs(tmp_path, df, uid, fixed_plan(SPLITS), _encoder(df))
    written = pd.read_parquet(out / "inputs.parquet")
    assert written["day"].max() <= SPLITS.validation.last
    assert set(written.columns) >= {"uid", "isFraud", "ProductCD", "TransactionDT"}
    assert plan_from_json((out / "plan.json").read_text()) == fixed_plan(SPLITS)


@pytest.mark.skipif(not TORCH_PY.exists(), reason=".venv-torch not set up")
def test_end_to_end_across_venvs(tmp_path, dev_table):
    """Main venv writes inputs; .venv-torch runs the GRU step; main venv joins its parquet."""
    df, features, table = dev_table
    plan = fixed_plan(SPLITS)
    temporal_step.write_inputs(tmp_path / "in", df, features["uid"], plan, _encoder(df),
                               n_steps=5, gru={"max_epochs": 3, "seed": 0})  # fmt: skip
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    done = subprocess.run(
        [str(TORCH_PY), "-m", "vaultic.views.temporal_step", str(tmp_path / "in"),
         str(tmp_path / "p_temporal.parquet")],  # fmt: skip
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    ext = pd.read_parquet(tmp_path / "p_temporal.parquet")
    out = view_table(df, features, _views(), plan, _encoder(df), SPLITS, {"temporal": ext})
    no_history = out["ctx_hist_n_past"].to_numpy() == 0
    assert np.array_equal(out["raw_temporal"].isna().to_numpy(), no_history)
    assert out.loc[~no_history, "raw_temporal"].between(0, 1).all()
    assert sys.executable != str(TORCH_PY)  # the assembling side really is another venv


def test_anomaly_view_scores_every_row_and_stays_label_free_inside():
    from vaultic.views.orchestrate import AnomalyScoreView

    df, features, _ = _synthetic(n=3000, n_uids=60)
    view = AnomalyScoreView(["amt", "V1"], max_iter=30)
    table = view_table(df, features, {"anomaly": view}, fixed_plan(SPLITS), _encoder(df), SPLITS)
    assert table["raw_anomaly"].between(0, 1).all()  # global scores exist for every row
    train = SPLITS.train.contains(df["day"].to_numpy())
    y = df["isFraud"].to_numpy()[train]
    assert view.view_.global_if.n_fit_ == int((y == 0).sum())  # anomaly models: legit rows only
