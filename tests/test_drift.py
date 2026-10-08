"""Phase 10 drift module on synthetic data with a known drift start."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.drift import inject
from vaultic.drift.detectors import ADWIN, PageHinkley, alarms
from vaultic.drift.label_delay import (
    detection_report,
    label_based_alarm_days,
    label_free_alarm_days,
    matured_error_stream,
)
from vaultic.drift.policies import simulate
from vaultic.drift.signals import DisagreementMonitor, feature_drift, psi, score_drift

# ---- signals ---------------------------------------------------------------------------------


def test_psi_by_hand_and_basic_properties():
    ref = ["a"] * 50 + ["b"] * 50
    cur = ["a"] * 75 + ["b"] * 25
    expected = 0.25 * math.log(1.5) + (-0.25) * math.log(0.5)
    assert psi(ref, cur) == pytest.approx(expected, abs=1e-6)
    rng = np.random.default_rng(0)
    base = rng.normal(size=20_000)
    assert psi(base, rng.normal(size=20_000)) < 0.01
    assert psi(base, rng.normal(1.0, 1, 20_000)) > 0.2
    with_nan = np.where(rng.random(20_000) < 0.3, np.nan, rng.normal(size=20_000))
    assert psi(base, with_nan) > 0.2  # 30% newly missing is drift, not ignored
    assert psi(ref, ["a"] * 50 + ["new"] * 50) > 1  # unseen categories land in "other"


def test_feature_drift_flags_only_the_shifted_feature():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame(rng.normal(size=(5000, 5)), columns=list("abcde"))
    cur = pd.DataFrame(rng.normal(size=(5000, 5)), columns=list("abcde"))
    cur["c"] += 0.8
    table = feature_drift(ref, cur).set_index("feature")
    assert table["psi_alert"].tolist() == [False, False, True, False, False]
    assert table["ks_alert"].tolist() == [False, False, True, False, False]
    assert (table["p_holm"] >= table["p_value"]).all()


def test_score_drift_and_disagreement_monitor():
    rng = np.random.default_rng(2)
    assert score_drift(rng.beta(1, 10, 5000), rng.beta(1, 10, 5000)) < 0.05
    assert score_drift(rng.beta(1, 10, 5000), rng.beta(2, 5, 5000)) > 0.2
    mon = DisagreementMonitor(k=3).fit([0.10, 0.12, 0.11, 0.09, 0.08])
    assert mon.mean_ == pytest.approx(0.10) and mon.std_ == pytest.approx(0.0158114, rel=1e-5)
    assert not mon.alarm(0.13) and mon.alarm(0.16)


# ---- detectors -------------------------------------------------------------------------------


def _bernoulli_change(p0=0.05, p1=0.25, n0=3000, n1=1000, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.random(n0) < p0, rng.random(n1) < p1]).astype(float)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adwin_detects_an_error_rate_change(seed):
    found = alarms(ADWIN(), _bernoulli_change(seed=seed))
    assert found and all(i >= 3000 for i in found)  # nothing before the change
    assert found[0] < 3400


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_page_hinkley_detects_an_increase(seed):
    found = alarms(PageHinkley(threshold=25), _bernoulli_change(seed=seed))
    assert found and found[0] >= 3000 and found[0] < 3600


def test_detectors_stay_quiet_without_change():
    stream = _bernoulli_change(p1=0.05, seed=3)
    assert alarms(ADWIN(), stream) == []
    assert alarms(PageHinkley(threshold=25), stream) == []


# ---- label delay -----------------------------------------------------------------------------


def test_matured_error_stream_by_hand():
    day = SECONDS_PER_DAY
    time = np.array([0, 10, 2 * day, 2 * day + 5])
    y = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.1, 0.2])  # threshold 0.5: right, wrong, right, wrong
    s = matured_error_stream(time, y, p, 0.5, delay_days=7)
    assert s["arrival_day"].tolist() == [7, 7, 9, 9]
    assert s["error"].tolist() == [0, 1, 0, 1] and s["row"].tolist() == [0, 1, 2, 3]
    assert (s["arrival"] - time[s["row"]] == 7 * day).all()


def test_detection_report_by_hand():
    r = detection_report([20, 95, 103, 110], start_day=100, first_day=10, last_day=200)
    assert r == {"delay_days": 3.0, "false_alarms": 2,
                 "false_alarms_per_6_months": pytest.approx(2 * 182 / 90), "missed": False}  # fmt: skip
    assert detection_report([20], 100, 10, 200)["missed"] is True


# ---- injected drifts -------------------------------------------------------------------------


def _transactions(days=240, per_day=150, seed=0, drift_day=None):
    """x1 drives fraud before drift_day; afterwards fraud hides in x1 and shows in x2."""
    rng = np.random.default_rng(seed)
    n = days * per_day
    day = np.repeat(np.arange(1, days + 1), per_day)
    time = day * SECONDS_PER_DAY + rng.integers(0, SECONDS_PER_DAY, n)
    order = np.argsort(time, kind="stable")
    day, time = day[order], time[order]
    y = (rng.random(n) < 0.05).astype(int)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    after = np.zeros(n, bool) if drift_day is None else day >= drift_day
    x1 = np.where((y == 1) & ~after, x1 + 2.5, x1)
    x2 = np.where((y == 1) & after, x2 + 2.5, x2)
    return pd.DataFrame(
        {
            "TransactionDT": time,
            "day": day,
            "isFraud": y,
            "TransactionAmt": rng.lognormal(4, 0.8, n),
            "ProductCD": rng.choice(["W", "C"], n),
            "P_emaildomain": rng.choice(["a.com", "b.com"], n),
            "x1": x1,
            "x2": x2,
            "vel": np.where(y == 1, rng.poisson(6, n), rng.poisson(1, n)).astype(float),
        }
    )


def test_generators_change_only_fraud_from_the_start_day():
    df = _transactions(days=60)
    before = (df["day"] < 30).to_numpy()
    for out, drift in [
        inject.amount_shift(df, "W", 3.0, 30),
        inject.new_email_domain(df, 30),
        inject.velocity_mimicry(df, 30, ["vel"]),
    ]:
        changed = np.zeros(len(df), bool)
        changed[drift.rows] = True
        assert drift.start_day == 30
        assert (df["isFraud"].to_numpy()[changed] == 1).all() and not changed[before].any()
        pd.testing.assert_frame_equal(out[~changed].astype(object), df[~changed].astype(object))
    out, drift = inject.amount_shift(df, "W", 3.0, 30)
    assert (df["ProductCD"].to_numpy()[drift.rows] == "W").all()
    assert np.allclose(out["TransactionAmt"].to_numpy()[drift.rows],
                       3 * df["TransactionAmt"].to_numpy()[drift.rows])  # fmt: skip
    out, drift = inject.new_email_domain(df, 30)
    assert (out["P_emaildomain"].to_numpy()[drift.rows] == inject.NEW_DOMAIN).all()
    out, drift = inject.velocity_mimicry(df, 30, ["vel"])
    legit_by_day = df[df["isFraud"] == 0].groupby("day")["vel"].apply(set)
    for r in drift.rows[:50]:  # each new value comes from a legit row of the same day
        assert out["vel"].iat[r] in legit_by_day[df["day"].iat[r]]
    assert out["vel"].to_numpy()[drift.rows].mean() < df["vel"].to_numpy()[drift.rows].mean()


def test_label_free_detects_before_labels_can_arrive():
    """Amount shift from day 150 (all rows of one product, so the score moves): the score PSI
    sees it within days; a label-based detector cannot fire before day 150 + L."""
    df = _transactions(days=240)
    df["x1"] = np.where(df["day"] >= 150, df["x1"] + 1.5, df["x1"])  # population shift
    train = (df["day"] < 120).to_numpy()
    model = LogisticRegression().fit(df.loc[train, ["x1", "x2"]], df.loc[train, "isFraud"])
    p = model.predict_proba(df[["x1", "x2"]])[:, 1]
    ref = (df["day"] >= 90) & (df["day"] < 120)
    free = label_free_alarm_days(df["day"], ref.to_numpy(), list(range(120, 241)),
                                 lambda r, c: score_drift(p[r], p[c]), threshold=0.2)  # fmt: skip
    free_report = detection_report(free.loc[free["alarm"], "day"], 150, 120, 240)
    assert free_report["delay_days"] <= 7 and free_report["false_alarms"] == 0
    for delay in (7, 30):
        stream = matured_error_stream(df["TransactionDT"], df["isFraud"], p, 0.5, delay)
        stream = stream[df["day"].to_numpy()[stream["row"]] >= 120]
        based = label_based_alarm_days(stream, ADWIN(), last_day=240)
        report = detection_report(based, 150, 120, 240)
        assert report["missed"] or report["delay_days"] >= delay


# ---- retraining policies ---------------------------------------------------------------------


def _fit(X, y):
    return LogisticRegression().fit(X, y)


@pytest.fixture(scope="module")
def drifted():
    return _transactions(days=240, drift_day=150)


def test_p0_never_and_p1_monthly(drifted):
    p0, _ = simulate(drifted, ["x1", "x2"], "P0", 120, 239, _fit)
    assert p0.n_attempts == 0 and set(p0.daily["model"]) == {0}
    p1, _ = simulate(drifted, ["x1", "x2"], "P1", 120, 239, _fit)
    assert [e["day"] for e in p1.events] == [150, 180, 210, 240]


def test_training_only_uses_matured_labels(drifted):
    seen = []

    def spy(X, y):
        seen.append(drifted.loc[X.index, "TransactionDT"].max())
        return _fit(X, y)

    result, _ = simulate(drifted, ["x1", "x2"], "P1", 120, 239, spy, delay_days=30)
    as_of = [120] + [e["day"] for e in result.events]
    for latest, day in zip(seen, as_of, strict=True):
        assert latest + 30 * SECONDS_PER_DAY <= day * SECONDS_PER_DAY


def test_gate_rejects_a_worse_challenger(drifted):
    class Constant:
        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

    calls = []

    def first_good_then_useless(X, y):
        calls.append(1)
        return _fit(X, y) if len(calls) == 1 else Constant()

    result, _ = simulate(drifted, ["x1", "x2"], "P1", 120, 239, first_good_then_useless)
    assert result.n_attempts == 4 and result.n_retrains == 0
    assert set(result.daily["model"]) == {0}
    assert all(e["pr_auc_challenger"] < e["pr_auc_champion"] for e in result.events)


def test_label_based_retraining_recovers_after_drift(drifted):
    # a 45-day training window: with all history the post-drift pattern stays diluted
    kw = {"delay_days": 30, "train_window_days": 45}
    p0, s0 = simulate(drifted, ["x1", "x2"], "P0", 120, 239, _fit, **kw)
    p3, s3 = simulate(drifted, ["x1", "x2"], "P3", 120, 239, _fit, **kw)
    assert p3.n_retrains >= 1
    assert p3.events[0]["day"] >= 150 + 30  # no label from the drift period before day 180
    late = lambda s: s[s["day"] >= 215]  # noqa: E731
    from vaultic.eval.metrics import pr_auc

    assert pr_auc(late(s3)["y"], late(s3)["p"]) > pr_auc(late(s0)["y"], late(s0)["p"]) + 0.1
    assert p3.total_cost < p0.total_cost
    blocks = p3.pr_auc_by_block(s3)
    assert list(blocks.columns) == ["block", "first_day", "pr_auc"] and len(blocks) == 4


def test_training_window_limits_history(drifted):
    seen = []

    def spy(X, y):
        t = drifted.loc[X.index, "TransactionDT"]
        seen.append((t.max() - t.min()) / SECONDS_PER_DAY)
        return _fit(X, y)

    simulate(drifted, ["x1", "x2"], "P1", 120, 239, spy, train_window_days=45, holdout_days=14)
    assert all(span <= 45 - 14 for span in seen)


def test_gate_needs_a_significant_gain_not_just_a_higher_number(drifted):
    """A challenger equal to the champion (same data, same model) never replaces it; P1 then
    keeps the first model even though challengers tie or win by noise."""
    calls = []

    def same_model_every_time(X, y):
        calls.append(1)
        if len(calls) == 1:
            same_model_every_time.first = _fit(X, y)
        return same_model_every_time.first

    result, _ = simulate(drifted, ["x1", "x2"], "P1", 120, 239, same_model_every_time,
                         gate_n_boot=200)  # fmt: skip
    assert result.n_attempts == 4 and result.n_retrains == 0
    for e in result.events:
        assert e["gain_ci_low"] <= 0 <= e["gain_ci_high"]


def test_accepted_challengers_have_a_significant_gain(drifted):
    result, _ = simulate(drifted, ["x1", "x2"], "P3", 120, 239, _fit, train_window_days=45)
    accepted = [e for e in result.events if e["accepted"]]
    assert accepted and all(e["gain_ci_low"] > 0 for e in accepted)
    assert all(e["cost_challenger"] <= e["cost_champion"] for e in accepted)
