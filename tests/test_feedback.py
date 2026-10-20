"""EXP-310 feedback simulation (D210, D211): the six pre-registered leakage tests (section 6) and
the engine's rules, on toy data with a fast logistic-regression fit."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from vaultic.learning.feedback import (
    DAY,
    MIN_FRAUD,
    Data,
    gate_rows,
    make_model,
    matured_rows,
    simulate,
    slots,
    split_matured,
)
from vaultic.learning.feedback_inputs import label_derived_at, load, require_last_day

POLICIES = ("P0", "P1", "P2", "P3", "P1_ungated", "P1_monthly", "P1_4weekly")


def toy(n_days=70, per_day=120, seed=0, delay=7, fraud_rate=0.12, drift_day=40):
    """Rows sorted by time; x1 separates fraud, and its relation flips sign after drift_day so
    retraining has something to learn."""
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(1, n_days + 1), per_day)
    time = day * DAY + np.tile(np.sort(rng.integers(0, DAY - 1, per_day)), n_days)
    order = np.argsort(time, kind="stable")
    day, time = day[order], time[order]
    y = (rng.random(len(day)) < fraud_rate).astype(int)
    sign = np.where(day < drift_day, 1.0, -1.0)
    x1 = rng.normal(size=len(day)) + 1.5 * y * sign
    x2 = rng.normal(size=len(day)) + 0.5 * y
    amount = rng.exponential(80, len(day)) + 5
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return Data(time, y, amount, X, delay)


def fit_lr(X, y):
    return LogisticRegression(max_iter=200).fit(X, y)


def champion_for(data, fit_days=(1, 26), cal_days=(27, 30)):
    rows = np.flatnonzero((data.day >= fit_days[0]) & (data.day <= fit_days[1]))
    cal = np.flatnonzero((data.day >= cal_days[0]) & (data.day <= cal_days[1]))
    return make_model(
        "champion", fit_lr(data.X.iloc[rows], data.y[rows]), data, rows, cal, fit_days[1]
    )


def run(data, end=70, fit=fit_lr, policies=POLICIES):
    return simulate(data, champion_for(data), 31, end, fit, policies)


# ---- (i) the label curtain ----------------------------------------------------------------
@pytest.mark.parametrize("delay", [0, 7, 30])
def test_i_no_training_threshold_or_gate_row_violates_the_curtain(delay):
    data = toy(delay=delay)
    for t in slots(31, 70):
        fit, thr, gate = split_matured(data, t)
        for rows in (fit, thr, gate):
            assert np.all(data.time[rows] + delay * DAY < t * DAY)  # strict (D214)
            assert np.all(data.time[rows] < t * DAY)
        assert set(fit).isdisjoint(thr) and set(thr).isdisjoint(gate)
    calls = []

    def recording_fit(X, y):
        calls.append(X.index.to_numpy())
        return fit_lr(X, y)

    _, events = run(data, fit=recording_fit)
    fitted_days = sorted({e["day"] for e in events if "challenger" in e})
    assert len(calls) == len(fitted_days)  # one fit per attempt day, shared by policies
    for rows, t in zip(calls, fitted_days, strict=True):
        assert np.all(data.time[rows] + delay * DAY < t * DAY)


# ---- (ii) flipping labels not yet known at t changes nothing up to t -------------------------
@pytest.mark.parametrize("delay", [0, 7])
def test_ii_future_label_flip_changes_no_model_or_score_up_to_t(delay):
    t = 52
    a = toy(delay=delay)
    b = toy(delay=delay)
    unknown = b.time + delay * DAY > t * DAY
    b.y = np.where(unknown, 1 - b.y, b.y)
    sa, ea = run(a)
    sb, eb = run(b)
    up = lambda s: s[s["day"] <= t].reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(up(sa), up(sb))
    att = lambda e: [x for x in e if x["kind"] == "attempt" and x["day"] <= t]  # noqa: E731
    alarms = lambda e: [x for x in e if x["kind"] == "alarm" and x["day"] < t]  # noqa: E731
    assert att(ea) == att(eb) and alarms(ea) == alarms(eb)


# ---- (iii) truncating the data after day t changes nothing up to t ---------------------------
def test_iii_truncation_after_t_changes_nothing_up_to_t():
    t = 55
    full = toy(delay=7)
    keep = full.day <= t
    cut = Data(
        full.time[keep], full.y[keep], full.amount[keep], full.X[keep].reset_index(drop=True), 7
    )
    sf, ef = run(full)
    sc, ec = simulate(cut, champion_for(cut), 31, t, fit_lr, POLICIES)
    up = lambda s: s[s["day"] <= t].reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(up(sf), sc.reset_index(drop=True))
    assert [x for x in ef if x["day"] <= t] == ec


# ---- (iv) recomputed label-derived inputs use no label not yet known -------------------------
def _input_frame():
    rows = []
    t0 = 10 * DAY
    for i, (uid_card, gap_days, fraud) in enumerate([(1, 0, 1), (1, 2, 0), (2, 3, 1), (1, 5, 1),
                                                     (2, 9, 0), (1, 12, 0), (1, 20, 0), (2, 25, 1),
                                                     (1, 33, 0), (2, 40, 0)]):  # fmt: skip
        t = t0 + gap_days * DAY + i
        rows.append({"TransactionID": i, "TransactionDT": t, "day": t // DAY, "TransactionAmt": 50.0,
                     "isFraud": fraud, "card1": float(uid_card), "card2": 1.0, "card3": 150.0,
                     "card5": 226.0, "addr1": 300.0, "D1": 0.0, "P_emaildomain": "x.com"})  # fmt: skip
    df = pd.DataFrame(rows)
    uid = pd.Series(df["card1"].astype(int).to_numpy())
    return df, uid


@pytest.mark.parametrize("delay", [0, 7, 30, 60])
def test_iv_recomputed_label_derived_inputs_use_no_unknown_label(delay):
    df, uid = _input_frame()
    train = np.ones(len(df), bool)
    base = label_derived_at(df, uid, delay, train)
    for r in range(len(df)):
        flipped = df.copy()
        unknown = flipped["TransactionDT"] + delay * DAY > flipped["TransactionDT"].iat[r]
        flipped.loc[unknown, "isFraud"] = 1 - flipped.loc[unknown, "isFraud"]
        again = label_derived_at(flipped, uid, delay, train)
        pd.testing.assert_series_equal(base.iloc[r], again.iloc[r], check_names=False)
    # hand-checked value: customer 1's 4th row (day 15) at delay 0 knows rows at days 10 and 12
    if delay == 0:
        assert base["uid_n_labels_known"].iat[3] == 2 and base["uid_fraud_known"].iat[3] == 1
    if delay == 60:
        assert (base["uid_n_labels_known"].fillna(0) == 0).all()  # nothing matures within 40 days


# ---- (v) the design run refuses any row after day 120 ----------------------------------------
def test_v_design_run_refuses_rows_after_day_120(tmp_path):
    require_last_day(np.array([1, 60, 120]), 120)
    with pytest.raises(RuntimeError, match="after day 120"):
        require_last_day(np.array([1, 121]), 120)
    with pytest.raises(RuntimeError, match="go-ahead"):
        load("final")
    from vaultic.learning.feedback_run import main

    cfg = tmp_path / "EXP-310-final.yaml"
    cfg.write_text("id: EXP-310-final\nstatus: frozen-final\n", "utf-8")
    with pytest.raises(SystemExit, match="design run"):
        main([str(cfg)])


# ---- (vi) the gate's row removal and evidence floor ------------------------------------------
def test_vi_gate_removes_both_models_training_rows_and_applies_the_evidence_floor():
    data = toy(delay=0)
    champ = champion_for(data)
    t = 45
    _, _, window = split_matured(data, t)
    other = make_model("other", champ.estimator, data, window[:100], window[100:110], t)
    rows = gate_rows(data, t, champ, other)
    assert not champ.train_mask[rows].any() and not other.train_mask[rows].any()
    assert set(rows) == set(window[100:]) - set(np.flatnonzero(champ.train_mask))
    # too few frauds in the window: rejected as insufficient evidence, no challenger fitted
    poor = toy(delay=0, fraud_rate=0.02)
    calls = []
    _, events = simulate(poor, champion_for(poor), 31, 70,
                         lambda X, y: calls.append(1) or fit_lr(X, y), ("P1",))  # fmt: skip
    att = [e for e in events if e["kind"] == "attempt"]
    assert att and all(
        e["reason"] == "insufficient evidence" and e["gate_frauds"] < MIN_FRAUD for e in att
    )
    assert calls == []


# ---- engine rules --------------------------------------------------------------------------
def test_policies_attempt_on_their_slots_and_p0_never():
    data = toy(delay=0)
    served, events = run(data)
    att = [e for e in events if e["kind"] == "attempt"]
    assert not [e for e in att if e["policy"] == "P0"]
    assert sorted({e["day"] for e in att if e["policy"] == "P1"}) == slots(31, 70)
    assert sorted({e["day"] for e in att if e["policy"] == "P1_monthly"}) == [61]  # 30 days (D214)
    assert sorted({e["day"] for e in att if e["policy"] == "P1_4weekly"}) == [59]  # design run only
    assert all(e["day"] in slots(31, 70) for e in att if e["policy"] in ("P2", "P3"))
    # every policy serves every row of the span exactly once
    for pol in POLICIES:
        s = served[served["policy"] == pol]
        assert (
            len(s) == len(np.flatnonzero((data.day >= 31) & (data.day <= 70))) == s["row"].nunique()
        )


def test_drift_makes_retraining_pay_and_the_gate_accept():
    data = toy(delay=0, drift_day=40)
    served, events = run(data)
    p1 = [e for e in events if e["kind"] == "attempt" and e["policy"] == "P1"]
    assert any(e["accepted"] for e in p1)  # the sign flip is learnable from matured labels
    ungated = [e for e in events if e["kind"] == "attempt" and e["policy"] == "P1_ungated"]
    assert all(e["accepted"] for e in ungated if e["reason"] != "insufficient evidence")
    from vaultic.eval.metrics import pr_auc

    late = served[served["day"] >= 60]
    auc = {p: pr_auc(data.y[g["row"]], g["p"]) for p, g in late.groupby("policy")}
    assert auc["P1"] > auc["P0"]


def test_no_drift_gate_rejects_most_challengers():
    data = toy(delay=0, drift_day=10_000)
    _, events = run(data, policies=("P1",))
    att = [e for e in events if e["kind"] == "attempt"]
    assert sum(e["accepted"] for e in att) <= len(att) // 2


def test_psi_alarm_needs_seven_days_of_the_new_champion():
    data = toy(delay=0, drift_day=40)
    _, events = run(data, policies=("P2",))
    promos = [e["day"] for e in events if e["kind"] == "attempt" and e["accepted"]]
    alarms = [e["day"] for e in events if e["kind"] == "alarm"]
    for p in promos:
        assert not [a for a in alarms if p <= a < p + 6]  # reference and window restarted (D211 b)


def test_matured_rows_follow_the_delay():
    data = toy(delay=30)
    rows = matured_rows(data, 50)
    assert rows.max() < np.flatnonzero(data.day == 21)[0]  # only days <= 19 / 20 are known


def test_strict_curtain_a_row_exactly_on_the_day_boundary_is_not_yet_known():
    """D214: a label is usable only if its_time + L < t; a row at exactly t * 86400 (L = 0) is
    known from day t + 1, not on day t."""
    time = np.array([10 * DAY - 1, 10 * DAY, 10 * DAY + 1, 11 * DAY])
    data = Data(time, [0, 1, 0, 1], [10.0] * 4, pd.DataFrame({"x": [0.0] * 4}), 0)
    assert matured_rows(data, 10).tolist() == [0]  # the boundary row (index 1) is not known
    assert matured_rows(data, 11).tolist() == [0, 1, 2]
    lagged = Data(time, [0, 1, 0, 1], [10.0] * 4, pd.DataFrame({"x": [0.0] * 4}), 7)
    assert matured_rows(lagged, 17).tolist() == [0]  # 10 days + 7 = day 17 exactly: not yet
