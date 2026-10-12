"""Phase 8 trust layer on synthetic data and hand-checked examples."""

import numpy as np
import pytest

from vaultic.eval.metrics import ece
from vaultic.trust.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    calibration_report,
    choose_calibrator,
    reliability_table,
)
from vaultic.trust.conformal import (
    MondrianConformal,
    SplitConformal,
    conformal_quantile,
    coverage,
    set_labels,
)
from vaultic.trust.routing import compare_policies, evaluate_policy, priority, select_daily


def _scores(n, seed, fraud_rate=0.03):
    """True probabilities q (rare positives) and labels drawn from them."""
    rng = np.random.default_rng(seed)
    q = 1 / (1 + np.exp(-(rng.normal(-4.5, 1.6, n))))
    y = (rng.random(n) < q).astype(int)
    return q, y


# ---- calibration ----------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [PlattCalibrator, IsotonicCalibrator])
def test_calibrators_fix_a_distorted_score(cls):
    q, y = _scores(40_000, 0)
    p = q**0.4  # overconfident: pushed towards 1
    half = len(q) // 2
    cal = cls().fit(p[:half], y[:half])
    after = cal.predict(p[half:])
    report = calibration_report(y[half:], p[half:], after)
    assert report["ece_after"] < report["ece_before"] / 3
    assert report["brier_after"] < report["brier_before"]


def test_choose_calibrator_uses_out_of_fold_ece():
    q, y = _scores(20_000, 1)
    model, scores = choose_calibrator(y, q**0.4, folds=5)
    assert set(scores) == {"platt", "isotonic"}
    assert model.name == min(scores, key=scores.get)
    assert ece(y, model.predict(q**0.4)) < ece(y, q**0.4)


def test_reliability_table():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.05, 0.15, 0.95, 1.0])
    t = reliability_table(y, p, bins=10)
    assert t["count"].sum() == 4 and len(t) == 10
    assert t.loc[9, "count"] == 2  # 0.95 and 1.0 share the last bin
    assert t.loc[9, "observed_rate"] == 0.5 and t.loc[1, "observed_rate"] == 1.0


# ---- conformal ------------------------------------------------------------------------------


def test_conformal_quantile_by_hand():
    scores = np.arange(1, 10) / 10  # 0.1 .. 0.9, n = 9
    assert conformal_quantile(scores, 0.2) == pytest.approx(0.8)  # ceil(10 * 0.8) = 8th
    assert conformal_quantile(scores, 0.05) == float("inf")  # ceil(10 * 0.95) = 10 > 9


def test_sets_from_known_thresholds():
    cp = SplitConformal()
    cp.thresholds_ = np.array([0.5, 0.5])
    sets = cp.predict_sets(np.array([0.95, 0.05, 0.5]))
    # p=0.95: legit score 0.95 > 0.5 (out), fraud score 0.05 (in) -> {fraud}
    assert set_labels(sets).tolist() == ["fraud", "legit", "uncertain"]
    cp.thresholds_ = np.array([0.1, 0.1])
    assert set_labels(cp.predict_sets(np.array([0.5]))).tolist() == ["empty"]


def test_split_conformal_meets_target_coverage():
    q, y = _scores(60_000, 2)
    cp = SplitConformal(alpha=0.1).fit(y[:30_000], q[:30_000])
    cov = coverage(y[30_000:], cp.predict_sets(q[30_000:]))
    assert cov["coverage"] >= 0.9 - 0.01


def test_mondrian_protects_the_rare_class():
    q, y = _scores(60_000, 3)
    cal, test = slice(0, 30_000), slice(30_000, None)
    split = coverage(y[test], SplitConformal(0.1).fit(y[cal], q[cal]).predict_sets(q[test]))
    mondrian = coverage(y[test], MondrianConformal(0.1).fit(y[cal], q[cal]).predict_sets(q[test]))
    assert mondrian["coverage_fraud"] >= 0.9 - 0.04
    assert mondrian["coverage_legit"] >= 0.9 - 0.01
    assert split["coverage_fraud"] < mondrian["coverage_fraud"]  # split under-covers fraud


# ---- routing ------------------------------------------------------------------------------


def _routing_example():
    #            day 1 (5 rows)                         day 2 (1 row)
    day = np.array([1, 1, 1, 1, 1, 2])
    p = np.array([0.9, 0.6, 0.3, 0.2, 0.1, 0.4])
    amount = np.array([10.0, 100.0, 1000.0, 50.0, 20.0, 70.0])
    y = np.array([1, 0, 1, 0, 1, 1])
    return day, p, amount, y


def test_priorities_by_hand():
    _, p, amount, _ = _routing_example()
    assert priority("R1", p, amount).tolist() == p.tolist()
    assert priority("R2", p, amount).tolist() == pytest.approx([9, 60, 300, 10, 2, 28])
    # u = 1 - |2p - 1| = 0.2, 0.8, 0.6, 0.4, 0.2, 0.8 ; R3 = R2 + u * amount
    assert priority("R3", p, amount).tolist() == pytest.approx([11, 140, 900, 30, 6, 84])
    d = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    r4 = priority("R4", p, amount, d=d, lambda_d=20.0)
    assert r4[3] == pytest.approx(30 + 20 * 50)  # disagreement bonus lifts row 3
    with pytest.raises(ValueError, match="disagreement"):
        priority("R4", p, amount)


def test_top_k_per_day_and_metrics():
    day, p, amount, y = _routing_example()
    assert select_daily(day, priority("R1", p, amount), 2).tolist() == [
        True, True, False, False, False, True,
    ]  # fmt: skip
    r1 = evaluate_policy(day, y, amount, p, "R1", k=2)
    # reviewed rows 0, 1, 5: frauds 0 (10) and 5 (70)
    assert (r1["reviews"], r1["fraud_caught"], r1["fraud_value_caught"]) == (3, 2, 80.0)
    r2 = evaluate_policy(day, y, amount, p, "R2", k=2)
    # R2 reviews rows 2, 1 on day 1 and row 5: frauds 2 (1000) and 5 (70)
    assert r2["fraud_value_caught"] == 1070.0 and r2["recall_at_k"] == pytest.approx(2 / 4)
    assert r2["value_recall"] == pytest.approx(1070 / 1100)
    assert r2["money_saved_per_review"] == pytest.approx(1070 / 3)


def test_ties_are_broken_by_row_order():
    day = np.zeros(4)
    assert select_daily(day, np.array([1.0, 1.0, 1.0, 1.0]), 2).tolist() == [
        True, True, False, False,
    ]  # fmt: skip


def test_compare_policies_skips_r4_without_disagreement():
    day, p, amount, y = _routing_example()
    table = compare_policies(day, y, amount, p, ks=(1, 2))
    assert sorted(table["policy"].unique()) == ["R1", "R2", "R3"] and len(table) == 6


# ---- adaptive conformal (8.2) -------------------------------------------------------------

from vaultic.trust.conformal import AdaptiveConformal, coverage_by_block, threshold_at  # noqa: E402


def test_threshold_at_edges():
    scores = np.arange(1, 10) / 10
    assert threshold_at(scores, 0.0) == float("inf") and threshold_at(scores, -0.2) == float("inf")
    assert threshold_at(scores, 0.2) == pytest.approx(0.8)
    assert threshold_at(scores, 0.95) == pytest.approx(0.1)  # ceil(10 * 0.05) = 1st
    assert threshold_at(scores, 1.0) == float("-inf")  # asks for no score at all


def _three_batches():
    # calibration: 100 legit rows with p = 0.1 -> every calibration score is 0.1
    # batch 1: a fraud scored p = 0 (fraud score 1.0 > 0.1: miss), batches 2-3: legit p = 0.05
    return np.array([1, 2, 3]), np.array([1, 0, 0]), np.array([0.0, 0.05, 0.05])


def test_aci_update_rule_by_hand():
    batch, y, p = _three_batches()
    aci = AdaptiveConformal(alpha=0.1, gamma=0.1, label_delay_days=0).fit(
        np.zeros(100, int), np.full(100, 0.1)
    )
    _, trace = aci.run(batch, y, p)
    # L = 0: a day's error is used from the next day.
    # 0.1 -> 0.1 + 0.1 * (0.1 - 1) = 0.01 -> 0.01 + 0.1 * (0.1 - 0) = 0.02
    assert [t["alpha_fraud"] for t in trace] == pytest.approx([0.1, 0.01, 0.02])
    assert [t["coverage"] for t in trace] == [0.0, 1.0, 1.0]


def test_aci_label_delay_postpones_the_update():
    """D55: day 1's error is known at day d only if 1 + L < d. L = 1: from day 3."""
    batch, y, p = _three_batches()
    aci = AdaptiveConformal(alpha=0.1, gamma=0.1, label_delay_days=1).fit(
        np.zeros(100, int), np.full(100, 0.1)
    )
    _, trace = aci.run(batch, y, p)
    assert [t["alpha_fraud"] for t in trace] == pytest.approx([0.1, 0.1, 0.01])


def test_aci_label_delay_is_in_days_not_batches():
    """Days 1, 2 and 40 with L = 30: both earlier days are known by day 40 (1 + 30 < 40 and
    2 + 30 < 40), however many batches lie between. Gaps in the stream do not shift arrival."""
    y, p = np.array([1, 1, 0]), np.array([0.0, 0.0, 0.05])
    aci = AdaptiveConformal(alpha=0.1, gamma=0.1, label_delay_days=30).fit(
        np.zeros(100, int), np.full(100, 0.1)
    )
    _, trace = aci.run(np.array([1, 2, 40]), y, p)
    # day 40 uses both misses: 0.1 -> 0.01 -> 0.01 + 0.1 * (0.1 - 1) = -0.08
    assert [t["alpha_fraud"] for t in trace] == pytest.approx([0.1, 0.1, -0.08])
    _, trace = aci.run(np.array([1, 2, 31]), y, p)  # 1 + 30 < 31 is false: nothing known yet
    assert trace[-1]["alpha_fraud"] == pytest.approx(0.1)


def test_aci_requires_the_label_delay():
    with pytest.raises(TypeError):
        AdaptiveConformal(alpha=0.1, gamma=0.1)
    with pytest.raises(ValueError, match="non-negative"):
        AdaptiveConformal(alpha=0.1, label_delay_days=-1)
    aci = AdaptiveConformal(alpha=0.1, label_delay_days=0).fit(np.zeros(10, int), np.full(10, 0.1))
    with pytest.raises(ValueError, match="time order"):
        aci.run(np.array([2, 1]), np.array([0, 0]), np.array([0.1, 0.1]))


def test_mondrian_aci_updates_only_the_class_that_erred():
    batch, y, p = _three_batches()
    cal_y = np.r_[np.zeros(100, int), np.ones(100, int)]
    cal_p = np.r_[np.full(100, 0.1), np.full(100, 0.9)]
    _, trace = (
        AdaptiveConformal(0.1, gamma=0.1, mondrian=True, label_delay_days=0)
        .fit(cal_y, cal_p)
        .run(batch, y, p)
    )
    assert trace[1]["alpha_fraud"] == pytest.approx(0.01)  # batch 1 missed its only fraud
    assert trace[1]["alpha_legit"] == pytest.approx(0.1)  # no legit rows in batch 1


def _shifted_stream(seed, days=150, per_day=400, shift_day=60, shift=2.0):
    """From day 61 on, fraud is more likely than the (unchanged) scores say."""
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(1, days + 1), per_day)
    q = 1 / (1 + np.exp(-rng.normal(-4.5, 1.6, len(day))))
    logit = np.log(q / (1 - q)) + np.where(day > shift_day, shift, 0.0)
    y = (rng.random(len(day)) < 1 / (1 + np.exp(-logit))).astype(int)
    return day, q, y


def test_aci_recovers_coverage_after_a_shift_where_split_does_not():
    day, p, y = _shifted_stream(0)
    _, cal_p, cal_y = _shifted_stream(1, days=30)
    static = coverage_by_block(day, y, SplitConformal(0.1).fit(cal_y, cal_p).predict_sets(p))
    sets, _ = (
        AdaptiveConformal(0.1, gamma=0.05, label_delay_days=0).fit(cal_y, cal_p).run(day, y, p)
    )
    adaptive = coverage_by_block(day, y, sets)
    assert len(static) == 5 and static["first_day"].tolist() == [1, 31, 61, 91, 121]
    assert (static["coverage"].iloc[:2] > 0.88).all()  # fine before the shift
    assert (static["coverage"].iloc[3:] < 0.85).all()  # under-covers after it
    assert (abs(adaptive["coverage"].iloc[3:] - 0.9) < 0.02).all()  # ACI is back on target
