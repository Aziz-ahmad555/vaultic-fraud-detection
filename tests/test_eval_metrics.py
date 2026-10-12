import numpy as np
import pytest

from vaultic.eval.bootstrap import holm, paired_bootstrap, seed_mean_ci
from vaultic.eval.metrics import (
    brier,
    choose_cost_threshold,
    choose_f1_threshold,
    cost_at,
    ece,
    f1_at,
    pr_auc,
    precision_at_k,
    recall_at_fpr,
)

Y = np.array([0, 0, 0, 0, 1, 0, 1, 1])
S = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


def test_pr_auc_perfect_and_known():
    assert pr_auc(Y, Y.astype(float)) == 1.0
    # ranked: 1 (0.8), 1 (0.7), 0 (0.6), 1 (0.5): AP = (1 + 1 + 3/4) / 3
    assert pr_auc(Y, S) == pytest.approx((1 + 1 + 0.75) / 3)


def test_recall_at_fpr():
    # 5 negatives: FPR 0 allows the top two positives only; FPR 0.2 allows all three
    assert recall_at_fpr(Y, S, 0.0) == pytest.approx(2 / 3)
    assert recall_at_fpr(Y, S, 0.2) == pytest.approx(1.0)


def test_precision_at_k_and_f1():
    assert precision_at_k(Y, S, 3) == pytest.approx(2 / 3)
    # threshold 0.5 flags 0.5, 0.6, 0.7, 0.8 -> tp 3, fp 1, fn 0
    assert f1_at(Y, S, 0.5) == pytest.approx(6 / 7)


def test_brier_and_ece():
    assert brier(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0
    assert brier(np.array([0, 1]), np.array([0.5, 0.5])) == 0.25
    # all scores 0.5 in one bin, observed rate 0.5 -> perfectly calibrated
    assert ece(np.array([0, 1]), np.array([0.5, 0.5])) == 0.0
    assert ece(np.array([0, 0]), np.array([0.9, 0.9])) == pytest.approx(0.9)


def test_cost_and_threshold_choice():
    amount = np.full(len(Y), 100.0)
    # threshold 0.65 flags 0.7 and 0.8: misses one fraud (100), no FP, 2 reviews (10)
    assert cost_at(Y, S, amount, 0.65) == pytest.approx(110.0)
    assert 0.0 < choose_f1_threshold(Y, S) <= 0.8
    t = choose_cost_threshold(Y, S, amount)
    assert cost_at(Y, S, amount, t) <= cost_at(Y, S, amount, 0.65)


def test_bootstrap_is_deterministic_and_brackets_point():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    s = [y * 0.5 + rng.random(2000) for _ in range(2)]
    ci1 = seed_mean_ci(y, s, pr_auc, n_boot=200, seed=0)
    ci2 = seed_mean_ci(y, s, pr_auc, n_boot=200, seed=0)
    assert ci1 == ci2
    point = np.mean([pr_auc(y, x) for x in s])
    assert ci1[0] <= point <= ci1[1]


def test_paired_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.1).astype(int)
    good = [y + rng.normal(0, 0.5, 3000)]
    bad = [y + rng.normal(0, 2.0, 3000)]
    r = paired_bootstrap(y, good, bad, pr_auc, n_boot=200)
    assert r["diff"] > 0 and r["ci_low"] > 0 and r["p_value"] < 0.05
    same = paired_bootstrap(y, good, good, pr_auc, n_boot=50)
    assert same["diff"] == 0.0 and same["p_value"] == 1.0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_recall_at_fpr_keeps_collinear_roc_points():
    """D49: each score is a tie of 1 positive + 1 negative, so the ROC points are collinear;
    the default roc_curve drops the middle ones. 100 negatives, budget 2% = 2 negatives:
    the top two tie pairs fit, so recall is 2/10."""
    pairs = 5
    y = np.array([1, 0] * pairs + [1] * 5 + [0] * (100 - pairs))
    s = np.concatenate(
        [np.repeat(np.linspace(0.99, 0.95, pairs), 2), np.full(5, 0.5), np.full(95, 0.1)]
    )
    assert recall_at_fpr(y, s, 0.02) == pytest.approx(0.2)
    assert recall_at_fpr(y, s, 0.04) == pytest.approx(0.4)


def test_recall_at_fpr_never_splits_a_tie():
    """100 negatives, budget 1% = 1 negative. The top tied block holds 2 positives and 2
    negatives (FPR 2%), so no threshold fits the budget with any positive: recall is 0."""
    y = np.array([1, 1, 0, 0] + [1] * 3 + [0] * 98)
    s = np.array([0.9] * 4 + [0.5] * 3 + [0.1] * 98)
    assert recall_at_fpr(y, s, 0.01) == 0.0
    # at 2% the top block fits (FPR 2%), and the next block adds 3 positives, no negatives
    assert recall_at_fpr(y, s, 0.02) == pytest.approx(1.0)


def test_precision_at_k_tie_at_the_boundary_is_order_independent():
    """Top-1 is a positive; then a 4-way tie holding 1 positive straddles k = 2:
    expected precision (1 + 1 * 1/4) / 2 = 0.625 whatever the row order."""
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1])
    for y in ([1, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]):
        assert precision_at_k(np.array(y), s, 2) == pytest.approx(0.625)
    assert precision_at_k(np.array([1, 0, 0, 0, 1, 0]), s, 5) == pytest.approx(
        2 / 5
    )  # no split tie
