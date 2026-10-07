"""Faithfulness metrics on a synthetic model whose true feature importance is known.

Model: p = sigmoid(3 x0 + 2 x1 + 1 x2 + 0 x3 + 0 x4), baseline 0. For a linear-logit model
the exact attribution of feature j is coef_j * (x_j - baseline_j).
"""

import numpy as np
import pytest

from vaultic.explain.faithfulness import (
    counterfactual_quality,
    curve_auc,
    deletion_curve,
    faithfulness_report,
    insertion_curve,
    jaccard_top_k,
    seed_consistency,
)

COEF = np.array([3.0, 2.0, 1.0, 0.0, 0.0])


def predict(X):
    return 1 / (1 + np.exp(-(np.asarray(X) @ COEF)))


def sig(z):
    return 1 / (1 + np.exp(-z))


def test_deletion_and_insertion_curves_by_hand():
    x = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    attr = COEF * x  # exact attributions: 3, 2, 1, 0, 0
    d = deletion_curve(predict, x, attr, np.zeros(5), max_k=3)
    assert d.tolist() == pytest.approx([sig(6), sig(3), sig(1), sig(0)])
    i = insertion_curve(predict, x, attr, np.zeros(5), max_k=3)
    assert i.tolist() == pytest.approx([sig(0), sig(3), sig(5), sig(6)])


def test_curve_auc_by_hand():
    assert curve_auc(np.array([1.0, 0.0])) == pytest.approx(0.5)
    assert curve_auc(np.array([1.0, 0.5, 0.0])) == pytest.approx(0.5)
    assert curve_auc(np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)


def test_true_attributions_beat_random_and_a_wrong_global_order():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 1.5, size=(300, 5))  # positive values: every true feature raises risk
    attr = COEF * X
    wrong_order = np.array([0.0, 0.0, 1.0, 2.0, 3.0])  # puts the useless features first
    r = faithfulness_report(predict, X, attr, np.zeros(5), wrong_order, max_k=5)
    assert r["deletion_auc"] < r["deletion_auc_random"] < r["deletion_auc_permutation"]
    assert r["insertion_auc"] > r["insertion_auc_random"] > r["insertion_auc_permutation"]
    # even the correct global order is no better than exact per-row attributions: per row,
    # coef_j * x_j can rank features differently (e.g. x0 = 0.5, x1 = 1.5 puts x1 first)
    right = faithfulness_report(predict, X, attr, np.zeros(5), COEF, max_k=5)
    assert right["deletion_auc"] <= right["deletion_auc_permutation"]
    assert right["insertion_auc"] >= right["insertion_auc_permutation"]


def test_jaccard_top3_by_hand():
    a = np.array([[5.0, 4.0, 3.0, 0.1, 0.0]])  # top-3 {0, 1, 2}
    b = np.array([[5.0, 4.0, 0.1, 3.0, 0.0]])  # top-3 {0, 1, 3}
    assert jaccard_top_k(a, b, 3).tolist() == [pytest.approx(2 / 4)]
    assert jaccard_top_k(a, -a, 3).tolist() == [1.0]  # sign does not matter, magnitude does


def test_seed_consistency_by_hand():
    a = np.array([[6.0, 5, 4, 3, 2, 1, 0]])  # top-5 {0..4}
    b = np.array([[6.0, 5, 4, 3, 0, 1, 2]])  # top-5 {0, 1, 2, 3, 6}: 4 shared of 6
    assert seed_consistency([a, a, a], k=5) == 1.0
    # pairs (a, a) = 1, (a, b) = 4/6, (a, b) = 4/6 -> mean (1 + 2 * 2/3) / 3
    assert seed_consistency([a, a, b], k=5) == pytest.approx((1 + 2 * 4 / 6) / 3)
    assert np.isnan(seed_consistency([a], k=5))


def test_counterfactual_quality_by_hand():
    X = np.array([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0]])
    X_cf = np.array(
        [
            [-1.0, 1.0, 1.0, 0.0, 0.0],  # x0 lowered by 2: logit 6 -> 0, risk 0.5 (not < 0.5)
            [1.0, -1.0, -1.0, 0.0, 0.0],  # x1, x2 lowered by 2: logit 6 -> 0 -> also 0.5
        ]
    )
    q = counterfactual_quality(
        predict, X, X_cf, threshold=0.6, scale=np.full(5, 2.0),
        actionable=np.array([False, True, True, True, True]),
    )  # fmt: skip
    assert q["validity"] == 1.0  # both reach 0.5 < 0.6
    assert q["sparsity"] == pytest.approx(1.5)  # 1 and 2 features changed
    assert q["proximity"] == pytest.approx((1.0 + 2.0) / 2)  # L1 in units of scale 2
    assert q["actionable_share"] == 0.5  # row 1 changes x0, which is not actionable
