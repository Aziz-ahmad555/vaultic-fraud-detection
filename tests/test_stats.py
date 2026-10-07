"""Phase 13 statistics tooling, checked against hand-worked examples."""

import numpy as np
import pytest

from vaultic.eval.metrics import cost_at
from vaultic.eval.stats import cost_breakdown, cost_sensitivity, effect_sizes, holm_table


def test_holm_table_by_hand():
    table = holm_table(
        [
            {"name": "MVAF vs F3", "diff": 0.02, "p_value": 0.01},
            {"name": "MVAF vs F4", "diff": 0.01, "p_value": 0.04},
            {"name": "MVAF vs F1", "diff": 0.05, "p_value": 0.03},
        ]
    )
    # sorted p: 0.01 * 3 = 0.03; 0.03 * 2 = 0.06; 0.04 * 1 = 0.04 -> running max 0.06
    assert table["p_holm"].tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert table["significant"].tolist() == [True, False, False]
    assert table["diff"].tolist() == [0.02, 0.01, 0.05]  # other columns are kept


def test_effect_sizes_by_hand():
    e = effect_sizes([0.70, 0.72, 0.74], [0.60, 0.61, 0.65])
    # differences 0.10, 0.11, 0.09: mean 0.10, sample sd 0.01; B mean 0.62
    assert e["diff"] == pytest.approx(0.10)
    assert e["cohens_d_paired"] == pytest.approx(10.0)
    assert e["relative_pct"] == pytest.approx(0.10 / 0.62 * 100)
    assert np.isnan(effect_sizes([0.7], [0.6])["cohens_d_paired"])  # one seed: undefined


def test_cost_breakdown_by_hand():
    y = np.array([1, 1, 0, 0, 1])
    flagged = np.array([True, False, True, False, False])
    amount = np.array([100.0, 50.0, 30.0, 20.0, 10.0])
    c = cost_breakdown(y, flagged, amount)  # defaults $10 / $5
    # missed fraud 50 + 10 = 60; one false positive; two reviews -> 60 + 10 + 10
    assert (c["missed_fraud_amount"], c["n_false_positives"], c["n_reviews"]) == (60.0, 1, 2)
    assert c["total_cost"] == 80.0
    assert cost_breakdown(y, flagged, amount, c_fp=2, c_rev=50)["total_cost"] == 60 + 2 + 100
    # consistent with the harness metric at the same threshold
    s = flagged.astype(float)
    assert cost_at(y, s, amount, 0.5) == 80.0 and cost_at(y, s, amount, 0.5, 2, 50) == 162.0


def test_cost_sensitivity_rechooses_the_threshold_on_validation():
    # scores rank all frauds first; cheap reviews -> flag more, expensive reviews -> flag fewer
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    amount = np.array([100.0, 8.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    table = cost_sensitivity(y, s, amount, y, s, amount, grid=(2.0, 50.0))
    assert len(table) == 4 and set(zip(table["c_fp"], table["c_rev"], strict=True)) == {
        (2.0, 2.0), (2.0, 50.0), (50.0, 2.0), (50.0, 50.0),
    }  # fmt: skip
    cheap = table[(table["c_fp"] == 2) & (table["c_rev"] == 2)].iloc[0]
    dear = table[(table["c_fp"] == 50) & (table["c_rev"] == 50)].iloc[0]
    # cheap reviews: catching the $8 fraud (cost 2 per review) pays -> both frauds flagged
    assert cheap["missed_fraud_amount"] == 0 and cheap["n_reviews"] == 2
    # $50 per review: flagging the $8 fraud costs more than missing it -> only the $100 one
    assert dear["missed_fraud_amount"] == 8.0 and dear["n_reviews"] == 1
    assert dear["total_cost"] == 8 + 50
