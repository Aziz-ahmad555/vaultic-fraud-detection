"""Phase B scenario framework: the history overlay equals rebuilding the indexes with the
synthetic rows included (hand-made examples), the synthetic-ID guard, and the scorecard."""

import numpy as np
import pandas as pd
import pytest

from vaultic.features.graph import EventIndex
from vaultic.scenarios import LABEL, SYNTHETIC_ID_START, require_no_synthetic
from vaultic.scenarios.overlay import OverlayFrame, OverlayIndex, _merge_positions, first_events
from vaultic.scenarios.run import summarise


def test_overlay_index_equals_full_index():
    rng = np.random.default_rng(0)
    keys = rng.choice(list("abcde"), 60).astype(object)
    times = rng.permutation(1000)[:60]
    vals = rng.integers(0, 2, 60).astype(float)
    base, extra = slice(0, 45), slice(45, 60)
    full = EventIndex(keys, times, vals)
    ov = OverlayIndex(EventIndex(keys[base], times[base], vals[base]), keys[extra], times[extra],
                      values=vals[extra])  # fmt: skip
    q = rng.choice(list("abcdez"), 40).astype(object)
    cut = rng.integers(0, 1100, 40)
    np.testing.assert_array_equal(ov.count_before(q, cut), full.count_before(q, cut))
    np.testing.assert_allclose(ov.sum_before(q, cut), full.sum_before(q, cut))


def _firsts(node, pair, t):
    df = pd.DataFrame({"node": node, "pair": pair, "t": t})
    f = df.groupby("pair").agg(node=("node", "first"), t=("t", "min"))
    return EventIndex(f["node"].to_numpy(object), f["t"].to_numpy())


def test_first_appearances_stay_exact():
    # pairs node|uid; the synthetic rows add a pair before, between and after stored ones
    node = np.array(["n1", "n1", "n2", "n2", "n1", "n2", "n3"], dtype=object)
    uid = np.array(["u1", "u2", "u1", "u3", "u3", "u3", "u9"], dtype=object)
    t = np.array([10, 20, 30, 40, 5, 35, 50])
    pair = node + "|" + uid
    base, syn = slice(0, 4), slice(4, 7)  # n1|u3 new; n2|u3 at 35 precedes stored 40; n3|u9 new
    full = _firsts(node, pair, t)
    b = _firsts(node[base], pair[base], t[base])
    fk, ft, fw = first_events(EventIndex(pair[base], t[base]), node[syn], pair[syn], t[syn])
    ov = OverlayIndex(b, fk, ft, weights=fw)
    q = np.array(["n1", "n2", "n3"] * 4, dtype=object)
    cut = np.repeat([6, 36, 41, 100], 3)
    np.testing.assert_array_equal(ov.count_before(q, cut), full.count_before(q, cut))


def test_merge_positions_in_time_order():
    times = np.array([5, 1, 9, 3, 7])  # positions 0-2 stored, 3-4 synthetic
    idx = {"a": np.array([1, 0, 2])}
    out = _merge_positions(idx, np.array(["a", "b"], dtype=object), np.array([3, 4]), times)
    assert out["a"].tolist() == [1, 3, 0, 2] and out["b"].tolist() == [4]
    assert idx["a"].tolist() == [1, 0, 2]  # the stored index is not changed


def test_overlay_frame_iloc_keeps_requested_order():
    base = pd.DataFrame({"x": [0, 1, 2]})
    f = OverlayFrame(base, pd.DataFrame({"x": [10, 11]}))
    assert f.iloc[[1, 3, 2, 4]]["x"].tolist() == [1, 10, 2, 11]
    assert len(f) == 5 and f["x"].tolist() == [0, 1, 2, 10, 11]


def test_synthetic_guard():
    require_no_synthetic([1, 2, SYNTHETIC_ID_START - 1])
    with pytest.raises(RuntimeError):
        require_no_synthetic([1, SYNTHETIC_ID_START])
    assert LABEL == "synthetic scenario - not the evaluation"


def test_scorecard_and_budgets():
    # 4 synthetic fraud rows, 10 legit rows; actions: 0 allow, 2 step-up, 3 hold
    t = pd.DataFrame({"synthetic": [True] * 4 + [False] * 10, "label": [1] * 4 + [0] * 10,
                      "model_action": [0, 0, 2, 2] + [2] + [0] * 9})  # fmt: skip
    rules = np.array([0, 3, 3, 0] + [0, 2] + [0] * 8)
    both = np.maximum(rules, t["model_action"].to_numpy())
    s = summarise(t, rules, both)
    assert s["model_only"]["recall_synthetic"] == 0.5 and s["model_only"]["fpr_legit"] == 0.1
    assert s["rules_only"]["recall_synthetic"] == 0.5 and s["rules_only"]["fpr_legit"] == 0.1
    assert s["rules_plus_model"]["recall_synthetic"] == 0.75  # row 1 has no alert
    assert s["rules_plus_model"]["fpr_legit"] == pytest.approx(0.2)
    assert s["added_fpr"] == pytest.approx(0.1) and not s["budget_added_fpr_ok"]
    assert not s["budget_rules_only_ok"]  # 10% > 2%
    assert s["rules_only"]["first_alerted_synthetic_row"] == 2
    assert s["model_only"]["first_alerted_synthetic_row"] == 3
