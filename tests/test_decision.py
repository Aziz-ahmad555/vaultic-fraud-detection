"""Phase 8.3 decision engine: rules and costs by hand, thresholds chosen on validation."""

import numpy as np
import pytest

from vaultic.trust.decision import (
    ACTIONS,
    ALLOW,
    BLOCK,
    HOLD,
    MONITOR,
    STEP_UP,
    CostModel,
    Thresholds,
    breakdown,
    choose_thresholds,
    cost_sensitivity,
    decide,
    total_cost,
)

TH = Thresholds(t_monitor=5, t_step=20, t_hold=100, t_block=400, t_dis=0.3)


def test_rules_by_hand():
    p = np.array([0.01, 0.10, 0.10, 0.50, 0.90, 0.90, 0.90, 0.02])
    amount = np.array([100, 100, 300, 300, 1000, 1000, 50, 100])
    # EL:            1,  10,  30, 150,  900,  900,  45,   2
    sets = ["legit", "legit", "legit", "uncertain", "fraud", "uncertain", "fraud", "uncertain"]
    d = [0, 0, 0, 0, 0, 0, 0.5, 0]
    got = decide(p, amount, TH, sets, d)
    assert [ACTIONS[a] for a in got] == [
        "allow",  # EL 1
        "monitor",  # EL 10 >= 5
        "step_up",  # EL 30 >= 20
        "hold",  # EL 150 >= 100 (uncertain set does not lower it)
        "block",  # EL 900 >= 400 and confident {fraud}
        "hold",  # EL 900 but the set is uncertain: never block without confidence
        "hold",  # EL 45, but views disagree (0.5 >= 0.3)
        "step_up",  # EL 2, but the conformal set is uncertain
    ]


def test_without_sets_or_disagreement_el_decides():
    got = decide([0.9, 0.5, 0.04], [1000, 100, 100], TH)
    assert got.tolist() == [BLOCK, STEP_UP, ALLOW]  # EL 900, 50, 4 (monitor needs >= 5)
    assert decide([0.05], [100], TH).tolist() == [MONITOR]


def test_threshold_order_is_enforced():
    with pytest.raises(ValueError):
        Thresholds(t_monitor=0, t_step=50, t_hold=10, t_block=400)


def test_costs_by_hand():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0])
    amount = np.array([200, 50, 80, 70, 500, 30, 40, 20, 10, 60.0])
    action = np.array([ALLOW, ALLOW, MONITOR, MONITOR, STEP_UP, STEP_UP, HOLD, HOLD, BLOCK, BLOCK])
    cost = CostModel()  # C_FP 10, C_rev 5, step = 5, step-up catches all
    # missed 200 + 80; step-ups 2 x 5; holds 2 x 5; blocks 2 x 5 + 2 FP x 10
    assert total_cost(action, y, amount, cost) == 200 + 80 + 10 + 10 + 10 + 20
    b = breakdown(action, y, amount, cost)
    assert b["missed_fraud_value"] == 280 and b["missed_fraud_count"] == 2
    assert b["false_positives"] == 2 and b["reviews"] == 4 and b["total_cost"] == 330
    leaky = CostModel(step_up_catch=0.5, c_step=1.0)
    assert total_cost(action, y, amount, leaky) == 280 + 2 * 1 + 0.5 * 500 + 10 + 30


def _validation(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    p = np.clip(np.where(y == 1, rng.beta(5, 2, n), rng.beta(1, 12, n)), 0, 1)
    amount = rng.lognormal(4, 1, n)
    return p, amount, y


def test_chosen_thresholds_beat_simple_policies_on_validation():
    p, amount, y = _validation()
    th, info = choose_thresholds(p, amount, y, grid_size=12)
    chosen = total_cost(decide(p, amount, th), y, amount)
    assert chosen == pytest.approx(info["validation_cost"])  # monitor costs like allow
    everything = {
        "allow all": np.full(len(y), ALLOW),
        "review all": np.full(len(y), HOLD),
        "block all": np.full(len(y), BLOCK),
    }
    for name, action in everything.items():
        assert chosen < total_cost(action, y, amount), name
    assert th.t_monitor <= th.t_step <= th.t_hold <= th.t_block
    allowed = decide(p, amount, th) <= MONITOR
    assert (decide(p, amount, th)[allowed] == MONITOR).mean() == pytest.approx(0.05, abs=0.01)


def test_disagreement_rule_is_only_used_when_it_pays():
    p, amount, y = _validation(seed=1)
    useless = np.random.default_rng(2).random(len(y))  # unrelated to fraud
    th, _ = choose_thresholds(p, amount, y, disagreement=useless, grid_size=8)
    assert th.t_dis == np.inf
    # a disagreement signal that marks frauds the score misses is worth a review
    hidden = (y == 1) & (p < 0.2)
    p2 = np.where(hidden, 0.01, p)
    signal = np.where(hidden, 0.9, 0.05)
    th2, _ = choose_thresholds(p2, amount, y, disagreement=signal, grid_size=8)
    assert th2.t_dis <= 0.9


def test_validation_only_api_and_fixed_thresholds_on_new_data():
    p, amount, y = _validation(seed=3)
    th, _ = choose_thresholds(p, amount, y, grid_size=8)
    p_new, amount_new, _ = _validation(seed=4)
    # applying to new rows needs no labels: thresholds are frozen, decide() takes none
    assert decide(p_new, amount_new, th).shape == (len(p_new),)


def test_cost_sensitivity_blocks_less_as_false_positives_get_dearer():
    p, amount, y = _validation(seed=5)
    out = cost_sensitivity(p, amount, y, c_fp_values=(2, 50), grid_size=8)
    assert [o["c_fp"] for o in out] == [2, 50]
    cheap, dear = (Thresholds(**o["thresholds"]) for o in out)
    blocks = [int((decide(p, amount, t) == BLOCK).sum()) for t in (cheap, dear)]
    assert blocks[1] <= blocks[0]
