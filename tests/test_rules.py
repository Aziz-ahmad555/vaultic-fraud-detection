"""Rules engine (Phase B, D192, D195): hand-made examples with known answers."""

import numpy as np
import pandas as pd

from vaultic.rules.calibrate import calibrate, combined, common_cap, most_permissive, thresholds_at
from vaultic.rules.engine import (
    BY_ID,
    RULES,
    SENSITIVITY_FILE,
    combine,
    evaluate,
    load_thresholds,
    rule_values,
)

TH = {"R1": 3, "R2": 10, "R3": 4, "R4": 0.5, "R5": 500, "R6": 5}


def test_most_permissive_on_ten_rows():
    # 10 legit rows, budget 20% -> at most 2 legit rows may fire
    v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 9], dtype=float)
    legit = np.ones(10, dtype=bool)
    assert most_permissive(v, legit, 0.2) == 9  # >=9 fires 2 rows; >=8 would fire 3
    assert most_permissive(v, legit, 0.1) == np.nextafter(9, np.inf)  # 9 ties: no value fits
    assert most_permissive(v, legit, 1.0) == 1


def test_most_permissive_ignores_missing_and_fraud_rows():
    v = np.array([np.nan, np.nan, 1, 2, 100, 100], dtype=float)
    legit = np.array([1, 1, 1, 1, 0, 0], dtype=bool)  # the 100s are fraud: never count as FP
    assert most_permissive(v, legit, 0.25) == 2  # 4 legit rows -> 1 may fire


def test_rules_fire_and_trace():
    row = {"uid_n_1h": 3, "uid_n_24h": 2, "ent_uids_7d_device": 1, "g_fraud_rate_card": 0.0,
           "g_fraud_rate_device": np.nan, "hist_n_past": 5, "TransactionAmt": 900,
           "new_device": 1, "amt_ratio_median": 6.0}  # fmt: skip
    out = evaluate(row, TH)
    fired = {t["rule"] for t in out["trace"] if t["fired"]}
    assert fired == {"R1", "R6"}  # R5 needs a new customer
    assert out["action"] == "hold" and out["alert"]
    assert len(out["trace"]) == len(RULES)


def test_missing_never_fires():
    out = evaluate({}, TH)
    assert out["action"] == "allow" and not out["alert"]
    assert all(not t["fired"] and t["value"] is None for t in out["trace"])


def test_new_customer_cap():
    row = {"hist_n_past": 0, "TransactionAmt": 500}
    assert [t["rule"] for t in evaluate(row, TH)["trace"] if t["fired"]] == ["R5"]
    assert evaluate({"hist_n_past": 1, "TransactionAmt": 5000}, TH)["action"] == "allow"


def test_combine_and_degraded_mode():
    assert combine("step_up", "allow")["action"] == "step_up"
    assert combine("allow", "hold")["action"] == "hold"
    assert combine("hold", "block")["action"] == "block"
    d = combine("step_up", None)
    assert d["action"] == "step_up" and d["mode"].startswith("degraded")


def test_vectorised_values_match_rowwise():
    rng = np.random.default_rng(0)
    n = 50
    f = pd.DataFrame({"uid_n_1h": rng.integers(0, 5, n), "uid_n_24h": rng.integers(0, 20, n),
                      "ent_uids_7d_device": rng.integers(0, 6, n),
                      "g_fraud_rate_card": rng.random(n), "g_fraud_rate_device": np.nan,
                      "hist_n_past": rng.integers(0, 3, n), "TransactionAmt": rng.random(n) * 900,
                      "new_device": rng.integers(0, 2, n), "amt_ratio_median": rng.random(n) * 9})  # fmt: skip
    for rule in RULES:
        vec = rule_values(f, rule)
        row = np.array([rule.value(r) for r in f.to_dict("records")])
        np.testing.assert_array_equal(np.isnan(vec), np.isnan(row))
        np.testing.assert_allclose(vec[~np.isnan(vec)], row[~np.isnan(row)])


def _frame(seed=1, n=2000):
    rng = np.random.default_rng(seed)
    n = 2000
    f = pd.DataFrame({"uid_n_1h": rng.poisson(1, n), "uid_n_24h": rng.poisson(3, n),
                      "ent_uids_7d_device": rng.poisson(1, n), "g_fraud_rate_card": rng.random(n),
                      "g_fraud_rate_device": rng.random(n), "hist_n_past": rng.integers(0, 4, n),
                      "TransactionAmt": rng.exponential(100, n), "new_device": rng.integers(0, 2, n),
                      "amt_ratio_median": rng.exponential(1, n),
                      "isFraud": (rng.random(n) < 0.05).astype(int)})  # fmt: skip
    return f


def test_calibrate_respects_budget():
    out = calibrate(_frame(), 0.005)
    for r in out["rules"].values():
        assert r["validation"]["fpr"] <= 0.005


def test_common_cap_is_largest_meeting_both_budgets():
    f = _frame()
    model = np.zeros(len(f), dtype=bool)
    cap, log = common_cap(f, model, rules_budget=0.015)
    assert cap is not None and log[-1]["ok"] and not any(r["ok"] for r in log[:-1])
    c = combined(f, thresholds_at(f, cap), model)
    assert c["rules_only"]["fpr"] <= 0.015
    assert c["rules_plus_model"]["fpr"] - c["model_only"]["fpr"] <= 0.005
    assert [r["cap"] for r in log] == [round(0.005 - i * 0.0001, 6) for i in range(len(log))]


def test_common_cap_none_when_unreachable():
    cap, log = common_cap(_frame(), np.zeros(2000, dtype=bool), rules_budget=0.0)
    assert cap is None and log[-1]["cap"] == 0.001 and len(log) == 41


def test_committed_thresholds_cover_every_rule():
    for path in (None, SENSITIVITY_FILE):  # D196 primary, D195 sensitivity
        th = load_thresholds() if path is None else load_thresholds(path)
        assert set(th) == set(BY_ID)
        assert all(np.isfinite(v) for v in th.values())
