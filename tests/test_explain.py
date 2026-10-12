"""Phase 9 explanation layer: reason-code templates, view contributions, narrative, and the
counterfactual immutability lock."""

import math

import numpy as np
import pandas as pd
import pytest

from vaultic.explain import counterfactual as cf
from vaultic.explain.narrative import Case, graph_evidence, narrative
from vaultic.explain.reason_codes import (
    Reason,
    describe,
    duration,
    lookup,
    to_points,
    top_reasons,
)
from vaultic.explain.view_contrib import describe_row, from_mvaf, view_contributions

# ---- reason codes --------------------------------------------------------------------------


def test_engineered_templates_by_hand():
    assert (
        describe("amt_ratio_median", 25.04)
        == "Amount is 25.0x this customer's usual (median) amount"
    )
    assert describe("vel_n_1h", 3) == "3 earlier transactions by this customer in the last hour"
    assert describe("uid_amt_sum_30d", 1234.5) == "Customer spent $1,234.50 in the last 30 days"
    assert describe("g_fraud_rate_device", 0.25) == (
        "25% of earlier labelled transactions on this device were confirmed fraud"
    )
    assert describe("g_shared_uids_address", 4) == "4 other customers share this billing address"
    assert describe("secs_since_prev", 480) == "8 min since this customer's previous transaction"
    assert describe("hour", 3) == "Transaction at relative hour 3 (dataset clock, not local time)"
    assert describe("anomaly_uid_if", 0.97).startswith("More unusual for this customer than 97%")
    assert describe("freq_uid", 12) == "This customer appeared 12 times in the training period"


def test_flags_and_missing_values():
    assert describe("new_device", 1.0) == "First time this customer uses this device"
    assert describe("new_device", 0.0) == "This customer has used this device before"
    assert describe("new_device", np.nan) == "No device on this transaction"
    assert describe("hist_n_past", 0) == "No earlier transactions for this customer (cold start)"
    assert describe("amt_z", np.nan) == "Too few earlier transactions to compare the amount"
    assert describe("uid_fraud_rate_known", None) == "No confirmed labels yet for this customer"


def test_masked_features_keep_honest_wording():
    assert describe("C13", 57.0, points=4) == "Unusual value in masked counter C13"
    assert describe("C13", 1.0, points=-3) == "Value in masked counter C13 lowers the risk score"
    assert describe("D4", np.nan) == "No value in masked time-delta field D4"
    assert describe("V258", 2.0, points=1) == "Unusual value in masked Vesta feature V258"
    assert describe("id_31", "x", points=1) == "Unusual value in masked identity field id_31"
    assert describe("M4", "M2", points=1) == "Unusual value in masked match flag M4"
    assert describe("card1", 9500, points=1) == "Unusual value in masked payment-card field card1"
    for name in ("C13", "D4", "V258", "id_31", "M4", "card1", "addr1", "dist1"):
        for pts in (5, -5):
            text = describe(name, 1.0, points=pts)
            assert "masked" in text and name in text  # never an invented meaning


def test_every_engineered_feature_has_a_template():
    from vaultic.features.behavioral import build_behavioral
    from vaultic.features.graph import build_graph_features
    from vaultic.features.pipeline import build_features
    from vaultic.features.pit import FrequencyEncoder

    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.sort(rng.integers(0, 40 * 86_400, n)),
            "TransactionAmt": rng.gamma(2, 40, n),
            "isFraud": (rng.random(n) < 0.2).astype(int),
            "ProductCD": rng.choice(["W", "C"], n),
            "P_emaildomain": rng.choice(["a.com", "b.com", None], n),
            "R_emaildomain": rng.choice(["a.com", None], n),
            "DeviceInfo": rng.choice(["d1", "d2", None], n),
            "DeviceType": rng.choice(["mobile", None], n),
            "addr1": rng.choice([100.0, 200.0], n),
            "addr2": rng.choice([87.0, np.nan], n),
            "D1": rng.integers(0, 30, n).astype(float),
            **{f"card{i}": rng.choice([1.0, 2.0], n) for i in range(1, 7)},
        }
    )
    df["day"] = df["TransactionDT"] // 86_400
    uid = pd.Series(rng.choice(["u1", "u2", "u3"], n))
    enc = {c: FrequencyEncoder().fit(df[c]) for c in ("card1", "addr1", "P_emaildomain")}
    enc["uid"] = FrequencyEncoder().fit(uid)
    names = [
        *build_features(df, uid, enc, 30).columns,
        *build_behavioral(df, uid).columns,
        *build_graph_features(df, uid, hub_thresholds={"card": 5.0, "device": 5.0}).columns,
        "anomaly_if",
        "anomaly_ae",
        "anomaly_uid_if",
    ]
    for name in names:
        if name == "TransactionID":
            continue
        lookup(name)  # raises KeyError if missing
        assert describe(name, 1.0, 1.0) and describe(name, np.nan)


def test_unknown_feature_raises():
    with pytest.raises(KeyError):
        lookup("made_up_feature")


def test_points_and_top_reasons():
    assert to_points(math.log(2)) == pytest.approx(20.0)  # doubling the odds = 20 points
    shap = {"amt_ratio_median": math.log(2), "new_device": 0.5 * math.log(2),
            "C13": -0.75 * math.log(2), "hour": float("nan")}  # fmt: skip
    values = {"amt_ratio_median": 25.0, "new_device": 1.0, "C13": 2.0}
    reasons = top_reasons(shap, values, k=2)
    assert [r.feature for r in reasons] == ["amt_ratio_median", "C13"]  # by |points|
    assert reasons[1].text == "Value in masked counter C13 lowers the risk score"
    assert reasons[0].render() == "Amount is 25.0x this customer's usual (median) amount (+20 pts)"
    up = top_reasons(shap, values, k=5, direction="up")
    assert [r.feature for r in up] == ["amt_ratio_median", "new_device"]


def test_duration_formatting():
    assert (
        duration(480) == "8 min" and duration(7200) == "2.0 h" and duration(3 * 86400) == "3.0 days"
    )


# ---- view contributions --------------------------------------------------------------------


def test_view_contributions_by_hand():
    names = ["tabular", "behavioral", "graph"]
    # logits: logit(0.8) = ln 4, logit(0.2) = -ln 4; graph missing
    views = np.array([[0.8, 0.2, np.nan], [0.5, 0.5, 0.5]])
    weights = np.array([[0.75, 0.25, 0.0], [0.2, 0.3, 0.5]])
    vc = view_contributions(weights, views, names, fused=np.array([0.8, 0.5]))
    ln4 = math.log(4)
    assert vc.contribution[0, :2] == pytest.approx([0.75 * ln4, -0.25 * ln4])
    assert np.isnan(vc.contribution[0, 2]) and np.isnan(vc.share[0, 2])  # missing, not 0%
    assert vc.share[0, :2] == pytest.approx([0.75, 0.25])
    assert vc.bias[0] == pytest.approx(ln4 - 0.5 * ln4)  # logit(fused) - sum of contributions
    assert vc.share[1] == pytest.approx([1 / 3] * 3)  # nothing drove it: equal shares
    assert describe_row(vc, 0) == (
        "Tabular 75% (raises risk), behavior 25% (lowers risk); not available: graph evidence"
    )


def test_view_contributions_from_mvaf_are_additive():
    from vaultic.fusion.mvaf import MVAF, _logit

    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.3).astype(int)
    views = np.clip(0.5 + 0.3 * (y[:, None] - 0.5) + rng.normal(0, 0.15, (n, 3)), 0.01, 0.99)
    views[rng.random((n, 3)) < 0.2] = np.nan
    views[np.isnan(views).all(axis=1), 0] = 0.5
    model = MVAF(epochs=5, seed=0).fit(views, y)
    vc = from_mvaf(model, views, ["tabular", "behavioral", "graph"])
    total = np.nansum(vc.contribution, axis=1) + vc.bias
    assert total == pytest.approx(_logit(model.predict_proba(views)))
    assert np.all(np.isnan(vc.share) == np.isnan(views))
    assert np.nansum(vc.share, axis=1) == pytest.approx(np.ones(n))


# ---- narrative -----------------------------------------------------------------------------


def test_graph_evidence_skips_missing_and_zero():
    feats = {"g_twohop_fraud": 3.0, "g_fraud_rate_device": np.nan, "g_fraud_rate_card": 0.0,
             "g_comm_fraud_rate": 0.4}  # fmt: skip
    assert graph_evidence(feats) == [
        "Linked through shared entities to 3 confirmed-fraud transactions (counted once per "
        "shared entity)",
        "40% of labelled transactions in its community were confirmed fraud",
    ]
    assert graph_evidence({}) == []


def test_narrative_combines_all_parts_and_states_gaps():
    reasons = [
        Reason("amt_ratio_median", "Amount is 25.0x this customer's usual (median) amount", 20.0)
    ]
    full = narrative(
        Case(
            transaction_id=7,
            risk=0.87,
            decision="send to review",
            reasons=reasons,
            views="Graph evidence 45% (raises risk), behavior 30% (raises risk)",
            graph=["3 other customers share this device"],
            counterfactual="If the amount were $120.00 instead of $950.00, the risk would fall "
            "from 87 to 31.",
            uncertain=True,
        )
    )
    assert full == (
        "Transaction 7: risk 87/100, decision: send to review. The model is uncertain about this "
        "case (its conformal set contains both classes). Main reasons: Amount is 25.0x this "
        "customer's usual (median) amount (+20 pts). Evidence by view: Graph evidence 45% (raises "
        "risk), behavior 30% (raises risk). Graph evidence: 3 other customers share this device. "
        "If the amount were $120.00 instead of $950.00, the risk would fall from 87 to 31."
    )
    bare = narrative(Case(transaction_id=8, risk=0.1, decision="approve"))
    assert "No feature-level reasons" in bare and "No graph evidence" in bare
    assert "No valid counterfactual" in bare and "uncertain" not in bare


# ---- counterfactual lock -------------------------------------------------------------------

COLUMNS = ["TransactionAmt", "DeviceInfo", "card1", "addr1", "uid_n_past", "amt_ratio_median",
           "g_twohop_fraud", "C13", "V258"]  # fmt: skip


class _AmountModel:
    """Risk = 0.9 when the amount is above 500, else 0.2."""

    def predict_proba(self, frame):
        p = np.where(frame["TransactionAmt"].to_numpy() > 500, 0.9, 0.2)
        return np.column_stack([1 - p, p])


def _query():
    return pd.DataFrame([{"TransactionAmt": 950.0, "DeviceInfo": "d9", "card1": 1111.0,
                          "addr1": 300.0, "uid_n_past": 12.0, "amt_ratio_median": 25.0,
                          "g_twohop_fraud": 3.0, "C13": 4.0, "V258": 1.0}])  # fmt: skip


def test_lock_classification():
    assert cf.immutable_features(COLUMNS) == ["card1", "addr1", "uid_n_past", "g_twohop_fraud",
                                              "C13"]  # fmt: skip
    assert cf.derived_features(COLUMNS) == ["amt_ratio_median"]
    assert cf.features_to_vary(COLUMNS) == ["TransactionAmt", "DeviceInfo"]
    with pytest.raises(ValueError, match="locked"):
        cf.features_to_vary(COLUMNS, vary=["TransactionAmt", "card1"])
    with pytest.raises(ValueError, match="locked"):
        cf.features_to_vary(COLUMNS, vary=["amt_ratio_median"])  # derived, no derive callback
    assert cf.features_to_vary(COLUMNS, vary=["amt_ratio_median"], derive=lambda c, q: c) == [
        "amt_ratio_median"
    ]


def test_generator_only_sees_mutable_features_and_locked_changes_are_rejected():
    seen = {}

    def backend(query, total, vary):
        seen["vary"] = vary
        good = query.assign(TransactionAmt=120.0)
        cheat_card = query.assign(TransactionAmt=100.0, card1=2222.0)  # changes a locked field
        cheat_hist = query.assign(TransactionAmt=90.0, uid_n_past=0.0)
        same_risk = query.assign(DeviceInfo="d1")  # changes only the device: still risky
        return pd.concat([good, cheat_card, cheat_hist, same_risk], ignore_index=True)

    ex = cf.CounterfactualExplainer(_AmountModel(), COLUMNS, backend)
    res = ex.explain(_query())
    assert seen["vary"] == ["TransactionAmt", "DeviceInfo"]
    assert res.rejected_locked == 2
    assert len(res.candidates) == 2
    for col in cf.immutable_features(COLUMNS):  # every accepted candidate keeps them
        assert (res.candidates[col] == _query()[col].iloc[0]).all()
    assert res.valid.tolist() == [True, False]
    assert res.changed == [["TransactionAmt"], ["DeviceInfo"]]
    assert cf.describe(res) == (
        "If the amount were $120.00 instead of $950.00, the risk would fall from 90 to 20."
    )
    q = cf.quality(res, scale={"TransactionAmt": 100.0})
    assert q == {"n": 2, "validity": 0.5, "proximity": pytest.approx((8.3 + 1.0) / 2),
                 "n_changed": 1.0, "all_actionable": 1.0}  # fmt: skip


def test_derive_recomputes_derived_features_and_validity_uses_the_model():
    def backend(query, total, vary):
        return query.assign(TransactionAmt=100.0)

    def derive(cands, query):  # amount / usual median (950 / 25 = 38)
        return cands.assign(amt_ratio_median=cands["TransactionAmt"] / 38.0)

    ex = cf.CounterfactualExplainer(_AmountModel(), COLUMNS, backend, derive=derive)
    res = ex.explain(_query())
    assert res.rejected_locked == 0
    assert res.candidates.at[0, "amt_ratio_median"] == pytest.approx(100 / 38)
    assert res.valid.tolist() == [True]


def test_no_valid_counterfactual_gives_none():
    ex = cf.CounterfactualExplainer(_AmountModel(), COLUMNS,
                                    lambda q, t, v: q.assign(DeviceInfo="d1"))  # fmt: skip
    res = ex.explain(_query())
    assert cf.describe(res) is None
    assert cf.quality(cf.CounterfactualResult(_query(), 0.9, _query().iloc[:0], np.array([]),
                                              np.array([], bool), 0))["n"] == 0  # fmt: skip


def test_dice_backend_respects_the_lock():
    dice_ml = pytest.importorskip("dice_ml")
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    n = 400
    train = pd.DataFrame(
        {
            "TransactionAmt": rng.uniform(5, 1000, n),
            "card1": rng.choice([1111.0, 2222.0, 3333.0], n),
            "uid_n_past": rng.integers(0, 50, n).astype(float),
        }
    )
    train["isFraud"] = (train["TransactionAmt"] > 600).astype(int)
    cols = ["TransactionAmt", "card1", "uid_n_past"]
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(
        train[cols], train["isFraud"]
    )
    backend = cf.DiceBackend(model, train, "isFraud", continuous=cols, seed=0)
    ex = cf.CounterfactualExplainer(model, cols, backend)
    query = pd.DataFrame([{"TransactionAmt": 900.0, "card1": 2222.0, "uid_n_past": 10.0}])
    res = ex.explain(query, total=4)
    assert dice_ml is not None and ex.vary == ["TransactionAmt"]
    assert res.rejected_locked == 0 and len(res.candidates) > 0
    assert (res.candidates["card1"] == 2222.0).all() and (
        res.candidates["uid_n_past"] == 10.0
    ).all()
    assert res.valid.mean() > 0.9
