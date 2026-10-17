"""D158 (1): the amount-derived inputs recomputed for one transaction equal the feature pipelines
rebuilt on the data with that transaction's amount changed."""

import numpy as np
import pandas as pd
import pytest

from vaultic.features.behavioral import build_behavioral
from vaultic.features.categories import CategoryEncoder
from vaultic.features.pipeline import build_features, fit_encoders
from vaultic.features.sequences import build_sequences
from vaultic.stage2.amount import FEATURES, AmountHistory, with_amount
from vaultic.views.orchestrate import CONTEXT


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.integers(0, 40 * 86_400, n))
    time[10] = time[9]  # a tie: same second, same customer below
    uid = rng.choice(["a", "b", "c", "d", "e"], n)
    uid[10] = uid[9]
    df = pd.DataFrame({
        "TransactionID": np.arange(n), "TransactionDT": time, "day": time // 86_400,
        "TransactionAmt": np.round(rng.lognormal(3, 1, n), 2), "isFraud": rng.integers(0, 2, n),
        "card1": rng.integers(1, 5, n), "card2": 1.0, "card3": 1.0, "card4": "visa", "card5": 1.0,
        "card6": "debit", "addr1": 1.0, "addr2": 1.0, "D1": 0.0, "P_emaildomain": "x.com",
        "DeviceInfo": rng.choice(["p", "q"], n), "ProductCD": rng.choice(["W", "C"], n),
    })  # fmt: skip
    df.loc[[20, 21, 22], "TransactionAmt"] = 50.0  # equal past amounts for some customers
    return df, pd.Series(uid)


def _pipelines(df, uid):
    enc = fit_encoders(df, uid)
    base = build_features(df, uid, enc, label_delay_days=30)
    beh = build_behavioral(df, uid)
    encoder = CategoryEncoder().fit(df, ("ProductCD",))
    seq = build_sequences(df, uid.astype(str), encoder, n_steps=10)
    cur = np.where(seq.has_history, seq.current[:, 0], np.nan).astype(np.float32)
    return pd.concat([base, beh.drop(columns=["TransactionID"])], axis=1).assign(
        TransactionAmt=df["TransactionAmt"].astype(np.float32), seq_cur_log_amount=cur
    )


@pytest.mark.parametrize("factor", [1.0, 0.95, 1.01, 3.0, 0.0])
def test_recomputed_inputs_equal_rebuilt_pipelines(factor):
    df, uid = _data()
    rows = np.array([0, 1, 9, 10, 23, 60, 119])  # first transaction, ties, equal amounts, late
    history = AmountHistory(df, uid, rows)
    new_amount = df["TransactionAmt"].to_numpy()[rows] * factor
    got = history.features(new_amount)
    for i, r in enumerate(rows):  # one transaction changed at a time
        changed = df.copy()
        changed.loc[r, "TransactionAmt"] = new_amount[i]
        want = _pipelines(changed, uid).iloc[r]
        for c in FEATURES:
            a, b = got.at[i, c], want[c]
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-6, abs=1e-7), (
                r, c, a, b,
            )  # fmt: skip


def test_change_does_not_reach_other_rows_inputs():
    df, uid = _data()
    before = _pipelines(df, uid)
    changed = df.copy()
    changed.loc[30, "TransactionAmt"] *= 4
    after = _pipelines(changed, uid)
    later_same = [r for r in range(31, len(df)) if uid[r] == uid[30]]
    # later transactions of the customer DO change in the pipelines (their history changed);
    # stage 2 perturbs only the explained row, so only row 30's inputs are recomputed
    assert not np.allclose(before.loc[later_same, "uid_amt_mean_past"],
                           after.loc[later_same, "uid_amt_mean_past"])  # fmt: skip
    history = AmountHistory(df, uid, np.array([30, later_same[0]]))
    unchanged = history.features(df["TransactionAmt"].to_numpy()[[30, later_same[0]]])
    assert unchanged.at[1, "amt_z"] == pytest.approx(before.at[later_same[0], "amt_z"], rel=1e-6)


def test_with_amount_keeps_missing_temporal_and_sets_context():
    df, uid = _data()
    feats = _pipelines(df, uid)
    rows = np.array([0, 50])  # row 0 has no history: temporal view missing
    frame = feats.loc[rows].reset_index(drop=True)
    ctx = np.zeros((2, len(CONTEXT)))
    out, new_ctx = with_amount(None, frame, ctx, AmountHistory(df, uid, rows), [10.0, 20.0])
    assert np.isnan(out.at[0, "seq_cur_log_amount"])
    assert out.at[1, "seq_cur_log_amount"] == pytest.approx(np.log1p(20.0))
    assert new_ctx[:, CONTEXT.index("ctx_log_amount")] == pytest.approx(np.log1p([10.0, 20.0]))
    assert list(out["TransactionAmt"]) == [10.0, 20.0]


def test_per_uid_scores_equal_the_original_transform():
    from vaultic.stage2.system import per_uid_scores
    from vaultic.views.anomaly import PerUidIsolationForest

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    X[rng.random((400, 3)) < 0.1] = np.nan
    uid = rng.choice(["a", "b", "c", "d"], 400)
    model = PerUidIsolationForest(min_history=50, n_estimators=20).fit(X, uid)
    n_past = rng.integers(0, 100, 400)
    want = model.transform(X, uid, n_past)
    got = per_uid_scores(model, X, uid, n_past)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)])
