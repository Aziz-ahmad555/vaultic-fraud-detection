"""Phase 5 prep: sequence builder on a 10-row example worked by hand.

uid A: rows 0, 1, 3, 5, 7, 8 at t = 0, 100, 400, 400, 1000, 2000 (rows 3 and 5 tie at t=400)
uid B: rows 2, 4, 6, 9 at t = 300, 400, 900, 2000
Amounts are compared after expm1 of float32 log1p values, so to 3 decimals.
"""

import numpy as np
import pandas as pd
import pytest

from vaultic.features.categories import CategoryEncoder
from vaultic.features.sequences import build_sequences


def _example():
    df = pd.DataFrame(
        {
            "TransactionDT": [0, 100, 300, 400, 400, 400, 900, 1000, 2000, 2000],
            "TransactionAmt": [9.0, 99.0, 1.0, 19.0, 4.0, 29.0, 2.0, 39.0, 49.0, 3.0],
            "ProductCD": ["W", "W", "C", "H", "C", None, "C", "W", "W", "C"],
            "C1": [1.0, 2.0, 1.0, 3.0, 1.0, np.nan, 1.0, 4.0, 5.0, 1.0],
        }
    )
    uid = pd.Series(["A", "A", "B", "A", "B", "A", "B", "A", "A", "B"])
    return df, uid


# fitted on the first 6 rows ("training period"): C, H, W seen -> codes 2, 3, 4
ENC = CategoryEncoder().fit(_example()[0].iloc[:6], columns=("ProductCD",))


def _amounts(log_amounts):
    return np.expm1(log_amounts.astype(np.float64)).round(3).tolist()


def test_left_padding_mask_and_order():
    df, uid = _example()
    s = build_sequences(df, uid, ENC, n_steps=4, extra_columns=("C1",))
    # find the row of uid A at t=1000 (amount 39): its strictly earlier A rows are
    # t=0 (9), 100 (99), 400 (19), 400 (29)
    r = int(np.flatnonzero((df["TransactionAmt"] == 39.0).to_numpy())[0])
    assert s.mask[r].tolist() == [True, True, True, True]
    amounts = _amounts(s.values[r, :, 0])
    assert amounts[:2] == [9.0, 99.0] and sorted(amounts[2:]) == [19.0, 29.0]
    assert s.feature_names == ["log_amount", "log_gap_next", "product_code", "C1"]


def test_most_recent_steps_kept_when_history_is_long():
    df, uid = _example()
    s = build_sequences(df, uid, ENC, n_steps=2)
    r = int(np.flatnonzero((df["TransactionAmt"] == 49.0).to_numpy())[0])  # A at t=2000
    # A's earlier rows: 9, 99, 19, 29, 39 -> the last two are 29/19 (t=400) and 39 (t=1000)
    assert _amounts(s.values[r, :, 0])[1] == 39.0
    assert s.mask[r].tolist() == [True, True]


def test_gap_to_next_step_and_to_the_transaction():
    df, uid = _example()
    s = build_sequences(df, uid, ENC, n_steps=3)
    r = int(np.flatnonzero((df["TransactionAmt"] == 2.0).to_numpy())[0])  # B at t=900
    # B's history before 900: t=300 (1.0) and t=400 (4.0) -> left-padded to 3 steps
    assert s.mask[r].tolist() == [False, True, True]
    gaps = np.expm1(s.values[r, :, 1]).round(3).tolist()
    assert gaps == [0.0, 100.0, 500.0]  # 300 -> 400, then 400 -> the transaction at 900
    assert (s.values[r, 0] == 0).all()  # padding is zeros


def test_ties_are_not_history_and_no_history_is_masked():
    df, uid = _example()
    s = build_sequences(df, uid, ENC, n_steps=3)
    first_a = int(np.flatnonzero((df["TransactionAmt"] == 9.0).to_numpy())[0])
    assert not s.has_history[first_a] and not s.mask[first_a].any()
    # the two A rows at t=400 do not see each other: each has exactly t=0 and t=100 before it
    for amount in (19.0, 29.0):
        r = int(np.flatnonzero((df["TransactionAmt"] == amount).to_numpy())[0])
        assert s.mask[r].sum() == 2
        assert sorted(_amounts(s.values[r, s.mask[r], 0])) == [9.0, 99.0]


def test_product_codes_missing_values_and_row_subset():
    df, uid = _example()
    s = build_sequences(df, uid, ENC, n_steps=3, extra_columns=("C1",), rows=np.array([8, 9]))
    assert s.values.shape == (2, 3, 4) and s.rows.tolist() == [8, 9]
    assert s.values.dtype == np.float32
    full = build_sequences(df, uid, ENC, n_steps=3, extra_columns=("C1",))
    assert np.array_equal(full.values[[8, 9]], s.values)
    # the missing ProductCD and the missing C1 (both on the 29.0 row) become 0
    r = int(np.flatnonzero((df["TransactionAmt"] == 39.0).to_numpy())[0])
    steps = full.values[r][full.mask[r]]
    row_29 = steps[np.isclose(np.expm1(steps[:, 0]), 29.0)][0]
    assert row_29[2] == 0.0 and row_29[3] == 0.0


def test_future_rows_never_change_a_sequence():
    """Leakage: rewriting or deleting everything at or after t leaves every earlier sequence
    (and every sequence at t, since ties are not history) unchanged."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "TransactionDT": np.sort(rng.integers(0, 5000, n)),
            "TransactionAmt": rng.gamma(2.0, 50.0, n),
            "ProductCD": rng.choice(["W", "C", "H", "R", "S"], n),
            "C1": rng.poisson(3, n).astype(float),
        }
    )
    uid = pd.Series(rng.choice([f"u{i}" for i in range(15)], n))
    enc = CategoryEncoder().fit(df.iloc[: n // 4], columns=("ProductCD",))
    base = build_sequences(df, uid, enc, n_steps=10, extra_columns=("C1",))
    cut = int(df["TransactionDT"].iloc[n // 2])
    keep = (df["TransactionDT"] <= cut).to_numpy()
    changed = df.copy()
    future = (changed["TransactionDT"] > cut).to_numpy()
    changed.loc[future, "TransactionAmt"] = 1e6
    changed.loc[future, "ProductCD"] = "S"
    changed.loc[future, "C1"] = -7.0
    after = build_sequences(changed, uid, enc, n_steps=10, extra_columns=("C1",))
    assert np.array_equal(base.values[keep], after.values[keep])
    truncated = build_sequences(df[keep], uid[keep], enc, n_steps=10, extra_columns=("C1",))
    assert np.array_equal(base.mask[keep], truncated.mask)
    assert np.array_equal(base.values[keep], truncated.values)  # one shared encoder


def test_rejects_unsorted_rows():
    df, uid = _example()
    with pytest.raises(ValueError, match="sorted"):
        build_sequences(df.iloc[::-1], uid.iloc[::-1], ENC)


def test_product_codes_come_from_the_training_encoder():
    df, uid = _example()
    s = build_sequences(df, uid, ENC, n_steps=4)
    r = int(np.flatnonzero((df["TransactionAmt"] == 39.0).to_numpy())[0])  # A at t=1000
    codes = dict(zip(_amounts(s.values[r, :, 0]), s.values[r, :, 2].tolist(), strict=True))
    assert codes == {9.0: 4.0, 99.0: 4.0, 19.0: 3.0, 29.0: 0.0}  # W, W, H, missing
    # a category never seen in training is 1 (unknown), not a new code
    later = df.copy()
    later.loc[0, "ProductCD"] = "R"
    s2 = build_sequences(later, uid, ENC, n_steps=4)
    assert s2.values[r, 0, 2] == 1.0
    # the same row gets the same code whether the frame is the full data or a slice
    part = build_sequences(df.iloc[:8].reset_index(drop=True), uid.iloc[:8].reset_index(drop=True),
                           ENC, n_steps=4)  # fmt: skip
    assert np.array_equal(part.values[r], s.values[r])


def test_current_step_is_the_scored_transaction_itself():
    df, uid = _example()
    s = build_sequences(df, uid, ENC, n_steps=3, extra_columns=("C1",))
    assert s.current.shape == (10, 4) and s.current.dtype == np.float32
    assert _amounts(s.current[:, 0]) == df["TransactionAmt"].tolist()
    assert (s.current[:, 1] == 0).all()  # no gap for the transaction being scored
    assert s.current[:, 2].tolist() == ENC.encode(df["ProductCD"], "ProductCD").tolist()
    assert s.current[:, 3].tolist() == df["C1"].fillna(0).tolist()


def test_temporal_dev_compare_groups_and_unfitted_fusion():
    """D70: GRU and rank-mean fusion vs B5 per history group, only on rows with history."""
    from vaultic.views.temporal_dev import compare

    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) < 0.1).astype(int)
    b5 = [y + rng.normal(0, 1.0, n) for _ in range(2)]
    gru = y + rng.normal(0, 0.5, n)
    gru[:200] = np.nan  # no history: masked
    n_past = rng.integers(1, 40, n).astype(float)
    n_past[:200] = 0
    t = compare(y, b5, gru, n_past).set_index("group")
    assert t.loc["all with history", "rows"] == n - 200
    assert sum(t.loc[g, "rows"] for g in ("1-4 past", "5-19 past", "20+ past")) == n - 200
    assert t.loc["all with history", "GRU"] > t.loc["all with history", "B5"]
    diff, lo, hi, p = t.loc["all with history", "GRU-B5"]
    assert lo > 0 and diff > 0
