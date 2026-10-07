"""Phase 5 prep: sequence builder on a 10-row example worked by hand.

uid A: rows 0, 1, 3, 5, 7, 8 at t = 0, 100, 400, 400, 1000, 2000 (rows 3 and 5 tie at t=400)
uid B: rows 2, 4, 6, 9 at t = 300, 400, 900, 2000
Amounts are compared after expm1 of float32 log1p values, so to 3 decimals.
"""

import numpy as np
import pandas as pd
import pytest

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


def _amounts(log_amounts):
    return np.expm1(log_amounts.astype(np.float64)).round(3).tolist()


def test_left_padding_mask_and_order():
    df, uid = _example()
    s = build_sequences(df, uid, n_steps=4, extra_columns=("C1",))
    # find the row of uid A at t=1000 (amount 39): its strictly earlier A rows are
    # t=0 (9), 100 (99), 400 (19), 400 (29)
    r = int(np.flatnonzero((df["TransactionAmt"] == 39.0).to_numpy())[0])
    assert s.mask[r].tolist() == [True, True, True, True]
    amounts = _amounts(s.values[r, :, 0])
    assert amounts[:2] == [9.0, 99.0] and sorted(amounts[2:]) == [19.0, 29.0]
    assert s.feature_names == ["log_amount", "log_gap_next", "product_code", "C1"]


def test_most_recent_steps_kept_when_history_is_long():
    df, uid = _example()
    s = build_sequences(df, uid, n_steps=2)
    r = int(np.flatnonzero((df["TransactionAmt"] == 49.0).to_numpy())[0])  # A at t=2000
    # A's earlier rows: 9, 99, 19, 29, 39 -> the last two are 29/19 (t=400) and 39 (t=1000)
    assert _amounts(s.values[r, :, 0])[1] == 39.0
    assert s.mask[r].tolist() == [True, True]


def test_gap_to_next_step_and_to_the_transaction():
    df, uid = _example()
    s = build_sequences(df, uid, n_steps=3)
    r = int(np.flatnonzero((df["TransactionAmt"] == 2.0).to_numpy())[0])  # B at t=900
    # B's history before 900: t=300 (1.0) and t=400 (4.0) -> left-padded to 3 steps
    assert s.mask[r].tolist() == [False, True, True]
    gaps = np.expm1(s.values[r, :, 1]).round(3).tolist()
    assert gaps == [0.0, 100.0, 500.0]  # 300 -> 400, then 400 -> the transaction at 900
    assert (s.values[r, 0] == 0).all()  # padding is zeros


def test_ties_are_not_history_and_no_history_is_masked():
    df, uid = _example()
    s = build_sequences(df, uid, n_steps=3)
    first_a = int(np.flatnonzero((df["TransactionAmt"] == 9.0).to_numpy())[0])
    assert not s.has_history[first_a] and not s.mask[first_a].any()
    # the two A rows at t=400 do not see each other: each has exactly t=0 and t=100 before it
    for amount in (19.0, 29.0):
        r = int(np.flatnonzero((df["TransactionAmt"] == amount).to_numpy())[0])
        assert s.mask[r].sum() == 2
        assert sorted(_amounts(s.values[r, s.mask[r], 0])) == [9.0, 99.0]


def test_product_codes_missing_values_and_row_subset():
    df, uid = _example()
    s = build_sequences(df, uid, n_steps=3, extra_columns=("C1",), rows=np.array([8, 9]))
    assert s.values.shape == (2, 3, 4) and s.rows.tolist() == [8, 9]
    assert s.values.dtype == np.float32
    full = build_sequences(df, uid, n_steps=3, extra_columns=("C1",))
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
    base = build_sequences(df, uid, n_steps=10, extra_columns=("C1",))
    cut = int(df["TransactionDT"].iloc[n // 2])
    keep = (df["TransactionDT"] <= cut).to_numpy()
    changed = df.copy()
    future = (changed["TransactionDT"] > cut).to_numpy()
    changed.loc[future, "TransactionAmt"] = 1e6
    changed.loc[future, "ProductCD"] = "S"
    changed.loc[future, "C1"] = -7.0
    after = build_sequences(changed, uid, n_steps=10, extra_columns=("C1",))
    assert np.array_equal(base.values[keep], after.values[keep])
    truncated = build_sequences(df[keep], uid[keep], n_steps=10, extra_columns=("C1",))
    assert np.array_equal(base.mask[keep], truncated.mask)
    # product codes are category codes of the whole frame, so compare the other features
    assert np.array_equal(base.values[keep][..., [0, 1, 3]], truncated.values[..., [0, 1, 3]])


def test_rejects_unsorted_rows():
    df, uid = _example()
    with pytest.raises(ValueError, match="sorted"):
        build_sequences(df.iloc[::-1], uid.iloc[::-1])
