import pandas as pd
import pytest

from vaultic.data.load import build_merged, merge_and_order


def _transactions():
    # Deliberately out of time order, with a tie on TransactionDT.
    return pd.DataFrame(
        {
            "TransactionID": [3, 1, 2, 4, 5],
            "isFraud": [0, 0, 1, 0, 1],
            "TransactionDT": [86_400 * 2 + 5, 86_400, 86_400 + 10, 86_400 * 2 + 5, 86_400 * 3],
            "TransactionAmt": [10.01, 20.5, 30.25, 40.0, 50.99],
            "ProductCD": ["W", "C", "W", "H", "W"],
            "V1": [1.0, None, 0.5, 1.0, 1.0],
        }
    )


def _identity():
    return pd.DataFrame(
        {
            "TransactionID": [2, 5],
            "id_01": [-5.0, 0.0],
            "DeviceType": ["mobile", "desktop"],
        }
    )


def test_merge_orders_by_time_with_id_tiebreak():
    df = merge_and_order(_transactions(), _identity())
    assert df["TransactionID"].tolist() == [1, 2, 3, 4, 5]
    assert df["TransactionDT"].is_monotonic_increasing


def test_merge_keeps_every_transaction_once_and_flags_identity():
    df = merge_and_order(_transactions(), _identity())
    assert len(df) == 5
    assert dict(zip(df["TransactionID"], df["has_identity"], strict=True)) == {
        1: 0,
        2: 1,
        3: 0,
        4: 0,
        5: 1,
    }
    assert df.loc[df["TransactionID"] == 2, "DeviceType"].item() == "mobile"
    assert df.loc[df["TransactionID"] == 1, "DeviceType"].isna().item()


def test_day_is_floor_of_seconds():
    df = merge_and_order(_transactions(), _identity())
    assert dict(zip(df["TransactionID"], df["day"], strict=True)) == {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}


def test_duplicate_ids_are_rejected():
    tx = _transactions()
    tx.loc[0, "TransactionID"] = 1
    with pytest.raises(ValueError, match="transaction file"):
        merge_and_order(tx, _identity())
    with pytest.raises(ValueError, match="identity file"):
        merge_and_order(_transactions(), pd.concat([_identity(), _identity()]))


def test_build_merged_dtypes_and_parquet_roundtrip(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _transactions().to_csv(raw / "train_transaction.csv", index=False)
    _identity().to_csv(raw / "train_identity.csv", index=False)
    out = tmp_path / "merged.parquet"

    df = build_merged(raw_dir=raw, out_path=out)
    assert df["TransactionAmt"].dtype == "float64"
    assert df["TransactionAmt"].tolist() == [20.5, 30.25, 10.01, 40.0, 50.99]  # cents exact
    assert df["TransactionID"].dtype == "int32"
    assert df["V1"].dtype == "float32"
    assert df["ProductCD"].dtype == "category"

    back = pd.read_parquet(out)
    pd.testing.assert_frame_equal(back, df)


def test_build_merged_nrows_reads_a_prefix(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _transactions().to_csv(raw / "train_transaction.csv", index=False)
    _identity().to_csv(raw / "train_identity.csv", index=False)
    df = build_merged(raw_dir=raw, out_path=tmp_path / "m.parquet", nrows=2)
    assert sorted(df["TransactionID"]) == [1, 3]
