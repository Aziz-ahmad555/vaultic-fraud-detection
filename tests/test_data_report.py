import pandas as pd

from vaultic.data.report import block_table, build_report, missing_by_group


def _df():
    # 4 rows in block 0 (days 1-30), 2 rows in block 1 (days 31-60)
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [86_400 * d for d in [1, 2, 10, 30, 31, 45]],
            "day": [1, 2, 10, 30, 31, 45],
            "isFraud": [0, 1, 0, 0, 1, 1],
            "has_identity": [1, 0, 0, 0, 1, 0],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "ProductCD": pd.Categorical(["W", "W", "C", "W", "C", "C"]),
            "C1": [1.0, None, 1.0, 1.0, 1.0, 1.0],
            "C2": [None, None, 1.0, 1.0, 1.0, 1.0],
            "V1": [1.0] * 6,
        }
    )


def test_block_table_counts_and_rates():
    t = block_table(_df())
    assert t["rows"].tolist() == [4, 2]
    assert t["fraud rate %"].tolist() == [25.0, 100.0]
    assert t["has_identity %"].tolist() == [25.0, 50.0]
    assert t["first day"].tolist() == [1, 31]


def test_missing_by_group():
    m = missing_by_group(_df())
    c = m.loc["C1-C14 (counts)"]
    assert c["columns"] == 2
    # C1 missing 1/6, C2 missing 2/6 -> mean 25%
    assert abs(c["mean missing %"] - 25.0) < 1e-9
    assert m.loc["V1-V339 (Vesta features)", "mean missing %"] == 0


def test_report_contains_overall_numbers():
    text = build_report(_df())
    assert "| Transactions | 6 |" in text
    assert "| Fraud | 3 (50.000%) |" in text
    assert "1 – 45" in text
    assert "| 0 | 1 | 30 | 4 | 4 | 25.00 | 25.00 |" in text  # integer columns print as integers
