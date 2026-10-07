import numpy as np
import pandas as pd
import pytest

from vaultic.data.uid import build_uids, choose_variant, history_carryover, uid_metrics


def _df():
    # Rows 0-2: one card holder (same card1/addr1, day - D1 = 10 on every row).
    # Row 3: same card1/addr1 but a different first-seen day -> a different person.
    # Row 4: missing addr1. Row 5-6: a second holder, both fraud.
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6, 7],
            "day": [10, 12, 20, 20, 21, 30, 31],
            "D1": [0.0, 2.0, 10.0, 0.0, 5.0, 1.0, 2.0],
            "isFraud": [0, 0, 0, 1, 0, 1, 1],
            "card1": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 2000.0, 2000.0],
            "card2": [111.0] * 7,
            "card3": [150.0] * 7,
            "card5": [226.0] * 7,
            "addr1": [300.0, 300.0, 300.0, 300.0, np.nan, 400.0, 400.0],
            "P_emaildomain": pd.Categorical(["a.com"] * 4 + [None, "b.com", "b.com"]),
        }
    )


def test_uid_groups_same_first_seen_day():
    u = build_uids(_df())
    assert u.loc[0, "uid"] == u.loc[1, "uid"] == u.loc[2, "uid"]
    assert u.loc[3, "uid"] != u.loc[0, "uid"]  # day - D1 = 20, not 10
    assert u.loc[5, "uid"] == u.loc[6, "uid"]  # day - D1 = 29 on both


def test_missing_component_is_a_token_and_flagged():
    u = build_uids(_df())
    assert u["uid_complete"].tolist() == [1, 1, 1, 1, 0, 1, 1]
    assert u.loc[4, "uid"] not in set(u.loc[[0, 1, 2, 3, 5], "uid"])


def test_card1_variant_ignores_time():
    u = build_uids(_df())
    assert u["card1"].nunique() == 2


def test_uid_metrics_on_known_example():
    df = _df()
    u = build_uids(df)
    m = uid_metrics(u["uid"], df["isFraud"], u["uid_complete"])
    # IDs: {rows 0,1,2: legit}, {3: fraud}, {4: legit}, {5,6: fraud} -> 4 IDs
    assert m["ids"] == 4
    assert m["repeat ids that are pure %"] == 100.0
    assert m["rows in repeat ids %"] == pytest.approx(5 / 7 * 100)
    assert m["fraud rate of singleton ids %"] == 50.0  # row 3 fraud, row 4 legit
    assert m["complete key %"] == pytest.approx(6 / 7 * 100)

    m1 = uid_metrics(u["card1"], df["isFraud"], u["card1_complete"])
    # card1 = 1000 mixes 1 fraud with 4 legit; card1 = 2000 is all fraud
    assert m1["repeat ids that are pure %"] == 50.0
    assert m1["rows in mixed ids %"] == pytest.approx(5 / 7 * 100)


def test_history_carryover_uses_no_labels():
    df = _df()
    u = build_uids(df)
    # rows after day 20: rows 4,5,6 (days 21,30,31); none of their uids appear by day 20
    assert history_carryover(df, u, "uid", split_day=20) == 0.0
    assert history_carryover(df, u, "card1", split_day=20) == pytest.approx(100 / 3)


def test_choose_variant_respects_coverage_floor():
    table = pd.DataFrame(
        {
            "repeat ids that are pure %": [80.0, 95.0, 99.0, 70.0],
            "rows in repeat ids %": [90.0, 70.0, 30.0, 99.0],
        },
        index=["uid_card", "uid", "uid2", "card1"],
    )
    assert choose_variant(table) == "uid"  # uid2 is purer but fails the 50% floor
