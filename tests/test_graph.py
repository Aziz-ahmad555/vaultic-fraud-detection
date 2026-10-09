"""Phase 4 graph features: a 10-row example worked by hand, settings A/B/C, leakage tests.

Label delay L = 1 day, window 30 days. Card key = card1 (card2-6 constant).

row  day      uid card email       address  device          fraud
r0   1        U1  c1   a.com       10|87    D1|desktop      1
r1   1 + 1h   U2  c1   b.com       20|87    -               0
r2   3        U1  c1   a.com,b.com 10|87    D1|desktop      0
r3   3 (tie)  U3  c2   a.com       -        D1|desktop      0
r4   5        U2  c1   b.com       20|87    D2|mobile       1
r5   6        U4  c3   c.com       30|87    -               0
r6   8        U1  c1   a.com       10|87    D1|desktop      0
r7   8 + 60s  U3  c2   a.com       -        D1|desktop      1
r8   40       U1  c1   a.com       10|87    D1|desktop      0
r9   40 (tie) U4  c3   c.com       30|87    -               0
"""

import numpy as np
import pandas as pd
import pytest

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.features.graph import EventIndex, build_graph_features, fit_hub_thresholds

D = SECONDS_PER_DAY
NAN = np.nan


def _example():
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(10),
            "TransactionDT": [D, D + 3600, 3 * D, 3 * D, 5 * D, 6 * D, 8 * D, 8 * D + 60,
                              40 * D, 40 * D],
            "isFraud": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            "card1": [1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0],
            "card2": 1.0, "card3": 1.0, "card4": "visa", "card5": 1.0, "card6": "debit",
            "P_emaildomain": ["a.com", "b.com", "a.com", "a.com", "b.com", "c.com", "a.com",
                              "a.com", "a.com", "c.com"],
            "R_emaildomain": [None, None, "b.com", None, None, None, None, None, None, None],
            "addr1": [10.0, 20.0, 10.0, NAN, 20.0, 30.0, 10.0, NAN, 10.0, 30.0],
            "addr2": [87.0, 87.0, 87.0, NAN, 87.0, 87.0, 87.0, NAN, 87.0, 87.0],
            "DeviceInfo": ["D1", None, "D1", "D1", "D2", None, "D1", "D1", "D1", None],
            "DeviceType": ["desktop", None, "desktop", "desktop", "mobile", None, "desktop",
                           "desktop", "desktop", None],
        }
    )  # fmt: skip
    uid = pd.Series(["U1", "U2", "U1", "U3", "U2", "U4", "U1", "U3", "U1", "U4"])
    return df, uid


def _features(setting="C"):
    df, uid = _example()
    return build_graph_features(df, uid, setting=setting, label_delay_days=1, window_days=30)


@pytest.fixture(scope="module")
def c():
    return _features("C")


def test_event_index_counts_and_sums_strictly_before():
    idx = EventIndex(["a", "a", "b", "a"], [10, 20, 20, 30], [1.0, 2.0, 5.0, 4.0])
    assert idx.count_before(["a", "a", "b", "z"], [20, 31, 21, 99]).tolist() == [1, 3, 1, 0]
    assert idx.sum_before(["a", "a", "b", "z"], [20, 31, 21, 99]).tolist() == [1.0, 7.0, 5.0, 0.0]


def test_card_degree_and_shared_uids(c):
    # c1 rows r0, r1, r2, r4, r6, r8; c2 r3, r7; c3 r5, r9
    assert c["g_deg_tx_card"].tolist() == [0, 1, 2, 0, 3, 0, 4, 1, 5, 1]
    assert c["g_deg_uid_card"].iloc[[1, 2, 4, 6, 8]].tolist() == [1, 2, 2, 2, 2]
    # other uids on the card: r1 sees U1; r2 (U1) sees U2; r0 sees nobody
    assert c["g_shared_uids_card"].iloc[[0, 1, 2, 4]].tolist() == [0, 1, 1, 1]


def test_card_fraud_rate_respects_label_delay(c):
    # r1 at day 1 + 1h: r0's label (day 1) is not 1 day old yet -> nothing known
    assert np.isnan(c.loc[1, "g_fraud_rate_card"])
    assert c.loc[2, "g_fraud_rate_card"] == pytest.approx(1 / 2)  # r0 (fraud), r1
    assert c.loc[4, "g_fraud_rate_card"] == pytest.approx(1 / 3)  # r0, r1, r2
    assert c.loc[6, "g_fraud_rate_card"] == pytest.approx(2 / 4)  # r0, r1, r2, r4
    assert c.loc[8, "g_fraud_rate_card"] == pytest.approx(2 / 5)


def test_device_features_ties_and_missing(c):
    # r2 and r3 share a second: each sees only r0 on device D1|desktop
    assert c.loc[2, "g_deg_tx_device"] == 1 and c.loc[3, "g_deg_tx_device"] == 1
    assert c.loc[3, "g_shared_uids_device"] == 1  # U1 (r0), not U1's tied r2
    assert (c.loc[6, "g_deg_tx_device"], c.loc[7, "g_deg_tx_device"]) == (3, 4)
    assert c.loc[6, "g_fraud_rate_device"] == pytest.approx(1 / 3)  # r0, r2, r3 known
    assert c.loc[7, "g_fraud_rate_device"] == pytest.approx(1 / 3)  # r6 not known yet
    assert c.loc[8, "g_fraud_rate_device"] == pytest.approx(2 / 5)
    assert c.loc[[1, 5, 9], "g_deg_tx_device"].isna().all()  # no device -> NaN


def test_email_two_edges_and_address(c):
    assert c.loc[2, "g_deg_tx_email"] == 1  # a.com: r0; b.com: r1 -> larger is 1
    assert c.loc[2, "g_fraud_rate_email"] == pytest.approx(1 / 2)  # pooled: r0 (1), r1 (0)
    assert c.loc[4, "g_deg_tx_email"] == 2 and c.loc[4, "g_fraud_rate_email"] == 0.0
    assert c.loc[6, "g_deg_tx_address"] == 2  # 10|87: r0, r2
    assert c.loc[[3, 7], "g_deg_tx_address"].isna().all()


def test_two_hop_fraud_paths(c):
    # r6 at day 8: known frauds via card c1 (r0, r4) + email a.com (r0) + address (r0)
    # + device (r0) = 5 paths
    assert c.loc[6, "g_twohop_fraud"] == 5


def test_components_use_the_30_day_window_before_the_day(c):
    # r2 / r3 (day 3): graph of r0, r1 -> {U1, U2, c1, D1}; r3 reaches it through D1 only
    assert (c.loc[2, "g_comp_tx"], c.loc[2, "g_comp_uids"]) == (2, 2)
    assert (c.loc[3, "g_comp_tx"], c.loc[3, "g_comp_uids"]) == (2, 2)
    assert c.loc[2, "g_comp_fraud_rate"] == pytest.approx(1 / 2)
    # r4 (day 5): r0-r3 in one component; known by day start: all four, one fraud
    assert c.loc[4, "g_comp_tx"] == 4 and c.loc[4, "g_comp_fraud_rate"] == pytest.approx(1 / 4)
    # r6 / r7 (day 8): r0-r4 connected through c1 and D1; r5 (U4, c3) is separate
    assert (c.loc[6, "g_comp_tx"], c.loc[6, "g_comp_uids"]) == (5, 3)
    assert c.loc[6, "g_comp_fraud_rate"] == pytest.approx(2 / 5)
    # r5 (day 6): its uid and card have no history; r8 (day 40): window is empty.
    # No group means missing (NaN), not an empty group (D51)
    structure = ["g_comp_tx", "g_comp_uids", "g_comp_fraud_rate", "g_comm_tx", "g_comm_fraud_rate"]
    assert c.loc[5, structure].isna().all() and c.loc[8, structure].isna().all()
    both = c[["g_comm_tx", "g_comp_tx"]].dropna()
    assert (both["g_comm_tx"] <= both["g_comp_tx"]).all()  # communities split components


def test_communities_match_components_on_separate_cliques():
    df, uid = _example()
    df = df.iloc[[0, 1, 5]].copy()  # U1/U2 share card c1; U4 alone on c3
    df["TransactionDT"] = [D, D + 10, D + 20]
    late = df.iloc[[0]].assign(TransactionID=99, TransactionDT=3 * D)
    df = pd.concat([df, late], ignore_index=True)
    uid = pd.Series(["U1", "U2", "U4", "U1"])
    f = build_graph_features(df, uid, "C", label_delay_days=1, window_days=30)
    assert f.loc[3, "g_comm_tx"] == f.loc[3, "g_comp_tx"] == 2  # {U1, U2, c1, D1}
    assert f.loc[3, "g_comm_fraud_rate"] == pytest.approx(1 / 2)


def test_setting_a_uses_everything_except_self():
    a = _features("A")
    # r6 card c1: the other five c1 rows, frauds r0 and r4
    assert a.loc[6, "g_deg_tx_card"] == 5 and a.loc[6, "g_fraud_rate_card"] == pytest.approx(2 / 5)
    # r8's static component: every row of U1/U2/U3 except r8 itself (7), frauds r0, r4, r7
    assert a.loc[8, "g_comp_tx"] == 7 and a.loc[8, "g_comp_fraud_rate"] == pytest.approx(3 / 7)


def test_setting_b_mixes_all_edges_with_delayed_labels():
    b, c = _features("B"), _features("C")
    assert b.loc[6, "g_deg_tx_card"] == 5  # all edges, like A
    assert b.loc[6, "g_fraud_rate_card"] == c.loc[6, "g_fraud_rate_card"]  # labels like C


def test_unknown_setting_and_unsorted_rows():
    df, uid = _example()
    with pytest.raises(ValueError, match="setting"):
        build_graph_features(df, uid, setting="D")
    with pytest.raises(ValueError, match="sorted"):
        build_graph_features(df.iloc[::-1], uid.iloc[::-1])


# ---- leakage (setting C never uses the future) ----------------------------------------------


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.integers(0, 60 * D, n))
    time[100:105] = time[100]
    users = rng.integers(0, 60, n)
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": time,
            "isFraud": (rng.random(n) < 0.1).astype(int),
            "card1": rng.integers(0, 25, n).astype(float),
            "card2": 1.0, "card3": 1.0, "card4": "visa", "card5": 1.0, "card6": "debit",
            "P_emaildomain": rng.choice(["a.com", "b.com", "c.com", None], n),
            "R_emaildomain": rng.choice(["a.com", None, None], n),
            "addr1": rng.choice([10.0, 20.0, np.nan], n),
            "addr2": 87.0,
            "DeviceInfo": rng.choice(["D1", "D2", "D3", None, None], n),
            "DeviceType": "desktop",
        }
    )  # fmt: skip
    return df, pd.Series(users.astype(str))


@pytest.mark.parametrize("q", [0.3, 0.7])
def test_setting_c_truncating_the_future_changes_nothing(q):
    df, uid = _synthetic()
    full = build_graph_features(df, uid, "C", label_delay_days=3, window_days=30)
    t = int(df["TransactionDT"].quantile(q))
    keep = (df["TransactionDT"] <= t).to_numpy()
    cut = build_graph_features(df[keep], uid[keep], "C", label_delay_days=3, window_days=30)
    pd.testing.assert_frame_equal(full[keep], cut, check_exact=True)


def test_setting_c_ignores_labels_not_yet_known():
    df, uid = _synthetic()
    full = build_graph_features(df, uid, "C", label_delay_days=3, window_days=30)
    t = int(df["TransactionDT"].quantile(0.5))
    unknown = (df["TransactionDT"] > t - 3 * D).to_numpy()
    flipped = df.copy()
    flipped.loc[unknown, "isFraud"] = 1 - flipped.loc[unknown, "isFraud"]
    rebuilt = build_graph_features(flipped, uid, "C", label_delay_days=3, window_days=30)
    upto = (df["TransactionDT"] <= t).to_numpy()
    pd.testing.assert_frame_equal(full[upto], rebuilt[upto], check_exact=True)


def test_setting_a_does_use_the_future():
    """Sanity check that the truncation test can detect leakage at all."""
    df, uid = _synthetic()
    full = build_graph_features(df, uid, "A", label_delay_days=3)
    t = int(df["TransactionDT"].quantile(0.5))
    keep = (df["TransactionDT"] <= t).to_numpy()
    cut = build_graph_features(df[keep], uid[keep], "A", label_delay_days=3)
    assert not full[keep].reset_index(drop=True).equals(cut.reset_index(drop=True))


def test_setting_c_group_fraud_rates_exist_with_a_30_day_delay_and_window():
    """D50: with L = 30 and a 30-day window no window transaction's label is known, so the
    group rates were always NaN. Known labels now come from all earlier transactions on the
    group's nodes; a non-trivial share of rows must get a value."""
    rng = np.random.default_rng(1)
    n = 3000
    time = np.sort(rng.integers(0, 150 * D, n))
    users = rng.integers(0, 80, n)
    cards = users % 40  # pairs of customers share a card: stable groups over time
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": time,
            "isFraud": (rng.random(n) < 0.1).astype(int),
            "card1": cards.astype(float), "card2": 1.0, "card3": 1.0, "card4": "visa",
            "card5": 1.0, "card6": "debit", "P_emaildomain": "a.com", "R_emaildomain": None,
            "addr1": np.nan, "addr2": np.nan, "DeviceInfo": None, "DeviceType": None,
        }
    )  # fmt: skip
    f = build_graph_features(
        df, pd.Series(users.astype(str)), "C", label_delay_days=30, window_days=30
    )
    late = (df["TransactionDT"] >= 70 * D).to_numpy()  # enough history for labels to mature
    for col in ("g_comp_fraud_rate", "g_comm_fraud_rate"):
        share = f.loc[late, col].notna().mean()
        assert share > 0.5, (col, share)
        assert f[col].dropna().between(0, 1).all()
    assert (
        f.loc[df["TransactionDT"] < 30 * D, "g_comp_fraud_rate"].isna().all()
    )  # nothing known yet


@pytest.mark.parametrize("q", [0.4, 0.8])
def test_setting_c_l30_truncation_with_long_label_history(q):
    """Labels from outside the window: still nothing from after t - L, nothing from the future."""
    df, uid = _synthetic(n=800)
    df["TransactionDT"] = np.sort(np.random.default_rng(3).integers(0, 120 * D, len(df)))
    full = build_graph_features(df, uid, "C", label_delay_days=30, window_days=30)
    t = int(df["TransactionDT"].quantile(q))
    keep = (df["TransactionDT"] <= t).to_numpy()
    cut = build_graph_features(df[keep], uid[keep], "C", label_delay_days=30, window_days=30)
    pd.testing.assert_frame_equal(full[keep], cut, check_exact=True)
    flipped = df.copy()
    recent = (df["TransactionDT"] > t - 30 * D).to_numpy()
    flipped.loc[recent, "isFraud"] = 1 - flipped.loc[recent, "isFraud"]
    again = build_graph_features(flipped, uid, "C", label_delay_days=30, window_days=30)
    pd.testing.assert_frame_equal(full[keep], again[keep], check_exact=True)


def test_setting_c_rows_without_a_group_are_missing_not_zero():
    """D51: in setting C a row none of whose entities is in the window graph gets NaN for every
    structure feature; rows in a group never get NaN counts."""
    df, uid = _synthetic(n=600)
    f = build_graph_features(df, uid, "C", label_delay_days=3, window_days=30)
    first_day = (df["TransactionDT"] // D == (df["TransactionDT"] // D).min()).to_numpy()
    assert f.loc[first_day, ["g_comp_tx", "g_comp_uids", "g_comm_tx"]].isna().all().all()
    grouped = f["g_comp_tx"].notna()
    assert grouped.any() and (f.loc[grouped, "g_comp_tx"] >= 1).all()
    assert (f.loc[grouped, "g_comp_uids"] >= 1).all() and f.loc[grouped, "g_comm_tx"].notna().all()
    a = build_graph_features(df, uid, "A", label_delay_days=3)
    assert a["g_comp_tx"].notna().all()  # A and B: every row is in the static graph


def test_nonhub_shared_evidence_ignores_hubs_and_own_history():
    """D52, by hand. Day 1: U1 on card k1 / device H; U2 on k1 / H; U3..U8 on device H only.
    Training thresholds: card 2 uids, device 2 uids (H, with 8 uids, is a hub).
    Day 3: U2 on k1 + H -> k1 shared with 1 other uid (U1), H is a hub -> 1.
           U9 on new card k9 + H -> only the hub -> 0 (no relational evidence).
           U1 again on its own card k5, no device -> 0 (own history is not shared evidence)."""
    rows = (
        [(1, "U1", 1.0, "H")]
        + [(1, "U2", 1.0, "H")]
        + [(1, f"U{i}", 50.0 + i, "H") for i in range(3, 9)]
    )
    rows += [(3, "U2", 1.0, "H"), (3, "U9", 9.0, "H"), (3, "U1", 5.0, None)]
    rows += [(2, "U1", 5.0, None)]
    rows.sort(key=lambda r: r[0])
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(len(rows)),
            "TransactionDT": [d * D + i for i, (d, *_rest) in enumerate(rows)],
            "isFraud": 0,
            "card1": [r[2] for r in rows], "card2": 1.0, "card3": 1.0, "card4": "visa",
            "card5": 1.0, "card6": "debit", "P_emaildomain": None, "R_emaildomain": None,
            "addr1": np.nan, "addr2": np.nan, "DeviceInfo": [r[3] for r in rows],
            "DeviceType": [None if r[3] is None else "mobile" for r in rows],
        }
    )  # fmt: skip
    uid = pd.Series([r[1] for r in rows])
    train = (df["TransactionDT"] < 2 * D).to_numpy()
    th = fit_hub_thresholds(df[train], uid[train], quantile=0.5)
    assert th == {"card": 1.0, "device": 8.0}  # median card has 1 uid; the only device has 8
    th = {"card": 2.0, "device": 2.0}  # the threshold the docstring walks through
    f = build_graph_features(df, uid, "C", label_delay_days=30, hub_thresholds=th)
    day3 = (df["TransactionDT"] // D == 3).to_numpy()
    got = dict(zip(uid[day3], f.loc[day3, "g_shared_nonhub"], strict=True))
    assert got == {"U2": 1.0, "U9": 0.0, "U1": 0.0}


def test_availability_share_per_split():
    """D52 availability: g_shared_nonhub > 0; NaN (no card or device) counts as unavailable."""
    from vaultic.features.graph import availability

    feats = pd.DataFrame({"g_shared_nonhub": [0.0, 2.0, np.nan, 1.0, 0.0, 3.0]})
    part = pd.Series(["train", "train", "train", "validation", "validation", "test"])
    share = availability(feats, part)
    assert share == {"train": 1 / 3, "validation": 0.5, "test": 1.0, "all": 0.5}
