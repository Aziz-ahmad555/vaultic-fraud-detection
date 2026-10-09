"""Phase 3 behavioral features: a hand-computed 10-row example and leakage tests.

uid A: rows 0,1,3,4,6,7,8 at t = 0, 3600, 7200, 10800, 90000, 93600, 691200 s,
       amounts 10, 20, 30, 30, 800, 15, 40.
uid B: rows 2,5,9 at t = 7200, 86400, 691200 s, amounts 100, 50, 60.
Rows 2/3 and 8/9 share a second (ties must not see each other).
Every expected value below was worked out by hand (comments show the arithmetic).
"""

import numpy as np
import pandas as pd
import pytest

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.features.behavioral import build_behavioral

D = SECONDS_PER_DAY


def _example():
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(10),
            "TransactionDT": [0, 3600, 7200, 7200, 10800, 86400, 90000, 93600, 8 * D, 8 * D],
            "TransactionAmt": [10.0, 20.0, 100.0, 30.0, 30.0, 50.0, 800.0, 15.0, 40.0, 60.0],
            "isFraud": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "P_emaildomain": ["a.com", "a.com", "b.com", "b.com", "a.com", "b.com", "a.com",
                              "c.com", "a.com", "b.com"],
            "DeviceInfo": ["D1", "D1", None, "D2", "D1", None, "D1", "D1", "D1", None],
            "addr2": [87.0, 87.0, 87.0, 87.0, np.nan, 87.0, 87.0, 87.0, 87.0, 87.0],
            "ProductCD": ["W", "W", "C", "W", "W", "C", "H", "W", "W", "C"],
            "card1": [1000.0] * 5 + [2000.0] + [1000.0] * 4,
            "card2": 111.0, "card3": 150.0, "card4": "visa", "card5": 226.0, "card6": "debit",
        }
    )  # fmt: skip
    uid = pd.Series(["A", "A", "B", "A", "A", "B", "A", "A", "A", "B"])
    return df, uid


@pytest.fixture(scope="module")
def f():
    df, uid = _example()
    return build_behavioral(df, uid)


def test_history_strength(f):
    assert f["hist_n_past"].tolist() == [0, 1, 0, 2, 3, 1, 4, 5, 6, 2]
    assert f.loc[0, "hist_days_since_first"] == 0  # cold start
    assert f.loc[8, "hist_days_since_first"] == pytest.approx(8.0)  # (691200 - 0) / 86400
    assert f.loc[9, "hist_days_since_first"] == pytest.approx((8 * D - 7200) / D)


def test_amount_deviation(f):
    # row 6: past amounts 10, 20, 30, 30 -> mean 22.5, sample std sqrt(275/3)
    assert f.loc[6, "amt_z"] == pytest.approx((800 - 22.5) / np.sqrt(275 / 3), rel=1e-5)
    # median 25, MAD median(15, 5, 5, 5) = 5
    assert f.loc[6, "amt_robust_z"] == pytest.approx(775 / (1.4826 * 5 + 0.01), rel=1e-5)
    assert f.loc[6, "amt_ratio_median"] == pytest.approx(32.0)
    assert f.loc[6, "amt_percentile"] == pytest.approx(1.0)
    # row 4: past 10, 20, 30 -> median 20; 30 is above two and equal to one -> 2.5 / 3
    assert f.loc[4, "amt_ratio_median"] == pytest.approx(1.5)
    assert f.loc[4, "amt_percentile"] == pytest.approx(2.5 / 3)
    # row 3: past 10, 20 (row 2 is another uid) -> mean 15, std sqrt(50); median 15, MAD 5
    assert f.loc[3, "amt_z"] == pytest.approx(15 / np.sqrt(50), rel=1e-5)
    assert f.loc[3, "amt_robust_z"] == pytest.approx(15 / (1.4826 * 5 + 0.01), rel=1e-5)
    # row 1: one past amount -> z needs two; ratio and percentile need one
    assert np.isnan(f.loc[1, "amt_z"]) and np.isnan(f.loc[1, "amt_robust_z"])
    assert f.loc[1, "amt_ratio_median"] == pytest.approx(2.0)
    assert f.loc[0, ["amt_z", "amt_ratio_median", "amt_percentile"]].isna().all()


def test_velocity_windows(f):
    # row 4 (t=10800): 1h window [7200, 10800) holds row 3 only; 24h holds rows 0, 1, 3
    assert (f.loc[4, "vel_n_1h"], f.loc[4, "vel_amt_1h"]) == (1, 30)
    assert (f.loc[4, "vel_n_24h"], f.loc[4, "vel_amt_24h"]) == (3, 60)
    # row 7 (t=93600): 1h = row 6; 24h [7200, 93600) = rows 3, 4, 6; 7d = all five
    assert (f.loc[7, "vel_n_1h"], f.loc[7, "vel_amt_1h"]) == (1, 800)
    assert (f.loc[7, "vel_n_24h"], f.loc[7, "vel_amt_24h"]) == (3, 860)
    assert (f.loc[7, "vel_n_7d"], f.loc[7, "vel_amt_7d"]) == (5, 890)
    # row 8 (t=8 days): 7d [86400, 691200) = rows 6, 7; 30d = all six (10+20+30+30+800+15)
    assert (f.loc[8, "vel_n_7d"], f.loc[8, "vel_amt_7d"]) == (2, 815)
    assert (f.loc[8, "vel_n_30d"], f.loc[8, "vel_amt_30d"]) == (6, 905)


def test_recency(f):
    assert f["secs_since_prev"].iloc[[1, 3, 4, 6, 7, 8]].tolist() == [
        3600, 3600, 3600, 79200, 3600, 597600,
    ]  # fmt: skip
    assert np.isnan(f.loc[0, "secs_since_prev"])
    assert f.loc[6, "mean_gap"] == pytest.approx(3600)  # past span 10800 over 3 gaps
    assert f.loc[6, "gap_ratio"] == pytest.approx(22.0)  # 79200 / 3600
    assert f.loc[7, "gap_ratio"] == pytest.approx(3600 / 22500)  # past span 90000 / 4 gaps
    assert np.isnan(f.loc[1, "mean_gap"])  # one past transaction: no gap yet
    assert f.loc[9, "gap_ratio"] == pytest.approx(604800 / 79200)


def test_escalation_slope(f):
    # row 6: 10, 20, 30, 30, 800 at x = -2..2 -> (-20 - 20 + 0 + 30 + 1600) / 10
    assert f.loc[6, "amt_slope_5"] == pytest.approx(159.0)
    # row 7: last four past (20, 30, 30, 800) + 15 -> (-40 - 30 + 0 + 800 + 30) / 10
    assert f.loc[7, "amt_slope_5"] == pytest.approx(76.0)
    assert f.loc[1, "amt_slope_5"] == pytest.approx(10.0)  # two points
    assert np.isnan(f.loc[0, "amt_slope_5"])  # one point


def test_novelty(f):
    cols = ["new_email", "new_device", "new_addr2", "new_product", "n_new_attributes"]
    assert f.loc[0, cols].tolist() == [1, 1, 1, 1, 4]
    assert f.loc[1, cols].tolist() == [0, 0, 0, 0, 0]
    # row 3: b.com and D2 are new for uid A (uid B's b.com does not count)
    assert f.loc[3, cols].tolist() == [1, 1, 0, 0, 2]
    # row 4: addr2 missing -> NaN flag, not counted
    assert np.isnan(f.loc[4, "new_addr2"]) and f.loc[4, "n_new_attributes"] == 0
    assert f.loc[6, "new_product"] == 1 and f.loc[7, "new_email"] == 1
    assert np.isnan(f.loc[2, "new_device"]) and f.loc[2, "n_new_attributes"] == 3


def test_entity_velocity(f):
    # card 1000: row 3 sees rows 0, 1 (uid A); row 2 shares its second and is excluded
    assert f.loc[3, "ent_uids_7d_card"] == 1
    assert f.loc[4, "ent_uids_7d_card"] == 2  # rows 0-3: uids A and B
    assert f.loc[0, "ent_uids_7d_card"] == 0  # entity present, nothing before
    # row 8 (t=8 days): window [86400, 691200) on card 1000 = rows 6, 7 (A); row 5 is card 2000
    assert f.loc[8, "ent_uids_7d_card"] == 1 and f.loc[9, "ent_uids_7d_card"] == 1
    assert f.loc[5, "ent_uids_7d_email"] == 2  # b.com used by B (row 2) and A (row 3)
    assert np.isnan(f.loc[5, "ent_uids_7d_device"])  # device missing


def test_time_pattern(f):
    assert f["hour"].iloc[[0, 1, 3, 4, 6, 7, 8]].tolist() == [0, 1, 2, 3, 1, 2, 0]
    assert f.loc[4, "hour_deviation"] == pytest.approx(2.0, abs=1e-5)  # usual hour 1
    assert f.loc[6, "hour_deviation"] == pytest.approx(0.5, abs=1e-5)  # usual hour 1.5
    assert f.loc[8, "hour_deviation"] == pytest.approx(1.5, abs=1e-5)  # past 0,1,2,3,1,2
    assert np.isnan(f.loc[0, "hour_deviation"])


def test_labels_are_not_used():
    df, uid = _example()
    flipped = df.assign(isFraud=1 - df["isFraud"])
    pd.testing.assert_frame_equal(build_behavioral(df, uid), build_behavioral(flipped, uid))


def test_rejects_unsorted_rows():
    df, uid = _example()
    with pytest.raises(ValueError, match="sorted"):
        build_behavioral(df.iloc[::-1], uid.iloc[::-1])


# ---------------------------------------------------------------------------------------
# Leakage tests on synthetic data (CLAUDE.md rule 12)
# ---------------------------------------------------------------------------------------


def _synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.integers(0, 40 * D, n))
    time[200:210] = time[200]  # ties
    users = rng.integers(0, 150, n)
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": time,
            "TransactionAmt": rng.lognormal(4, 1, n).round(2),
            "isFraud": (rng.random(n) < 0.05).astype(int),
            "P_emaildomain": rng.choice(["a.com", "b.com", "c.com", None], n),
            "DeviceInfo": rng.choice(["D1", "D2", None], n),
            "addr2": rng.choice([87.0, 60.0, np.nan], n),
            "ProductCD": rng.choice(["W", "C", "H"], n),
            "card1": rng.integers(1000, 1030, n).astype(float),
        }
    )
    return df, pd.Series(users)


@pytest.mark.parametrize("q", [0.3, 0.6, 0.9])
def test_truncating_the_future_changes_nothing(q):
    df, uid = _synthetic()
    full = build_behavioral(df, uid)
    t = int(df["TransactionDT"].quantile(q))
    keep = (df["TransactionDT"] <= t).to_numpy()
    truncated = build_behavioral(df[keep], uid[keep])
    pd.testing.assert_frame_equal(full[keep], truncated, check_exact=True)


def test_changing_future_rows_changes_nothing_before_them():
    df, uid = _synthetic()
    full = build_behavioral(df, uid)
    t = int(df["TransactionDT"].quantile(0.5))
    future = (df["TransactionDT"] > t).to_numpy()
    rng = np.random.default_rng(3)
    altered = df.copy()
    altered.loc[future, "TransactionAmt"] = rng.lognormal(6, 1, future.sum())
    altered.loc[future, "P_emaildomain"] = "new.com"
    new_uid = uid.copy()
    new_uid[future] = rng.integers(0, 150, future.sum())
    rebuilt = build_behavioral(altered, new_uid)
    pd.testing.assert_frame_equal(full[~future], rebuilt[~future], check_exact=True)


def test_constant_history_gives_finite_values():
    """D66: equal past amounts (std 0) and past rows in one second (mean gap 0) stay finite."""
    D = 86_400
    df = pd.DataFrame(
        {
            "TransactionID": range(4),
            "TransactionDT": [0, 0, 0, D],
            "TransactionAmt": [10.0, 10.0, 10.0, 50.0],
            "P_emaildomain": ["a"] * 4,
            "DeviceInfo": ["d"] * 4,
            "addr2": [1.0] * 4,
            "ProductCD": ["W"] * 4,
            **{f"card{i}": [1.0] * 4 for i in range(1, 7)},
        }
    )
    f = build_behavioral(df, pd.Series(["u"] * 4))
    assert np.isfinite(
        f.drop(columns="TransactionID").to_numpy(dtype=float)[
            ~f.drop(columns="TransactionID").isna().to_numpy()
        ]
    ).all()
    assert f.loc[3, "amt_z"] == pytest.approx(40 / 0.01)
    assert f.loc[3, "gap_ratio"] == pytest.approx(D / 1.0)
