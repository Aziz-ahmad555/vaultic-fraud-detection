"""Unit tests on small hand-made examples with known answers."""

import numpy as np
import pandas as pd
import pytest

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.features.pipeline import build_features, fit_encoders
from vaultic.features.pit import FrequencyEncoder, PastIndex, label_cutoff

D = SECONDS_PER_DAY


def test_count_and_sum_before_exclude_ties_and_other_keys():
    key = np.array(["a", "a", "b", "a", "a", "b"])
    time = np.array([10, 20, 20, 20, 30, 40])
    value = np.array([1.0, 2.0, 100.0, 4.0, np.nan, 200.0])
    p = PastIndex(key, time)
    # rows 1 and 3 share time 20 with key a: neither sees the other
    assert p.count_before(time).tolist() == [0, 1, 0, 1, 3, 1]
    assert p.sum_before(value, time).tolist() == [0.0, 1.0, 0.0, 1.0, 7.0, 100.0]


def test_last_time_before():
    key = np.array(["a", "a", "a", "b"])
    time = np.array([5, 9, 9, 9])
    p = PastIndex(key, time)
    last = p.last_time_before(time)
    assert np.isnan(last[0]) and last[1] == 5 and last[2] == 5 and np.isnan(last[3])


def test_max_last_k_and_previous_value():
    key = np.array(["a", "a", "a", "a", "b", "a"])
    time = np.array([1, 2, 3, 3, 3, 9])
    value = np.array([5.0, 1.0, 7.0, 2.0, 100.0, 4.0])
    dev = np.array(["x", "y", "y", "z", "q", "x"])
    p = PastIndex(key, time)
    assert np.isnan(p.max_before(value, time)[0])
    # rows 2 and 3 tie at t=3: both see only rows 0-1 (max 5); row 5 sees 5, 1, 7, 2 -> 7
    assert p.max_before(value, time)[[2, 3, 5]].tolist() == [5.0, 5.0, 7.0]
    s, c = p.last_k_sum_count(value, time, k=2)
    # row 5: last two earlier rows of key a are rows 2 and 3 (7 + 2)
    assert (s[5], c[5]) == (9.0, 2) and (s[2], c[2]) == (6.0, 2) and (s[0], c[0]) == (0.0, 0)
    prev = p.value_before(dev, time, fill="")
    assert prev.tolist() == ["", "x", "y", "y", "", "z"]


def test_label_cutoff_respects_delay_and_strict_past():
    t = np.array([40 * D])
    # label from day 10 is known on day 40 with L=30 (10 + 30 <= 40); from day 10 + 1s it is not
    assert label_cutoff(t, 30)[0] == 10 * D + 1
    assert label_cutoff(t, 0)[0] == 40 * D  # L=0: strictly earlier rows only


def test_frequency_encoder_unseen_and_missing():
    enc = FrequencyEncoder().fit(pd.Series(["x", "x", None, "y"]))
    out = enc.transform(pd.Series(["x", "z", None]))
    assert out.tolist() == [2.0, 0.0, 1.0]


def _frame():
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "TransactionDT": [0, 3_600, 2 * D, 40 * D, 40 * D + 60],
            "day": [0, 0, 2, 40, 40],
            "TransactionAmt": [10.0, 30.0, 20.0, 100.0, 5.0],
            "isFraud": [1, 0, 0, 0, 1],
            "card1": [1.0, 1.0, 1.0, 1.0, 2.0],
            "addr1": [5.0, 5.0, 5.0, 5.0, 6.0],
            "P_emaildomain": ["a.com"] * 5,
        }
    )


def test_build_features_known_values():
    df = _frame()
    uid = pd.Series(["u", "u", "u", "u", "v"])
    enc = fit_encoders(df.iloc[:3], uid.iloc[:3])
    f = build_features(df, uid, enc, label_delay_days=30)

    assert f["uid_n_past"].tolist() == [0, 1, 2, 3, 0]
    assert f["uid_amt_mean_past"].iloc[3] == pytest.approx(20.0)
    assert f["uid_amt_vs_mean"].iloc[3] == pytest.approx(5.0)
    assert f["uid_secs_since_prev"].iloc[1] == 3_600
    # 24h window at row 2 (day 2) excludes rows 0-1 (more than 24h earlier)
    assert f["uid_n_24h"].tolist() == [0, 1, 0, 0, 0]
    assert f["uid_n_7d"].iloc[2] == 2 and f["uid_amt_sum_7d"].iloc[2] == 40.0
    # On day 40 all three earlier labels of u are 30+ days old: one fraud of three
    assert f["uid_n_labels_known"].tolist() == [0, 0, 0, 3, 0]
    assert f["uid_fraud_rate_known"].iloc[3] == pytest.approx(1 / 3)
    # encoders fitted on rows 0-2 only: uid v and card1=2 are unseen
    assert f["freq_uid"].tolist() == [3, 3, 3, 3, 0]
    assert f["freq_card1"].tolist() == [3, 3, 3, 3, 0]


def test_build_features_rejects_unsorted_rows():
    df = _frame().iloc[[1, 0, 2, 3, 4]]
    uid = pd.Series(["u"] * 5, index=df.index)
    with pytest.raises(ValueError, match="sorted"):
        build_features(df, uid, fit_encoders(df, uid), label_delay_days=30)
