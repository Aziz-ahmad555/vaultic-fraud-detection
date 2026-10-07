"""Leakage tests (roadmap Phase 1.4). Must pass before any feature change is merged.

1. Truncation: deleting every row after time t must not change any feature at or before t.
2. Future labels: shuffling labels of rows after t must not change any feature at or before t.
3. Label delay: shuffling labels of rows whose label is not yet known at t (time > t - L)
   must not change any feature at or before t.

Each test runs on synthetic data (always) and on the real 50,000-row sample
(data/interim/merged_sample.parquet, built by `python -m vaultic.data.load --nrows 50000
--out data/interim/merged_sample.parquet`) when it exists; the raw data is not in git, so
CI only has the synthetic case.
"""

import numpy as np
import pandas as pd
import pytest

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.data.uid import build_uids
from vaultic.features.pipeline import INPUT_COLUMNS, build_features, fit_encoders
from vaultic.paths import INTERIM_DIR

SAMPLE_PATH = INTERIM_DIR / "merged_sample.parquet"
QUANTILES = [0.25, 0.5, 0.75]


def _synthetic(n=5_000, seed=0):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.integers(0, 60 * SECONDS_PER_DAY, n))
    time[100:110] = time[100]  # a block of ties
    customers = rng.integers(0, 300, n)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": time,
            "day": time // SECONDS_PER_DAY,
            "TransactionAmt": rng.lognormal(4, 1, n),
            "isFraud": (rng.random(n) < 0.05 + 0.3 * (customers % 17 == 0)).astype(int),
            "card1": (customers % 50).astype(float),
            "addr1": (customers % 7).astype(float),
            "P_emaildomain": pd.Categorical(rng.choice(["a.com", "b.com", None], n)),
        }
    ), pd.Series(customers, name="uid")


def _real_sample():
    df = pd.read_parquet(SAMPLE_PATH, columns=INPUT_COLUMNS)
    uid = build_uids(df)["uid2"]
    return df, uid


DATASETS = {
    "synthetic": _synthetic,
    "real_50k": pytest.param(
        _real_sample,
        marks=pytest.mark.skipif(not SAMPLE_PATH.exists(), reason="real 50k sample not built"),
    ),
}


@pytest.fixture(params=list(DATASETS.values()), ids=list(DATASETS))
def data(request):
    df, uid = request.param()
    # Encoders are fitted once, on the first 60% of the time range, and reused: the tests
    # check the per-row computation, not the (fixed) training-period fit.
    train = df["TransactionDT"] <= df["TransactionDT"].quantile(0.6)
    return df, uid, fit_encoders(df[train], uid[train])


def _cut_times(df):
    return [int(df["TransactionDT"].quantile(q)) for q in QUANTILES]


@pytest.mark.parametrize("delay_days", [1, 30])
def test_truncating_the_future_changes_nothing(data, delay_days):
    df, uid, enc = data
    full = build_features(df, uid, enc, delay_days)
    for t in _cut_times(df):
        keep = (df["TransactionDT"] <= t).to_numpy()
        truncated = build_features(df[keep], uid[keep], enc, delay_days)
        pd.testing.assert_frame_equal(full[keep], truncated, check_exact=True)


@pytest.mark.parametrize("delay_days", [1, 30])
def test_shuffling_future_labels_changes_nothing(data, delay_days):
    df, uid, enc = data
    full = build_features(df, uid, enc, delay_days)
    rng = np.random.default_rng(1)
    for t in _cut_times(df):
        future = (df["TransactionDT"] > t).to_numpy()
        shuffled = df.copy()
        shuffled.loc[future, "isFraud"] = rng.permutation(1 - df.loc[future, "isFraud"].to_numpy())
        rebuilt = build_features(shuffled, uid, enc, delay_days)
        past = ~future
        pd.testing.assert_frame_equal(full[past], rebuilt[past], check_exact=True)


@pytest.mark.parametrize("delay_days", [1, 30])
def test_shuffling_unmatured_labels_changes_nothing(data, delay_days):
    df, uid, enc = data
    full = build_features(df, uid, enc, delay_days)
    rng = np.random.default_rng(2)
    for t in _cut_times(df):
        unknown_at_t = (df["TransactionDT"] > t - delay_days * SECONDS_PER_DAY).to_numpy()
        shuffled = df.copy()
        flipped = 1 - df.loc[unknown_at_t, "isFraud"].to_numpy()
        shuffled.loc[unknown_at_t, "isFraud"] = rng.permutation(flipped)
        rebuilt = build_features(shuffled, uid, enc, delay_days)
        upto_t = (df["TransactionDT"] <= t).to_numpy()
        pd.testing.assert_frame_equal(full[upto_t], rebuilt[upto_t], check_exact=True)


def test_the_label_feature_is_not_trivially_constant(data):
    """Guard against the tests passing only because no label is ever used."""
    df, uid, enc = data
    f = build_features(df, uid, enc, label_delay_days=1)
    assert f["uid_n_labels_known"].max() > 0
