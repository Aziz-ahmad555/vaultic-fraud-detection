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

import functools

import numpy as np
import pandas as pd
import pytest

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.data.splits import load_splits
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


# the uid variant chosen in D18 (splits.yaml uid_variant), as every pipeline uses (review N7)
UID_VARIANT = load_splits().uid_variant


def _real_sample():
    df = pd.read_parquet(SAMPLE_PATH, columns=INPUT_COLUMNS)
    uid = build_uids(df)[UID_VARIANT]
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


# ---- real-data graph and sequence leakage (review L1, D60) -----------------------------------
#
# The 50k sample spans days 1-13, so with L = 30 no label matures inside it: the L = 30 runs
# check edge and structure leakage, and an L = 1 run exercises matured labels on the same rows.
# A slice of the full merged data (days 1-75, a fixed 1-in-15 subset of customers, whole
# histories kept) exercises L = 30 with labels that do mature.

GRAPH_COLUMNS = ["TransactionID", "TransactionDT", "day", "isFraud", "TransactionAmt", "ProductCD",
                 "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "dist1",
                 "P_emaildomain", "R_emaildomain", "DeviceInfo", "DeviceType", "D1", "C1", "C13"]  # fmt: skip


MERGED_FULL = INTERIM_DIR / "merged.parquet"
GRAPH_DATASETS = {"real_50k": SAMPLE_PATH, "real_days_1_75": MERGED_FULL}


@functools.cache
def _graph_data(name: str):
    """Loaded on first use only (never at collection time)."""
    if name == "real_50k":
        df = pd.read_parquet(SAMPLE_PATH, columns=GRAPH_COLUMNS)
        return df, build_uids(df)[UID_VARIANT].astype(str)
    full = pd.read_parquet(MERGED_FULL, columns=GRAPH_COLUMNS)
    full = full[full["day"] <= 75].reset_index(drop=True)
    uid = build_uids(full)[UID_VARIANT]
    keep = (uid % 15 == 0).to_numpy()  # whole customers, deterministic
    return full[keep].reset_index(drop=True), uid[keep].reset_index(drop=True).astype(str)


GRAPH_PARAMS = [
    pytest.param(
        name, marks=pytest.mark.skipif(not path.exists(), reason=f"{path.name} not available")
    )
    for name, path in GRAPH_DATASETS.items()
]


def _cut(df, q):
    t = int(df["TransactionDT"].quantile(q))
    return t, (df["TransactionDT"] <= t).to_numpy()


@pytest.mark.parametrize("delay", [30, 1])
@pytest.mark.parametrize("name", GRAPH_PARAMS)
def test_real_graph_features_truncation_and_label_flip(name, delay):
    from vaultic.features.graph import build_graph_features, fit_hub_thresholds_on_training_period

    df, uid = _graph_data(name)
    # review N7: with hub thresholds passed in (fitted once on the training period), so the
    # availability feature g_shared_nonhub is covered by truncation and label flips too
    kw = dict(label_delay_days=delay, window_days=30,
              hub_thresholds=fit_hub_thresholds_on_training_period(df, uid, load_splits()))  # fmt: skip
    full = build_graph_features(df, uid, "C", **kw)
    assert (full["g_shared_nonhub"].fillna(0) > 0).mean() > 0.05  # availability is non-trivial
    if name == "real_days_1_75" and delay == 30:  # labels must actually mature here
        assert full["g_fraud_rate_card"].notna().mean() > 0.05
        assert full["g_comp_fraud_rate"].notna().mean() > 0.05
    for q in (0.5, 0.85):
        t, keep = _cut(df, q)
        cut = build_graph_features(df[keep], uid[keep], "C", **kw)
        pd.testing.assert_frame_equal(full[keep], cut, check_exact=True)
    t, keep = _cut(df, 0.7)
    flipped = df.copy()
    unknown = (df["TransactionDT"] > t - delay * SECONDS_PER_DAY).to_numpy()
    flipped.loc[unknown, "isFraud"] = 1 - flipped.loc[unknown, "isFraud"]
    again = build_graph_features(flipped, uid, "C", **kw)
    pd.testing.assert_frame_equal(full[keep], again[keep], check_exact=True)


@pytest.mark.parametrize("name", GRAPH_PARAMS)
def test_real_sequences_truncation_and_future_changes(name):
    from vaultic.features.categories import CategoryEncoder
    from vaultic.features.sequences import build_sequences

    df, uid = _graph_data(name)
    t0 = int(df["TransactionDT"].quantile(0.3))
    enc = CategoryEncoder().fit(df[df["TransactionDT"] <= t0], columns=("ProductCD",))
    extra = ("C1", "C13", "D1", "dist1")
    full = build_sequences(df, uid, enc, n_steps=20, extra_columns=extra)
    for q in (0.5, 0.85):
        _, keep = _cut(df, q)
        cut = build_sequences(df[keep].reset_index(drop=True), uid[keep].reset_index(drop=True), enc,
                              n_steps=20, extra_columns=extra)  # fmt: skip
        assert np.array_equal(full.values[keep], cut.values) and np.array_equal(
            full.mask[keep], cut.mask
        )
    t, keep = _cut(df, 0.6)
    changed = df.copy()
    future = ~keep
    changed.loc[future, "TransactionAmt"] = 1e6
    changed.loc[future, ["C1", "C13"]] = -5.0
    again = build_sequences(changed, uid, enc, n_steps=20, extra_columns=extra)
    assert np.array_equal(full.values[keep], again.values[keep])


@pytest.mark.parametrize("name", GRAPH_PARAMS)
def test_real_behavioral_features_truncation_and_future_changes(name):
    """Phase 3 (D66): behavioral features on real rows depend only on strictly earlier rows of
    the same uid (and, for entity velocity, earlier rows of the same card/email/device)."""
    from vaultic.features.behavioral import build_behavioral

    df, uid = _graph_data(name)
    full = build_behavioral(df, uid)
    assert (
        full["amt_z"].notna().mean() > 0.01
    )  # some real history exists (the 50k sample spans ~13 days)
    for q in (0.5, 0.85):
        _, keep = _cut(df, q)
        cut = build_behavioral(df[keep], uid[keep])
        pd.testing.assert_frame_equal(full[keep], cut, check_exact=True)
    _, keep = _cut(df, 0.6)
    changed = df.copy()
    future = ~keep
    changed.loc[future, "TransactionAmt"] = 1e6
    changed["DeviceInfo"] = changed["DeviceInfo"].astype(object)
    changed.loc[future, "DeviceInfo"] = "future-device"
    changed.loc[future, "P_emaildomain"] = np.nan
    again = build_behavioral(changed, uid)
    pd.testing.assert_frame_equal(full[keep], again[keep], check_exact=True)
