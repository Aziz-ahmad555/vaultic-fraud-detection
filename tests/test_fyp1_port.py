import numpy as np
import pandas as pd
import pytest

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.views.fyp1 import IF_FEATURES, MIN_HISTORY, FYP1Model, build_fyp1_frame


def _df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.integers(0, 100 * SECONDS_PER_DAY, n))
    users = np.where(np.arange(n) < 400, 7, rng.integers(0, 200, n))  # user 7 is heavy
    y = (rng.random(n) < 0.08).astype(int)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": time,
            "day": time // SECONDS_PER_DAY,
            "isFraud": y,
            "TransactionAmt": rng.lognormal(4, 1, n) * (1 + 3 * y),
            "ProductCD": rng.choice(["W", "C"], n),
            "card1": rng.integers(1000, 1010, n).astype(float),
            "card2": np.nan,
            "card3": 150.0,
            "card4": rng.choice(["visa", None], n),
            "card5": 226.0,
            "card6": "debit",
            "addr1": 300.0,
            "addr2": 87.0,
            "P_emaildomain": rng.choice(["a.com", None], n),
            "C1": rng.integers(0, 4, n).astype(float),
            "C2": 1.0,
            "C13": 1.0,
            "DeviceInfo": rng.choice(["Windows", "iOS", None], n),
        }
    ), pd.Series(users)


def test_behavioural_features_are_point_in_time():
    df, uid = _df()
    f = build_fyp1_frame(df, uid)
    first = uid.groupby(uid).head(1).index  # each user's first transaction
    assert (f.loc[first, "_n_past"] == 0).all()
    assert (f.loc[first, "rolling_count_10"] == 0).all()
    assert (f.loc[first, "delta_prev_seconds"] == 0).all()
    assert f["rolling_count_10"].max() == 10
    assert set(IF_FEATURES) <= set(f.columns)


def test_model_trains_personal_forest_only_for_heavy_users_and_returns_probabilities():
    df, uid = _df()
    X = build_fyp1_frame(df, uid)
    model = FYP1Model(seed=0).fit(X, df["isFraud"])
    assert set(model.personal_) == {7}  # only user 7 has >= MIN_HISTORY rows
    assert (uid.value_counts() >= MIN_HISTORY).sum() == 1
    p = model.predict_proba(X)
    assert p.shape == (len(df), 2) and np.allclose(p.sum(axis=1), 1)
    assert ((p[:, 1] >= 0) & (p[:, 1] <= 1)).all()


def test_extreme_amount_rule_forces_high_score():
    df, uid = _df()
    X = build_fyp1_frame(df, uid)
    model = FYP1Model(seed=0).fit(X, df["isFraud"])
    row = X[(X["_n_past"] >= 3) & (X["_uid"] != 7)].iloc[[0]].copy()
    row["amount"] = row["_hist_max"] * 6
    assert model.predict_proba(row)[0, 1] >= 0.95


@pytest.mark.parametrize("seed", [0, 1])
def test_same_seed_same_scores(seed):
    df, uid = _df()
    X = build_fyp1_frame(df, uid)
    a = FYP1Model(seed).fit(X, df["isFraud"]).predict_proba(X)
    b = FYP1Model(seed).fit(X, df["isFraud"]).predict_proba(X)
    assert np.array_equal(a, b)
