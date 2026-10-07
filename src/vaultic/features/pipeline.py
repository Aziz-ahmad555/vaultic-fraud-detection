"""Phase 1.4: point-in-time base features (customer history, velocity, matured labels,
frequency encodings).

Every feature of a transaction at time t uses only transactions with time < t; label-derived
features use only labels with its_time + L <= t; frequency encoders are fitted on the
training period only and applied forward. tests/test_leakage.py checks all three.

Run:  python -m vaultic.features.pipeline [--uid-variant uid]
                                       (writes data/features/base_features_<variant>.parquet)
      python -m vaultic.features.pipeline --sample   (uses data/interim/merged_sample.parquet)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vaultic.data.load import load_merged
from vaultic.data.splits import Splits, load_splits
from vaultic.data.uid import UID_PATH, build_uids
from vaultic.features.pit import FrequencyEncoder, PastIndex, label_cutoff
from vaultic.paths import FEATURES_DIR, INTERIM_DIR, MERGED_PATH


def features_path(uid_variant: str) -> Path:
    return FEATURES_DIR / f"base_features_{uid_variant}.parquet"


WINDOWS = {"1h": 3_600, "24h": 86_400, "7d": 7 * 86_400, "30d": 30 * 86_400}
FREQ_COLUMNS = ["card1", "addr1", "P_emaildomain"]
INPUT_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "day",
    "TransactionAmt",
    "isFraud",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "D1",
    "P_emaildomain",
]


def fit_encoders(train: pd.DataFrame, uid: pd.Series) -> dict[str, FrequencyEncoder]:
    """Fit frequency encoders on training-period rows only."""
    encoders = {col: FrequencyEncoder().fit(train[col]) for col in FREQ_COLUMNS}
    encoders["uid"] = FrequencyEncoder().fit(uid)
    return encoders


def build_features(
    df: pd.DataFrame,
    uid: pd.Series,
    encoders: dict[str, FrequencyEncoder],
    label_delay_days: int,
) -> pd.DataFrame:
    """Base features for every row of df (rows must be sorted by TransactionDT)."""
    time = df["TransactionDT"].to_numpy(dtype=np.int64)
    if np.any(np.diff(time) < 0):
        raise ValueError("df must be sorted by TransactionDT")
    amount = df["TransactionAmt"].to_numpy(dtype=np.float64)
    past = PastIndex(uid.to_numpy(), time)

    out = pd.DataFrame({"TransactionID": df["TransactionID"].to_numpy()}, index=df.index)

    # Customer history, strictly before t
    n_past = past.count_before(time)
    amt_sum = past.sum_before(amount, time)
    amt_sq = past.sum_before(amount**2, time)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_past > 0, amt_sum / n_past, np.nan)
        var = np.where(n_past > 1, (amt_sq - n_past * mean**2) / np.maximum(n_past - 1, 1), np.nan)
    out["uid_n_past"] = n_past.astype(np.float32)
    out["uid_amt_mean_past"] = mean.astype(np.float32)
    out["uid_amt_std_past"] = np.sqrt(np.clip(var, 0, None)).astype(np.float32)
    out["uid_amt_vs_mean"] = (amount / mean).astype(np.float32)
    out["uid_secs_since_prev"] = (time - past.last_time_before(time)).astype(np.float32)

    # Velocity windows [t - w, t)
    for name, seconds in WINDOWS.items():
        start = time - seconds
        out[f"uid_n_{name}"] = (past.count_before(time) - past.count_before(start)).astype(
            np.float32
        )
        out[f"uid_amt_sum_{name}"] = (
            past.sum_before(amount, time) - past.sum_before(amount, start)
        ).astype(np.float32)

    # Labels known at t: its_time + L <= t
    cutoff = label_cutoff(time, label_delay_days)
    n_known = past.count_before(cutoff)
    fraud_known = past.sum_before(df["isFraud"].to_numpy(dtype=np.float64), cutoff)
    out["uid_n_labels_known"] = n_known.astype(np.float32)
    out["uid_fraud_known"] = fraud_known.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["uid_fraud_rate_known"] = np.where(n_known > 0, fraud_known / n_known, np.nan).astype(
            np.float32
        )

    # Frequency encodings fitted on the training period
    for col in FREQ_COLUMNS:
        out[f"freq_{col}"] = encoders[col].transform(df[col])
    out["freq_uid"] = encoders["uid"].transform(uid)
    return out


def build_for_splits(
    df: pd.DataFrame, uids: pd.DataFrame, splits: Splits, uid_variant: str | None = None
) -> pd.DataFrame:
    uid = uids[uid_variant or splits.uid_variant]
    is_train = splits.train.contains(df["day"])
    encoders = fit_encoders(df[is_train], uid[is_train])
    return build_features(df, uid, encoders, splits.label_delay_days)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="store_true", help="use merged_sample.parquet")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--uid-variant", default=None, help="default: uid_variant in splits.yaml")
    args = parser.parse_args()
    splits = load_splits()
    variant = args.uid_variant or splits.uid_variant
    if args.sample:
        df = load_merged(INTERIM_DIR / "merged_sample.parquet", columns=INPUT_COLUMNS)
        uids = build_uids(df)
        out = args.out or FEATURES_DIR / "base_features_sample.parquet"
    else:
        df = load_merged(MERGED_PATH, columns=INPUT_COLUMNS)
        uids = pd.read_parquet(UID_PATH)
        if not (uids["TransactionID"].to_numpy() == df["TransactionID"].to_numpy()).all():
            raise ValueError("uids.parquet is out of date; rerun python -m vaultic.data.uid")
        out = args.out or features_path(variant)
    features = build_for_splits(df, uids, splits, variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    print(f"wrote {out}: {len(features):,} rows x {features.shape[1] - 1} features")


if __name__ == "__main__":
    main()
