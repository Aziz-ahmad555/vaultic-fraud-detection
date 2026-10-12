"""Phase 3: behavioral and velocity view: how unusual a transaction is for its own customer.

Every feature of a transaction at time t uses only that customer's (uid's) transactions with
time strictly before t; rows sharing the same second never see each other. No labels are used.
Groups (roadmap Phase 3):

  amount deviation  amt_z (mean/std, std 0 -> one cent; >= 2 past), amt_robust_z = (a - median) /
                    (1.4826 * MAD + 0.01) (>= 2 past), amt_ratio_median (>= 1 past),
                    amt_percentile (mid-rank in past amounts, >= 1 past)
  velocity          vel_n_{1h,24h,7d,30d}, vel_amt_{...}: count / amount sum in [t - w, t)
  recency           secs_since_prev, mean_gap (mean gap between past transactions, >= 2 past),
                    gap_ratio = secs_since_prev / mean_gap (mean_gap 0 -> 1 s)
  escalation        amt_slope_5: least-squares slope of amount over the last 5 transactions
                    including the current one (>= 2 points)
  novelty           new_{email,device,addr2,product}: 1 if the uid never used this value
                    before t, 0 if it did, NaN if the value is missing; n_new_attributes
  entity velocity   ent_uids_7d_{card,email,device}: distinct uids seen with this card /
                    email domain / device in [t - 7d, t); NaN if the entity is missing
  history strength  hist_n_past, hist_days_since_first (0 = cold start, as in the roadmap)
  time pattern      hour (relative: IEEE-CIS has no calendar time), hour_deviation: circular
                    distance in hours from the uid's mean past hour (NaN without history)

Features that need history are NaN when there is none (CLAUDE.md rule 11), except the history
strength pair, where 0 is the cold-start signal.
"""

from __future__ import annotations

import bisect
from collections import Counter, deque

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.features.pit import PastIndex

WINDOWS = {"1h": 3_600, "24h": 86_400, "7d": 7 * 86_400, "30d": 30 * 86_400}
MAD_SCALE = 1.4826
MAD_EPS = 0.01  # one cent: keeps robust z finite when all past amounts are equal
SLOPE_POINTS = 5
NOVELTY = {
    "email": "P_emaildomain",
    "device": "DeviceInfo",
    "addr2": "addr2",
    "product": "ProductCD",
}
ENTITIES = {
    "card": ["card1", "card2", "card3", "card4", "card5", "card6"],
    "email": ["P_emaildomain"],
    "device": ["DeviceInfo"],
}
ENTITY_WINDOW = 7 * 86_400


def _check_sorted(time: np.ndarray) -> None:
    if np.any(np.diff(time) < 0):
        raise ValueError("rows must be sorted by TransactionDT")


def _amount_history_features(uid: np.ndarray, time: np.ndarray, amount: np.ndarray) -> dict:
    """Median / MAD / percentile / slope features: an expanding sorted list per uid."""
    n = len(amount)
    out = {k: np.full(n, np.nan) for k in ("median", "mad", "percentile", "slope")}
    order = np.lexsort((time, pd.factorize(uid, use_na_sentinel=False)[0]))
    codes = pd.factorize(uid, use_na_sentinel=False)[0][order]
    bounds = np.flatnonzero(np.diff(codes)) + 1
    for rows in np.split(order, bounds):
        t, a = time[rows], amount[rows]
        past_sorted: list[float] = []
        added = 0
        for i, row in enumerate(rows):
            p = int(np.searchsorted(t, t[i], side="left"))  # rows strictly before t
            while added < p:
                bisect.insort(past_sorted, a[added])
                added += 1
            current = a[i]
            if p >= 1:
                past = np.asarray(past_sorted)
                med = float(np.median(past))
                out["median"][row] = med
                out["mad"][row] = float(np.median(np.abs(past - med)))
                below = bisect.bisect_left(past_sorted, current)
                equal = bisect.bisect_right(past_sorted, current) - below
                out["percentile"][row] = (below + 0.5 * equal) / p
            recent = np.append(a[max(0, p - (SLOPE_POINTS - 1)) : p], current)
            if len(recent) >= 2:
                x = np.arange(len(recent), dtype=float)
                x -= x.mean()
                out["slope"][row] = float((x * recent).sum() / (x * x).sum())
    return out


def _entity_distinct_uids(entity: np.ndarray, uid: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Distinct uids per entity in [t - 7d, t), processed in time blocks so ties are excluded."""
    n = len(time)
    out = np.full(n, np.nan)
    windows: dict = {}
    start = 0
    while start < n:
        end = start
        while end < n and time[end] == time[start]:
            end += 1
        t = time[start]
        for i in range(start, end):  # read before inserting this second's rows
            e = entity[i]
            if e is None:
                continue
            events, counts = windows.get(e, (deque(), Counter()))
            while events and events[0][0] < t - ENTITY_WINDOW:
                _, old = events.popleft()
                counts[old] -= 1
                if counts[old] == 0:
                    del counts[old]
            windows[e] = (events, counts)
            out[i] = len(counts)
        for i in range(start, end):
            e = entity[i]
            if e is None:
                continue
            events, counts = windows.setdefault(e, (deque(), Counter()))
            events.append((t, uid[i]))
            counts[uid[i]] += 1
        start = end
    return out


def _entity_key(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    present = [c for c in cols if c in df]
    parts = df[present].astype(object)
    missing = parts.isna().all(axis=1).to_numpy()
    key = parts.astype(str).agg("|".join, axis=1).to_numpy(dtype=object)
    key[missing] = None
    return key


def build_behavioral(df: pd.DataFrame, uid: pd.Series) -> pd.DataFrame:
    """Behavioral features for every row of df (rows sorted by TransactionDT)."""
    time = df["TransactionDT"].to_numpy(dtype=np.int64)
    _check_sorted(time)
    amount = df["TransactionAmt"].to_numpy(dtype=np.float64)
    uid_values = uid.to_numpy()
    past = PastIndex(uid_values, time)
    out = pd.DataFrame({"TransactionID": df["TransactionID"].to_numpy()}, index=df.index)

    # history strength
    n_past = past.count_before(time)
    first_time = past.first_time_before(time)
    out["hist_n_past"] = n_past.astype(np.float32)
    out["hist_days_since_first"] = np.where(
        n_past > 0, (time - first_time) / SECONDS_PER_DAY, 0.0
    ).astype(np.float32)

    # amount deviation (mean/std from running sums; median, MAD, percentile from the loop)
    total = past.sum_before(amount, time)
    squares = past.sum_before(amount**2, time)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_past > 0, total / n_past, np.nan)
        var = np.where(n_past > 1, (squares - n_past * mean**2) / (n_past - 1), np.nan)
        std = np.sqrt(np.clip(var, 0, None))
        # a constant history (std 0) gets one cent as its spread, like amt_robust_z (D66)
        spread = np.where(std > 0, std, MAD_EPS)
        out["amt_z"] = np.where(n_past > 1, (amount - mean) / spread, np.nan).astype(np.float32)
    hist = _amount_history_features(uid_values, time, amount)
    with np.errstate(invalid="ignore", divide="ignore"):
        robust = (amount - hist["median"]) / (MAD_SCALE * hist["mad"] + MAD_EPS)
        out["amt_robust_z"] = np.where(n_past > 1, robust, np.nan).astype(np.float32)
        out["amt_ratio_median"] = (amount / hist["median"]).astype(np.float32)
    out["amt_percentile"] = hist["percentile"].astype(np.float32)

    # velocity
    for name, seconds in WINDOWS.items():
        start = time - seconds
        out[f"vel_n_{name}"] = (past.count_before(time) - past.count_before(start)).astype(
            np.float32
        )
        out[f"vel_amt_{name}"] = (
            past.sum_before(amount, time) - past.sum_before(amount, start)
        ).astype(np.float32)

    # recency
    since_prev = time - past.last_time_before(time)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_gap = np.where(
            n_past > 1, (past.last_time_before(time) - first_time) / (n_past - 1), np.nan
        )
        out["secs_since_prev"] = since_prev.astype(np.float32)
        out["mean_gap"] = mean_gap.astype(np.float32)
        # past transactions all in the same second: a 1-second mean gap keeps the ratio finite
        out["gap_ratio"] = (since_prev / np.where(mean_gap > 0, mean_gap, 1.0)).astype(np.float32)

    # escalation
    out["amt_slope_5"] = hist["slope"].astype(np.float32)

    # novelty
    flags = []
    for name, col in NOVELTY.items():
        values = df[col].astype(object)
        missing = values.isna().to_numpy()
        pair = pd.Series(uid_values).astype(str).to_numpy() + "\x1f" + values.astype(str).to_numpy()
        seen_before = PastIndex(pair, time).count_before(time) > 0
        flag = np.where(missing, np.nan, (~seen_before).astype(float))
        out[f"new_{name}"] = flag.astype(np.float32)
        flags.append(flag)
    stacked = np.vstack(flags)
    out["n_new_attributes"] = np.where(
        np.isnan(stacked).all(axis=0), np.nan, np.nansum(stacked, axis=0)
    ).astype(np.float32)

    # entity velocity
    for name, cols in ENTITIES.items():
        out[f"ent_uids_7d_{name}"] = _entity_distinct_uids(
            _entity_key(df, cols), uid_values, time
        ).astype(np.float32)

    # time pattern (relative hour)
    hour = (time // 3600) % 24
    angle = 2 * np.pi * hour / 24
    sin_sum = past.sum_before(np.sin(angle), time)
    cos_sum = past.sum_before(np.cos(angle), time)
    usual = (np.arctan2(sin_sum, cos_sum) * 24 / (2 * np.pi)) % 24
    diff = np.abs(hour - usual) % 24
    out["hour"] = hour.astype(np.float32)
    out["hour_deviation"] = np.where(n_past > 0, np.minimum(diff, 24 - diff), np.nan).astype(
        np.float32
    )
    return out


def main() -> None:
    """Write data/features/behavioral_<uid variant>.parquet for the full merged data."""
    import argparse

    from vaultic.data.load import load_merged
    from vaultic.data.splits import load_splits
    from vaultic.data.uid import UID_PATH
    from vaultic.paths import FEATURES_DIR, MERGED_PATH

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--uid-variant", default=None, help="default: uid_variant in splits.yaml")
    args = parser.parse_args()
    variant = args.uid_variant or load_splits().uid_variant
    columns = ["TransactionID", "TransactionDT", "TransactionAmt", *NOVELTY.values()]
    columns += [c for cols in ENTITIES.values() for c in cols if c not in columns]
    df = load_merged(MERGED_PATH, columns=columns)
    uids = pd.read_parquet(UID_PATH, columns=["TransactionID", variant])
    if not (uids["TransactionID"].to_numpy() == df["TransactionID"].to_numpy()).all():
        raise ValueError("uids.parquet is out of date; rerun python -m vaultic.data.uid")
    features = build_behavioral(df, uids[variant])
    out = FEATURES_DIR / f"behavioral_{variant}.parquet"
    features.to_parquet(out, index=False)
    print(f"wrote {out}: {len(features):,} rows x {features.shape[1] - 1} features")


if __name__ == "__main__":
    main()
