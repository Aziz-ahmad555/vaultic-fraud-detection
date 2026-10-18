"""Seeded scenario generators (Phase B). Each returns the synthetic rows (stored columns and
dtypes), their customer codes, labels and a description; every row is marked synthetic by its
TransactionID (>= SYNTHETIC_ID_START) and by the `synthetic` list the runner keeps.

Masked fields follow the D184 rules for an existing customer: C*, M*, V* and id_* plus the card
and billing-address fields are copied from the customer's most recent real transaction before the
attack starts, D1 is set so the reconstructed uid is that customer, every other D field and
dist1 / dist2 are missing.

card_testing   a burst of small amounts on one existing customer's card, card-not-present: product
               W, no device or identity fields, P_emaildomain copied; gaps of 15-90 s, amounts
               1.00-9.99; every row fraud. Intensity sets the burst length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.scenarios import SYNTHETIC_ID_START

INTENSITY = {"low": 5, "medium": 12, "high": 25}  # card testing: rows in the burst
SCENARIOS = ("card_testing",)


@dataclass
class Scenario:
    name: str
    seed: int
    intensity: str
    day: int
    rows: pd.DataFrame  # synthetic rows, stored columns and dtypes
    uids: np.ndarray
    labels: np.ndarray  # 1 = fraud (known by construction)
    info: dict = field(default_factory=dict)


def _copy_from_customer(store, row: pd.DataFrame, src: pd.Series, day: int) -> None:
    from vaultic.live.custom import CARD_ADDR, COPIED, _set

    for c in row.columns:
        if COPIED.match(c) or c in CARD_ADDR:
            _set(row, c, None if pd.isna(src[c]) else src[c])
    d1n = float(src["day"]) - float(src["D1"])
    _set(row, "D1", day - d1n)


def pick_customer(store, rng, t0: int, min_rows: int = 3, active_days: int = 30) -> int:
    """A seeded existing customer with >= min_rows real rows before t0, one in the last
    active_days, and a known D1 (so the uid can be reconstructed)."""
    t = store.time
    recent = (t < t0) & (t >= t0 - active_days * SECONDS_PER_DAY)
    cand = pd.unique(store.uid[recent])
    rng.shuffle(cand)
    d1 = store.df["D1"].to_numpy(float)
    for u in cand:
        prev = store.before(store.uid_rows.get(int(u)), t0)
        if len(prev) >= min_rows and np.isfinite(d1[prev[-1]]):
            return int(u)
    raise RuntimeError("no customer fits")


def card_testing(store, seed: int = 0, intensity: str = "medium", day: int | None = None,
                 scenario_index: int = 0) -> Scenario:  # fmt: skip
    from vaultic.live.custom import _set, empty_row
    from vaultic.live.store import uid_key_strings

    rng = np.random.default_rng(seed)
    lo, hi = int(store.splits.test.first), int(store.splits.test.last)
    day = int(rng.integers(lo, hi + 1)) if day is None else int(day)
    t0 = day * SECONDS_PER_DAY + int(rng.integers(8 * 3600, 20 * 3600))
    uid = pick_customer(store, rng, t0)
    src = store.df.iloc[int(store.before(store.uid_rows[uid], t0)[-1])]
    n = INTENSITY[intensity]
    gaps = rng.integers(15, 91, size=n)
    gaps[0] = 0
    times = t0 + np.cumsum(gaps)
    amounts = np.round(rng.uniform(1.0, 9.99, size=n), 2)
    out = []
    for i in range(n):
        row = empty_row(store)
        row["TransactionID"] = SYNTHETIC_ID_START + scenario_index * 1_000_000 + i
        row["TransactionDT"] = int(times[i])
        row["day"] = int(times[i] // SECONDS_PER_DAY)
        row["TransactionAmt"] = float(amounts[i])
        row["isFraud"] = 1
        _set(row, "ProductCD", "W")
        _copy_from_customer(store, row, src, int(row["day"].iloc[0]))
        for c in ("card4", "card6", "P_emaildomain"):
            _set(row, c, None if pd.isna(src[c]) else src[c])
        for c in row.columns:  # card-not-present: no device or identity fields
            if c.startswith("id_") or c in ("DeviceType", "DeviceInfo"):
                _set(row, c, None)
        row["has_identity"] = np.int8(0)
        out.append(row)
    rows = pd.concat(out, ignore_index=True).astype(store.df.dtypes.to_dict())
    keys = uid_key_strings(rows)
    resolved = {store.key_to_uid.get(k) for k in keys}
    if resolved != {uid}:
        raise RuntimeError(f"synthetic rows resolve to {resolved}, not customer {uid}")
    return Scenario("card_testing", seed, intensity, day, rows, np.full(n, uid), np.ones(n, int),
                    {"customer": uid, "source_transaction": int(src["TransactionID"]),
                     "start": int(t0), "n_rows": n,
                     "amounts": [float(a) for a in amounts]})  # fmt: skip


GENERATORS = {"card_testing": card_testing}
