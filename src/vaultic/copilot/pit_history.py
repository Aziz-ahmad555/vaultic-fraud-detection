"""The case customer's point-in-time history from the Test Lab's history store (D228), read-only.

The demo bundle holds only the test period, so a customer's earlier rows from the training,
validation and calibration days are not in it. When the Test Lab is running, its history store
(every merged IEEE-CIS row, indexed for "strictly before t") supplies them:

  - only rows of the same uid with time strictly before the case's time;
  - usual amount = median of those amounts; device seen before = the case's DeviceInfo among them;
  - transactions in the 24 hours before; other customers on the same card (card1-card6) in the
    7 days before;
  - known fraud among the earlier rows only once its_time + L days <= as-of (label-delay curtain);
  - the last earlier rows, newest first, with time, amount and whether their label is known.

Nothing is written; the store and src/vaultic/live are not changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DAY = 86400
CARDS = ("card1", "card2", "card3", "card4", "card5", "card6")


@dataclass
class PitHistory:
    total: int  # earlier rows of the customer (all periods)
    usual_amount: float | None
    device_known: bool | None  # None: the case has no device, or no earlier rows
    last_24h: int
    shared_card_7d: int
    known_fraud: int  # earlier rows whose fraud label is known at the as-of
    first_day: int | None
    recent: list[dict] = field(default_factory=list)  # newest first: tid, time, amount, label|None


def _same(values: pd.Series, value) -> np.ndarray:
    """values == value, with NaN equal to NaN (a missing card field matches a missing one)."""
    if pd.isna(value):
        return values.isna().to_numpy()
    return (values == value).to_numpy()


def store_history(store: Any, position: int, as_of: float, L: int, n_recent: int = 5) -> PitHistory:
    """History of the customer of the store row `position` strictly before its own time."""
    df = store.df
    t_case = int(df["TransactionDT"].iat[position])
    times = df["TransactionDT"].to_numpy(dtype=np.int64)
    uid = int(store.uid[position])
    earlier = store.before(store.uid_rows.get(uid), t_case)  # strictly before the case's time
    earlier = earlier[times[earlier] < t_case]  # belt and braces: never the case's own second
    amounts = df["TransactionAmt"].to_numpy(dtype=float)[earlier]
    device = df["DeviceInfo"].iat[position] if "DeviceInfo" in df else None
    if device is None or pd.isna(device) or not len(earlier):
        known_device = None
    else:
        known_device = bool(device in set(df["DeviceInfo"].iloc[earlier].dropna()))
    last_24h = int(np.sum(times[earlier] >= t_case - DAY))
    # other customers on the same card in [t - 7 days, t)
    window = (times >= t_case - 7 * DAY) & (times < t_case)
    same = window.copy()
    for c in CARDS:
        same &= _same(df[c], df[c].iat[position])
    shared = len(set(store.uid[same].tolist()) - {uid})
    labels = df["isFraud"].to_numpy()[earlier]
    matured = times[earlier] + L * DAY <= as_of
    recent = []
    for p in earlier[::-1][:n_recent]:
        known = times[p] + L * DAY <= as_of
        recent.append({"tid": int(df["TransactionID"].iat[p]), "time": int(times[p]),
                       "amount": float(df["TransactionAmt"].iat[p]),
                       "label": int(df["isFraud"].iat[p]) if known else None})  # fmt: skip
    return PitHistory(
        total=int(len(earlier)),
        usual_amount=float(np.median(amounts)) if len(amounts) else None,
        device_known=known_device,
        last_24h=last_24h,
        shared_card_7d=int(shared),
        known_fraud=int(np.sum((labels == 1) & matured)),
        first_day=int(times[earlier[0]] // DAY) if len(earlier) else None,
        recent=recent,
    )


def from_testlab(svc: Any, tid: int, as_of: float, L: int) -> PitHistory | None:
    """The history through a ready Test Lab service; None when it is not mounted or not ready."""
    if svc is None or getattr(svc, "state", None) != "ready":
        return None
    engine = svc._ready()
    return store_history(engine.store, svc._position(tid), as_of, L)
