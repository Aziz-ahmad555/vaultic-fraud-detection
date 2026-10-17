"""Custom ("build your own") transactions for the Test Lab (D184). Custom input has no ground truth.

From the form: customer (an existing uid or new), amount, product, card type (card4 network and
card6 kind), purchaser email domain, device (DeviceType, DeviceInfo) and a date and time inside
the test period. Every other IEEE-CIS column is filled by these rules (Aziz, 2026-10-17):

  existing customer  C*, M*, V* and id_* from the customer's most recent transaction BEFORE the
                     chosen time; card1 and addr1 (the uid's parts) and card2 / card3 / card5 /
                     addr2 (the rest of the card and billing address) from that transaction as
                     well; D1 = day - (that customer's day - D1), so the reconstructed uid
                     (card1 + addr1 + day - D1) is exactly that customer; every other D field NaN
                     (stale time gaps are not copied); dist1 / dist2 NaN. A customer with no
                     transaction before the chosen time is refused (copying from a later one would
                     use the future).
  new customer       an unseen card1 (larger than any stored value), addr1 NaN, D1 = 0 (the
                     account starts that day), every masked field NaN; a new uid
  both               R_emaildomain NaN; has_identity = 1 when any identity field (DeviceType,
                     DeviceInfo, id_*) is present; isFraud unknown (the transaction's own label is
                     never used by its features).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY

COPIED = re.compile(r"^(C\d+|M\d+|V\d+|id_\d+)$")
CARD_ADDR = ("card1", "card2", "card3", "card5", "addr1", "addr2")
IDENTITY = re.compile(r"^(DeviceType|DeviceInfo|id_\d+)$")


class CustomInputError(ValueError):
    """A custom transaction the Test Lab cannot build (message shown to the user)."""


def empty_row(store) -> pd.DataFrame:
    """One all-missing row with the stored columns and dtypes (categories kept)."""
    row = store.df.iloc[[0]].copy().reset_index(drop=True)
    for c in row.columns:
        if isinstance(row[c].dtype, pd.CategoricalDtype):
            row[c] = pd.Categorical([np.nan], categories=row[c].cat.categories)
        elif pd.api.types.is_float_dtype(row[c].dtype):
            row[c] = np.nan
        else:  # the integer columns (id, time, day, label, has_identity) are always set
            row[c] = 0
    return row


def _set(row: pd.DataFrame, col: str, value) -> None:
    if col not in row:
        return
    dtype = row[col].dtype
    if isinstance(dtype, pd.CategoricalDtype):
        if value is not None and value == value and value not in dtype.categories:
            raise CustomInputError(f"{col} = {value!r} is not a value seen in the data")
        row[col] = pd.Categorical([value], categories=dtype.categories)
    else:
        row[col] = np.nan if value is None else value


def options(store) -> dict:
    """Allowed form values (the categories seen in the data) and the test-period day range."""
    df = store.df

    def cats(c, top=None):
        vc = df[c].value_counts()
        vals = [str(v) for v in vc.index if vc[v] > 0]
        return vals[:top] if top else vals

    return {"products": cats("ProductCD"), "card4": cats("card4"), "card6": cats("card6"),
            "email_domains": cats("P_emaildomain", 60), "device_types": cats("DeviceType"),
            "device_infos": cats("DeviceInfo", 40),
            "days": [int(store.splits.test.first), int(store.splits.test.last)]}  # fmt: skip


def build_custom(store, spec: dict) -> tuple[pd.DataFrame, int, dict]:
    """(one-row transaction, uid code, notes) from the form spec. Raises CustomInputError."""
    from vaultic.live.store import uid_key_strings

    day = int(spec["day"])
    lo, hi = int(store.splits.test.first), int(store.splits.test.last)
    if not lo <= day <= hi:
        raise CustomInputError(f"the date must be inside the test period (days {lo}-{hi})")
    seconds = int(spec.get("seconds", 12 * 3600))
    if not 0 <= seconds < SECONDS_PER_DAY:
        raise CustomInputError("the time of day must be within the day")
    t = day * SECONDS_PER_DAY + seconds
    amount = float(spec["amount"])
    if not amount > 0:
        raise CustomInputError("the amount must be positive")
    row = empty_row(store)
    tid = store.info.get("next_tid", int(store.df["TransactionID"].max()) + 1)
    store.info["next_tid"] = tid + 1
    row["TransactionID"] = tid
    row["TransactionDT"] = t
    row["day"] = day
    row["TransactionAmt"] = amount
    row["isFraud"] = 0  # unknown; never used by the transaction's own features
    for col, key in (("ProductCD", "product"), ("card4", "card4"), ("card6", "card6"),
                     ("P_emaildomain", "email"), ("DeviceType", "device_type"),
                     ("DeviceInfo", "device_info")):  # fmt: skip
        _set(row, col, spec.get(key) or None)
    notes = {"ground_truth": "custom input - no ground truth"}

    customer = spec.get("customer")
    if customer in (None, "", "new"):
        _set(row, "card1", float(np.nanmax(store.df["card1"].to_numpy(float)) + 1 + tid % 1000))
        _set(row, "D1", 0.0)
        notes["customer"] = "new customer: unseen card, D1 = 0, masked fields missing"
        uid = None
    else:
        uid = int(customer)
        if uid not in store.uid_rows:
            raise CustomInputError(f"customer {uid} does not exist")
        prev = store.before(store.uid_rows[uid], t)
        if not len(prev):
            raise CustomInputError(f"customer {uid} has no transaction before that time; "
                                   "choose a later time or a new customer")  # fmt: skip
        src = store.df.iloc[int(prev[-1])]
        for c in row.columns:
            if COPIED.match(c) or c in CARD_ADDR:
                _set(row, c, None if pd.isna(src[c]) else src[c])
        d1n = float(src["day"]) - float(src["D1"]) if pd.notna(src["D1"]) else np.nan
        _set(row, "D1", day - d1n if np.isfinite(d1n) else None)
        notes["customer"] = (f"existing customer {uid}: masked C/M/V/id fields from their "
                             f"transaction {int(src['TransactionID'])}; D1 set so the uid is "
                             "theirs; other D fields missing")  # fmt: skip
    has_id = any(row[c].notna().iloc[0] for c in row.columns if IDENTITY.match(c))
    row["has_identity"] = np.int8(1 if has_id else 0)
    key = uid_key_strings(row)[0]
    resolved = store.key_to_uid.get(key)
    if uid is not None and resolved != uid:
        raise CustomInputError(f"internal: the transaction resolves to uid {resolved}, not {uid}")
    if uid is None:
        uid = int(resolved) if resolved is not None else store.new_uid()
    notes["uid"] = int(uid)
    return row.astype(store.df.dtypes.to_dict()), int(uid), notes
