"""Seeded scenario generators (Phase B, D197, D199, D203; v2 = the primary since D203, see apply_v2). Each returns the synthetic rows (stored columns
and dtypes), their customer codes, labels and a description; every row is marked synthetic by its
TransactionID (>= SYNTHETIC_ID_START). Same seed + intensity -> same rows.

Masked fields follow D184:
  existing customer  C*, M*, V*, id_* and the card / billing-address fields copied from the
                     customer's most recent REAL transaction before the attack starts; D1 set so
                     the reconstructed uid is that customer; other D fields and dist1 / dist2
                     missing
  new customer       an unseen card1, addr1 missing, D1 = days since the account's first
                     transaction (0 on that day, so the uid stays the same), every masked field
                     missing
A scenario on a device new to the customer takes the device's identity fields (DeviceType,
DeviceInfo, id_*) from a real "donor" transaction of the training period that used that
DeviceInfo, so the device fingerprint is a real one.

  card_testing     burst of 1.00-9.99 charges on one existing customer's card, card-not-present
                   (ProductCD W, no identity), gaps 15-90 s; rows low 5 / medium 12 / high 25
  account_takeover established customer (>= 10 earlier rows), a device the customer never used
                   (DeviceInfo seen 10-1000 times) with its donor identity fields and email domain
                   (another common domain if equal to the customer's), amounts = customer's median
                   amount x low 5 / medium 10 / high 20, rows 1 / 2 / 3 within 30 minutes
  mule_ring        new cards (new customers) all on ONE rare DeviceInfo (seen 2-5 times in the
                   data) as a stand-in for one device, one transaction each, 100-400, gaps 2-20
                   minutes, the device's donor product, email and identity fields; cards low 5 /
                   medium 10 / high 20
  bust_out         one new card-not-present account (ProductCD W) that builds up with ordinary
                   purchases (lognormal, median 50) on build-up days low 3 / medium 7 / high 14,
                   1-2 a day, labelled LEGITIMATE (D190), then cashes out with large purchases
                   (500-2000) within 2 hours on the next day: low 3 / medium 5 / high 8, labelled
                   fraud
  mimicry          existing customer (>= 10 earlier rows): every field copied from the customer
                   (D184, including the customer's device), amounts and hours of day drawn from
                   the customer's own earlier transactions; rows low 1 / medium 3 / high 6 over the
                   day
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.scenarios import SYNTHETIC_ID_START

INTENSITIES = ("low", "medium", "high")
CARD_TESTING_ROWS = {"low": 5, "medium": 12, "high": 25}
ATO = {"low": (1, 5), "medium": (2, 10), "high": (3, 20)}  # rows, amount x customer median
MULE_CARDS = {"low": 5, "medium": 10, "high": 20}
BUST_OUT = {"low": (3, 3), "medium": (7, 5), "high": (14, 8)}  # build-up days, cash-out rows
MIMICRY_ROWS = {"low": 1, "medium": 3, "high": 6}
COMMON_EMAILS = ("gmail.com", "yahoo.com", "hotmail.com", "aol.com", "outlook.com")
MULE_DEVICE_NOTE = ("IEEE-CIS has no true device ID; DeviceInfo is a model/OS string, so a rare "
                    "value stands in for one device.")  # fmt: skip
SCENARIOS = ("card_testing", "account_takeover", "mule_ring", "bust_out", "mimicry")


@dataclass
class Scenario:
    name: str
    seed: int
    intensity: str
    day: int  # the attack (cash-out) day
    rows: pd.DataFrame  # synthetic rows, stored columns and dtypes
    uids: np.ndarray
    labels: np.ndarray  # 1 = fraud, 0 = legitimate (bust-out build-up), known by construction
    info: dict = field(default_factory=dict)


# ---- helpers -----------------------------------------------------------------------------------
def _identity(c: str) -> bool:
    return c.startswith("id_") or c in ("DeviceType", "DeviceInfo")


def _new_row(store, tid: int, t: int, amount: float, label: int) -> pd.DataFrame:
    from vaultic.live.custom import empty_row

    row = empty_row(store)
    row["TransactionID"] = tid
    row["TransactionDT"] = int(t)
    row["day"] = int(t // SECONDS_PER_DAY)
    row["TransactionAmt"] = float(amount)
    row["isFraud"] = int(label)
    return row


def _set(row, c, v) -> None:
    from vaultic.live.custom import _set as set_

    set_(row, c, None if v is None or (not isinstance(v, str) and pd.isna(v)) else v)


def _copy_existing(row: pd.DataFrame, src: pd.Series) -> None:
    """D184 existing customer: masked fields, card and address from src; D1 resolves the uid."""
    from vaultic.live.custom import CARD_ADDR, COPIED

    for c in row.columns:
        if COPIED.match(c) or c in CARD_ADDR:
            _set(row, c, src[c])
    for c in ("ProductCD", "card4", "card6", "P_emaildomain", "DeviceType", "DeviceInfo"):
        _set(row, c, src[c])
    day = int(row["day"].iloc[0])
    _set(row, "D1", day - (float(src["day"]) - float(src["D1"])))


def _copy_identity(row: pd.DataFrame, donor: pd.Series) -> None:
    for c in row.columns:
        if _identity(c):
            _set(row, c, donor[c])


def _clear_identity(row: pd.DataFrame) -> None:
    for c in row.columns:
        if _identity(c):
            _set(row, c, None)


def _finish(store, rows: list[pd.DataFrame]) -> pd.DataFrame:
    from vaultic.live.custom import IDENTITY

    out = pd.concat(rows, ignore_index=True)
    ident = [c for c in out.columns if IDENTITY.match(c)]
    out["has_identity"] = out[ident].notna().any(axis=1).astype(np.int8)
    return out.astype(store.df.dtypes.to_dict())


def _resolve(store, rows: pd.DataFrame, expected: np.ndarray) -> None:
    from vaultic.live.store import uid_key_strings

    keys = uid_key_strings(rows)
    for k, u in zip(keys, expected, strict=True):
        got = store.key_to_uid.get(k)
        if got is not None and int(got) != int(u):
            raise RuntimeError(f"synthetic row resolves to customer {got}, not {u}")
        if got is None and int(u) in store.uid_rows:
            raise RuntimeError(f"synthetic row for existing customer {u} does not resolve")


def _tid(scenario_index: int, i: int) -> int:
    return SYNTHETIC_ID_START + scenario_index * 1_000_000 + i


def _attack_start(store, rng, day: int | None) -> tuple[int, int]:
    lo, hi = int(store.splits.test.first), int(store.splits.test.last)
    day = int(rng.integers(lo, hi + 1)) if day is None else int(day)
    return day, day * SECONDS_PER_DAY + int(rng.integers(8 * 3600, 20 * 3600))


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


def pick_donor(store, rng, lo: int, hi: int, exclude=()) -> pd.Series:
    """A training-period transaction whose DeviceInfo appears lo-hi times in the data and is not
    in exclude (its identity fields become the attacker's device)."""
    df = store.df
    info = df["DeviceInfo"].astype(object)
    counts = info.value_counts()
    ok = counts[(counts >= lo) & (counts <= hi)].index.difference(pd.Index(list(exclude)))
    train = store.splits.train.contains(df["day"].to_numpy())
    cand = np.flatnonzero(train & info.isin(ok).to_numpy())
    return df.iloc[int(rng.choice(cand))]


def _new_uids(store, n: int) -> np.ndarray:
    return int(store.uid.max()) + 1 + np.arange(n)


def _new_card1(store, i: int) -> float:
    return float(np.nanmax(store.df["card1"].to_numpy(float)) + 1 + i)


# ---- generators --------------------------------------------------------------------------------
def card_testing(store, seed=0, intensity="medium", day=None, scenario_index=0) -> Scenario:
    rng = np.random.default_rng(seed)
    day, t0 = _attack_start(store, rng, day)
    uid = pick_customer(store, rng, t0)
    src = store.df.iloc[int(store.before(store.uid_rows[uid], t0)[-1])]
    n = CARD_TESTING_ROWS[intensity]
    gaps = rng.integers(15, 91, size=n)
    gaps[0] = 0
    times = t0 + np.cumsum(gaps)
    amounts = np.round(rng.uniform(1.0, 9.99, size=n), 2)
    out = []
    for i in range(n):
        row = _new_row(store, _tid(scenario_index, i), times[i], amounts[i], 1)
        _copy_existing(row, src)
        _set(row, "ProductCD", "W")
        _clear_identity(row)  # card-not-present: no device or identity fields
        out.append(row)
    rows = _finish(store, out)
    uids = np.full(n, uid)
    _resolve(store, rows, uids)
    return Scenario("card_testing", seed, intensity, day, rows, uids, np.ones(n, int),
                    {"customer": uid, "source_transaction": int(src["TransactionID"]),
                     "start": int(t0), "n_rows": n})  # fmt: skip


def account_takeover(store, seed=0, intensity="medium", day=None, scenario_index=1) -> Scenario:
    rng = np.random.default_rng(seed)
    day, t0 = _attack_start(store, rng, day)
    uid = pick_customer(store, rng, t0, min_rows=10)
    prev = store.before(store.uid_rows[uid], t0)
    src = store.df.iloc[int(prev[-1])]
    own = set(store.df["DeviceInfo"].astype(object).iloc[prev].dropna())
    donor = pick_donor(store, rng, 10, 1000, exclude=own)
    email = donor["P_emaildomain"]
    if pd.isna(email) or email == src["P_emaildomain"]:
        email = rng.choice([e for e in COMMON_EMAILS if e != src["P_emaildomain"]])
    n, mult = ATO[intensity]
    median = float(np.median(store.df["TransactionAmt"].to_numpy(float)[prev]))
    times = t0 + np.r_[0, np.sort(rng.integers(60, 1800, size=n - 1))]
    out = []
    for i in range(n):
        amount = round(median * mult * rng.uniform(0.9, 1.1), 2)
        row = _new_row(store, _tid(scenario_index, i), times[i], amount, 1)
        _copy_existing(row, src)
        _copy_identity(row, donor)
        _set(row, "ProductCD", donor["ProductCD"])
        _set(row, "P_emaildomain", email)
        out.append(row)
    rows = _finish(store, out)
    uids = np.full(n, uid)
    _resolve(store, rows, uids)
    return Scenario("account_takeover", seed, intensity, day, rows, uids, np.ones(n, int),
                    {"customer": uid, "customer_median_amount": median,
                     "device": str(donor["DeviceInfo"]), "email": str(email),
                     "start": int(t0), "n_rows": n})  # fmt: skip


def mule_ring(store, seed=0, intensity="medium", day=None, scenario_index=2) -> Scenario:
    rng = np.random.default_rng(seed)
    day, t0 = _attack_start(store, rng, day)
    donor = pick_donor(store, rng, 2, 5)
    k = MULE_CARDS[intensity]
    times = t0 + np.r_[0, np.cumsum(rng.integers(120, 1201, size=k - 1))]
    uids = _new_uids(store, k)
    out = []
    for i in range(k):
        row = _new_row(store, _tid(scenario_index, i), times[i],
                       round(float(rng.uniform(100, 400)), 2), 1)  # fmt: skip
        _set(row, "card1", _new_card1(store, i))
        _set(row, "D1", 0.0)
        for c in ("card4", "card6"):
            _set(row, c, donor[c])
        _copy_identity(row, donor)
        _set(row, "ProductCD", donor["ProductCD"])
        _set(row, "P_emaildomain", donor["P_emaildomain"])
        out.append(row)
    rows = _finish(store, out)
    _resolve(store, rows, uids)
    return Scenario("mule_ring", seed, intensity, day, rows, uids, np.ones(k, int),
                    {"device": str(donor["DeviceInfo"]), "device_note": MULE_DEVICE_NOTE,
                     "device_rows_in_data": int((store.df["DeviceInfo"].astype(object)
                                                 == donor["DeviceInfo"]).sum()),
                     "n_cards": k, "start": int(t0)})  # fmt: skip


def bust_out(store, seed=0, intensity="medium", day=None, scenario_index=3) -> Scenario:
    rng = np.random.default_rng(seed)
    build_days, n_cash = BUST_OUT[intensity]
    lo, hi = int(store.splits.test.first), int(store.splits.test.last)
    day = int(rng.integers(lo + build_days, hi + 1)) if day is None else int(day)
    first = day - build_days
    donor = store.df.iloc[int(rng.choice(np.flatnonzero(
        store.splits.train.contains(store.df["day"].to_numpy())
        & (store.df["ProductCD"].astype(object) == "W").to_numpy())))]  # fmt: skip
    email = rng.choice(COMMON_EMAILS)
    times, amounts, labels = [], [], []
    for d in range(first, day):
        for _ in range(int(rng.integers(1, 3))):
            times.append(d * SECONDS_PER_DAY + int(rng.integers(8 * 3600, 22 * 3600)))
            amounts.append(round(float(rng.lognormal(np.log(50), 0.5)), 2))
            labels.append(0)
    t_cash = day * SECONDS_PER_DAY + int(rng.integers(8 * 3600, 20 * 3600))
    for i in range(n_cash):
        times.append(t_cash + (0 if i == 0 else int(rng.integers(60, 7200))))
        amounts.append(round(float(rng.uniform(500, 2000)), 2))
        labels.append(1)
    order = np.argsort(times, kind="stable")
    times = np.asarray(times)[order]
    amounts, labels = np.asarray(amounts)[order], np.asarray(labels)[order]
    if len(np.unique(times)) != len(times):
        times = times + np.arange(len(times))  # keep times distinct
    uid = _new_uids(store, 1)[0]
    card1 = _new_card1(store, 0)
    out = []
    for i, (t, a, y) in enumerate(zip(times, amounts, labels, strict=True)):
        row = _new_row(store, _tid(scenario_index, i), t, a, y)
        _set(row, "card1", card1)
        _set(row, "D1", float(int(t // SECONDS_PER_DAY) - first))
        for c in ("card2", "card3", "card4", "card5", "card6"):
            _set(row, c, donor[c])
        _set(row, "ProductCD", "W")
        _set(row, "P_emaildomain", email)
        out.append(row)
    rows = _finish(store, out)
    uids = np.full(len(rows), uid)
    _resolve(store, rows, uids)
    return Scenario("bust_out", seed, intensity, day, rows, uids, labels.astype(int),
                    {"first_day": first, "cash_out_day": day, "build_up_rows": int((labels == 0).sum()),
                     "cash_out_rows": int(labels.sum()),
                     "first_cash_out_time": int(times[labels == 1].min())})  # fmt: skip


def mimicry(store, seed=0, intensity="medium", day=None, scenario_index=4) -> Scenario:
    rng = np.random.default_rng(seed)
    day, t0 = _attack_start(store, rng, day)
    t0 = day * SECONDS_PER_DAY  # times come from the customer's own hours of day
    uid = pick_customer(store, rng, t0, min_rows=10)
    prev = store.before(store.uid_rows[uid], t0)
    src = store.df.iloc[int(prev[-1])]
    amt = store.df["TransactionAmt"].to_numpy(float)[prev]
    sec = store.time[prev] % SECONDS_PER_DAY
    n = MIMICRY_ROWS[intensity]
    times = np.sort(t0 + rng.choice(sec, size=n, replace=True))
    times = times + np.arange(n)  # distinct
    out = []
    for i in range(n):
        row = _new_row(store, _tid(scenario_index, i), times[i], float(rng.choice(amt)), 1)
        _copy_existing(row, src)
        out.append(row)
    rows = _finish(store, out)
    uids = np.full(n, uid)
    _resolve(store, rows, uids)
    return Scenario(
        "mimicry",
        seed,
        intensity,
        day,
        rows,
        uids,
        np.ones(n, int),
        {"customer": uid, "n_rows": n, "customer_rows_before": int(len(prev))},
    )


GENERATORS_V1 = {"card_testing": card_testing, "account_takeover": account_takeover,
                 "mule_ring": mule_ring, "bust_out": bust_out, "mimicry": mimicry}  # fmt: skip


# ---- generator v2 (D203): masked fields from real validation-period rows ----------------------
MASKED_V2 = re.compile(r"^(C\d+|D(?:[2-9]|1[0-5])|M\d+|V\d+|dist1|dist2)$")
IDENTITY_COLS = re.compile(r"^(DeviceType|DeviceInfo|id_\d+)$")
KEEPS_DEVICE = ("account_takeover", "mule_ring")  # the scenario's own device stays (D203)
NEW_CUSTOMER = ("mule_ring", "bust_out")


def _pools(store) -> dict:
    df = store.df
    day = df["day"].to_numpy()
    val = (day >= store.splits.validation.first) & (day <= store.splits.validation.last)
    d1 = df["D1"].to_numpy(float)
    n_past = np.nan_to_num(store.features["hist_n_past"].to_numpy(float))
    return {"val": val, "young": val & (d1 >= 0) & (d1 <= 13), "existing": val & (n_past >= 3),
            "product": df["ProductCD"].astype(object).to_numpy()}  # fmt: skip


def apply_v2(store, sc: Scenario) -> Scenario:
    """Generator v2 (D203): each synthetic row's masked block (C, D2-D15, M, V, dist1/dist2 and,
    unless the scenario sets the device, DeviceType / DeviceInfo / id_*) copied from ONE real
    validation-period row: the hijacked customer's own rows with the row's product (else existing
    customers' rows with that product) for existing customers; young accounts' rows with the row's
    product for new customers. Own random stream (seed, 2): the attack structure equals v1's."""
    rng = np.random.default_rng([sc.seed, 2])
    df, pools = store.df, _pools(store)
    rows = sc.rows.copy()
    cols = [c for c in rows.columns if MASKED_V2.match(c)]
    if sc.name not in KEEPS_DEVICE:
        cols += [c for c in rows.columns if IDENTITY_COLS.match(c)]
    picks, tiers = [], {"own": 0, "pool": 0}
    for i in range(len(rows)):
        product = str(rows["ProductCD"].iat[i])
        if sc.name in NEW_CUSTOMER:
            cand = np.flatnonzero(pools["young"] & (pools["product"] == product))
            tiers["pool"] += 1
        else:
            own = store.uid_rows.get(int(sc.uids[i]), np.array([], dtype=np.int64))
            own = own[pools["val"][own]]
            if sc.name != "mimicry":
                own = own[pools["product"][own] == product]
            if len(own):
                cand = own
                tiers["own"] += 1
            else:
                cand = np.flatnonzero(pools["existing"] & (pools["product"] == product))
                tiers["pool"] += 1
        picks.append(int(rng.choice(cand)))
    src = df.iloc[picks].reset_index(drop=True)
    for c in cols:
        rows[c] = src[c].to_numpy()
    if sc.name == "mimicry":  # the customer's own product and device come with the sampled row
        rows["ProductCD"] = src["ProductCD"].to_numpy()
    ident = [c for c in rows.columns if IDENTITY_COLS.match(c)]
    rows["has_identity"] = rows[ident].notna().any(axis=1).astype(np.int8)
    rows = rows.astype(store.df.dtypes.to_dict())
    _resolve(store, rows, sc.uids)
    info = dict(sc.info, generator="v2", masked_source=tiers,
                masked_source_transactions=[int(t) for t in src["TransactionID"]])  # fmt: skip
    return Scenario(sc.name, sc.seed, sc.intensity, sc.day, rows, sc.uids, sc.labels, info)


def _v2(gen):
    def run(store, seed=0, intensity="medium", day=None):
        return apply_v2(store, gen(store, seed=seed, intensity=intensity, day=day))

    run.__name__ = f"{gen.__name__}_v2"
    return run


GENERATORS_V2 = {name: _v2(gen) for name, gen in GENERATORS_V1.items()}
GENERATORS = GENERATORS_V2  # the primary generator since D203; v1 stays available and recorded
BY_VERSION = {"v1": GENERATORS_V1, "v2": GENERATORS_V2}
