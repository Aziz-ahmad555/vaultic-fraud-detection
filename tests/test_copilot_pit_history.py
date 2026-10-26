"""The copilot's point-in-time customer history from the Test Lab's store (D228), on a hand-made
store of 10 rows with known answers: nothing at or after the case's time, labels only after the
delay, the same-card window, and how get_case / get_customer_history render it with the Test Lab
on and off."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from test_copilot_api import AS_OF, mount
from test_copilot_api import demo as demo  # noqa: F401  (fixture)

from vaultic.copilot.llm import ScriptedLLM
from vaultic.copilot.pit_history import DAY, from_testlab, store_history
from vaultic.copilot.tools import Ctx, get_case, get_customer_history

L = 30
T = 200 * DAY  # the case's time
HISTORY_Q = "Has this customer done something like this before?"


class FakeStore:
    """The parts of vaultic.live.store.HistoryStore that pit_history reads."""

    def __init__(self, rows):
        self.df = (
            pd.DataFrame(rows).sort_values("TransactionDT", kind="stable").reset_index(drop=True)
        )
        self.time = self.df["TransactionDT"].to_numpy(dtype=np.int64)
        self.uid = self.df["uid"].to_numpy(dtype=np.int64)
        self.uid_rows = {int(u): np.flatnonzero(self.uid == u) for u in np.unique(self.uid)}

    def before(self, positions, t):  # as HistoryStore.before: strictly before t
        if positions is None or not len(positions):
            return np.array([], dtype=np.int64)
        return positions[: np.searchsorted(self.time[positions], t, side="left")]


def row(
    tid,
    t,
    uid=1,
    amt=10.0,
    fraud=0,
    device="dev-a",
    card=(100, 200.0, 150.0, "visa", 226.0, "debit"),
):
    cards = dict(zip(("card1", "card2", "card3", "card4", "card5", "card6"), card, strict=True))
    return {"TransactionID": tid, "TransactionDT": t, "uid": uid, "TransactionAmt": amt,
            "isFraud": fraud, "DeviceInfo": device, **cards}  # fmt: skip


ROWS = [
    row(1, T - 100 * DAY, amt=20.0, fraud=1, device="dev-old"),  # matured fraud
    row(2, T - 40 * DAY, amt=30.0),  # matured legit
    row(3, T - 10 * DAY, amt=40.0, fraud=1),  # fraud, label not known yet at T
    row(4, T - 3600, amt=50.0),  # within the last 24 hours
    row(5, T, amt=999.0, fraud=1, device="dev-new"),  # same second as the case: never used
    row(6, T, amt=60.0, device="dev-a"),  # THE CASE
    row(7, T + 60, amt=999.0, fraud=1),  # after the case: never used
    row(8, T - 2 * DAY, uid=2),  # another customer, same card, inside 7 days
    row(9, T - 8 * DAY, uid=3),  # another customer, same card, outside 7 days
    row(10, T - DAY, uid=4, card=(101, 200.0, 150.0, "visa", 226.0, "debit")),  # another card
]


def case_position(store, tid=6):
    return int(np.flatnonzero(store.df["TransactionID"].to_numpy() == tid)[0])


def test_only_rows_strictly_before_the_case_time_are_used():
    store = FakeStore(ROWS)
    h = store_history(store, case_position(store), as_of=T, L=L)
    assert h.total == 4  # rows 1-4; not 5 (same second), not 7 (later)
    assert h.usual_amount == 35.0  # median of 20, 30, 40, 50: 999 never enters
    assert h.last_24h == 1  # row 4
    assert h.first_day == 100
    assert [e["tid"] for e in h.recent] == [4, 3, 2, 1]  # newest first, all strictly before T
    assert all(e["time"] < T for e in h.recent)
    assert h.device_known is True  # dev-a seen on rows 2-4; "dev-new" of row 5 never counted


def test_labels_are_known_only_after_the_delay():
    store = FakeStore(ROWS)
    h = store_history(store, case_position(store), as_of=T, L=L)
    assert h.known_fraud == 1  # row 1 only; row 3's fraud matures at T + 20 days
    labels = {e["tid"]: e["label"] for e in h.recent}
    assert labels == {4: None, 3: None, 2: 0, 1: 1}
    later = store_history(store, case_position(store), as_of=T + 20 * DAY, L=L)
    assert later.known_fraud == 2 and {e["tid"]: e["label"] for e in later.recent}[3] == 1
    assert later.total == 4  # a later as-of reveals labels, never later rows


def test_other_customers_on_the_same_card_in_the_last_7_days():
    store = FakeStore(ROWS)
    assert store_history(store, case_position(store), as_of=T, L=L).shared_card_7d == 1  # uid 2
    nan_card = [row(1, T - DAY, uid=1, card=(100, np.nan, 150.0, "visa", 226.0, None)),
                row(2, T - DAY, uid=2, card=(100, np.nan, 150.0, "visa", 226.0, None)),
                row(3, T, uid=1, card=(100, np.nan, 150.0, "visa", 226.0, None))]  # fmt: skip
    store = FakeStore(nan_card)
    h = store_history(store, case_position(store, 3), as_of=T, L=L)
    assert h.shared_card_7d == 1  # missing card fields match missing ones


def test_a_customer_without_earlier_rows_is_a_cold_start():
    store = FakeStore([row(1, T, uid=9), row(2, T + 5, uid=9)])
    h = store_history(store, case_position(store, 1), as_of=T, L=L)
    assert (h.total, h.usual_amount, h.device_known, h.first_day, h.recent) == (
        0,
        None,
        None,
        None,
        [],
    )


def test_the_history_is_read_only_through_a_ready_test_lab():
    store = FakeStore(ROWS)
    assert from_testlab(None, 6, T, L) is None
    assert from_testlab(SimpleNamespace(state="loading"), 6, T, L) is None
    lab = SimpleNamespace(state="ready", _ready=lambda: SimpleNamespace(store=store),
                          _position=lambda tid: case_position(store, tid))  # fmt: skip
    before = store.df.copy()
    assert from_testlab(lab, 6, T, L).total == 4
    pd.testing.assert_frame_equal(store.df, before)


# --- rendering in the copilot's evidence (the demo backend's synthetic bundle) ---


def demo_store(demo):
    """A store holding the demo case's customer: 3 rows before the bundle, plus the case itself."""
    case = demo.df[demo.df["TransactionID"] == demo.tid].iloc[0]
    t = int(case["TransactionDT"])
    rows = [row(901, t - 90 * DAY, uid=5, amt=12.0, fraud=1), row(902, t - 50 * DAY, uid=5, amt=14.0),
            row(903, t - 5 * DAY, uid=5, amt=16.0, fraud=1), row(int(demo.tid), t, uid=5)]  # fmt: skip
    store = FakeStore(rows)
    lab = SimpleNamespace(state="ready", _ready=lambda: SimpleNamespace(store=store),
                          _position=lambda tid: case_position(store, tid))  # fmt: skip
    return lab


def test_with_the_test_lab_on_the_customer_history_comes_from_the_store(demo):  # noqa: F811
    _, cop = mount(demo, ScriptedLLM("Nothing."), testlab=demo_store(demo))
    block = cop.start("question", demo.tid, HISTORY_Q, AS_OF, L).evidence.block
    assert "3 earlier transactions in total; usual amount 14.00; device seen before yes" in block
    assert "(point-in-time history store, Test Lab)" in block
    assert "known fraud among them: 1 (labels known at this time)" in block  # 903 not matured
    assert "earlier transaction 903:" in block and "outcome label not known yet" in block
    assert (
        "earlier transaction 901:" in block
        and "outcome label: fraud (known after the 30-day" in block
    )
    assert "no frozen decision (before the test period)" in block


def off_ctx(history_count, rows=()):
    case = {"amount": 60.0, "product": "W", "row": {"TransactionDT": T},
            "decision": {"action": "hold", "p": 0.5, "expected_loss": 1.0, "conformal_set": "{1}",
                         "disagreement": 0.1},
            "inputs": {"historyCount": history_count, "usualAmount": None, "deviceKnown": None,
                       "velocity24h": None, "sharedEntities": None},
            "entities": {"uid": "5"}, "label": None}  # fmt: skip
    read = SimpleNamespace(history=lambda uid, t, L: list(rows))
    return Ctx(read=read, tid=6, time=T, L=L, case=case, testlab=None)


def test_with_the_test_lab_off_missing_values_say_why():
    ctx = off_ctx(20)
    cust = get_case(ctx)[1].text
    assert "20 earlier transactions in total; usual amount not available" in cust
    assert cust.endswith("Not available: the Test Lab is off and the earlier rows are outside the "
                         "test-period data.")  # fmt: skip
    hist = get_customer_history(ctx)[0].text
    assert (
        "20 earlier transactions in total (all periods); none of them in the test-period data"
        in hist
    )
    assert "history store, which is off" in hist
    cold = off_ctx(0)
    assert "Not available" not in get_case(cold)[1].text  # nothing earlier: nothing to explain
    assert "which is off" not in get_customer_history(cold)[0].text
