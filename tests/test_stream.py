"""The stream scorer's history and messages: no future information (Sprint 3 step 3.5, D245).

On a hand-made store (a few training-period rows and stored rows after the replay start):
- the history holds only the pre-replay rows and the consumed events, never a stored later row;
- an event's view of the history changes when an EARLIER event changes and never when a LATER one changes
  or is removed (the plan's no-leakage test 1);
- out-of-order messages and sequence gaps stop the scorer (2);
- a label is invisible before its curtain, and a consumed row without a label stops the scorer once its
  curtain passes (3);
- the history refuses to be asked at a time before what it holds, and before() never returns a row at or
  after t (4);
- the growing index equals the offline EventIndex;
- a replay message carries only the raw columns and round-trips to the same row.
On a real Kafka (marker 'integration'): a replay through the broker comes back complete and in order.
The full real-data proof is the 10,000-transaction stream parity (research/tables/platform).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaultic.data.load import SECONDS_PER_DAY as DAY
from vaultic.features.graph import EventIndex
from vaultic.infra import stream as S

START = 151 * DAY


def _rows(spec):
    """spec: (day, second, card1, addr1, D1, email, device, fraud) per row."""
    out = []
    for i, (day, sec, card, addr, d1, email, device, fraud) in enumerate(spec):
        out.append({"TransactionID": 1000 + i, "TransactionDT": day * DAY + sec, "isFraud": fraud,
                    "TransactionAmt": 10.0 + i, "card1": float(card), "card2": 100.0, "card3": 150.0,
                    "card4": "visa", "card5": 226.0, "card6": "debit", "addr1": float(addr),
                    "addr2": 87.0, "P_emaildomain": email, "R_emaildomain": None,
                    "DeviceInfo": device, "DeviceType": "desktop" if device else None,
                    "D1": float(d1), "id_01": None, "day": day, "has_identity": int(bool(device))})  # fmt: skip
    df = pd.DataFrame(out).sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    for c in (
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceInfo",
        "DeviceType",
        "id_01",
    ):
        df[c] = pd.Categorical(df[c], categories=sorted({"visa", "debit", "a.com", "b.com", "dev1",
                                                         "dev2", "desktop", "x"}))  # fmt: skip
    df["isFraud"] = df["isFraud"].astype("int8")
    df["day"] = df["day"].astype("int16")
    df["has_identity"] = df["has_identity"].astype("int8")
    return df


def _store(df):
    from vaultic.data.splits import load_splits
    from vaultic.features.behavioral import ENTITIES, _entity_key
    from vaultic.live.store import HistoryStore, _positions_by_key, uid_key_strings

    keys = uid_key_strings(df)
    uid = pd.factorize(keys, sort=False)[0].astype(np.int64)
    t = df["TransactionDT"].to_numpy(dtype=np.int64)
    entity_keys = {n: _entity_key(df, cols) for n, cols in ENTITIES.items()}
    return HistoryStore(df=df, uid=uid, features=pd.DataFrame(columns=["f"]), splits=load_splits(),
                        uid_rows=_positions_by_key(uid.astype(object), t), uid_keys={},
                        key_to_uid=dict(zip(keys, uid, strict=True)), entity_keys=entity_keys,
                        entity_rows={n: _positions_by_key(k, t) for n, k in entity_keys.items()},
                        graph={}, structure={})  # fmt: skip


# training-period rows (pre-replay), then STORED rows inside the replay window: the stream history must
# never see the stored ones; it sees the consumed messages instead
BASE = [(1, 10, 7, 300, 0, "a.com", "dev1", 0), (2, 10, 7, 300, 1, "a.com", "dev1", 1),
        (3, 10, 8, 310, 0, "b.com", "dev2", 0), (100, 10, 9, 320, 0, "a.com", None, 1),
        (151, 50, 7, 300, 150, "a.com", "dev1", 1), (152, 10, 8, 310, 152 - 3, "b.com", "dev2", 1)]  # fmt: skip
# the stream: 10 events of day 151, several by the customer of card 7 / addr 300 (D1 = day - 1)
STREAM = [(151, 5 * i, 7 if i % 3 else 8, 300 if i % 3 else 310, (151 if i % 3 else 151 - 3) - (1 if i % 3 else 0),
           "a.com" if i % 2 else "b.com", "dev1" if i % 3 else "dev2", 0) for i in range(10)]  # fmt: skip


def _history(stream=STREAM):
    base = _store(_rows(BASE))
    h = S.StreamHistory.from_store(base, START, log=lambda *_: None)
    msgs = _rows(stream)
    return h, base, msgs


def _view(h, q):
    """What live/features.py reads for one transaction, through the history's own API."""
    from vaultic.features.behavioral import ENTITIES, _entity_key
    from vaultic.features.graph import build_edges

    t = int(q["TransactionDT"].iloc[0])
    uid = h.uid_for(q)
    hist = h.before(h.uid_rows.get(uid), t)
    rows = h.df.iloc[hist]
    ent = {}
    for name, cols in ENTITIES.items():
        key = _entity_key(q, cols)[0]
        ent[name] = len(h.before(h.entity_rows[name].get(key), t)) if key is not None else None
    e = build_edges(q.reset_index(drop=True), pd.Series([uid]))
    node = e["node"].to_numpy()
    times = np.full(len(e), t)
    return (tuple(rows["TransactionID"]), tuple(rows["TransactionAmt"]), tuple(sorted(ent.items())),
            tuple(h.graph["edge"].count_before(node, times)),
            tuple(h.graph["firsts"].count_before(node, times)))  # fmt: skip


def _play(stream, change=None):
    """Feed the stream; return each event's view, taken before the event is added."""
    h, _, msgs = _history(stream)
    views = []
    for i in range(len(msgs)):
        q = msgs.iloc[[i]].reset_index(drop=True)
        views.append(_view(h, q))
        h.consume(i, q, h.uid_for(q))
    return views


def test_the_history_holds_only_pre_replay_rows_and_consumed_events():
    h, base, msgs = _history()
    assert h.n_pre == 4 and len(base.df) == 6  # the two stored rows from day 151 on are not held
    assert int(h.time.max()) < START
    first = msgs.iloc[[0]].reset_index(drop=True)
    uid = h.uid_for(first)
    # the stored day-151 row of this customer (t = 151d + 50 s) exists in the base store, never here
    assert 1004 not in set(h.df.iloc[h.before(h.uid_rows.get(uid), START + 10**6)]["TransactionID"])
    assert h.next_uid > int(base.uid.max())  # new customers never get an offline code


def test_an_event_sees_earlier_events_and_never_later_ones():
    views = _play(STREAM)
    assert len(set(views)) > 3  # the views really differ along the stream
    # change an EARLIER event (event 4's amount): the later events of that customer see it
    changed = _play_with_amount(STREAM, 4, 999.0)
    assert changed[:5] == views[:5]  # event 4 itself and everything before it: unchanged
    assert any(a != b for a, b in zip(changed[5:], views[5:], strict=True))
    # change or remove a LATER event (event 7): nothing before it moves
    later = _play_with_amount(STREAM, 7, 12345.0)
    assert later[:8] == views[:8]
    removed = _play(STREAM[:7])
    assert removed == views[:7]


def _play_with_amount(stream, k, amount):
    h, _, msgs = _history(stream)
    msgs.loc[k, "TransactionAmt"] = amount
    views = []
    for i in range(len(msgs)):
        q = msgs.iloc[[i]].reset_index(drop=True)
        views.append(_view(h, q))
        h.consume(i, q, h.uid_for(q))
    return views


def test_out_of_order_messages_and_gaps_stop_the_scorer():
    h, _, msgs = _history()
    q = [msgs.iloc[[i]].reset_index(drop=True) for i in range(3)]
    h.consume(0, q[1], h.uid_for(q[1]))
    with pytest.raises(S.StreamOrderError, match="earlier"):
        h.consume(1, q[0], h.uid_for(q[0]))
    with pytest.raises(S.StreamOrderError, match="sequence 2 received, 1 expected"):
        h.consume(2, q[2], h.uid_for(q[2]))
    with pytest.raises(S.StreamOrderError, match="sequence 0 received"):
        h.consume(0, q[2], h.uid_for(q[2]))  # a repeat
    h2, _, _ = _history()
    with pytest.raises(S.StreamOrderError, match="before the replay start"):
        h2.check_next(0, START - 1)


def test_the_history_is_never_asked_about_a_time_before_what_it_holds():
    h, _, msgs = _history()
    q = msgs.iloc[[5]].reset_index(drop=True)
    h.consume(0, q, h.uid_for(q))
    t = int(q["TransactionDT"].iloc[0])
    with pytest.raises(S.StreamLeakError):
        h.before(h.uid_rows.get(h.uid_for(q)), t - 1)
    got = h.before(np.arange(len(h.time)), t)
    assert (h.time[got] < t).all()  # the event itself (time t) is not "before t"
    h.checks = True
    h._time._a[h._time.n - 1] = t - 10  # a corrupted (future-looking) row is caught, not used
    h.last_time = t - 10
    with pytest.raises(S.StreamLeakError):
        h.before(np.array([h.n_pre]), t - 20)


def test_a_label_is_invisible_before_its_curtain_and_a_missing_one_stops_the_scorer():
    h, _, msgs = _history()
    q = msgs.iloc[[0]].reset_index(drop=True)
    t0 = int(q["TransactionDT"].iloc[0])
    h.consume(0, q, h.uid_for(q))
    node = "uid:" + str(h.uid_for(q))
    L = h.label_delay * DAY
    h.label(1000, 1)
    cut = lambda t: t - L  # noqa: E731 - the feature code's label cutoff (labels known by t - L)
    assert (
        h.graph["labels"].sum_before([node], [cut(t0 + L)])[0] == 0
    )  # exactly at the curtain: not yet
    assert h.graph["labels"].sum_before([node], [cut(t0 + L + 1)])[0] == 1  # after it: known
    h2, _, _ = _history()
    h2.consume(0, q, h2.uid_for(q))
    h2.require_labels(t0 + L - 1)  # before the curtain: fine without a label
    with pytest.raises(S.CurtainError, match="label curtain"):
        h2.require_labels(t0 + L)
    h2.label(1000, 0)
    h2.require_labels(t0 + L)
    with pytest.raises(KeyError):
        h2.label(999, 1)


def test_the_growing_index_equals_the_offline_event_index():
    rng = np.random.default_rng(0)
    keys = rng.choice(["a", "b", "c", "d"], 300)
    times = np.sort(rng.integers(0, 1000, 300))
    vals = rng.integers(0, 2, 300).astype(float)
    ref = EventIndex(keys, times, vals)
    g = S.GrowingIndex()
    order = rng.permutation(300)  # also out of time order (late labels)
    for i in order:
        g.add(keys[i], times[i], vals[i])
    qk, qt = rng.choice(["a", "b", "c", "d", "z"], 200), rng.integers(-5, 1100, 200)
    assert (g.count_before(qk, qt) == ref.count_before(qk, qt)).all()
    assert np.allclose(g.sum_before(qk, qt), ref.sum_before(qk, qt))


def test_a_message_carries_only_raw_columns_and_round_trips_to_the_same_row():
    df = _rows(BASE)
    cols = S.raw_columns(df.columns)
    assert not {"isFraud", "day", "has_identity"} & set(cols)
    row = df.iloc[4]
    msg = S.decode(S.encode(7, "run1", row, cols))
    assert msg["seq"] == 7 and msg["run"] == "run1" and "isFraud" not in msg["tx"]
    back = S.row_from_message(msg["tx"], df.iloc[:0])
    expect = df.iloc[[4]].reset_index(drop=True).copy()
    expect["isFraud"] = np.int8(S.UNKNOWN_LABEL)  # the label never travels
    pd.testing.assert_frame_equal(back, expect)
    with pytest.raises(ValueError, match="must not carry"):
        S.decode(b'{"run": "r", "seq": 0, "tx": {"isFraud": 1}}')
    with pytest.raises(ValueError, match="missing"):
        S.row_from_message({k: v for k, v in msg["tx"].items() if k != "card1"}, df.iloc[:0])
    with pytest.raises(ValueError, match="unknown"):
        S.row_from_message({**msg["tx"], "secret": 1}, df.iloc[:0])


# ---- on a real Kafka -----------------------------------------------------------------------------------
@pytest.mark.integration
def test_a_replay_through_kafka_comes_back_complete_and_in_order():
    from vaultic.infra import kafka
    from vaultic.infra.replay import publish

    try:
        kafka.admin().list_topics(timeout=3)
    except Exception:  # noqa: BLE001 - no Kafka here
        pytest.skip("no Kafka reachable (start it with `docker compose --profile stream up -d`)")
    from confluent_kafka import KafkaError, TopicPartition

    topics = (
        "vaultic-test.tx.raw",
        "vaultic-test.tx.scored",
        "vaultic-test.alerts",
        "vaultic-test.feedback",
    )
    df = _rows(STREAM * 2)
    sent = publish(df, speed=None, topics=topics, log=lambda *_: None)
    c = kafka.consumer(f"vaultic-test-{sent['run']}")
    c.assign([TopicPartition(topics[0], 0, 0)])
    got = []
    try:
        while len(got) < len(df):
            m = c.poll(10.0)
            assert m is not None, "timed out"
            if m.error() and m.error().code() == KafkaError._PARTITION_EOF:
                continue
            got.append(S.decode(m.value()))
    finally:
        c.close()
        kafka.admin().delete_topics(list(topics))
    assert [m["seq"] for m in got] == list(range(len(df)))
    assert [m["tx"]["TransactionID"] for m in got] == df["TransactionID"].tolist()
    assert all("isFraud" not in m["tx"] for m in got)
