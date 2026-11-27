"""The stream scorer's history: only the pre-replay rows and the events consumed so far (Sprint 3 step 3.5,
D245).

StreamHistory answers every query the Test Lab's online feature code (live/features.py) makes of a history
store, but it holds, by construction, nothing else:

- the PRE-REPLAY history: the stored rows strictly before the replay's first event (labels included; the
  feature code applies the label delay to them as offline);
- the CONSUMED events: rebuilt from their stream messages (raw IEEE-CIS columns only), with an unknown
  label until one arrives on the feedback topic.

Its customer and entity indexes, its uid-key map and its graph event indexes (the pre-replay EventIndex plus
a growing index of consumed events) are built from those two parts only; a day's window-graph partitions are
computed from them when the day's first event is scored. New customers get new uid codes, never an offline
code. Guards (D245, 5):
- consume() refuses a sequence gap and an earlier TransactionDT than the last one (StreamOrderError);
- before() refuses to answer at a time earlier than the newest consumed event and, with `checks` on, asserts
  that no returned row has time >= t (StreamLeakError);
- a label is visible only when its_time + L <= t (the queries' label cutoff), and require_labels() stops the
  scorer when a consumed row without a label passes that curtain (CurtainError): a missing label would read
  as "legitimate".

The module lives in vaultic.infra, not vaultic.live: the Test Lab's store cache is keyed by the code under
live/ (D221), and the stream history does not change that store.
"""

from __future__ import annotations

import bisect
import json
import math
from collections import deque

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY

DERIVED = ("day", "has_identity")  # computed by the consumer, never sent
NOT_SENT = ("isFraud", *DERIVED)  # no label leaves the producer
UNKNOWN_LABEL = 0  # placeholder in the int isFraud column; never read before the curtain (guarded)


class StreamOrderError(RuntimeError):
    """A message out of order or a gap in the sequence: the scorer stops (D245)."""


class StreamLeakError(AssertionError):
    """The history was asked something that would need information from after t (D245)."""


class CurtainError(RuntimeError):
    """A consumed row passed its label curtain without a label (D245)."""


# ---- indexes -----------------------------------------------------------------------------------------
class GrowingIndex:
    """Per-key events added one at a time; count and sum of events with time < cutoff, as EventIndex."""

    def __init__(self):
        self.times: dict = {}
        self.cum: dict = {}

    def add(self, key, time: int, value: float = 1.0) -> None:
        ts = self.times.setdefault(key, [])
        cs = self.cum.setdefault(key, [])
        if not ts or int(time) >= ts[-1]:  # in time order: the stream's normal case
            ts.append(int(time))
            cs.append((cs[-1] if cs else 0.0) + float(value))
            return
        i = bisect.bisect_right(ts, int(time))  # a late label: insert, then redo the running sums
        ts.insert(i, int(time))
        vals = np.diff([0.0, *cs]).tolist()
        vals.insert(i, float(value))
        self.cum[key] = np.cumsum(vals).tolist()

    def _pos(self, key, cutoff) -> int:
        ts = self.times.get(key)
        return 0 if not ts else bisect.bisect_left(ts, int(cutoff))

    def count_before(self, keys, cutoff) -> np.ndarray:
        keys, cutoff = _broadcast(keys, cutoff)
        return np.array(
            [self._pos(k, c) for k, c in zip(keys, cutoff, strict=True)], dtype=np.int64
        )

    def sum_before(self, keys, cutoff) -> np.ndarray:
        keys, cutoff = _broadcast(keys, cutoff)
        out = np.zeros(len(keys))
        for j, (k, c) in enumerate(zip(keys, cutoff, strict=True)):
            i = self._pos(k, c)
            if i:
                out[j] = self.cum[k][i - 1]
        return out


def _broadcast(keys, cutoff):
    keys = np.asarray(keys, dtype=object)
    cutoff = np.broadcast_to(np.asarray(cutoff, dtype=np.int64), keys.shape)
    return keys, cutoff


class CombinedIndex:
    """The pre-replay EventIndex plus the growing index of consumed events."""

    def __init__(self, pre, grow: GrowingIndex):
        self.pre, self.grow = pre, grow

    def count_before(self, keys, cutoff) -> np.ndarray:
        return self.pre.count_before(keys, cutoff) + self.grow.count_before(keys, cutoff)

    def sum_before(self, keys, cutoff) -> np.ndarray:
        return self.pre.sum_before(keys, cutoff) + self.grow.sum_before(keys, cutoff)


class _Growing:
    """A numpy array that grows by appending (amortised doubling)."""

    def __init__(self, initial: np.ndarray):
        self._a = np.array(initial, copy=True)
        self.n = len(initial)

    def append(self, value) -> None:
        if self.n == len(self._a):
            self._a = np.concatenate(
                [self._a, np.empty(max(1024, len(self._a)), dtype=self._a.dtype)]
            )
        self._a[self.n] = value
        self.n += 1

    def __getitem__(self, idx):
        return self._a[: self.n][idx]

    def __len__(self):
        return self.n

    def view(self) -> np.ndarray:
        return self._a[: self.n]


class _Positions:
    """key -> time-sorted row positions: the pre-replay ones (from the stored index, cut at n_pre) and
    the consumed ones."""

    def __init__(self, stored: dict, n_pre: int):
        self.stored, self.n_pre, self.new = stored, n_pre, {}

    def get(self, key, default=None):
        pre = self.stored.get(key)
        if pre is not None and len(pre):
            pre = pre[: np.searchsorted(pre, self.n_pre, side="left")]  # positions rise with time
        new = self.new.get(key)
        if new:
            pre = np.concatenate([pre if pre is not None else np.empty(0, np.int64), new])
        if pre is None or not len(pre):
            return default
        return pre

    def add(self, key, pos: int) -> None:
        self.new.setdefault(key, []).append(pos)


class _Rows:
    """`df.iloc[positions]` over the pre-replay rows and the consumed rows (same columns and dtypes)."""

    def __init__(self, pre: pd.DataFrame):
        self.pre, self.n_pre, self.consumed = pre, len(pre), []

    @property
    def iloc(self):
        return self

    @property
    def columns(self):
        return self.pre.columns

    @property
    def dtypes(self):
        return self.pre.dtypes

    def __getitem__(self, positions) -> pd.DataFrame:
        positions = np.atleast_1d(np.asarray(positions, dtype=np.int64))
        old = positions[positions < self.n_pre]
        parts = [self.pre.iloc[old]] if len(old) else [self.pre.iloc[:0]]
        parts += [self.consumed[p - self.n_pre] for p in positions[positions >= self.n_pre]]
        return pd.concat(parts) if len(parts) > 1 else parts[0]


class _Structure:
    """Day -> the window-graph partitions, from the visible edges only, computed once per day."""

    def __init__(self, history: StreamHistory, stored: dict):
        self.h, self.stored, self.days = history, stored, {}

    def get(self, day, default=None):
        day = int(day)
        if day not in self.days:
            self.days[day] = self.h._snapshot(day)
        return self.days[day] if self.days[day] is not None else default


# ---- the history ---------------------------------------------------------------------------------------
class StreamHistory:
    """See the module docstring. Built with from_store(); fed with consume() and label()."""

    def __init__(self):
        self.checks = True

    @classmethod
    def from_store(cls, base, start_time: int, log=print) -> StreamHistory:
        """The pre-replay part of a full HistoryStore (live/store.py): its rows strictly before
        start_time, with indexes built from those rows only."""
        from vaultic.live.store import _graph_indexes, uid_key_strings

        h = cls()
        h.start_time = int(start_time)
        n_pre = int(np.searchsorted(base.time, h.start_time, side="left"))
        h.n_pre = n_pre
        pre = base.df.iloc[:n_pre]
        h.df = _Rows(pre)
        h._uid = _Growing(np.asarray(base.uid[:n_pre], dtype=np.int64))
        h._time = _Growing(np.asarray(base.time[:n_pre], dtype=np.int64))
        h.uid_rows = _Positions(base.uid_rows, n_pre)
        h.entity_keys = {n: _Growing(np.asarray(k[:n_pre], dtype=object))
                         for n, k in base.entity_keys.items()}  # fmt: skip
        h.entity_rows = {n: _Positions(r, n_pre) for n, r in base.entity_rows.items()}
        h.key_to_uid = dict(zip(uid_key_strings(pre), h._uid.view().tolist(), strict=True))
        h.next_uid = int(np.max(base.uid)) + 1  # never an offline code
        log(f"stream history: {n_pre:,} pre-replay rows (before t = {h.start_time})")
        pre_graph = _graph_indexes(pre.reset_index(drop=True), h._uid.view(), base.splits, log)
        h.grow = {k: GrowingIndex() for k in ("edge", "firsts", "pair", "labels")}
        h.graph = {k: CombinedIndex(pre_graph[k], h.grow[k]) for k in h.grow}
        h.graph["hub_thresholds"] = pre_graph["hub_thresholds"]
        h._pre_pairs = pre_graph["pair"]
        h._pre_struct = pre_graph["struct_edges"]
        h._new_struct: list[pd.DataFrame] = []
        h._new_pairs: set = set()
        h._row_edges: dict[int, pd.DataFrame] = {}
        h.structure = _Structure(h, base.structure)
        h._stored_structure = base.structure
        h.features = base.features.iloc[:0]  # the column list only, never a row
        h.splits, h.encoders, h.product_encoder = base.splits, base.encoders, base.product_encoder
        h.info = {"start_time": h.start_time, "pre_rows": n_pre}
        h.last_time, h.last_seq = None, None
        h.by_tid: dict[int, int] = {}
        h.unlabeled: deque = deque()  # (time, pos) of consumed rows without a label, in time order
        h.labeled: set = set()
        h.label_delay = int(base.splits.label_delay_days)
        return h

    # the attributes live/features.py reads
    @property
    def uid(self) -> np.ndarray:
        return self._uid.view()

    @property
    def time(self) -> np.ndarray:
        return self._time.view()

    def before(self, positions, t: int) -> np.ndarray:
        """Positions (time-sorted) with time strictly before t; never anything at or after t."""
        t = int(t)
        if self.last_time is not None and self.last_time > t:
            raise StreamLeakError(
                f"asked at t = {t}, but the history already holds t = {self.last_time}"
            )
        if positions is None or not len(positions):
            return np.array([], dtype=np.int64)
        positions = np.asarray(positions, dtype=np.int64)
        k = np.searchsorted(self.time[positions], t, side="left")
        out = positions[:k]
        if self.checks and len(out) and int(self.time[out].max()) >= t:
            raise StreamLeakError(f"a row at or after t = {t} would be used")
        return out

    def new_uid(self) -> int:
        code = self.next_uid
        self.next_uid += 1
        return code

    def uid_for(self, query: pd.DataFrame) -> int:
        """The customer of a message: a known key's code, else a new code (never an offline code)."""
        from vaultic.live.store import uid_key_strings

        key = uid_key_strings(query)[0]
        if key not in self.key_to_uid:
            self.key_to_uid[key] = self.new_uid()
        return int(self.key_to_uid[key])

    # ---- feeding ---------------------------------------------------------------------------------------
    def check_next(self, seq: int, t: int) -> None:
        """Refuse a gap or an earlier time, BEFORE the event is scored."""
        expected = 0 if self.last_seq is None else self.last_seq + 1
        if int(seq) != expected:
            raise StreamOrderError(f"sequence {seq} received, {expected} expected: a message is missing "
                                   "or repeated; the scorer stops")  # fmt: skip
        if self.last_time is not None and int(t) < self.last_time:
            raise StreamOrderError(f"TransactionDT {t} is earlier than the last one ({self.last_time}); "
                                   "the scorer stops")  # fmt: skip
        if int(t) < self.start_time:
            raise StreamOrderError(
                f"TransactionDT {t} is before the replay start ({self.start_time})"
            )

    def require_labels(self, t: int) -> None:
        """Stop if a consumed row without a label has passed its curtain (its_time + L <= t)."""
        while self.unlabeled and self.unlabeled[0][1] in self.labeled:
            self.unlabeled.popleft()
        if self.unlabeled and self.unlabeled[0][0] + self.label_delay * SECONDS_PER_DAY <= int(t):
            tid = int(self.df.consumed[self.unlabeled[0][1] - self.n_pre]["TransactionID"].iloc[0])
            raise CurtainError(f"transaction {tid} passed its {self.label_delay}-day label curtain without "
                               "a label on the feedback topic; scoring on would read it as legitimate")  # fmt: skip

    def consume(self, seq: int, query: pd.DataFrame, uid_code: int) -> int:
        """Add an already scored event to the history; returns its position."""
        from vaultic.features.behavioral import ENTITIES, _entity_key
        from vaultic.features.graph import COMPONENT_TYPES, GRAPH_INPUT_COLUMNS, build_edges

        t = int(query["TransactionDT"].iloc[0])
        self.check_next(seq, t)
        pos = self.n_pre + len(self.df.consumed)
        row = query.reset_index(drop=True).copy()
        row["isFraud"] = np.asarray([UNKNOWN_LABEL], dtype=row["isFraud"].dtype)
        row.index = [pos]
        self.df.consumed.append(row)
        self._uid.append(int(uid_code))
        self._time.append(t)
        self.uid_rows.add(int(uid_code), pos)
        for name, cols in ENTITIES.items():
            key = _entity_key(row, cols)[0]
            self.entity_keys[name].append(key)
            if key is not None:
                self.entity_rows[name].add(key, pos)
        g = row[[c for c in GRAPH_INPUT_COLUMNS if c in row]].reset_index(drop=True)
        edges = build_edges(g, pd.Series([uid_code]))
        edges["label"] = np.nan  # unknown until the feedback topic says otherwise
        edges["row"] = pos
        for node, uid in zip(edges["node"], edges["uid"], strict=True):
            pair = node + "\x1f" + uid
            self.grow["edge"].add(node, t)
            self.grow["pair"].add(pair, t)
            seen = self._pre_pairs.count_before([pair], [2**62])[0] > 0 or pair in self._new_pairs
            if not seen:
                self.grow["firsts"].add(node, t)
                self._new_pairs.add(pair)
        self._row_edges[pos] = edges
        struct = edges[edges["type"].isin(COMPONENT_TYPES)]
        self._new_struct.append(struct[self._pre_struct.columns])
        self.by_tid[int(row["TransactionID"].iloc[0])] = pos
        self.unlabeled.append((t, pos))
        self.last_time, self.last_seq = t, int(seq)
        return pos

    def label(self, tid: int, label: int) -> None:
        """A label from the feedback topic. It is visible to a query at time t only when
        its_time + L <= t: the feature code's label cutoffs, applied to the event's own time."""
        pos = self.by_tid.get(int(tid))
        if pos is None:
            raise KeyError(f"transaction {tid} has not been consumed")
        if pos in self.labeled:
            return
        label = int(label)
        if label not in (0, 1):
            raise ValueError("a label is 0 or 1")
        row = self.df.consumed[pos - self.n_pre]
        row["isFraud"] = np.asarray([label], dtype=row["isFraud"].dtype)
        edges = self._row_edges[pos]
        t = int(row["TransactionDT"].iloc[0])
        for node in edges["node"]:
            self.grow["labels"].add(node, t, float(label))
        edges["label"] = float(label)
        for part in self._new_struct:
            if len(part) and (part["row"] == pos).any():
                part.loc[part["row"] == pos, "label"] = float(label)
        self.labeled.add(pos)

    # ---- window graphs ---------------------------------------------------------------------------------
    def _snapshot(self, day: int):
        """Day d's partitions over the visible edges of days [d - 30, d) (the offline code). When the whole
        window is pre-replay the stored snapshot is the same computation on the same rows."""
        from vaultic.live.store import _structure_snapshot

        if day * SECONDS_PER_DAY <= self.start_time and day in self._stored_structure:
            return self._stored_structure[day]
        struct = pd.concat([self._pre_struct, *self._new_struct], ignore_index=True)
        if self.checks and len(struct) and int(struct["time"].max()) >= (day + 1) * SECONDS_PER_DAY:
            raise StreamLeakError(
                f"the window graph of day {day} was asked after later days were held"
            )
        return _structure_snapshot(struct, day, self.label_delay)


# ---- messages ------------------------------------------------------------------------------------------
def raw_columns(columns) -> list[str]:
    """The columns a replay message carries: every stored IEEE-CIS column except the label and the
    derived day and has_identity (D245)."""
    return [c for c in columns if c not in NOT_SENT]


def encode(seq: int, run: str, row: pd.Series, columns) -> bytes:
    """One replay message: the run id, the sequence number and the raw columns (NaN as null)."""
    values = {}
    for c in columns:
        v = row[c]
        if v is None or (isinstance(v, float) and math.isnan(v)) or (v is pd.NA) or (v is pd.NaT):
            values[c] = None
        elif isinstance(v, np.integer | int) and not isinstance(v, bool):
            values[c] = int(v)
        elif isinstance(v, np.floating | float):
            f = float(v)
            values[c] = None if math.isnan(f) else f
        else:
            values[c] = str(v)
    return json.dumps({"run": run, "seq": int(seq), "tx": values}, allow_nan=False).encode()


def decode(payload: bytes) -> dict:
    msg = json.loads(payload)
    if set(msg) != {"run", "seq", "tx"}:
        raise ValueError("not a replay message")
    if any(c in msg["tx"] for c in NOT_SENT):
        raise ValueError(f"a replay message must not carry {', '.join(NOT_SENT)}")
    return msg


def row_from_message(tx: dict, template: pd.DataFrame) -> pd.DataFrame:
    """The message's transaction as one row with the stored columns and dtypes; day and has_identity are
    derived, the label is unknown (UNKNOWN_LABEL)."""
    from vaultic.live.custom import IDENTITY

    unknown = sorted(set(tx) - set(template.columns))
    if unknown:
        raise ValueError(f"unknown columns in a replay message: {unknown[:10]}")
    missing = sorted(set(raw_columns(template.columns)) - set(tx))
    if (
        missing
    ):  # never filled in silently: a column the producer did not send would read as missing
        raise ValueError(f"columns missing from a replay message: {missing[:10]}")
    cols = {}
    for c, dtype in template.dtypes.items():
        if c in NOT_SENT:
            continue
        v = tx.get(c)
        if isinstance(dtype, pd.CategoricalDtype):
            cols[c] = pd.Categorical([v], categories=dtype.categories)
        else:
            cols[c] = [np.nan if v is None else v]
    row = pd.DataFrame(cols)
    row["day"] = int(row["TransactionDT"].iloc[0]) // SECONDS_PER_DAY
    ident = [c for c in row.columns if IDENTITY.match(c)]
    row["has_identity"] = int(row[ident].notna().any(axis=1).iloc[0])
    row["isFraud"] = UNKNOWN_LABEL
    return row[template.columns].astype(template.dtypes.to_dict())
