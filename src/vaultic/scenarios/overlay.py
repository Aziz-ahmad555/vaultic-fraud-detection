"""History overlay: the Test Lab's history store plus a scenario's synthetic rows (Phase B).

The live features read history only through the store (live/features.py): the customer's earlier
rows, the entity rows of the 7 days before t, the graph EventIndex queries and the day's
window-graph snapshot. `with_synthetic` returns a store that answers every one of those queries
as if the synthetic rows had been in the data, without copying the stored rows:

  rows        synthetic rows get positions after the stored ones; a row frame proxy serves
              .iloc over both (live/features._slice)
  indexes     customer and entity position lists are merged in time order for the keys the
              synthetic rows touch
  graph       each EventIndex is wrapped: stored counts / sums plus the synthetic events before
              the cutoff; a (node, customer) pair's first appearance moves earlier when a
              synthetic row precedes the stored first one (+1 at the synthetic time, -1 at the
              stored first time), so distinct-customer counts stay exact
  labels      synthetic labels enter the known-label index at their own time (used only from
              t + 30 days, the label delay, like every label)
  snapshots   the day's window-graph partitions are recomputed with the synthetic rows for every
              day after the first synthetic day that the scenario reaches (D194); a same-day
              scenario changes no snapshot (a day's snapshot uses only earlier days)

Because every feature uses rows strictly before t, all of a scenario's synthetic rows are added
at once; each row still sees only the synthetic rows before it. Synthetic times are distinct
from each other.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY


class OverlayFrame:
    """Read-only stand-in for store.df over stored + synthetic rows (positions >= n_base are
    synthetic). Supports what the live features use: .iloc[positions] and column access."""

    def __init__(self, base: pd.DataFrame, extra: pd.DataFrame):
        self.base, self.extra, self.n_base = base, extra.reset_index(drop=True), len(base)
        self.iloc = _ILoc(self)
        self.columns, self.dtypes = base.columns, base.dtypes

    def __len__(self) -> int:
        return self.n_base + len(self.extra)

    def __getitem__(self, col):
        return pd.concat([self.base[col], self.extra[col]], ignore_index=True)


class _ILoc:
    def __init__(self, frame: OverlayFrame):
        self.f = frame

    def __getitem__(self, positions):
        pos = np.atleast_1d(np.asarray(positions, dtype=np.int64))
        syn = pos >= self.f.n_base
        parts = [self.f.base.iloc[pos[~syn]], self.f.extra.iloc[pos[syn] - self.f.n_base]]
        out = pd.concat(parts)
        order = np.concatenate([np.flatnonzero(~syn), np.flatnonzero(syn)])
        out = out.iloc[np.argsort(order, kind="stable")]
        return out


class OverlayIndex:
    """EventIndex + extra weighted events: count_before / sum_before add the extra events with
    the same key and time < cutoff (count weight, value)."""

    def __init__(self, base, keys, times, weights=None, values=None):
        self.base = base
        self.keys = np.asarray(keys, dtype=object)
        self.times = np.asarray(times, dtype=np.int64)
        n = len(self.keys)
        self.weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        self.values = np.zeros(n) if values is None else np.asarray(values, dtype=float)

    def _extra(self, keys, cutoff, what):
        keys = np.asarray(keys, dtype=object)
        cutoff = np.broadcast_to(np.asarray(cutoff, dtype=np.int64), keys.shape)
        if not len(self.keys):
            return np.zeros(len(keys))
        match = (keys[:, None] == self.keys[None, :]) & (self.times[None, :] < cutoff[:, None])
        return (match * what[None, :]).sum(axis=1)

    def count_before(self, keys, cutoff) -> np.ndarray:
        return self.base.count_before(keys, cutoff) + self._extra(keys, cutoff, self.weights)

    def sum_before(self, keys, cutoff) -> np.ndarray:
        return self.base.sum_before(keys, cutoff) + self._extra(keys, cutoff, self.values)


def _merge_positions(index: dict, keys, positions, times_all) -> dict:
    """Copy of a key -> positions dict with the new positions merged in time order."""
    out = dict(index)
    s = pd.Series(np.asarray(positions), index=pd.Index(keys, dtype=object))
    s = s[s.index.notna()]
    for k, g in s.groupby(level=0, sort=False):
        old = out.get(k, np.array([], dtype=np.int64))
        merged = np.concatenate([old, g.to_numpy()])
        order = np.lexsort((merged, times_all[merged]))
        out[k] = merged[order]
    return out


def with_synthetic(store, rows: pd.DataFrame, uids: np.ndarray, log=print):
    """A store whose history includes the synthetic rows (rows: stored columns and dtypes)."""
    from vaultic.features.behavioral import ENTITIES, _entity_key
    from vaultic.features.graph import build_edges
    from vaultic.live.store import _structure_snapshot

    rows = rows.astype(store.df.dtypes.to_dict()).reset_index(drop=True)
    t_syn = rows["TransactionDT"].to_numpy(dtype=np.int64)
    if len(np.unique(t_syn)) != len(t_syn):
        raise ValueError("synthetic rows need distinct times")
    uids = np.asarray(uids, dtype=store.uid.dtype)
    n = len(store.df)
    pos = np.arange(n, n + len(rows))
    time_all = np.concatenate([store.time, t_syn])

    from vaultic.live.store import HistoryStore

    cls = type("OverlayStore", (_Overlay, HistoryStore), {})
    out = cls(**{f.name: getattr(store, f.name) for f in fields(HistoryStore)})
    out.df = OverlayFrame(store.df, rows)
    out.uid = np.concatenate([store.uid, uids])
    out._time = time_all
    out.uid_rows = _merge_positions(store.uid_rows, uids.astype(object), pos, time_all)
    out.entity_keys, out.entity_rows = dict(store.entity_keys), dict(store.entity_rows)
    for name, cols in ENTITIES.items():
        k = _entity_key(rows, cols)
        out.entity_keys[name] = np.concatenate([store.entity_keys[name], k]).astype(object)
        out.entity_rows[name] = _merge_positions(store.entity_rows[name], k, pos, time_all)
    out.info = dict(store.info, next_uid=max(int(store.info.get("next_uid", 0)),
                                             int(out.uid.max()) + 1))  # fmt: skip

    edges = build_edges(rows, pd.Series(uids))
    node, et = edges["node"].to_numpy(), edges["time"].to_numpy(dtype=np.int64)
    pair = (edges["node"] + "\x1f" + edges["uid"]).to_numpy()
    g = dict(store.graph)
    g["edge"] = OverlayIndex(store.graph["edge"], node, et)
    g["pair"] = OverlayIndex(store.graph["pair"], pair, et)
    g["labels"] = OverlayIndex(store.graph["labels"], node, et, values=edges["label"].to_numpy())
    fk, ft, fw = first_events(store.graph["pair"], node, pair, et)
    g["firsts"] = OverlayIndex(store.graph["firsts"], fk, ft, weights=fw)
    se = edges[edges["type"].isin(("uid", "card", "device"))]
    se = se.assign(row=se["row"].to_numpy() + n)[["row", "type", "key", "time", "uid", "label",
                                                   "node"]]  # fmt: skip
    g["struct_edges"] = pd.concat([store.graph["struct_edges"], se], ignore_index=True)
    out.graph = g

    out.structure = dict(store.structure)
    first_day = int(t_syn.min() // SECONDS_PER_DAY)
    for d in sorted(store.structure):
        if d > first_day:  # D194: days after a synthetic row see it in their window
            log(f"recomputing window-graph snapshot for day {d} with synthetic rows")
            out.structure[d] = _structure_snapshot(g["struct_edges"], d,
                                                   store.splits.label_delay_days)  # fmt: skip
    return out


def first_events(base_pair, node, pair, times) -> tuple[list, list, list]:
    """Extra (node, time, weight) events for the first-appearance index: +1 at a pair's first
    synthetic time when the pair had not appeared before it in the stored data, and -1 at the
    pair's stored first time when that comes later (it is no longer the first)."""
    fk, ft, fw = [], [], []
    first = pd.DataFrame({"node": node, "pair": pair, "t": np.asarray(times, dtype=np.int64)})
    first = first.groupby("pair", sort=True).agg(node=("node", "first"), t=("t", "min"))
    never = np.iinfo(np.int64).max // 4
    for p, r in first.iterrows():
        t = int(r["t"])
        if base_pair.count_before([p], [t])[0] > 0:
            continue  # the pair already appeared before its first synthetic row
        fk.append(r["node"]), ft.append(t), fw.append(1.0)
        if base_pair.count_before([p], [never])[0] > 0:
            fk.append(r["node"]), ft.append(_first_time(base_pair, p)), fw.append(-1.0)
    return fk, ft, fw


def _first_time(index, key) -> int:
    """Time of the first stored event of key in an EventIndex."""
    code = index.index.get_indexer([key])[0]
    return int(index.times_sorted[index.start[code]])


class _Overlay:
    """Mixin: the store's row times come from the merged array (stored + synthetic)."""

    @property
    def time(self) -> np.ndarray:
        return self._time


__all__ = ["OverlayFrame", "OverlayIndex", "with_synthetic"]
