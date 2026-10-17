"""The stored history the live features are computed from (D182).

Built once from the same data files the frozen final run read (stage2/system.load_parts: the
merged IEEE-CIS rows, the uid reconstruction and every offline feature file) and cached on drive
E. Everything is indexed for "strictly before t" queries:

  customer rows    per uid, row positions in time order (base, behavioral, sequence features)
  entity rows      per behavioral entity key (card, email, device), positions in time order
                   (the 7-day distinct-uid counts)
  graph events     the offline graph code's own EventIndex structures over every stored edge
                   (degree, distinct uids, own-pair, known labels) and the training-period hub
                   thresholds (setting C, label delay L = 30)
  structure        per test day d, the window graph of days [d - 30, d) partitioned with the
                   offline code (components, Louvain, seed 0) and its group statistics (labels
                   known by the day's start, D50): looked up, never recomputed per request
  encoders         the training-period frequency encoders and ProductCD encoder

The stored test-period rows are opened only behind the post-hoc guard (a completed final run on
record, D144), as the demo backend does (D154).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("E:/dev-cache/testlab")  # never in the repository
SEED = 0
WINDOW_DAYS = 30


def _positions_by_key(keys: np.ndarray, times: np.ndarray) -> dict:
    """key -> row positions sorted by (time, position); None keys are skipped."""
    s = pd.Series(np.arange(len(keys)), index=pd.Index(keys, dtype=object))
    s = s[s.index.notna()]
    order = np.lexsort((s.to_numpy(), times[s.to_numpy()]))
    s = s.iloc[order]
    return {k: g.to_numpy() for k, g in s.groupby(level=0, sort=False)}


@dataclass
class HistoryStore:
    df: pd.DataFrame  # merged rows, sorted by TransactionDT (as stored)
    uid: np.ndarray  # int uid code per row
    features: pd.DataFrame  # the offline features of every row (parity reference)
    splits: object
    uid_rows: dict
    uid_keys: dict  # uid code -> uid key string (card1_addr1_D1n tokens)
    key_to_uid: dict  # uid key string -> uid code
    entity_keys: dict  # behavioral entity type -> key per row
    entity_rows: dict  # behavioral entity type -> {key: positions}
    graph: dict  # EventIndex structures and hub thresholds
    structure: dict = field(default_factory=dict)  # day -> {partition: (node_group, stats)}
    encoders: dict = field(default_factory=dict)
    product_encoder: object = None
    info: dict = field(default_factory=dict)

    @property
    def time(self) -> np.ndarray:
        return self.df["TransactionDT"].to_numpy(dtype=np.int64)

    def before(self, positions: np.ndarray, t: int) -> np.ndarray:
        """Positions (time-sorted) with time strictly before t."""
        if positions is None or not len(positions):
            return np.array([], dtype=np.int64)
        k = np.searchsorted(self.time[positions], t, side="left")
        return positions[:k]

    def new_uid(self) -> int:
        code = int(self.info.get("next_uid", int(self.uid.max()) + 1))
        self.info["next_uid"] = code + 1
        return code


def uid_key_strings(df: pd.DataFrame) -> np.ndarray:
    """The uid variant's key string per row, exactly as data/uid.build_uids forms it before
    factorising (card1 + addr1 + (day - D1))."""
    from vaultic.data.uid import D1N, VARIANTS, _token

    parts = df.assign(**{D1N: df["day"] - df["D1"]})
    cols = VARIANTS["uid"]
    key = _token(parts[cols[0]])
    for col in cols[1:]:
        key = key + "_" + _token(parts[col])
    return key.to_numpy(dtype=object)


def _graph_indexes(df: pd.DataFrame, uid: np.ndarray, splits, log) -> dict:
    """The offline graph code's setting-C event indexes over every stored edge."""
    from vaultic.features.graph import (
        GRAPH_INPUT_COLUMNS,
        EventIndex,
        build_edges,
        fit_hub_thresholds_on_training_period,
    )
    from vaultic.features.pit import label_cutoff  # noqa: F401  (used at query time)

    g = df[[c for c in GRAPH_INPUT_COLUMNS if c in df]].reset_index(drop=True)
    u = pd.Series(uid)
    edges = build_edges(g, u)
    log(f"graph: {len(edges):,} edges")
    node, etime, label = (
        edges["node"].to_numpy(),
        edges["time"].to_numpy(),
        edges["label"].to_numpy(),
    )
    pair = (edges["node"] + "\x1f" + edges["uid"]).to_numpy()
    first = edges.assign(pair=pair).groupby("pair")["time"].transform("min").to_numpy()
    is_first = edges.assign(pair=pair, first=first).drop_duplicates("pair").index
    out = {
        "edge": EventIndex(node, etime),
        "firsts": EventIndex(node[is_first], first[is_first]),
        "pair": EventIndex(pair, etime),
        "labels": EventIndex(node, etime, label),
        "hub_thresholds": fit_hub_thresholds_on_training_period(g, u, splits),
    }
    struct = edges[edges["type"].isin(("uid", "card", "device"))]
    out["struct_edges"] = struct[["row", "type", "key", "time", "uid", "label", "node"]].copy()
    return out


def _structure_snapshot(struct: pd.DataFrame, d: int, delay_days: int) -> dict:
    """Day d's window graph partitions and group statistics, exactly as
    features/graph._structure_features / _apply_partitions compute them for day d (setting C)."""
    from vaultic.data.load import SECONDS_PER_DAY
    from vaultic.features.graph import (
        _first_node_in,
        _group_stats,
        _known_by_day_start,
        _partitions,
    )

    day = struct.groupby("row")["time"].first() // SECONDS_PER_DAY
    window = day[(day >= d - WINDOW_DAYS) & (day < d)].index
    tx_nodes = struct[struct["row"].isin(window)]
    if tx_nodes.empty:
        return {}
    nodes, parts = _partitions(tx_nodes, SEED)
    past = struct[struct["row"].isin(day[day < d].index)]
    known_fn = _known_by_day_start(d, delay_days)
    per_tx = tx_nodes.drop_duplicates("row").set_index("row")
    known = known_fn(per_tx)
    per_label = _first_node_in(past, nodes)
    snap = {}
    for name, labels in parts.items():
        node_group = pd.Series(labels, index=nodes)
        group_of_tx = node_group.reindex(per_tx["node"]).set_axis(per_tx.index)
        stats = _group_stats(group_of_tx, per_tx, known)
        group_of_label = node_group.reindex(per_label["node"]).set_axis(per_label.index)
        lab = _group_stats(group_of_label, per_label, known_fn(per_label))
        stats[["known", "fraud"]] = lab[["known", "fraud"]].reindex(stats.index).fillna(0.0)
        snap[name] = (node_group, stats)
    return snap


def build_store(log=print, days=None) -> HistoryStore:
    """Build the store from the frozen data files (opened behind the post-hoc guard)."""
    from vaultic.eval.posthoc_guard import require_final_on_record
    from vaultic.features.behavioral import ENTITIES, _entity_key
    from vaultic.features.categories import fit_on_training_period
    from vaultic.features.pipeline import fit_encoders
    from vaultic.stage2.system import load_parts

    require_final_on_record()  # D144: test-period rows only after the final run is on record
    started = time.perf_counter()
    parts = load_parts(dev=False)
    df, features, splits = parts["df"], parts["features"], parts["splits"]
    uid = features["uid"].to_numpy()
    t = df["TransactionDT"].to_numpy(dtype=np.int64)
    log(f"data: {len(df):,} rows ({time.perf_counter() - started:.0f} s)")
    keys = uid_key_strings(df)
    key_to_uid = dict(zip(keys, uid, strict=True))
    uid_keys = {int(u): k for k, u in key_to_uid.items()}
    uid_rows = _positions_by_key(uid.astype(object), t)
    entity_keys = {name: _entity_key(df, cols) for name, cols in ENTITIES.items()}
    entity_rows = {name: _positions_by_key(k, t) for name, k in entity_keys.items()}
    log(f"customer and entity indexes ({time.perf_counter() - started:.0f} s)")
    graph = _graph_indexes(df, uid, splits, log)
    log(f"graph indexes ({time.perf_counter() - started:.0f} s)")
    store = HistoryStore(df=df, uid=uid, features=features, splits=splits, uid_rows=uid_rows,
                         uid_keys=uid_keys, key_to_uid=key_to_uid, entity_keys=entity_keys,
                         entity_rows=entity_rows, graph=graph)  # fmt: skip
    train = splits.train.contains(df["day"].to_numpy())
    store.encoders = fit_encoders(df[train], pd.Series(uid[train]))
    store.product_encoder = fit_on_training_period(df, splits, columns=("ProductCD",))
    for d in (days if days is not None else range(splits.test.first, splits.test.last + 1)):
        store.structure[int(d)] = _structure_snapshot(graph["struct_edges"], int(d),
                                                      splits.label_delay_days)  # fmt: skip
        log(f"structure snapshot day {d} ({time.perf_counter() - started:.0f} s)")
    store.info.update(built_seconds=round(time.perf_counter() - started), rows=int(len(df)))
    return store
