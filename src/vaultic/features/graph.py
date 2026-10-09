"""Phase 4 graph view, features first (no GNN): a heterogeneous entity graph and its
point-in-time features, with the settings A / B / C switch for the leakage study (Paper 1).

Schema (roadmap 4.1): transactions linked to their entities, each edge carrying the
transaction's timestamp. Entity types:
  card     card1-card6 combined          (every row)
  email    P_emaildomain, R_emaildomain  (one node space; up to two edges per transaction)
  address  addr1 + addr2                 (where present)
  device   DeviceInfo + DeviceType       (where present; ~24% of rows)
  uid      the reconstructed customer    (used for components / communities)

Features (roadmap 4.2), per entity type T in card / email / address / device:
  g_deg_tx_T          transactions touching the entity
  g_deg_uid_T         distinct uids touching it
  g_shared_uids_T     other uids (not this one) touching it
  g_fraud_rate_T      known-fraud rate of the transactions touching it
and per transaction:
  g_twohop_fraud      known-fraud transactions reachable transaction -> entity -> transaction,
                      counted per path (a fraud sharing two entities counts twice; an exact
                      distinct count would need set unions over hub entities)
  g_comp_tx / g_comp_uids / g_comp_fraud_rate   connected component of the uid / card / device
                      entity graph: transactions, distinct uids, known-fraud rate
  g_comm_tx / g_comm_fraud_rate                 Louvain community on the same graph
For email (two edges) degree-type features take the larger edge, and fraud rates pool both.
A missing entity gives NaN for its type (CLAUDE.md rule 11). The transaction's own row and own
label are never counted, in any setting.

Settings (roadmap 4.4):
  A  all edges (past and future) + all labels                    -> expected to be inflated
  B  all edges + only labels known by t - L                       -> partly inflated
  C  only edges before t + only labels known by t - L             -> honest (the default)
In C the entity features use transactions strictly before t (exact timestamps); components
and communities use a daily snapshot of the `window_days` days before the transaction's day,
so they never see the current day. In A and B they use one static graph of all transactions.
Email domains and addresses are hubs (one domain or region links most transactions), so
components and communities use only uid, card and device (research/decisions.md D34).
In C the group's known-fraud rate counts every earlier transaction on the group's nodes whose
label is known by the day's start, not only the window's (D50): with L = 30 and a 30-day
window, no window label would ever be known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.features.pit import label_cutoff

ENTITY_COLUMNS = {
    "card": ["card1", "card2", "card3", "card4", "card5", "card6"],
    "address": ["addr1", "addr2"],
    "device": ["DeviceInfo", "DeviceType"],
}
EMAIL_COLUMNS = ["P_emaildomain", "R_emaildomain"]
FEATURE_TYPES = ("card", "email", "address", "device")
NONHUB_TYPES = ("card", "device")  # entities that can carry relational evidence (D52)
HUB_QUANTILE = 0.99
COMPONENT_TYPES = ("uid", "card", "device")
SETTINGS = ("A", "B", "C")


# ---- events --------------------------------------------------------------------------------


class EventIndex:
    """Per-key events with times; counts and sums of events with time < cutoff for any query
    (key, cutoff). Running sums restart per key, so results are exact and order-independent."""

    def __init__(self, keys, times, values=None):
        keys = np.asarray(keys, dtype=object)
        self.index = pd.Index(pd.unique(keys))
        codes = self.index.get_indexer(keys)
        times = np.asarray(times, dtype=np.int64)
        order = np.lexsort((times, codes))
        self.codes_sorted = codes[order]
        self.times_sorted = times[order]
        self.span = int(times.max()) + 2 if len(times) else 2
        self.composite = self.codes_sorted * self.span + self.times_sorted
        sizes = np.bincount(codes, minlength=len(self.index))
        self.start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        v = np.ones(len(times)) if values is None else np.asarray(values, dtype=float)[order]
        self.running = pd.Series(v).groupby(self.codes_sorted).cumsum().to_numpy()

    def _positions(self, keys, cutoff):
        codes = self.index.get_indexer(np.asarray(keys, dtype=object))
        cutoff = np.clip(np.asarray(cutoff, dtype=np.int64), 0, self.span - 1)
        safe = np.maximum(codes, 0)
        pos = np.searchsorted(self.composite, safe * self.span + cutoff, side="left")
        start = self.start[safe] if len(self.start) else np.zeros_like(safe)
        return codes, pos, start

    def count_before(self, keys, cutoff) -> np.ndarray:
        codes, pos, start = self._positions(keys, cutoff)
        return np.where(codes >= 0, pos - start, 0)

    def sum_before(self, keys, cutoff) -> np.ndarray:
        codes, pos, start = self._positions(keys, cutoff)
        has = (codes >= 0) & (pos > start)
        out = np.zeros(len(codes))
        out[has] = self.running[pos[has] - 1]
        return out


def _combined_key(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = df[[c for c in cols if c in df]].astype(object)
    missing = parts.isna().all(axis=1).to_numpy()
    key = parts.astype(str).agg("|".join, axis=1).to_numpy(dtype=object)
    key[missing] = None
    return key


def build_edges(df: pd.DataFrame, uid: pd.Series) -> pd.DataFrame:
    """One row per (transaction, entity) edge: row position, type, entity key, time, uid, label."""
    frames = []
    base = {
        "row": np.arange(len(df)),
        "time": df["TransactionDT"].to_numpy(dtype=np.int64),
        "uid": uid.astype(str).to_numpy(),
        "label": df["isFraud"].to_numpy(dtype=float),
    }
    for etype, cols in ENTITY_COLUMNS.items():
        frames.append(pd.DataFrame({**base, "type": etype, "key": _combined_key(df, cols)}))
    for col in EMAIL_COLUMNS:
        if col in df:
            frames.append(
                pd.DataFrame({**base, "type": "email", "key": df[col].astype(object).to_numpy()})
            )
    frames.append(pd.DataFrame({**base, "type": "uid", "key": base["uid"]}))
    edges = pd.concat(frames, ignore_index=True)
    edges = edges[edges["key"].notna()]
    edges = edges.drop_duplicates(["row", "type", "key"])  # P == R domain: one edge
    edges["node"] = edges["type"] + ":" + edges["key"].astype(str)
    return edges.reset_index(drop=True)


# ---- entity features -----------------------------------------------------------------------


def _edge_features(edges: pd.DataFrame, setting: str, delay_days: int) -> pd.DataFrame:
    """Per-edge degree, distinct uids, other uids, known-fraud count and known-label count."""
    node, time, label = (
        edges["node"].to_numpy(),
        edges["time"].to_numpy(),
        edges["label"].to_numpy(),
    )
    pair = (edges["node"] + "\x1f" + edges["uid"]).to_numpy()
    out = pd.DataFrame(index=edges.index)

    if setting == "C":
        edge_index = EventIndex(node, time)
        out["deg_tx"] = edge_index.count_before(node, time)
        first = edges.assign(pair=pair).groupby("pair")["time"].transform("min").to_numpy()
        is_first = edges.assign(pair=pair, first=first).drop_duplicates("pair").index
        firsts = EventIndex(node[is_first], first[is_first])
        out["deg_uid"] = firsts.count_before(node, time)
        own_seen = EventIndex(pair, time).count_before(pair, time) > 0
        out["shared_uids"] = out["deg_uid"] - own_seen
    else:  # A and B: every edge, past and future, except the transaction's own
        counts = edges.groupby("node")["row"].transform("size").to_numpy()
        out["deg_tx"] = counts - 1
        distinct = edges.groupby("node")["uid"].transform("nunique").to_numpy()
        out["deg_uid"] = distinct
        out["shared_uids"] = distinct - 1

    if setting == "A":  # every other transaction's label
        out["n_known"] = edges.groupby("node")["row"].transform("size").to_numpy() - 1
        out["fraud_known"] = edges.groupby("node")["label"].transform("sum").to_numpy() - label
    else:  # labels known by t - L only (its_time + L <= t)
        cutoff = label_cutoff(time, delay_days)
        labels = EventIndex(node, time, label)
        out["n_known"] = labels.count_before(node, cutoff)
        out["fraud_known"] = labels.sum_before(node, cutoff)
    return out


# ---- non-hub relational evidence (graph-view availability, D52) -----------------------------


def fit_hub_thresholds_on_training_period(
    df: pd.DataFrame, uid: pd.Series, splits, quantile: float = HUB_QUANTILE
) -> dict[str, float]:
    """Hub threshold per entity type from the TRAINING-period rows of df only (D52, review N3).

    The rows are selected here from splits.train, never by the caller; `_fit_hub_thresholds`
    then refuses any row after the training period's last day as a second guard."""
    day = df["TransactionDT"].to_numpy(dtype=np.int64) // SECONDS_PER_DAY
    train = splits.train.contains(day)
    if not train.any():
        raise ValueError("no training-period rows to fit hub thresholds on")
    uid = pd.Series(np.asarray(uid))
    return _fit_hub_thresholds(df[train], uid[train], quantile, last_day=splits.train.last)


def _fit_hub_thresholds(
    df: pd.DataFrame, uid: pd.Series, quantile: float, last_day: int
) -> dict[str, float]:
    """The `quantile` of the number of distinct uids per card / device. An entity whose distinct
    uids before t exceed it is a hub (a shared terminal, a default device string) and is not
    evidence. Raises if any row lies after `last_day` (the training period's last day)."""
    days = df["TransactionDT"].to_numpy(dtype=np.int64) // SECONDS_PER_DAY
    if len(days) and days.max() > last_day:
        raise ValueError(
            f"hub thresholds must be fitted on training-period rows only (day <= {last_day}); "
            f"got rows up to day {int(days.max())}"
        )
    edges = build_edges(df.reset_index(drop=True), uid.reset_index(drop=True))
    out = {}
    for etype in NONHUB_TYPES:
        counts = edges[edges["type"] == etype].groupby("node")["uid"].nunique()
        out[etype] = float(np.quantile(counts, quantile)) if len(counts) else float("inf")
    return out


def _nonhub_shared(edges: pd.DataFrame, n_rows: int, thresholds: dict[str, float]) -> np.ndarray:
    """Per row: most other uids sharing one of its non-hub cards / devices (NaN if the row has
    neither a card nor a device). Uses the edge features (point-in-time in setting C)."""
    e = edges[edges["type"].isin(NONHUB_TYPES)]
    limit = e["type"].map(thresholds).to_numpy(dtype=float)
    shared = np.where(
        e["deg_uid"].to_numpy(dtype=float) <= limit, e["shared_uids"].to_numpy(dtype=float), 0.0
    )
    per_row = pd.Series(shared, index=e["row"].to_numpy()).groupby(level=0).max()
    return per_row.reindex(np.arange(n_rows)).to_numpy(dtype=float)


# ---- components and communities ------------------------------------------------------------


def _node_graph(tx_nodes: pd.DataFrame) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    """Entity graph: nodes linked when one transaction touches both (weight = transactions)."""
    nodes = pd.Index(pd.unique(tx_nodes["node"]))
    codes = nodes.get_indexer(tx_nodes["node"])
    by_row = pd.DataFrame({"row": tx_nodes["row"].to_numpy(), "code": codes})
    pairs = by_row.merge(by_row, on="row")
    pairs = pairs[pairs["code_x"] < pairs["code_y"]]
    weights = pairs.groupby(["code_x", "code_y"]).size()
    a = weights.index.get_level_values(0).to_numpy()
    b = weights.index.get_level_values(1).to_numpy()
    return nodes, a, b, weights.to_numpy()


def _components(nodes, a, b) -> np.ndarray:
    n = len(nodes)
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _communities(nodes, a, b, w, seed: int) -> np.ndarray:
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(len(nodes)))
    g.add_weighted_edges_from(zip(a.tolist(), b.tolist(), w.tolist(), strict=True))
    labels = np.empty(len(nodes), dtype=np.int64)
    for i, members in enumerate(nx.community.louvain_communities(g, weight="weight", seed=seed)):
        labels[list(members)] = i
    return labels


def _group_stats(group_of_row: pd.Series, rows: pd.DataFrame, known: np.ndarray) -> pd.DataFrame:
    """Transactions, distinct uids, known labels and known frauds per group."""
    frame = pd.DataFrame(
        {
            "group": group_of_row.to_numpy(),
            "uid": rows["uid"].to_numpy(),
            "known": known.astype(float),
            "fraud": rows["label"].to_numpy() * known,
        }
    )
    return frame.groupby("group").agg(
        tx=("uid", "size"), uids=("uid", "nunique"), known=("known", "sum"), fraud=("fraud", "sum")
    )


def _query_union(query_nodes: pd.DataFrame, node_group: pd.Series, stats: pd.DataFrame):
    """Per query row: union of the groups its entity nodes fall in (each group counted once)."""
    q = query_nodes.assign(group=node_group.reindex(query_nodes["node"]).to_numpy())
    q = q.dropna(subset=["group"]).drop_duplicates(["row", "group"])
    joined = q.join(stats, on="group")
    return joined.groupby("row")[["tx", "uids", "known", "fraud"]].sum()


def _partitions(tx_nodes: pd.DataFrame, seed: int):
    """Entity nodes and their component / Louvain community labels for one graph."""
    nodes, a, b, w = _node_graph(tx_nodes)
    return nodes, {"comp": _components(nodes, a, b), "comm": _communities(nodes, a, b, w, seed)}


def _first_node_in(label_nodes: pd.DataFrame, nodes: pd.Index) -> pd.DataFrame:
    """One row per transaction: its first entity node (card first) that is in the graph."""
    inside = label_nodes[label_nodes["node"].isin(nodes)]
    return inside.drop_duplicates("row").set_index("row")


def _apply_partitions(out, query_nodes, tx_nodes, nodes, partitions, known_fn, exclude_self,
                      label_nodes=None):  # fmt: skip
    """Write group statistics for the query rows. Each transaction is counted once, in the
    group of its first entity node (its card): its nodes share one component, but may fall in
    different Louvain communities.

    Transactions and uids are counted on the graph's own transactions (tx_nodes). Known labels
    and frauds are counted on `label_nodes` when given (setting C: every past transaction on
    the group's nodes, D50), else on tx_nodes."""
    per_tx = tx_nodes.drop_duplicates("row").set_index("row")
    known = known_fn(per_tx)
    per_label = None if label_nodes is None else _first_node_in(label_nodes, nodes)
    for name, labels in partitions.items():
        node_group = pd.Series(labels, index=nodes)
        group_of_tx = node_group.reindex(per_tx["node"]).set_axis(per_tx.index)
        stats = _group_stats(group_of_tx, per_tx, known)
        if per_label is not None:
            group_of_label = node_group.reindex(per_label["node"]).set_axis(per_label.index)
            lab = _group_stats(group_of_label, per_label, known_fn(per_label))
            stats[["known", "fraud"]] = lab[["known", "fraud"]].reindex(stats.index).fillna(0.0)
        union = _query_union(query_nodes, node_group, stats)
        if union.empty:
            continue
        if exclude_self:  # the query transaction is part of the static graph
            own = per_tx.loc[union.index]
            own_known = known_fn(own)
            union["tx"] -= 1
            union["known"] -= own_known
            union["fraud"] -= own["label"].to_numpy() * own_known
        out.loc[union.index, f"{name}_tx"] = union["tx"].to_numpy()
        out.loc[union.index, f"{name}_known"] = union["known"].to_numpy()
        out.loc[union.index, f"{name}_fraud"] = union["fraud"].to_numpy()
        if name == "comp":
            out.loc[union.index, "comp_uids"] = union["uids"].to_numpy()


def _known_by_day_start(d: int, delay_days: int):
    """Labels known at the start of day d: its_time + L <= day start."""
    cutoff = int(label_cutoff(np.array([int(d) * SECONDS_PER_DAY]), delay_days)[0])
    return lambda tx: (tx["time"].to_numpy() < cutoff).astype(float)


def _structure_features(edges, n_rows, setting, delay_days, window_days, seed):
    """Component and community features for every transaction row."""
    struct = edges[edges["type"].isin(COMPONENT_TYPES)]
    # NaN = no group for this row (setting C: none of its entities is in the window graph);
    # never 0, which would claim an empty group (rule 11, D51)
    out = pd.DataFrame(
        np.nan,
        index=np.arange(n_rows),
        columns=["comp_tx", "comp_uids", "comp_known", "comp_fraud", "comm_tx", "comm_known",
                 "comm_fraud"],  # fmt: skip
    )
    day = (edges.groupby("row")["time"].first() // SECONDS_PER_DAY).reindex(np.arange(n_rows))

    if setting == "C":  # a fresh graph per day: the window_days days before it
        for d in np.unique(day):
            window = day[(day >= d - window_days) & (day < d)].index
            tx_nodes = struct[struct["row"].isin(window)]
            if tx_nodes.empty:
                continue
            nodes, parts = _partitions(tx_nodes, seed)
            query = struct[struct["row"].isin(day[day == d].index)]
            # structure from the window; known labels from ALL earlier transactions on the
            # group's nodes (the label rule still applies: known by the day's start, D50)
            past = struct[struct["row"].isin(day[day < d].index)]
            _apply_partitions(out, query, tx_nodes, nodes, parts,
                              _known_by_day_start(d, delay_days), exclude_self=False,
                              label_nodes=past)  # fmt: skip
        return out

    nodes, parts = _partitions(struct, seed)  # A and B: one static graph of all transactions
    if setting == "A":
        _apply_partitions(out, struct, struct, nodes, parts, lambda tx: np.ones(len(tx)), True)
    else:  # B: same graph, labels known by the start of each transaction's day
        for d in np.unique(day):
            query = struct[struct["row"].isin(day[day == d].index)]
            _apply_partitions(out, query, struct, nodes, parts,
                              _known_by_day_start(d, delay_days), exclude_self=True)  # fmt: skip
    return out


# ---- public API ----------------------------------------------------------------------------


def build_graph_features(
    df: pd.DataFrame,
    uid: pd.Series,
    setting: str = "C",
    label_delay_days: int = 30,
    window_days: int = 30,
    seed: int = 0,
    hub_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Graph features for every row of df (rows sorted by TransactionDT).

    With `hub_thresholds` (fit_hub_thresholds_on_training_period) the output also has
    g_shared_nonhub: other uids sharing a non-hub card or device before t; the graph view is
    available only where it is > 0 (D52). Only in setting C: A and B raise (D79)."""
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}")
    if hub_thresholds is not None and setting != "C":
        # availability decides whether the graph view exists for a row; settings A and B see
        # later edges, so their g_shared_nonhub would let the future decide that (N4, D79)
        raise ValueError(
            "graph-view availability (g_shared_nonhub) comes from setting C edge features only"
        )
    time = df["TransactionDT"].to_numpy(dtype=np.int64)
    if np.any(np.diff(time) < 0):
        raise ValueError("rows must be sorted by TransactionDT")
    edges = build_edges(df.reset_index(drop=True), uid.reset_index(drop=True))
    feats = _edge_features(edges, setting, label_delay_days)
    edges = pd.concat([edges, feats], axis=1)

    out = pd.DataFrame({"TransactionID": df["TransactionID"].to_numpy()})
    for etype in FEATURE_TYPES:
        e = edges[edges["type"] == etype].groupby("row")
        agg = pd.DataFrame(
            {
                "deg_tx": e["deg_tx"].max(),
                "deg_uid": e["deg_uid"].max(),
                "shared_uids": e["shared_uids"].max(),
                "n_known": e["n_known"].sum(),
                "fraud_known": e["fraud_known"].sum(),
            }
        ).reindex(
            np.arange(len(df))
        )  # rows without this entity -> NaN
        out[f"g_deg_tx_{etype}"] = agg["deg_tx"].to_numpy(dtype=float)
        out[f"g_deg_uid_{etype}"] = agg["deg_uid"].to_numpy(dtype=float)
        out[f"g_shared_uids_{etype}"] = agg["shared_uids"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = agg["fraud_known"] / agg["n_known"]
        out[f"g_fraud_rate_{etype}"] = rate.where(agg["n_known"] > 0).to_numpy(dtype=float)
    if hub_thresholds is not None:
        out["g_shared_nonhub"] = _nonhub_shared(edges, len(df), hub_thresholds)
    entity_edges = edges[edges["type"].isin(FEATURE_TYPES)]
    out["g_twohop_fraud"] = (
        entity_edges.groupby("row")["fraud_known"].sum().reindex(np.arange(len(df)), fill_value=0)
    ).to_numpy(dtype=float)

    s = _structure_features(edges, len(df), setting, label_delay_days, window_days, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["g_comp_tx"] = s["comp_tx"].to_numpy()
        out["g_comp_uids"] = s["comp_uids"].to_numpy()
        out["g_comp_fraud_rate"] = np.where(
            s["comp_known"] > 0, s["comp_fraud"] / s["comp_known"], np.nan
        )
        out["g_comm_tx"] = s["comm_tx"].to_numpy()
        out["g_comm_fraud_rate"] = np.where(
            s["comm_known"] > 0, s["comm_fraud"] / s["comm_known"], np.nan
        )
    out.index = df.index
    return out
