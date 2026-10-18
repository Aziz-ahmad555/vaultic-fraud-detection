"""Online features for ONE transaction at time t (D182): every feature computed point-in-time from
the stored history strictly before t, with the offline feature code itself, so the live features
equal the offline ones (checked by live/parity.py).

  base, behavioral   features/pipeline.build_features and features/behavioral.build_behavioral on
                     the customer's earlier rows plus the transaction; the 7-day distinct-uid
                     counts per card / email / device with behavioral._entity_distinct_uids on that
                     entity's rows of the 7 days before t
  sequence           features/sequences.build_sequences + views/temporal_diag.flatten on the
                     customer's earlier rows (20 steps, the temporal view's channels)
  graph              the offline EventIndex queries (strictly before t; labels known by t - L)
                     and the day's precomputed window-graph partitions (live/store.py)
  tabular / B5       features/sets.design_matrix("b5") on the transaction and its base features
  anomaly            the frozen anomaly models (stage2/system.FrozenSystem.anomaly_scores)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY


def _slice(store, query: pd.DataFrame, uid_code: int) -> tuple[pd.DataFrame, pd.Series]:
    """The customer's rows before t, then the transaction (sorted by time)."""
    t = int(query["TransactionDT"].iloc[0])
    hist = store.before(store.uid_rows.get(uid_code), t)
    rows = pd.concat([store.df.iloc[hist], query], ignore_index=True)
    uid = pd.Series(np.concatenate([store.uid[hist], [uid_code]]))
    return rows, uid


def base_features(store, rows, uid) -> pd.DataFrame:
    from vaultic.features.pipeline import INPUT_COLUMNS, build_features

    out = build_features(rows[INPUT_COLUMNS], uid, store.encoders, store.splits.label_delay_days)
    return out.iloc[[-1]].reset_index(drop=True)


def behavioral_features(store, rows, uid, query) -> pd.DataFrame:
    from vaultic.features.behavioral import (
        ENTITIES,
        ENTITY_WINDOW,
        _entity_distinct_uids,
        _entity_key,
        build_behavioral,
    )

    out = build_behavioral(rows, uid).iloc[[-1]].reset_index(drop=True)
    t = int(query["TransactionDT"].iloc[0])
    u = int(uid.iloc[-1])
    for name, cols in ENTITIES.items():
        key = _entity_key(query, cols)[0]
        if key is None:
            out[f"ent_uids_7d_{name}"] = np.float32(np.nan)
            continue
        pos = store.before(store.entity_rows[name].get(key), t)
        pos = pos[store.time[pos] >= t - ENTITY_WINDOW]
        ent = np.concatenate([store.entity_keys[name][pos], [key]]).astype(object)
        uids = np.concatenate([store.uid[pos], [u]])
        times = np.concatenate([store.time[pos], [t]])
        out[f"ent_uids_7d_{name}"] = np.float32(_entity_distinct_uids(ent, uids, times)[-1])
    return out


def sequence_features(store, rows, uid) -> pd.DataFrame:
    from vaultic.features.sequences import build_sequences
    from vaultic.views.temporal_dev import EXTRA, N_STEPS
    from vaultic.views.temporal_diag import flatten

    seq = build_sequences(rows, uid.astype(str), store.product_encoder, n_steps=N_STEPS,
                          extra_columns=EXTRA, rows=np.array([len(rows) - 1]))  # fmt: skip
    names = [f"seq{t:02d}_{f}" for t in range(N_STEPS) for f in seq.feature_names]
    names += ["seq_n_steps", *[f"seq_cur_{f}" for f in seq.feature_names]]
    out = pd.DataFrame(flatten(seq), columns=names)
    if not seq.has_history[0]:
        out.loc[:, names] = np.nan  # no history: the temporal view is missing (rule 11)
    return out


def graph_features(store, query: pd.DataFrame, uid_code: int) -> pd.DataFrame:
    """features/graph.build_graph_features (setting C) for one transaction."""
    from vaultic.features.graph import FEATURE_TYPES, NONHUB_TYPES, _query_union, build_edges
    from vaultic.features.pit import label_cutoff

    g = store.graph
    t = int(query["TransactionDT"].iloc[0])
    edges = build_edges(query.reset_index(drop=True), pd.Series([uid_code]))
    node = edges["node"].to_numpy()
    pair = (edges["node"] + "\x1f" + edges["uid"]).to_numpy()
    times = np.full(len(edges), t)
    edges["deg_tx"] = g["edge"].count_before(node, times)
    edges["deg_uid"] = g["firsts"].count_before(node, times)
    edges["shared_uids"] = edges["deg_uid"] - (g["pair"].count_before(pair, times) > 0)
    cut = label_cutoff(times, store.splits.label_delay_days)
    edges["n_known"] = g["labels"].count_before(node, cut)
    edges["fraud_known"] = g["labels"].sum_before(node, cut)

    out = {}
    for etype in FEATURE_TYPES:
        e = edges[edges["type"] == etype]
        if e.empty:
            for c in ("deg_tx", "deg_uid", "shared_uids", "fraud_rate"):
                out[f"g_{c}_{etype}"] = np.nan
            continue
        out[f"g_deg_tx_{etype}"] = float(e["deg_tx"].max())
        out[f"g_deg_uid_{etype}"] = float(e["deg_uid"].max())
        out[f"g_shared_uids_{etype}"] = float(e["shared_uids"].max())
        n, f = float(e["n_known"].sum()), float(e["fraud_known"].sum())
        out[f"g_fraud_rate_{etype}"] = f / n if n > 0 else np.nan
    nh = edges[edges["type"].isin(NONHUB_TYPES)]
    if nh.empty:
        out["g_shared_nonhub"] = np.nan
    else:
        limit = nh["type"].map(g["hub_thresholds"]).to_numpy(dtype=float)
        shared = np.where(
            nh["deg_uid"].to_numpy(float) <= limit, nh["shared_uids"].to_numpy(float), 0.0
        )
        out["g_shared_nonhub"] = float(shared.max())
    out["g_twohop_fraud"] = float(edges.loc[edges["type"].isin(FEATURE_TYPES), "fraud_known"].sum())

    s = {k: np.nan for k in ("comp_tx", "comp_uids", "comp_known", "comp_fraud", "comm_tx",
                             "comm_known", "comm_fraud")}  # fmt: skip
    snap = store.structure.get(t // SECONDS_PER_DAY, {})
    struct = edges[edges["type"].isin(("uid", "card", "device"))].assign(row=0)
    for name, (node_group, stats) in snap.items():
        union = _query_union(struct, node_group, stats)
        if union.empty:
            continue
        s[f"{name}_tx"], s[f"{name}_known"] = float(union["tx"].iloc[0]), float(
            union["known"].iloc[0]
        )
        s[f"{name}_fraud"] = float(union["fraud"].iloc[0])
        if name == "comp":
            s["comp_uids"] = float(union["uids"].iloc[0])
    out["g_comp_tx"], out["g_comp_uids"] = s["comp_tx"], s["comp_uids"]
    out["g_comp_fraud_rate"] = s["comp_fraud"] / s["comp_known"] if s["comp_known"] > 0 else np.nan
    out["g_comm_tx"] = s["comm_tx"]
    out["g_comm_fraud_rate"] = s["comm_fraud"] / s["comm_known"] if s["comm_known"] > 0 else np.nan
    return pd.DataFrame([out])


def online_features(store, system, query: pd.DataFrame, uid_code: int) -> pd.DataFrame:
    """One row with every column of the offline feature frame (stage2 parts['features'])."""
    from vaultic.features.sets import design_matrix

    if len(query) != 1:
        raise ValueError("online_features scores one transaction at a time")
    query = query.reset_index(drop=True)
    rows, uid = _slice(store, query, uid_code)
    base = base_features(store, rows, uid)
    beh = behavioral_features(store, rows, uid, query)
    seq = sequence_features(store, rows, uid)
    graph = graph_features(store, query, uid_code)
    tab = design_matrix(query, base.assign(TransactionID=query["TransactionID"].to_numpy()), "b5")
    frame = pd.concat([tab.reset_index(drop=True), beh.drop(columns=["TransactionID"]), graph, seq],
                      axis=1)  # fmt: skip
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame["uid"] = uid_code
    anomaly = system.anomaly_scores(frame)
    for c in anomaly.columns:
        frame[c] = anomaly[c].to_numpy(dtype=np.float32)
    cols = [c for c in store.features.columns if c in frame.columns]
    missing = sorted(set(store.features.columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"online features lack columns of the offline frame: {missing[:10]}")
    out = frame[cols]
    num = [c for c in cols if c != "uid"]
    out = out.astype({c: np.float32 for c in num})
    return out
