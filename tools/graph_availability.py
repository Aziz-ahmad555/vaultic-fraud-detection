"""The script behind D52's real-data graph-view availability numbers (review N3, D78).

Run (repo root, real data):  python tools/graph_availability.py

Share of rows with the graph view available under the old rule (D42: any entity of the row seen
before) and the non-hub rule (D52: a card or device shared with another uid before t, below the
hub threshold), per split. Hub thresholds come from fit_hub_thresholds_on_training_period
(training rows only); the edge features are setting C's (point-in-time, L = splits.yaml's label
delay). It reads no labels' values for availability and reports no metric. D52's numbers
(train 62.8%, validation 49.0%, test 46.0%, all 58.2%; old rule 100%) were produced by this
script's first version, which called the hub fit on training rows directly.
"""

import time

import numpy as np
import pandas as pd

from vaultic.data.load import load_merged
from vaultic.data.splits import load_splits
from vaultic.data.uid import UID_PATH
from vaultic.features.graph import (
    FEATURE_TYPES,
    _edge_features,
    _nonhub_shared,
    build_edges,
    fit_hub_thresholds_on_training_period,
)
from vaultic.paths import MERGED_PATH


def main() -> None:
    t0 = time.perf_counter()
    splits = load_splits()
    df = load_merged(MERGED_PATH)
    uid = pd.read_parquet(UID_PATH, columns=["TransactionID", splits.uid_variant])
    if not (uid["TransactionID"].to_numpy() == df["TransactionID"].to_numpy()).all():
        raise ValueError("uids.parquet is out of date")
    uid = uid[splits.uid_variant].astype(str).reset_index(drop=True)
    part = splits.assign(df["day"])
    th = fit_hub_thresholds_on_training_period(df, uid, splits)
    print("hub thresholds (99th pct of distinct uids per entity, training period):", th)

    edges = build_edges(df.reset_index(drop=True), uid)
    edges = pd.concat([edges, _edge_features(edges, "C", splits.label_delay_days)], axis=1)
    new = np.nan_to_num(_nonhub_shared(edges, len(df), th)) > 0
    ent = edges[edges["type"].isin(FEATURE_TYPES)]
    old = np.zeros(len(df), bool)
    old[ent.loc[ent["deg_tx"] > 0, "row"].to_numpy()] = True
    for name in ("train", "validation", "test"):
        m = part == name
        print(f"{name:10s} rows {m.sum():7d}  old rule {old[m].mean():6.1%}  "
              f"non-hub rule {new[m].mean():6.1%}")  # fmt: skip
    print(f"all        rows {len(df):7d}  old {old.mean():6.1%}  non-hub {new.mean():6.1%}")
    hubs = {t: int((edges[edges["type"] == t].groupby("node")["uid"].nunique() > th[t]).sum())
            for t in ("card", "device")}  # fmt: skip
    print("entities above the hub threshold (whole period):", hubs)
    print(f"{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
