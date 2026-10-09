"""Feature sets the harness can request by name (config key `features`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Never model inputs: identifiers, the label, and raw time (time trends would leak the split).
NON_FEATURES = {"TransactionID", "isFraud", "TransactionDT", "day"}
# B5 plus point-in-time view features precomputed in data/features/<kind>_<uid variant>.parquet
# (Phase 3 behavioral, Phase 4 graph); the harness attaches them to the base features (D66).
EXTRA_FEATURES = {
    "b5_behavioral": ("behavioral",),
    "b5_graph": ("graph",),
    "b5_anomaly": ("anomaly",),
}
# One view's precomputed features ALONE (a standalone view model for MVAF, D75): the harness
# passes data/features/<kind>_<uid>.parquet as `base` and nothing else is used.
VIEW_ONLY = {"behavioral_only": "behavioral", "graph_only": "graph", "anomaly_only": "anomaly"}
# Feature sets that include the point-in-time base features.
NEEDS_BASE = {"raw_base", "b5", *EXTRA_FEATURES}
# Feature sets that need the customer id (passed as `base` with a `uid` column).
NEEDS_UID = {"fyp1"}


def raw_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def _codes(df: pd.DataFrame) -> pd.DataFrame:
    """Categoricals as integer codes (NaN where missing); numbers as float32.

    Codes come from the category list, which carries no label information.
    """
    out = {}
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            codes = df[col].cat.codes.astype(np.float32)
            out[col] = codes.where(codes >= 0)
        else:
            out[col] = df[col].astype(np.float32)
    return pd.DataFrame(out, index=df.index)


def design_matrix(df: pd.DataFrame, base: pd.DataFrame | None, name: str) -> pd.DataFrame:
    """Build the model input for feature set `name`.

    raw         every original column (B2-B4)
    raw_lr      numeric columns + one-hot ProductCD (B1)
    raw_base    raw + point-in-time base features (B5 without V-reduction)
    b5          raw with V columns reduced (experiments/configs/v_columns.yaml) + base features
    fyp1        FYP-1's global columns + point-in-time behavioural features + uid (B6)
    b5_<kind>   b5 + the precomputed <kind> features (EXTRA_FEATURES), already attached to base
    <kind>_only only the precomputed <kind> features (VIEW_ONLY), passed as base
    """
    raw = df[raw_columns(df)]
    if name == "fyp1":
        from vaultic.views.fyp1 import build_fyp1_frame

        if base is None or "uid" not in base:
            raise ValueError("feature set fyp1 needs a uid column")
        return build_fyp1_frame(df, base["uid"].set_axis(df.index))
    if name in EXTRA_FEATURES:
        return design_matrix(df, base, "b5")
    if name in VIEW_ONLY:
        if base is None or not np.array_equal(
            base["TransactionID"].to_numpy(), df["TransactionID"].to_numpy()
        ):
            raise ValueError(f"feature set {name} needs its feature file aligned with the data")
        return base.drop(columns="TransactionID").set_axis(df.index).astype(np.float32)
    if name == "b5":
        from vaultic.features.vreduce import load_kept, v_columns

        dropped = set(v_columns(raw.columns)) - set(load_kept())
        df = df.drop(columns=sorted(dropped))
        return design_matrix(df, base, "raw_base")
    if name == "raw":
        return _codes(raw)
    if name == "raw_lr":
        numeric = raw.select_dtypes(include="number").astype(np.float32)
        onehot = pd.get_dummies(df["ProductCD"], prefix="ProductCD", dtype=np.float32)
        return pd.concat([numeric, onehot], axis=1)
    if name == "raw_base":
        if base is None:
            raise ValueError("feature set raw_base needs base features")
        if not (base["TransactionID"].to_numpy() == df["TransactionID"].to_numpy()).all():
            raise ValueError("base features are not aligned with the data")
        extra = base.drop(columns="TransactionID").set_axis(df.index)
        return pd.concat([_codes(raw), extra.astype(np.float32)], axis=1)
    raise ValueError(f"unknown feature set {name!r}")


def attach_features(base: pd.DataFrame, extra: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Base features plus the columns of a precomputed feature file, row-aligned by id."""
    if not np.array_equal(base["TransactionID"].to_numpy(), extra["TransactionID"].to_numpy()):
        raise ValueError(f"{kind} features are not aligned with the base features")
    clash = sorted(set(base.columns) & set(extra.columns) - {"TransactionID"})
    if clash:
        raise ValueError(f"{kind} features repeat base feature names: {clash}")
    return pd.concat([base, extra.drop(columns="TransactionID").set_axis(base.index)], axis=1)


def drop_features(X: pd.DataFrame, names) -> pd.DataFrame:
    """Remove features named in a config's `drop_features` (ablations); unknown names raise."""
    names = list(names or [])
    missing = sorted(set(names) - set(X.columns))
    if missing:
        raise ValueError(f"drop_features names columns that are not in the feature set: {missing}")
    return X.drop(columns=names)
