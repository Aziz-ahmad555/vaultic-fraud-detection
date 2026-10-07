"""Feature sets the harness can request by name (config key `features`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Never model inputs: identifiers, the label, and raw time (time trends would leak the split).
NON_FEATURES = {"TransactionID", "isFraud", "TransactionDT", "day"}
# Feature sets that include the point-in-time base features.
NEEDS_BASE = {"raw_base", "b5"}


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
    """
    raw = df[raw_columns(df)]
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
