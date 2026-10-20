"""Inputs of the feedback simulation (EXP-310; D210 section 6, D211).

The B5 inputs come from the frozen point-in-time feature pipeline (stage2.system.load_parts). The
three label-derived B5 columns (uid_fraud_known, uid_fraud_rate_known, uid_n_labels_known, D89)
are stored with the system's 30-day label delay; for a simulation at delay L != 30 they are
recomputed with the same code (features.pipeline.build_features) at delay L, so the model's inputs
never use a label the loop is not allowed to know (for L = 60 the stored values would). Every other
input is label-free. Recomputed columns are cached under the dev cache on drive E.

The design run (EXP-310-dev) reads days 1-120 only: rows after day 120 are dropped right after
loading and `require_last_day` refuses any that get through.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaultic.views.definitions import LABEL_DERIVED

STORED_DELAY = 30
DEV_LAST_DAY = 120


def require_last_day(day, last: int) -> None:
    if len(day) and int(np.max(day)) > last:
        raise RuntimeError(f"a row after day {last} reached the design run")


def label_derived_at(df: pd.DataFrame, uid: pd.Series, delay: int, train_mask) -> pd.DataFrame:
    """The three label-derived columns recomputed at label delay `delay` (days), with the
    pipeline's own point-in-time code; df sorted by TransactionDT."""
    from vaultic.features.pipeline import INPUT_COLUMNS, build_features, fit_encoders

    train_mask = np.asarray(train_mask, bool)
    enc = fit_encoders(df.loc[train_mask], uid[train_mask])
    out = build_features(df[INPUT_COLUMNS].reset_index(drop=True), uid.reset_index(drop=True),
                         enc, delay)  # fmt: skip
    return out[list(LABEL_DERIVED)].reset_index(drop=True)


def load(mode: str, final_started: bool = False) -> dict:
    """Data and B5 inputs for 'dev' (days 1-120) or 'final' (every day). 'final' is refused unless
    the caller has passed the final-run guards and pushed FINAL-STARTED (feedback_final.main)."""
    from vaultic.stage2.system import load_parts

    if mode not in ("dev", "final"):
        raise ValueError("mode must be dev or final")
    if mode == "final":
        if not final_started:
            raise RuntimeError("EXP-310-final needs Aziz's go-ahead and the final-run guards")
        parts = load_parts(dev=False)  # every day, test period included (after FINAL-STARTED)
        return {"df": parts["df"], "features": parts["features"], "b5_cols": list(parts["b5_cols"]),
                "splits": parts["splits"], "mode": mode}  # fmt: skip
    parts = load_parts(dev=True)  # test-period rows dropped at load (D180)
    keep = parts["df"]["day"].to_numpy() <= DEV_LAST_DAY
    df = parts["df"][keep].reset_index(drop=True)
    features = parts["features"][keep].reset_index(drop=True)
    require_last_day(df["day"].to_numpy(), DEV_LAST_DAY)
    return {"df": df, "features": features, "b5_cols": list(parts["b5_cols"]),
            "splits": parts["splits"], "mode": mode}  # fmt: skip


def inputs_for_delay(loaded: dict, delay: int, cache_dir=None) -> pd.DataFrame:
    """B5 inputs with the label-derived columns at delay L (stored values when L = 30)."""
    X = loaded["features"][loaded["b5_cols"]].copy()
    if delay == STORED_DELAY:
        return X
    path = None
    if cache_dir is not None:
        path = cache_dir / f"label_derived_{loaded['mode']}_L{delay}.parquet"
        if path.exists():
            cols = pd.read_parquet(path)
            if len(cols) == len(X):
                for c in LABEL_DERIVED:
                    X[c] = cols[c].to_numpy(dtype=np.float32)
                return X
    df = loaded["df"]
    train = loaded["splits"].train.contains(df["day"].to_numpy())
    cols = label_derived_at(df, loaded["features"]["uid"], delay, train)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cols.to_parquet(path, index=False)
    for c in LABEL_DERIVED:
        X[c] = cols[c].to_numpy(dtype=np.float32)
    return X
