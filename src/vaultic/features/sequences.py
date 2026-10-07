"""Phase 5 prep: per-transaction sequences of the uid's previous transactions (no model yet).

For each transaction at time t: the uid's last `n_steps` transactions strictly before t (rows in
the same second are not history), oldest first. Each step holds
  log_amount     log1p(amount)
  log_gap_next   log1p(seconds from this step to the next one; the last step's next one is the
                 transaction being scored)
  product_code   ProductCD code from the shared CategoryEncoder fitted on the training period
                 (0 = missing, 1 = unseen in training; for an embedding later)
  + any extra per-row columns (e.g. C/D counters, behavioral novelty flags), NaN -> 0;
    an extra column the encoder knows is replaced by its code
Short histories are left-padded with zeros; `mask` is True for real steps. A transaction whose
uid has no history gets an all-False mask and has_history = False: the temporal view is masked
for it, never given a fake sequence (CLAUDE.md rule 11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vaultic.features.categories import CategoryEncoder
from vaultic.features.pit import PastIndex

BASE_FEATURES = ("log_amount", "log_gap_next", "product_code")


def _column(df: pd.DataFrame, col: str, encoder: CategoryEncoder) -> np.ndarray:
    if col in encoder.columns:
        return encoder.encode(df[col], col).astype(np.float64)
    return df[col].to_numpy(dtype=np.float64)


@dataclass
class Sequences:
    values: np.ndarray  # (rows, n_steps, features) float32
    mask: np.ndarray  # (rows, n_steps) bool, True for real steps
    has_history: np.ndarray  # (rows,) bool
    feature_names: list[str]
    rows: np.ndarray  # positions in the input frame these sequences belong to


def build_sequences(
    df: pd.DataFrame,
    uid: pd.Series,
    encoder: CategoryEncoder,
    n_steps: int = 10,
    extra_columns: tuple[str, ...] = (),
    rows: np.ndarray | None = None,
) -> Sequences:
    """Sequences for the transactions at `rows` (positions in df; default: all).

    `encoder` must be the one fitted on the training period (features.categories), so codes
    are the same whichever frame (train, validation, a single replayed day) is passed in.
    """
    time = df["TransactionDT"].to_numpy(dtype=np.int64)
    if np.any(np.diff(time) < 0):
        raise ValueError("rows must be sorted by TransactionDT")
    n = len(df)
    rows = np.arange(n) if rows is None else np.asarray(rows)

    # per-row step features
    step = np.column_stack(
        [
            np.log1p(np.clip(df["TransactionAmt"].to_numpy(dtype=np.float64), 0, None)),
            np.zeros(n),  # log_gap_next is filled per sequence below
            encoder.encode(df["ProductCD"], "ProductCD").astype(np.float64),
            *[_column(df, c, encoder) for c in extra_columns],
        ]
    )
    step = np.nan_to_num(step, nan=0.0)

    # rows of each uid sorted by time; p = number of strictly earlier rows of the same uid
    past = PastIndex(uid.to_numpy(), time)
    order = past.order  # sorted by uid, then time
    start = past.group_start[past.codes[rows]]
    p = past.count_before(time)[rows]

    offsets = p[:, None] - n_steps + np.arange(n_steps)[None, :]  # left-padded positions
    mask = offsets >= 0
    sorted_pos = np.where(mask, start[:, None] + offsets, 0)
    src = order[sorted_pos]  # original row of each step

    values = step[src].astype(np.float64)
    # gap to the next step: the following step, or the transaction itself for the last step
    step_time = time[src].astype(np.float64)
    next_time = np.concatenate([step_time[:, 1:], time[rows].astype(np.float64)[:, None]], axis=1)
    values[:, :, 1] = np.log1p(np.clip(next_time - step_time, 0, None))
    values[~mask] = 0.0

    return Sequences(
        values=values.astype(np.float32),
        mask=mask,
        has_history=mask.any(axis=1),
        feature_names=[*BASE_FEATURES, *extra_columns],
        rows=rows,
    )
