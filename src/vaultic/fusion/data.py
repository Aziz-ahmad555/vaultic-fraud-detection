"""The one place fusion rows are chosen (review M1, research/decisions.md D53).

From the view-prediction table (views/orchestrate.py):
  fit        gate-training rows before the inner time split: every gate is fitted on these
  tune       the latest `tune_fraction` of gate-training DAYS (inner time split): epochs,
             hidden size, dropout, F2's fixed weights, ... are chosen here, never on calibrate
  calibrate  the calibrate tail of validation (days 144-150): per-view and fused calibration,
             conformal calibration, decision thresholds and routing lambdas only
  test       evaluation (final runs only)

Every fusion method F1-F7 and MVAF gets exactly these rows (`fit_all`), so differences between
methods cannot come from different data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vaultic.views.orchestrate import CONTEXT, VIEWS, require_calibrated


@dataclass(frozen=True)
class Rows:
    ids: np.ndarray  # TransactionID
    views: np.ndarray  # (n, 5) view probabilities, NaN where missing
    context: np.ndarray  # (n, len(CONTEXT))
    y: np.ndarray
    day: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class FusionSplit:
    fit: Rows
    tune: Rows
    calibrate: Rows
    test: Rows


def _rows(t: pd.DataFrame) -> Rows:
    t = t.sort_values(["TransactionDT", "TransactionID"], kind="stable")
    return Rows(
        ids=t["TransactionID"].to_numpy(),
        views=t[[f"p_{v}" for v in VIEWS]].to_numpy(dtype=float),
        context=t[list(CONTEXT)].to_numpy(dtype=float),
        y=t["label"].to_numpy(),
        day=t["day"].to_numpy(),
    )


def fusion_split(table: pd.DataFrame, tune_fraction: float = 0.2) -> FusionSplit:
    """Split the view table by role; the tune rows are the latest whole days of gate_train."""
    require_calibrated(table)  # p_* must be calibrated views (D56)
    if not 0 < tune_fraction < 1:
        raise ValueError("tune_fraction must be in (0, 1)")
    gate = table[table["role"] == "gate_train"]
    days = np.sort(gate["day"].unique())
    if len(days) < 2:
        raise ValueError("gate-training rows need at least two days for the inner time split")
    n_tune = max(1, int(round(len(days) * tune_fraction)))
    tune_days = days[-n_tune:]
    is_tune = gate["day"].isin(tune_days)
    return FusionSplit(
        fit=_rows(gate[~is_tune]),
        tune=_rows(gate[is_tune]),
        calibrate=_rows(table[table["role"] == "calibrate"]),
        test=_rows(table[table["role"] == "test"]),
    )


def fit_all(split: FusionSplit, methods: Mapping[str, Callable[[], object]]) -> dict[str, object]:
    """Fit every fusion method on exactly split.fit (views, y, context)."""
    fitted = {}
    for name, make in methods.items():
        model = make()
        model.fit(split.fit.views, split.fit.y, split.fit.context)  # same signature for F1-F7, MVAF
        fitted[name] = model
    return fitted
