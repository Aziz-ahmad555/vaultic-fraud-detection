"""Phase 10.2 injected drifts with a known start day, so detection delay can be measured.

  amount_shift       fraud in one ProductCD changes its amounts by a factor
  new_email_domain   fraud moves to an email domain never seen before
  velocity_mimicry   fraud copies legitimate velocity: its velocity features are replaced by
                     values of legitimate transactions from the same days

Only fraud rows on or after `start_day` change; everything earlier is untouched, which the
tests check. The generators change the given columns only: features derived from them (e.g.
amount ratios) must be rebuilt afterwards with the point-in-time feature code, or listed in
`columns` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NEW_DOMAIN = "drift-new-domain.example"


@dataclass(frozen=True)
class Drift:
    kind: str
    start_day: int
    rows: np.ndarray  # positions of the changed rows


def _targets(df: pd.DataFrame, start_day: int, extra=None, share: float = 1.0, seed: int = 0):
    target = (df["isFraud"].to_numpy() == 1) & (df["day"].to_numpy() >= start_day)
    if extra is not None:
        target &= extra
    if share < 1.0:
        target &= np.random.default_rng(seed).random(len(df)) < share
    return target


def amount_shift(df: pd.DataFrame, product: str, factor: float, start_day: int,
                 columns=("TransactionAmt",)) -> tuple[pd.DataFrame, Drift]:  # fmt: skip
    out = df.copy()
    rows = _targets(df, start_day, (df["ProductCD"].astype(object) == product).to_numpy())
    for col in columns:
        out.loc[rows, col] = out.loc[rows, col].to_numpy(dtype=float) * factor
    return out, Drift("amount_shift", start_day, np.flatnonzero(rows))


def new_email_domain(df: pd.DataFrame, start_day: int, column: str = "P_emaildomain",
                     domain: str = NEW_DOMAIN, share: float = 1.0,
                     seed: int = 0) -> tuple[pd.DataFrame, Drift]:  # fmt: skip
    out = df.copy()
    rows = _targets(df, start_day, share=share, seed=seed)
    out[column] = out[column].astype(object)
    out.loc[rows, column] = domain
    return out, Drift("new_email_domain", start_day, np.flatnonzero(rows))


def velocity_mimicry(df: pd.DataFrame, start_day: int, columns: list[str],
                     seed: int = 0) -> tuple[pd.DataFrame, Drift]:  # fmt: skip
    """Each targeted fraud row takes the velocity values of a random legitimate row of the
    same day (same day, so the replacement is still point-in-time plausible)."""
    out = df.copy()
    rows = _targets(df, start_day)
    rng = np.random.default_rng(seed)
    day = df["day"].to_numpy()
    legit = df["isFraud"].to_numpy() == 0
    for i in np.flatnonzero(rows):
        pool = np.flatnonzero(legit & (day == day[i]))
        if not len(pool):
            continue
        donor = rng.choice(pool)
        out.iloc[i, [out.columns.get_loc(c) for c in columns]] = df.iloc[donor][columns].to_numpy()
    return out, Drift("velocity_mimicry", start_day, np.flatnonzero(rows))
