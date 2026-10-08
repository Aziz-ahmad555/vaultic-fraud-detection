"""Phase 10.2 label-delay simulator and detection scoring.

A transaction's label becomes known L days after the transaction (L = 0, 7, 30, 60). Label-
based detectors therefore see the error of a transaction only at time t + L, in that order;
label-free detectors see each day's scores and features the same day.

Detection scoring for an injected drift with a known start day:
  delay         days from the drift start to the first alarm on or after it
  false alarms  alarms before the start, scaled to a 6-month (182-day) period
  missed        no alarm on or after the start
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY

DELAYS = (0, 7, 30, 60)


def matured_error_stream(time, y, p, threshold: float, delay_days: int) -> pd.DataFrame:
    """The 0/1 error of each scored transaction, ordered by when its label arrives
    (time + L days; ties keep transaction order)."""
    time = np.asarray(time, dtype=np.int64)
    arrival = time + int(delay_days) * SECONDS_PER_DAY
    error = ((np.asarray(p) >= threshold).astype(int) != np.asarray(y).astype(int)).astype(int)
    order = np.argsort(arrival, kind="stable")
    return pd.DataFrame({"arrival": arrival[order], "arrival_day": arrival[order] // SECONDS_PER_DAY,
                         "error": error[order], "row": order})  # fmt: skip


def label_based_alarm_days(
    stream: pd.DataFrame, detector, last_day: int | None = None
) -> list[int]:
    """Days on which the detector fired while reading the matured error stream."""
    out = []
    for day, err in zip(stream["arrival_day"].to_numpy(), stream["error"].to_numpy(), strict=True):
        if last_day is not None and day > last_day:
            break
        if detector.update(err):
            out.append(int(day))
    return sorted(set(out))


def label_free_alarm_days(
    day,
    reference: np.ndarray,
    days: list[int],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    threshold: float,
    window_days: int = 7,
) -> pd.DataFrame:
    """For each monitored day d: statistic(reference rows, rows of the window_days days ending
    at d) and whether it exceeds the threshold. `reference` is a row mask of the reference
    period (e.g. the champion's last training month)."""
    day = np.asarray(day)
    rows = []
    for d in days:
        current = (day > d - window_days) & (day <= d)
        value = statistic(reference, current) if current.any() else np.nan
        rows.append({"day": int(d), "statistic": value, "alarm": bool(value > threshold)})
    return pd.DataFrame(rows)


def detection_report(alarm_days, start_day: int, first_day: int, last_day: int) -> dict:
    a = np.asarray(sorted(alarm_days), dtype=float)
    a = a[(a >= first_day) & (a <= last_day)]
    before, after = a[a < start_day], a[a >= start_day]
    pre_days = max(start_day - first_day, 1)
    return {
        "delay_days": float(after[0] - start_day) if len(after) else np.nan,
        "false_alarms": int(len(before)),
        "false_alarms_per_6_months": float(len(before) * 182 / pre_days),
        "missed": not len(after),
    }
