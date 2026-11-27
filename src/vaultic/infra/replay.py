"""Replay producer (Sprint 3 step 3.5, D245): test-period transactions into Kafka's tx.raw, in time order.

    python -m vaultic.infra.replay --start-day 151 --rows 3000 --speed max
    python -m vaultic.infra.replay --start-day 151 --end-day 152 --speed 1000

- Reads the merged IEEE-CIS rows behind the post-hoc guard (D144: test-period rows only once the final
  run is on record), keeps the test period, and sorts by TransactionDT with ties by TransactionID (the
  stored order).
- Publishes ONLY the raw transaction and identity columns: no isFraud, no derived day or has_identity, no
  feature, no score. Each message carries the run id and a sequence number from 0.
- Speed: 1, 100 or 1000 times real time, or `max` (as fast as possible).
- Recreates the four topics first, so tx.raw holds exactly this run.
"""

from __future__ import annotations

import argparse
import time
import uuid

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.infra.stream import encode, raw_columns

SPEEDS = {"1": 1.0, "100": 100.0, "1000": 1000.0, "max": None}


def test_rows(start_day: int | None = None, rows: int | None = None, end_day: int | None = None,
              columns: list[str] | None = None) -> pd.DataFrame:  # fmt: skip
    """The replay window: test-period rows from start_day, `rows` of them or up to end_day."""
    from vaultic.data.load import MERGED_PATH, load_merged
    from vaultic.data.splits import load_splits
    from vaultic.eval.posthoc_guard import require_final_on_record

    require_final_on_record()  # D144: before any test-period row is read
    splits = load_splits()
    df = load_merged(MERGED_PATH, columns=columns)
    day = (df["TransactionDT"].to_numpy() // SECONDS_PER_DAY).astype(int)
    first = splits.test.first if start_day is None else int(start_day)
    if first < splits.test.first:
        raise ValueError(f"the replay starts in the test period (day {splits.test.first} or later)")
    keep = (day >= first) & (day <= (splits.test.last if end_day is None else int(end_day)))
    df = df[keep].sort_values(["TransactionDT", "TransactionID"], kind="stable")
    if rows is not None:
        df = df.iloc[: int(rows)]
    if df.empty:
        raise ValueError("the replay window is empty")
    return df.reset_index(drop=True)


def publish(df: pd.DataFrame, speed: float | None = None, run: str | None = None, env=None, log=print,
            recreate: bool = True, topics=None) -> dict:  # fmt: skip
    """Send the window's rows to tx.raw (raw columns only); returns the run id and counts. `topics`
    (default kafka.TOPICS, the first being the raw topic) lets the tests use their own topics."""
    from vaultic.infra import kafka

    topics = tuple(topics or kafka.TOPICS)
    run = run or uuid.uuid4().hex[:12]
    cols = raw_columns(df.columns)
    if recreate:
        kafka.recreate_topics(env, topics)
    p = kafka.producer(env)
    t = df["TransactionDT"].to_numpy(dtype=np.int64)
    started = time.perf_counter()
    for i, (_, row) in enumerate(df.iterrows()):
        if speed is not None and i:
            due = (t[i] - t[0]) / speed
            wait = due - (time.perf_counter() - started)
            if wait > 0:
                time.sleep(wait)
        p.produce(topics[0], encode(i, run, row, cols))
        if i % 1000 == 0:
            p.poll(0)
    left = p.flush(60)
    if left:
        raise RuntimeError(f"{left} replay messages were not delivered")
    out = {"run": run, "sent": int(len(df)), "first_time": int(t[0]), "last_time": int(t[-1]),
           "seconds": round(time.perf_counter() - started, 1)}  # fmt: skip
    log(f"replay {run}: {out['sent']:,} messages in {out['seconds']} s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start-day", type=int, default=None, help="first day (default: test start)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rows", type=int, default=None)
    g.add_argument("--end-day", type=int, default=None)
    ap.add_argument("--speed", choices=sorted(SPEEDS), default="max")
    a = ap.parse_args()
    publish(test_rows(a.start_day, a.rows, a.end_day), SPEEDS[a.speed])


if __name__ == "__main__":
    main()
