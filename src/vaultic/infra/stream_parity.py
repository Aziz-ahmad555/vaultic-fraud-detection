"""Stream parity (Sprint 3 step 3.5, D245): the first N test-period transactions through Kafka, compared with
the frozen offline run.

    python -m vaultic.infra.stream_parity --n 10000        (one heavy job; Kafka running)

The replay producer sends the window (raw columns only) to tx.raw; the stream scorer reads it back in
order and, per transaction, computes the features from its StreamHistory (pre-replay rows + consumed events)
and scores them. Each is compared with the offline frozen features (every column within 1e-6, NaN on the
same columns) and the stored EXP-200-final MVAF and B5 scores (within 1e-12), as D183/D189. The uid
grouping must be the same as offline (codes may differ: a new customer never gets an offline code), and
tx.scored must hold one message per message sent. The history's checks are on (StreamHistory.before()
asserts no row at or after t is used).

The comparison is the only place the stored features, uids and scores are read; the scorer never sees
them. The result (counts, largest differences, failing columns, timings; no row data) goes to
research/tables/platform/stream_parity.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from vaultic.live.parity import FEATURE_TOLERANCE, SCORE_TOLERANCE, _diff

N = 10_000
OUT = Path("research") / "tables" / "platform" / "stream_parity.json"


class Grouping:
    """Checks that the stream's uid codes group transactions exactly as the offline codes do."""

    def __init__(self):
        self.fwd: dict[int, int] = {}
        self.back: dict[int, int] = {}
        self.mismatches = 0

    def add(self, stream_uid: int, offline_uid: int) -> None:
        a, b = self.fwd.setdefault(stream_uid, offline_uid), self.back.setdefault(
            offline_uid, stream_uid
        )
        if a != offline_uid or b != stream_uid:
            self.mismatches += 1


def run(n: int = N, log=print, out: Path | None = OUT, env=None) -> dict:
    from confluent_kafka import KafkaError, TopicPartition

    from vaultic.infra import kafka
    from vaultic.infra.replay import publish, test_rows
    from vaultic.infra.scorer import StreamScorer, load_base
    from vaultic.infra.stream import decode
    from vaultic.paths import REPO_ROOT
    from vaultic.stage2.final import E10_RUN

    t0 = time.perf_counter()
    store, system = load_base(log)
    stored = __import__("pandas").read_parquet(REPO_ROOT / E10_RUN / "predictions.parquet")
    stored = stored.set_index("TransactionID")
    pos_of = dict(
        zip(store.df["TransactionID"].to_numpy().tolist(), range(len(store.df)), strict=True)
    )
    loaded = time.perf_counter() - t0
    log(f"system and store loaded ({loaded:.0f} s)")

    window = test_rows(rows=n)
    stored_raw = set(c for c in store.df.columns if c not in ("isFraud", "day", "has_identity"))
    sent_raw = set(c for c in window.columns if c not in ("isFraud", "day", "has_identity"))
    if stored_raw != sent_raw:
        raise RuntimeError(
            f"replay columns differ from the stored ones: {sorted(stored_raw ^ sent_raw)[:10]}"
        )
    sent = publish(window, speed=None, env=env, log=log)

    scorer = StreamScorer(store, system, decisions=None, log=log)
    c = kafka.consumer(f"vaultic-parity-{sent['run']}", env)
    c.assign([TopicPartition("tx.raw", 0, 0)])
    p = kafka.producer(env)
    cols = None
    worst: dict[str, float] = {}
    score_diff = {"MVAF": 0.0, "B5": 0.0}
    grouping = Grouping()
    decisions: dict[str, int] = {}
    alerts = 0
    done = 0
    started = time.perf_counter()
    try:
        while done < sent["sent"]:
            m = c.poll(5.0)
            if m is None:
                continue
            if m.error():
                if m.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise RuntimeError(str(m.error()))
            result, query, frame = scorer.process(decode(m.value()))
            if done == 0:
                log(f"stream history built ({time.perf_counter() - started:.0f} s)")
                started_scoring = time.perf_counter()
            tid = result["TransactionID"]
            pos = pos_of[tid]
            if cols is None:
                cols = [x for x in frame.columns if x != "uid"]
                missing = sorted(set(store.features.columns) - set(frame.columns))
                if missing:
                    raise RuntimeError(f"stream features lack offline columns: {missing[:10]}")
            a = frame[cols].to_numpy(dtype=float)[0]
            b = store.features.iloc[pos][cols].to_numpy(dtype=float)
            for col, v in zip(cols, _diff(a, b), strict=True):
                if v > worst.get(col, 0.0):
                    worst[col] = float(v)
            score_diff["MVAF"] = max(
                score_diff["MVAF"], abs(result["risk"] - float(stored.at[tid, "score_MVAF"]))
            )
            score_diff["B5"] = max(
                score_diff["B5"], abs(result["b5_score"] - float(stored.at[tid, "score_B5"]))
            )
            grouping.add(int(frame["uid"].iloc[0]), int(store.uid[pos]))
            decisions[result["decision"]] = decisions.get(result["decision"], 0) + 1
            alerts += int(result["alert"])
            p.produce("tx.scored", json.dumps(result).encode())
            if result["alert"]:
                p.produce("alerts", json.dumps(result).encode())
            p.poll(0)
            done += 1
            if done % 250 == 0:
                el = time.perf_counter() - started_scoring
                log(f"stream parity {done:,}/{sent['sent']:,} ({el:.0f} s, {done / el:.2f} tx/s); "
                    f"max feature diff {max(worst.values()) if worst else 0.0:.2e}, "
                    f"MVAF {score_diff['MVAF']:.2e}, B5 {score_diff['B5']:.2e}, uid mismatches {grouping.mismatches}")  # fmt: skip
    finally:
        p.flush(60)
        c.close()
    scored = _count("tx.scored", env)
    lat = np.asarray(scorer.latency[1:])  # the first message also builds the history
    failing = {k: v for k, v in worst.items() if v > FEATURE_TOLERANCE}
    passed = (not failing and all(v <= SCORE_TOLERANCE for v in score_diff.values())
              and grouping.mismatches == 0 and scored == sent["sent"] == done)  # fmt: skip
    h = scorer.history
    result = {
        "step": "Sprint 3 step 3.5 (D245)",
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "passed": bool(passed),
        "n": int(done),
        "window": {"first_day": int(window["TransactionDT"].iloc[0] // 86400),
                   "last_day": int(window["TransactionDT"].iloc[-1] // 86400),
                   "pre_replay_rows": int(h.n_pre)},  # fmt: skip
        "feature_tolerance": FEATURE_TOLERANCE, "score_tolerance": SCORE_TOLERANCE,
        "features_compared": len(cols or []),
        "max_feature_diff": max(worst.values()) if worst else 0.0,
        "n_failing_features": len(failing),
        "failing_features": dict(sorted(failing.items(), key=lambda x: -x[1])[:25]),
        "max_score_diff": score_diff,
        "uid_grouping_mismatches": grouping.mismatches,
        "new_customers": int(h.next_uid - (int(np.max(store.uid)) + 1)),
        "messages_sent": sent["sent"], "tx_scored": int(scored),
        "decisions": decisions, "alerts": alerts,
        "checks": {"before_assertion": h.checks, "order_and_gaps": True, "label_curtain": True},
        "seconds": {"load": round(loaded), "replay_publish": sent["seconds"],
                    "scoring": round(time.perf_counter() - started),
                    "per_tx_p50_ms": float(np.percentile(lat, 50)) if len(lat) else None,
                    "per_tx_p95_ms": float(np.percentile(lat, 95)) if len(lat) else None},  # fmt: skip
        "note": "measured on one laptop CPU with Docker running; per-transaction time is features + "
                "scoring in the scorer process (no explanation), not end-to-end latency",
    }  # fmt: skip
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        log(f"wrote {out}")
    return result


def _count(topic: str, env=None) -> int:
    from vaultic.infra import kafka

    c = kafka.consumer(f"vaultic-count-{time.time_ns()}", env)
    try:
        lo, hi = c.get_watermark_offsets(
            __import__("confluent_kafka").TopicPartition(topic, 0), timeout=10
        )
        return int(hi - lo)
    finally:
        c.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    from disk_space import problems

    bad = problems()
    if bad:
        raise SystemExit("not starting (D239): " + "; ".join(bad))
    r = run(a.n, out=a.out)
    print(json.dumps({k: r[k] for k in ("passed", "n", "max_feature_diff", "max_score_diff",
                                        "uid_grouping_mismatches", "tx_scored")}))  # fmt: skip
    raise SystemExit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
