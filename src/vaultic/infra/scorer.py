"""Stream scorer (Sprint 3 step 3.5, D245): tx.raw in order -> features -> the frozen system -> tx.scored.

    python -m vaultic.infra.scorer            (Kafka running: docker compose --profile stream up -d)

For each message, in order:
1. the sequence must be the next one and TransactionDT must not decrease (else it stops);
2. no consumed row may have passed its label curtain without a label (else it stops);
3. the features are computed from the StreamHistory (pre-replay rows + consumed events only) with the
   Test Lab's online feature code, BEFORE the event is added;
4. the frozen stage-2 system scores it: MVAF risk (5-seed mean), the E12 decision and its alert (step-up or
   higher, D159), B5;
5. the result goes to tx.scored, an alert to `alerts`, and with VAULTIC_STORE=postgres a row to
   `decisions` (source stream, D240); then the event is added to the history.
Labels arrive on `feedback` ({"TransactionID": ..., "label": 0 or 1}) and become visible only after their
curtain (its_time + L <= t).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from vaultic.infra.stream import StreamHistory, decode, row_from_message

ACTIONS = ("allow", "monitor", "step_up", "hold", "block")
MODEL_VERSION = "stage2-frozen MVAF + E12 (D179, D182)"


def score_frame(system, query: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """The Test Lab engine's scoring steps (live/engine.LiveEngine.score) on a computed feature row."""
    from vaultic.fusion.mvaf import view_inputs
    from vaultic.trust.conformal import set_labels
    from vaultic.trust.decision import STEP_UP, decide

    q = query.reset_index(drop=True)
    ctx = system.context(q, frame)
    avail = system.availability(frame)
    s = system.score(frame, ctx, avail)
    pe = system.fused_calibrator.predict(s["risk"])
    conf = set_labels(system.conformal.predict_sets(pe))
    dis = view_inputs(s["p"])[3]
    action = int(decide(pe, q["TransactionAmt"].to_numpy(float), system.thresholds, conf, dis)[0])
    return {"risk": float(s["risk"][0]), "calibrated": float(pe[0]), "conformal_set": str(conf[0]),
            "decision": ACTIONS[action], "alert": bool(action >= STEP_UP),
            "b5_score": float(system.b5(frame)[0][0])}  # fmt: skip


class StreamScorer:
    """Scores replay messages one at a time over a StreamHistory built at the first message."""

    def __init__(self, base_store, system, decisions=None, log=print):
        self.base, self.system, self.decisions, self.log = base_store, system, decisions, log
        self.history: StreamHistory | None = None
        self.run: str | None = None
        self.latency: list[float] = []
        self.pending: dict[int, int] = {}  # labels that arrived before their transaction

    def start(self, start_time: int, run: str) -> None:
        self.history = StreamHistory.from_store(self.base, start_time, log=self.log)
        self.run = run

    def process(self, msg: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        """Score one decoded message and add it to the history. Returns (result, query row, features)."""
        from vaultic.live.features import online_features

        started = time.perf_counter()
        query = row_from_message(msg["tx"], self.base.df.iloc[:0])
        t = int(query["TransactionDT"].iloc[0])
        if self.history is None:
            self.start(t, msg["run"])
        if msg["run"] != self.run:
            raise RuntimeError(f"message of run {msg['run']} in run {self.run}: tx.raw mixes runs")
        h = self.history
        h.check_next(msg["seq"], t)  # 1. order, before anything is computed
        h.require_labels(t)  # 2. the curtain
        uid = h.uid_for(query)
        frame = online_features(h, self.system, query, uid)  # 3. before the event is added
        result = score_frame(self.system, query, frame)  # 4.
        h.consume(msg["seq"], query, uid)  # 5. only now is the event part of the history
        tid = int(query["TransactionID"].iloc[0])
        if tid in self.pending:
            h.label(tid, self.pending.pop(tid))
        result.update(TransactionID=int(query["TransactionID"].iloc[0]), TransactionDT=t,
                      seq=int(msg["seq"]), run=self.run,
                      latency_ms=(time.perf_counter() - started) * 1000)  # fmt: skip
        self.latency.append(result["latency_ms"])
        if self.decisions is not None:
            self.decisions.record(result["TransactionID"], result["decision"], result["risk"],
                                  MODEL_VERSION)  # fmt: skip
        return result, query, frame

    def feedback(self, payload: bytes) -> None:
        """A label for a consumed transaction; one for a transaction not consumed yet waits for it."""
        m = json.loads(payload)
        tid, label = int(m["TransactionID"]), int(m["label"])
        if self.history is None or tid not in self.history.by_tid:
            self.pending[tid] = label
            return
        self.history.label(tid, label)


def load_base(log=print):
    """The frozen system and the full history store (both from their caches on drive E); the stream
    history is cut from the store at the first message."""
    from vaultic.live.engine import load_system
    from vaultic.live.parity import require_inputs
    from vaultic.live.store import cached_store

    require_inputs()
    return cached_store(log), load_system(log)


def run(env=None, log=print, max_messages: int | None = None) -> dict:
    """Consume tx.raw from its start until stopped (or max_messages); feedback is applied as it comes."""
    from confluent_kafka import KafkaError

    from vaultic.infra import kafka
    from vaultic.infra.stores import store_mode

    store, system = load_base(log)
    decisions = None
    if store_mode() == "postgres":
        from vaultic.infra.stores import DecisionStore

        decisions = DecisionStore()
    scorer = StreamScorer(store, system, decisions, log)
    c = kafka.consumer(f"vaultic-scorer-{int(time.time())}", env)
    c.subscribe(["tx.raw", "feedback"])
    p = kafka.producer(env)
    n = 0
    try:
        while max_messages is None or n < max_messages:
            m = c.poll(1.0)
            if m is None:
                continue
            if m.error():
                if m.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise RuntimeError(str(m.error()))
            if m.topic() == "feedback":
                scorer.feedback(m.value())
                continue
            result, _, _ = scorer.process(decode(m.value()))
            p.produce("tx.scored", json.dumps(result).encode())
            if result["alert"]:
                p.produce("alerts", json.dumps(result).encode())
            p.poll(0)
            n += 1
            if n % 500 == 0:
                lat = np.percentile(scorer.latency[-500:], [50, 95])
                log(f"scored {n:,} (last 500: p50 {lat[0]:.0f} ms, p95 {lat[1]:.0f} ms)")
    finally:
        p.flush(30)
        c.close()
    return {"scored": n}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
