"""Kafka settings and topics of the stream (Sprint 3 step 3.5, D245).

Single node (compose service `kafka`, profile `stream`), host listener 127.0.0.1:VAULTIC_KAFKA_PORT (9094).
Four topics, 1 partition each so the order is kept, replication factor 1. A replay run recreates them, so
tx.raw always holds exactly one run.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping

TOPICS = ("tx.raw", "tx.scored", "alerts", "feedback")


def bootstrap(env: Mapping[str, str] | None = None) -> str:
    if env is None:
        from vaultic.infra import db

        db._load_dotenv()
        env = os.environ
    return f"127.0.0.1:{int(env.get('VAULTIC_KAFKA_PORT') or 9094)}"


def admin(env=None):
    from confluent_kafka.admin import AdminClient

    return AdminClient({"bootstrap.servers": bootstrap(env)})


def producer(env=None):
    from confluent_kafka import Producer

    # idempotent, one in-flight batch: the broker keeps the send order exactly
    return Producer({"bootstrap.servers": bootstrap(env), "enable.idempotence": True, "linger.ms": 5,
                     "compression.type": "lz4", "queue.buffering.max.messages": 200_000})  # fmt: skip


def consumer(group: str, env=None):
    from confluent_kafka import Consumer

    return Consumer({"bootstrap.servers": bootstrap(env), "group.id": group,
                     "auto.offset.reset": "earliest", "enable.auto.commit": False,
                     "enable.partition.eof": True})  # fmt: skip


def recreate_topics(env=None, topics=TOPICS, timeout: float = 30.0) -> None:
    """Delete the stream's topics (if present) and create them again, empty: 1 partition, RF 1."""
    from confluent_kafka.admin import NewTopic

    a = admin(env)
    present = [t for t in topics if t in a.list_topics(timeout=10).topics]
    if present:
        for f in a.delete_topics(present, operation_timeout=timeout).values():
            f.result()
        deadline = time.monotonic() + timeout
        while any(t in a.list_topics(timeout=10).topics for t in present):
            if time.monotonic() > deadline:
                raise TimeoutError(f"topics not deleted after {timeout:.0f} s: {present}")
            time.sleep(0.2)
    new = [NewTopic(t, num_partitions=1, replication_factor=1) for t in topics]
    deadline = time.monotonic() + timeout
    while True:
        results = a.create_topics(new, operation_timeout=timeout)
        try:
            for f in results.values():
                f.result()
            return
        except Exception as e:  # noqa: BLE001 - a deletion may still be finishing
            if "already exists" not in str(e).lower() or time.monotonic() > deadline:
                raise
            time.sleep(0.5)
