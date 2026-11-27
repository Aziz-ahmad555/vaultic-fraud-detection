"""Which stores the platform uses, and the PostgreSQL label and decision stores (Sprint 3 step 3.3, D240).

VAULTIC_STORE (read from the environment and .env, like VAULTIC_AUTH) chooses, and fails closed:
- empty: today's behaviour, the SQLite audit log on drive E and labels in memory for the session;
- `postgres`: the audit log, analyst labels and copilot records in PostgreSQL. The start is refused if
  the database is unreachable or migration 0004 is missing: never a silent fallback to SQLite, which
  would split the log in two;
- anything else refuses the start.

Labels (roadmap 11.2): fraud / legit / unsure with time and analyst, one append-only row per label action,
written in the same transaction as its audit entry (fail-closed, D209). "unsure" never enters any
training, and the training reader needs an as-of time and the label delay (D241, A4); nothing here feeds
a model, a metric or a research table (D208).

Decisions (Q10): only the stream scorer's decisions (source `stream`, step 3.5) are written; the frozen
E12 decisions stay in the read-only bundle (D129) and are never copied.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timedelta

from vaultic.audit.pg_store import LABELS, PostgresAuditStore

MODES = {"": "files", "postgres": "postgres"}
ACTIONS = ("allow", "monitor", "step_up", "hold", "block")  # the decisions table's CHECK (0001)


def store_mode(env: Mapping[str, str] | None = None) -> str:
    """'files' or 'postgres'; anything else refuses the start (D240)."""
    if env is None:
        from vaultic.infra import db

        db._load_dotenv()
        env = os.environ
    value = (env.get("VAULTIC_STORE") or "").strip().lower()
    if value not in MODES:
        raise RuntimeError(
            f"VAULTIC_STORE must be empty (SQLite audit log, labels in memory) or 'postgres', not {value!r}"
        )
    return MODES[value]


def postgres_audit_store(url: str | None = None) -> PostgresAuditStore:
    from vaultic.infra import db

    return PostgresAuditStore(url or db.db_url())


def store_banner(store) -> str:
    if isinstance(store, PostgresAuditStore):
        from vaultic.infra.db import redact

        return (
            f"Vaultic store: PostgreSQL ({redact(store.url)}): audit log, labels, copilot records"
        )
    return "Vaultic store: files (SQLite audit log on drive E; labels in memory for this session)"


class LabelStore:
    """Analyst labels in `feedback_labels`, each with its audit entry in one transaction."""

    def __init__(self, audit: PostgresAuditStore):
        self.audit = audit

    def label(self, analyst, role, tid: int, label: str, time: str | None = None) -> dict:
        """Record a label; returns the audit entry. Refused (ValueError) for any other label. `time`
        (ISO, default now) is the label time, as for any audit entry."""
        if label not in LABELS:
            raise ValueError(f"label must be one of {', '.join(LABELS)}")
        tid = int(tid)
        with self.audit.transaction() as conn:
            before = self._current(conn, tid)
            was = before["label"] if before else None
            entry = self.audit.append_in(conn, analyst, role, f"Marked transaction {tid} as {label}",
                                         target=f"tx:{tid}", before=was, after=label,
                                         detail=f"was {was}" if was else "first label",
                                         time=time)  # fmt: skip
            conn.execute("INSERT INTO feedback_labels (audit_id, transaction_id, label, analyst, role, "
                         "labelled_at) VALUES (%s, %s, %s, %s, %s, %s)",
                         (entry["id"], tid, label, entry["analyst"], entry["role"], entry["time"]))  # fmt: skip
        return entry

    @staticmethod
    def _current(conn, tid: int) -> dict | None:
        row = conn.execute("SELECT transaction_id, label, analyst, role, labelled_at, audit_id FROM "
                           "feedback_labels WHERE transaction_id = %s ORDER BY id DESC LIMIT 1",
                           (tid,)).fetchone()  # fmt: skip
        return None if row is None else dict(row)

    def current(self, tid: int) -> dict | None:
        with self.audit.connect() as conn:
            return self._current(conn, int(tid))

    def history(self, tid: int) -> list[dict]:
        """Every label of a transaction, oldest first."""
        with self.audit.connect() as conn:
            rows = conn.execute("SELECT transaction_id, label, analyst, role, labelled_at, audit_id FROM "
                                "feedback_labels WHERE transaction_id = %s ORDER BY id",
                                (int(tid),)).fetchall()  # fmt: skip
        return [dict(r) for r in rows]

    def training_labels(self, as_of, label_delay_days: float) -> dict[int, str]:
        """The labels a model may learn from at `as_of` (CLAUDE.md rule 4, D241 A4): per transaction the
        latest label whose label time is at or before as_of minus the label delay (by label time; 30 days
        elsewhere in the project), and only fraud / legit: a transaction whose latest matured label is
        "unsure" is left out. Both parameters are required. `as_of`: an aware datetime or ISO text.
        """
        cutoff = _aware(as_of) - timedelta(days=float(label_delay_days))
        if label_delay_days < 0:
            raise ValueError("the label delay cannot be negative")
        with self.audit.connect() as conn:
            rows = conn.execute("SELECT DISTINCT ON (transaction_id) transaction_id, label FROM "
                                "feedback_labels WHERE labelled_at::timestamptz <= %s "
                                "ORDER BY transaction_id, labelled_at::timestamptz DESC, id DESC",
                                (cutoff,)).fetchall()  # fmt: skip
        return {int(r["transaction_id"]): r["label"] for r in rows if r["label"] != "unsure"}


def _aware(t) -> datetime:
    d = datetime.fromisoformat(t) if isinstance(t, str) else t
    if not isinstance(d, datetime) or d.tzinfo is None:
        raise ValueError("as_of must be a timezone-aware datetime or ISO text with an offset")
    return d


class DecisionStore:
    """Stream-scored decisions in `decisions` (written from step 3.5). The frozen E12 decisions are
    never written here (Q10, D129)."""

    def __init__(self, url: str | None = None):
        from vaultic.infra import db

        self.url = url or db.db_url()

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.url, autocommit=True, row_factory=dict_row, connect_timeout=5)

    def record(self, tid: int, action: str, risk: float | None, model_version: str,
               source: str = "stream") -> int:  # fmt: skip
        if source != "stream":
            raise ValueError("only stream decisions are written; the frozen E12 decisions stay in the "
                             "read-only bundle (D129, D240)")  # fmt: skip
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {', '.join(ACTIONS)}")
        if not model_version:
            raise ValueError("a decision needs its model version")
        with self._connect() as conn:
            row = conn.execute("INSERT INTO decisions (transaction_id, source, action, risk, model_version) "
                               "VALUES (%s, 'stream', %s, %s, %s) RETURNING id",
                               (int(tid), action, risk, model_version)).fetchone()  # fmt: skip
        return int(row["id"])

    def for_transaction(self, tid: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, transaction_id, source, action, risk, model_version, "
                                "decided_at FROM decisions WHERE transaction_id = %s ORDER BY id",
                                (int(tid),)).fetchall()  # fmt: skip
        return [dict(r) for r in rows]
