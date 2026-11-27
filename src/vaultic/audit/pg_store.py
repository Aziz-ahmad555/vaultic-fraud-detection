"""The audit store on PostgreSQL (Sprint 3 step 3.3, D240): the AuditStore protocol of store.py, with the
same hash chain (the same `entry_hash` over the same canonical JSON, so a migrated entry keeps its hash)
and the same policies (labels fail-closed, everything else fail-open, D209).

- **Appends** are serialised by a transaction-level advisory lock, across threads and processes.
- **The truncation anchor** is the one row of `audit_anchor`, updated in the same transaction as each
  append, so the log and its anchor never disagree after a commit.
- **Linked records (Q13).** A copilot entry (`Copilot question` / `Copilot summary`) gets its
  `copilot_audit` row in the same transaction; a label entry written through `labels()` gets its
  `feedback_labels` row. Each row is derived from the content of its hash-chained entry, and `verify()`
  recomputes every row from its entry, so the chain covers the records too. Both tables are append-only
  by trigger (migration 0004).

Tamper-EVIDENT, not tamper-proof, as the SQLite log: someone with database rights can drop the triggers
and rewrite the log, the anchor and the records together.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from datetime import UTC, datetime

from vaultic.audit.store import FIELDS, GENESIS, _value, clean_actor, entry_hash

LOCK_KEY = 2310_0004  # pg_advisory_xact_lock key of the appends (D240); the runner uses 2310_0001
# the copilot's audit actions (src/vaultic/copilot/service.py ACTIONS, E20-frozen) -> request kind
COPILOT_ACTIONS = {"Copilot question": "question", "Copilot summary": "summary"}
LABELS = ("fraud", "legit", "unsure")
LABEL_ACTION = re.compile(r"^Marked transaction (-?\d+) as (fraud|legit|unsure)$")
_COLS = ", ".join(f'"{k}"' for k in FIELDS)


class StoreSetupError(RuntimeError):
    """The database cannot hold the PostgreSQL stores (unreachable, or migration 0004 missing)."""


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def copilot_record(entry: dict) -> dict | None:
    """The copilot record of an entry, derived only from what the entry's hash covers (Q13); None for
    any other entry."""
    kind = COPILOT_ACTIONS.get(entry.get("action"))
    if kind is None:
        return None
    after = entry.get("after")
    return {"audit_id": int(entry["id"]), "kind": kind, "target": entry.get("target"),
            "question": entry.get("detail"), "analyst": entry.get("analyst"),
            "role": entry.get("role"), "time": entry.get("time"),
            "result": None if after is None else json.loads(after)}  # fmt: skip


def record_sha256(record: dict) -> str:
    return hashlib.sha256(canonical(record).encode("utf-8")).hexdigest()


def label_of(entry: dict) -> tuple[int, str] | None:
    """(transaction id, label) of a label entry, else None."""
    m = LABEL_ACTION.match(entry.get("action") or "")
    return (int(m.group(1)), m.group(2)) if m else None


class PostgresAuditStore:
    """Append-only PostgreSQL audit log with a hash chain, an anchor row and linked records. One
    connection per call (autocommit; each write is one explicit transaction)."""

    def __init__(self, url: str, check: bool = True):
        self.url = url
        if check:
            self.check_schema()

    # ---- connections ------------------------------------------------------------------------
    def connect(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.url, autocommit=True, row_factory=dict_row, connect_timeout=5)

    def check_schema(self) -> None:
        try:
            with self.connect() as conn:
                found = conn.execute("SELECT to_regclass('audit_anchor') AS t").fetchone()["t"]
        except Exception as e:  # noqa: BLE001 - reported as a start-up refusal
            raise StoreSetupError(f"PostgreSQL is not reachable ({type(e).__name__})") from e
        if found is None:
            raise StoreSetupError(
                "migration 0004_records.sql is not applied: run `python -m vaultic.infra.db migrate`"
            )

    @contextmanager
    def transaction(self, conn=None):
        """A write transaction holding the append lock; yields its connection. Everything written in
        it (the entry, the anchor, linked records) commits together or not at all."""
        if conn is not None:
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
                yield conn
            return
        with self.connect() as own, own.transaction():
            own.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
            yield own

    # ---- writes -----------------------------------------------------------------------------
    def append_in(self, conn, analyst, role, action, target=None, before=None, after=None,
                  detail="", time=None) -> dict:  # fmt: skip
        """Append inside a transaction() that already holds the lock."""
        if not action:
            raise ValueError("an audit entry needs an action")
        name, role = clean_actor(analyst, role)
        entry = {"time": time or datetime.now(UTC).isoformat(timespec="seconds"), "analyst": name,
                 "role": role, "action": str(action), "target": None if target is None else str(target),
                 "before": _value(before), "after": _value(after), "detail": str(detail or "")}  # fmt: skip
        last = conn.execute("SELECT id, hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        entry["id"] = 1 if last is None else int(last["id"]) + 1
        entry["prev_hash"] = GENESIS if last is None else last["hash"]
        entry["hash"] = entry_hash(entry["prev_hash"], entry)
        self._insert(conn, entry)
        self._set_anchor(conn, entry["id"], entry["hash"])
        return entry

    def _insert(self, conn, entry: dict) -> None:
        """One entry exactly as given (id and hashes included) plus its copilot record, if any."""
        conn.execute(f"INSERT INTO audit_log ({_COLS}) VALUES ({', '.join('%s' for _ in FIELDS)})",
                     [entry[k] for k in FIELDS])  # fmt: skip
        record = copilot_record(entry)
        if record is not None:
            conn.execute(
                "INSERT INTO copilot_audit (audit_id, record, record_sha256) VALUES (%s, %s, %s)",
                (entry["id"], canonical(record), record_sha256(record)),
            )

    @staticmethod
    def _set_anchor(conn, entries: int, last_hash: str) -> None:
        conn.execute("INSERT INTO audit_anchor (one, entries, hash) VALUES (TRUE, %s, %s) "
                     "ON CONFLICT (one) DO UPDATE SET entries = EXCLUDED.entries, hash = EXCLUDED.hash",
                     (entries, last_hash))  # fmt: skip

    def append(self, analyst, role, action, target=None, before=None, after=None, detail="",
               time=None) -> dict:  # fmt: skip
        with self.transaction() as conn:
            return self.append_in(conn, analyst, role, action, target=target, before=before,
                                  after=after, detail=detail, time=time)  # fmt: skip

    def export(self, analyst, role) -> tuple[dict, list[dict]]:
        """The export entry and the snapshot in one locked transaction: the count is exact."""
        with self.transaction() as conn:
            n = int(conn.execute("SELECT COUNT(*) AS n FROM audit_log").fetchone()["n"])
            entry = self.append_in(conn, analyst, role, "Exported the audit log", target="audit_log",
                                   detail=f"{n + 1} entries (CSV)")  # fmt: skip
            rows = conn.execute(f"SELECT {_COLS} FROM audit_log ORDER BY id DESC").fetchall()
        return entry, [dict(r) for r in rows]

    # ---- reads ------------------------------------------------------------------------------
    def query(self, analyst=None, action=None, since=None, until=None, limit=500, offset=0):
        with self.connect() as conn:
            return self.query_in(conn, analyst, action, since, until, limit, offset)

    @staticmethod
    def query_in(conn, analyst=None, action=None, since=None, until=None, limit=500, offset=0):
        where, args = [], []
        if analyst:
            where.append("analyst = %s")
            args.append(analyst)
        if action:  # SQLite's LIKE ignores ASCII case; ILIKE matches that
            where.append("action ILIKE %s")
            args.append(f"%{action}%")
        if since:
            where.append('"time" >= %s')
            args.append(since)
        if until:
            where.append('"time" <= %s')
            args.append(until)
        sql = f"SELECT {_COLS} FROM audit_log" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY id DESC LIMIT %s OFFSET %s"
        return [dict(r) for r in conn.execute(sql, [*args, int(limit), int(offset)]).fetchall()]

    def count(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM audit_log").fetchone()["n"])

    def read_anchor(self) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT entries, hash FROM audit_anchor").fetchone()
        return None if row is None else {"entries": int(row["entries"]), "hash": row["hash"]}

    def verify(self) -> dict:
        """The chain, the anchor and the linked records, read in one consistent snapshot."""
        with self.connect() as conn:
            return self.verify_in(conn)

    @staticmethod
    def verify_in(conn, snapshot: bool = True) -> dict:
        """Walk the chain in id order as SQLiteAuditStore.verify does, compare the anchor row (in
        PostgreSQL it is written in the same transaction, so it must match exactly), then check every
        copilot_audit and feedback_labels row against its entry (Q13). Tamper-evident only.
        snapshot=False runs inside the caller's open transaction (the migration checks before commit).
        """
        records = {"copilot": 0, "labels": 0}

        def bad(n, first, reason):
            return {"ok": False, "entries": n, "first_bad": first, "reason": reason,
                    "records": records}  # fmt: skip

        prev, expected_id, hashes, copilot, labels = GENESIS, 1, {}, {}, {}
        with conn.transaction():  # a savepoint inside the caller's transaction when snapshot=False
            if snapshot:
                conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            with conn.cursor(name="audit_walk") as cur:
                cur.execute(f"SELECT {_COLS} FROM audit_log ORDER BY id")
                for e in cur:
                    e = dict(e)
                    if e["id"] != expected_id:
                        return bad(expected_id - 1, expected_id, f"entry {expected_id} is missing")
                    if e["prev_hash"] != prev or entry_hash(prev, e) != e["hash"]:
                        return bad(expected_id - 1, e["id"],
                                   f"entry {e['id']} does not match the chain")  # fmt: skip
                    hashes[e["id"]] = e["hash"]
                    rec = copilot_record(e)
                    if rec is not None:
                        copilot[e["id"]] = rec
                    lab = label_of(e)
                    if lab is not None:
                        labels[e["id"]] = (*lab, e["analyst"], e["role"], e["time"])
                    prev, expected_id = e["hash"], expected_id + 1
            n = expected_id - 1
            anchor = conn.execute("SELECT entries, hash FROM audit_anchor").fetchone()
            crows = conn.execute(
                "SELECT audit_id, record, record_sha256 FROM copilot_audit ORDER BY audit_id"
            ).fetchall()
            lrows = conn.execute("SELECT audit_id, transaction_id, label, analyst, role, labelled_at "
                                 "FROM feedback_labels ORDER BY audit_id").fetchall()  # fmt: skip
        if anchor is None:
            if n:
                return bad(n, None, "the anchor row is missing")
        else:
            a_n = int(anchor["entries"])
            if n < a_n:
                return bad(n, n + 1, f"the log has {n} entries but the anchor records {a_n}: "
                                     "entries were removed from the end")  # fmt: skip
            if a_n and hashes.get(a_n) != anchor["hash"]:
                return bad(n, a_n, f"entry {a_n} does not match the anchor")
            if n > a_n:
                return bad(n, a_n + 1, f"the anchor records {a_n} entries but the log has {n}")
        seen = set()
        for r in crows:
            aid = int(r["audit_id"])
            want = copilot.get(aid)
            if want is None:
                return bad(n, aid, f"copilot record {aid} has no copilot entry")
            if r["record"] != json.loads(canonical(want)) or r["record_sha256"] != record_sha256(
                want
            ):
                return bad(n, aid, f"copilot record {aid} does not match its entry")
            seen.add(aid)
        missing = sorted(set(copilot) - seen)
        if missing:
            return bad(n, missing[0], f"copilot entry {missing[0]} has no copilot record")
        for r in lrows:
            aid = int(r["audit_id"])
            row = (int(r["transaction_id"]), r["label"], r["analyst"], r["role"], r["labelled_at"])
            if labels.get(aid) != row:
                return bad(n, aid, f"label row {aid} does not match its entry")
        records.update(copilot=len(crows), labels=len(lrows))
        return {"ok": True, "entries": n, "first_bad": None, "reason": None, "records": records}
