"""The audit store (D208): one interface, SQLite behind it now (PostgreSQL later).

Every entry is kept. There is no update or delete path: the SQLite table has triggers that refuse
UPDATE and DELETE. Each entry carries a hash chain:

    hash = SHA-256(prev_hash + canonical JSON of the entry's content)

so `verify()` finds the first entry that was altered, inserted or removed (ids are contiguous from
1). The file lives under the dev cache on drive E (VAULTIC_AUDIT_DB overrides), never in the
repository.

    store = default_store()
    store.append(analyst="Aziz", role="analyst", action="label", target="tx:3500000",
                 before=None, after="fraud", detail="first label")
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

GENESIS = "0" * 64
FIELDS = ("id", "time", "analyst", "role", "action", "target", "before", "after", "detail",
          "prev_hash", "hash")  # fmt: skip
UNKNOWN = "unknown"
ROLES = ("analyst", "admin")
MAX_NAME = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY,
    time      TEXT NOT NULL,
    analyst   TEXT NOT NULL,
    role      TEXT NOT NULL,
    action    TEXT NOT NULL,
    target    TEXT,
    before    TEXT,
    after     TEXT,
    detail    TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_time ON audit_log (time);
CREATE INDEX IF NOT EXISTS audit_analyst ON audit_log (analyst);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'the audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'the audit log is append-only'); END;
"""


def entry_hash(prev_hash: str, entry: dict) -> str:
    content = {k: entry.get(k) for k in FIELDS if k not in ("prev_hash", "hash")}
    payload = prev_hash + json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clean_actor(analyst: str | None, role: str | None) -> tuple[str, str]:
    """A named analyst and a role; anything missing or malformed is recorded as 'unknown'."""
    name = " ".join(str(analyst or "").split())[:MAX_NAME]
    name = name if name and name.isprintable() else UNKNOWN
    role = role if role in ROLES else UNKNOWN
    return name, role


def _value(x) -> str | None:
    """before / after are stored as JSON text (None stays NULL)."""
    return None if x is None else json.dumps(x, sort_keys=True, default=str)


class AuditStore(Protocol):
    def append(self, analyst: str | None, role: str | None, action: str, target: str | None = None,
    before=None, after=None, detail: str = "", time: str | None = None) -> dict: ...  # fmt: skip

    def query(self, analyst: str | None = None, action: str | None = None, since: str | None = None,
    until: str | None = None, limit: int = 500, offset: int = 0) -> list[dict]: ...  # fmt: skip

    def count(self) -> int: ...

    def verify(self) -> dict: ...


class SQLiteAuditStore:
    """Append-only SQLite audit log with a hash chain. Thread-safe within a process (one lock);
    across processes each append runs in an immediate (write-locked) transaction."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def append(self, analyst, role, action, target=None, before=None, after=None, detail="",
               time=None) -> dict:  # fmt: skip
        if not action:
            raise ValueError("an audit entry needs an action")
        name, role = clean_actor(analyst, role)
        entry = {"time": time or datetime.now(UTC).isoformat(timespec="seconds"), "analyst": name,
                 "role": role, "action": str(action), "target": None if target is None else str(target),
                 "before": _value(before), "after": _value(after), "detail": str(detail or "")}  # fmt: skip
        with self.lock:
            con = self._connect()
            try:
                con.execute("BEGIN IMMEDIATE")
                last = con.execute(
                    "SELECT id, hash FROM audit_log ORDER BY id DESC LIMIT 1"
                ).fetchone()
                entry["id"] = 1 if last is None else int(last["id"]) + 1
                entry["prev_hash"] = GENESIS if last is None else last["hash"]
                entry["hash"] = entry_hash(entry["prev_hash"], entry)
                con.execute(f"INSERT INTO audit_log ({', '.join(FIELDS)}) VALUES "
                            f"({', '.join('?' for _ in FIELDS)})", [entry[k] for k in FIELDS])  # fmt: skip
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            finally:
                con.close()
        return entry

    def query(self, analyst=None, action=None, since=None, until=None, limit=500, offset=0):
        where, args = [], []
        if analyst:
            where.append("analyst = ?")
            args.append(analyst)
        if action:
            where.append("action LIKE ?")
            args.append(f"%{action}%")
        if since:
            where.append("time >= ?")
            args.append(since)
        if until:
            where.append("time <= ?")
            args.append(until)
        sql = "SELECT * FROM audit_log" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        with self._connect() as con:
            rows = con.execute(sql, [*args, int(limit), int(offset)]).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0])

    def verify(self) -> dict:
        """Walk the chain in id order: every id contiguous from 1, every prev_hash equal to the
        previous entry's hash, every hash equal to its recomputed value."""
        prev, expected_id = GENESIS, 1
        with self._connect() as con:
            for r in con.execute("SELECT * FROM audit_log ORDER BY id"):
                e = dict(r)
                if e["id"] != expected_id:
                    return {"ok": False, "entries": expected_id - 1, "first_bad": expected_id,
                            "reason": f"entry {expected_id} is missing"}  # fmt: skip
                if e["prev_hash"] != prev or entry_hash(prev, e) != e["hash"]:
                    return {"ok": False, "entries": expected_id - 1, "first_bad": e["id"],
                            "reason": f"entry {e['id']} does not match the chain"}  # fmt: skip
                prev, expected_id = e["hash"], expected_id + 1
        return {"ok": True, "entries": expected_id - 1, "first_bad": None, "reason": None}


def to_csv(entries: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(FIELDS), extrasaction="ignore")
    w.writeheader()
    for e in entries:
        w.writerow(e)
    return buf.getvalue()


def default_path() -> Path:
    env = os.environ.get("VAULTIC_AUDIT_DB")
    if env:
        return Path(env)
    from vaultic.live.store import cache_root

    return cache_root() / "audit" / "audit.sqlite"


def default_store() -> SQLiteAuditStore:
    return SQLiteAuditStore(default_path())
