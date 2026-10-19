"""The audit store (D208, D209): one interface, SQLite behind it now (PostgreSQL later).

Tamper-EVIDENT, not tamper-proof. Every entry is kept and the SQLite table has triggers that
refuse UPDATE and DELETE. Each entry carries a hash chain,

    hash = SHA-256(prev_hash + canonical JSON of the entry's content),

and a small anchor file next to the database (<db>.anchor.json) holds the latest entry count and
hash, rewritten in the same append under the store's lock. `verify()` therefore detects an entry
that was edited, removed, inserted or reordered inside the chain (hash chain, contiguous ids) and a
tail that was cut off (database behind the anchor).

What it does NOT protect against: someone with file access can drop the triggers and rewrite the
whole chain, and someone who edits both the database and the anchor can hide any change. The log
is attribution and evidence of accidental or casual change, not proof against a determined
attacker with access to the files.

The file lives under the dev cache on drive E (VAULTIC_AUDIT_DB overrides), never in the
repository.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import threading
from contextlib import closing
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
    """A named analyst and a role; anything missing or malformed is recorded as 'unknown'.
    Self-declared by the client, not authenticated (D209): attribution, not proof."""
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

    def export(self, analyst: str | None, role: str | None) -> tuple[dict, list[dict]]: ...


class SQLiteAuditStore:
    """Append-only SQLite audit log with a hash chain and a truncation anchor. Thread-safe within a
    process (one re-entrant lock); each append runs in an immediate (write-locked) transaction."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.anchor_path = self.path.with_name(self.path.name + ".anchor.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with closing(self._connect()) as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    # ---- anchor -----------------------------------------------------------------------------
    def _write_anchor(self, entries: int, last_hash: str) -> None:
        tmp = self.anchor_path.with_name(self.anchor_path.name + ".tmp")
        tmp.write_text(json.dumps({"entries": entries, "hash": last_hash}), "utf-8")
        os.replace(tmp, self.anchor_path)  # atomic on the same volume

    def read_anchor(self) -> dict | None:
        try:
            return json.loads(self.anchor_path.read_text("utf-8"))
        except FileNotFoundError:
            return None

    # ---- writes -----------------------------------------------------------------------------
    def append(self, analyst, role, action, target=None, before=None, after=None, detail="",
               time=None) -> dict:  # fmt: skip
        if not action:
            raise ValueError("an audit entry needs an action")
        name, role = clean_actor(analyst, role)
        entry = {"time": time or datetime.now(UTC).isoformat(timespec="seconds"), "analyst": name,
                 "role": role, "action": str(action), "target": None if target is None else str(target),
                 "before": _value(before), "after": _value(after), "detail": str(detail or "")}  # fmt: skip
        with self.lock, closing(self._connect()) as con:
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
            self._write_anchor(entry["id"], entry["hash"])
        return entry

    def export(self, analyst, role) -> tuple[dict, list[dict]]:
        """Record the export and take the snapshot under one lock, so the count in the entry is
        exactly the number of rows exported (the export entry included)."""
        with self.lock:
            n = self.count()
            entry = self.append(analyst, role, "Exported the audit log", target="audit_log",
                                detail=f"{n + 1} entries (CSV)")  # fmt: skip
            return entry, self.query(limit=10**9)

    # ---- reads ------------------------------------------------------------------------------
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
        with closing(self._connect()) as con:
            rows = con.execute(sql, [*args, int(limit), int(offset)]).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with closing(self._connect()) as con:
            return int(con.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0])

    def verify(self) -> dict:
        """Walk the chain in id order (ids contiguous from 1, each prev_hash equal to the previous
        hash, each hash equal to its recomputed value), then compare with the anchor (truncation).
        Tamper-evident only: see the module docstring for what this cannot detect."""
        prev, expected_id, hashes = GENESIS, 1, {}
        with closing(self._connect()) as con:
            for r in con.execute("SELECT * FROM audit_log ORDER BY id"):
                e = dict(r)
                if e["id"] != expected_id:
                    return {"ok": False, "entries": expected_id - 1, "first_bad": expected_id,
                            "reason": f"entry {expected_id} is missing"}  # fmt: skip
                if e["prev_hash"] != prev or entry_hash(prev, e) != e["hash"]:
                    return {"ok": False, "entries": expected_id - 1, "first_bad": e["id"],
                            "reason": f"entry {e['id']} does not match the chain"}  # fmt: skip
                hashes[e["id"]] = e["hash"]
                prev, expected_id = e["hash"], expected_id + 1
        n = expected_id - 1
        anchor = self.read_anchor()
        if anchor is not None:
            a_n = int(anchor.get("entries", 0))
            if n < a_n:
                return {"ok": False, "entries": n, "first_bad": n + 1,
                        "reason": f"the log has {n} entries but the anchor records {a_n}: "
                                  "entries were removed from the end"}  # fmt: skip
            if a_n and hashes.get(a_n) != anchor.get("hash"):
                return {"ok": False, "entries": n, "first_bad": a_n,
                        "reason": f"entry {a_n} does not match the anchor"}  # fmt: skip
            if n > a_n + 1:
                return {"ok": False, "entries": n, "first_bad": a_n + 1,
                        "reason": f"the anchor records {a_n} entries but the log has {n}"}  # fmt: skip
            if n == a_n + 1:
                return {"ok": True, "entries": n, "first_bad": None,
                        "reason": "the last entry is not yet in the anchor (interrupted write)"}  # fmt: skip
        elif n:
            return {"ok": False, "entries": n, "first_bad": None,
                    "reason": "the anchor file is missing"}  # fmt: skip
        return {"ok": True, "entries": n, "first_bad": None, "reason": None}


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
