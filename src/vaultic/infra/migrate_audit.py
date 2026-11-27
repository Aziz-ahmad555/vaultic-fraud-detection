"""Copy the SQLite audit log into PostgreSQL, verified at both ends (Sprint 3 step 3.3, D240).

"Both ends" (D241, wording note): the SQLite source before the copy, and the PostgreSQL copy inside the
migration's transaction, before it commits.

    python -m vaultic.infra.migrate_audit --from <audit.sqlite> [--to <db url>] --dry-run
    python -m vaultic.infra.migrate_audit --from <audit.sqlite> [--to <db url>]

Stop the API server first: a server still writing to the SQLite log would make it grow apart from
its copy (the migration then refuses to finish).

1. The SQLite file is opened read-only (never created, changed or deleted) and its chain and anchor are
   verified; a broken source, a source entry containing NUL (PostgreSQL text cannot hold it, D241 A2) and
   an already migrated source (its marker file exists) are refused.
2. The target must have migration 0004 applied and an empty audit log, anchor and record tables; anything
   else is refused.
3. --dry-run stops here and prints what would be copied.
4. Every entry is copied in id order with its id and hashes, plus the copilot record derived from each
   copilot entry (Q13), and the anchor row is set: all in ONE transaction, which also re-verifies the
   PostgreSQL chain and compares the entry count, the last hash and the sha256 of the full CSV export
   with the source and verifies the chain, anchor and copilot records before it commits. The label
   boundary is set to the copied count (D241 A1: those label entries have no label rows).
5. Before the commit (D241, A3): the marker `<log>.migrated.json` is written next to the SQLite log (the
   log itself is unchanged; from now on SQLiteAuditStore refuses to append to it), then the source is
   re-read under SQLite's write lock (BEGIN IMMEDIATE, then ROLLBACK: nothing is written). If it grew or
   changed during the copy, the marker is removed and everything is rolled back.
   Any difference rolls everything back.

The real audit DB is migrated only with Aziz's go-ahead (a persistent change). --to defaults to the
connection of vaultic.infra.db (VAULTIC_DB_URL or the POSTGRES_* keys); the password is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import threading
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from vaultic.audit.pg_store import PostgresAuditStore, copilot_record, has_nul
from vaultic.audit.store import SQLiteAuditStore, to_csv


class MigrationRefused(Exception):
    pass


class ReadOnlySQLiteAuditStore(SQLiteAuditStore):
    """The SQLite log opened read-only: no schema, no directory, no write."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.anchor_path = self.path.with_name(self.path.name + ".anchor.json")
        self.lock = threading.RLock()  # verify() takes the store's lock (D242)
        if not self.path.is_file():
            raise MigrationRefused(f"{self.path} does not exist")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def entries(self) -> list[dict]:
        with closing(self._connect()) as con:
            return [dict(r) for r in con.execute("SELECT * FROM audit_log ORDER BY id")]


def marker_path(source: str | Path) -> Path:
    """The marker that ends the SQLite log's life (D241, A3); SQLiteAuditStore.migrated_marker()."""
    source = Path(source)
    return source.with_name(source.name + ".migrated.json")


def entries_locked(path: Path) -> list[dict]:
    """The source entries read under SQLite's write lock: an append in progress finishes first and no
    new one can start until the ROLLBACK (nothing is written; the file content is unchanged)."""
    con = sqlite3.connect(path, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    try:
        con.execute("BEGIN IMMEDIATE")
        return [dict(r) for r in con.execute("SELECT * FROM audit_log ORDER BY id")]
    finally:
        con.execute("ROLLBACK")
        con.close()


def digest(rows: list[dict]) -> str:
    """sha256 of the CSV export of the rows, newest first as the export endpoint writes them."""
    ordered = sorted(rows, key=lambda r: r["id"], reverse=True)
    return hashlib.sha256(to_csv(ordered).encode("utf-8")).hexdigest()


def summary(rows: list[dict]) -> dict:
    last = max(rows, key=lambda r: r["id"]) if rows else None  # any order in, the highest id out
    return {"entries": len(rows), "last_hash": last["hash"] if last else None,
            "export_sha256": digest(rows),
            "copilot_records": sum(copilot_record(r) is not None for r in rows)}  # fmt: skip


def _target_empty(conn) -> None:
    for table in ("audit_log", "audit_anchor", "copilot_audit", "feedback_labels"):
        n = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        if n:
            raise MigrationRefused(f"the target is not empty: {table} has {n} row(s)")


def migrate(source: str | Path, target: PostgresAuditStore, dry_run: bool = False,
            log=print) -> dict:  # fmt: skip
    src = ReadOnlySQLiteAuditStore(source)
    marker = marker_path(src.path)
    if marker.exists():
        raise MigrationRefused(f"{src.path} was already migrated ({marker.name} exists)")
    check = src.verify()
    if not check["ok"]:
        raise MigrationRefused(f"the source log does not verify: {check['reason']}")
    if check["reason"]:
        log(f"note: {check['reason']}")
    rows = src.entries()
    bad = [e["id"] for e in rows if has_nul(e)]
    if bad:
        raise MigrationRefused(f"source entries {bad[:5]} contain NUL characters, which PostgreSQL text "
                               "cannot hold; their hashes cover the raw character, so they cannot be "
                               "copied unchanged (D241, A2)")  # fmt: skip
    want = summary(rows)
    log(f"source {src.path}: {want['entries']} entries, last hash {want['last_hash']}, "
        f"{want['copilot_records']} copilot record(s), export sha256 {want['export_sha256']}")  # fmt: skip
    with target.connect() as conn:
        _target_empty(conn)
    if dry_run:
        log("dry run: nothing was written")
        return {**want, "dry_run": True}
    written = False
    try:
        with target.transaction() as conn:
            _copy(conn, target, rows, want)
            after = target.verify_in(
                conn, snapshot=False
            )  # chain, anchor and records, before commit
            if not after["ok"]:
                raise MigrationRefused(f"the copy does not verify ({after['reason']}); rolled back")
            marker.write_text(json.dumps({
                "target": _redacted(target.url), "entries": want["entries"],
                "last_hash": want["last_hash"], "export_sha256": want["export_sha256"],
                "time": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2), "utf-8")  # fmt: skip
            written = True
            if summary(entries_locked(src.path)) != want:  # A3: a writer appended during the copy
                raise MigrationRefused("the SQLite log changed during the copy (stop the server "
                                       "first); rolled back")  # fmt: skip
    except BaseException:
        if written:
            marker.unlink(missing_ok=True)
        raise
    log(f"copied {want['entries']} entries; PostgreSQL verifies: {after}; marker {marker}")
    return {**want, "dry_run": False, "verify": after, "marker": str(marker)}


def _redacted(url: str) -> str:
    from vaultic.infra.db import redact

    return redact(url)


def _copy(conn, target: PostgresAuditStore, rows: list[dict], want: dict) -> None:
    """Inside the target's locked transaction: the entries, the anchor, the label boundary, and the
    comparison with the source."""
    _target_empty(conn)  # again, under the append lock
    for e in rows:
        target._insert(conn, e)
    if rows:
        target._set_anchor(conn, rows[-1]["id"], rows[-1]["hash"])
    # D241 A1: the copied label entries have no label rows (their labels were session-only)
    conn.execute("UPDATE audit_label_boundary SET after_id = %s", (len(rows),))
    got = summary(target.query_in(conn, limit=10**9))
    if got != want:
        raise MigrationRefused(f"the copy differs from the source ({got} vs {want}); rolled back")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from", dest="source", required=True, help="the SQLite audit log")
    p.add_argument("--to", dest="url", default=None, help="PostgreSQL URL (default: .env)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    from vaultic.infra import db
    from vaultic.infra.stores import postgres_audit_store

    try:
        url = a.url or db.db_url()
        print(f"target {db.redact(url)}")
        migrate(a.source, postgres_audit_store(url), dry_run=a.dry_run)
    except (MigrationRefused, db.MigrationError, RuntimeError) as e:
        raise SystemExit(f"refused: {e}") from None


if __name__ == "__main__":
    main()
