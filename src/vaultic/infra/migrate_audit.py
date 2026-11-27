"""Copy the SQLite audit log into PostgreSQL, verified at both ends (Sprint 3 step 3.3, D240).

    python -m vaultic.infra.migrate_audit --from <audit.sqlite> [--to <db url>] --dry-run
    python -m vaultic.infra.migrate_audit --from <audit.sqlite> [--to <db url>]

1. The SQLite file is opened read-only (never created, changed or deleted) and its chain and anchor are
   verified; a broken source is refused.
2. The target must have migration 0004 applied and an empty audit log, anchor and record tables; anything
   else is refused.
3. --dry-run stops here and prints what would be copied.
4. Every entry is copied in id order with its id and hashes, plus the copilot record derived from each
   copilot entry (Q13), and the anchor row is set: all in ONE transaction, which also re-verifies the
   PostgreSQL chain and compares the entry count, the last hash and the sha256 of the full CSV export
   with the source and verifies the chain, anchor and copilot records before it commits. Any
   difference rolls everything back.

The real audit DB is migrated only with Aziz's go-ahead (a persistent change). --to defaults to the
connection of vaultic.infra.db (VAULTIC_DB_URL or the POSTGRES_* keys); the password is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

from vaultic.audit.pg_store import PostgresAuditStore, copilot_record
from vaultic.audit.store import SQLiteAuditStore, to_csv


class MigrationRefused(Exception):
    pass


class ReadOnlySQLiteAuditStore(SQLiteAuditStore):
    """The SQLite log opened read-only: no schema, no directory, no write."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.anchor_path = self.path.with_name(self.path.name + ".anchor.json")
        if not self.path.is_file():
            raise MigrationRefused(f"{self.path} does not exist")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def entries(self) -> list[dict]:
        with closing(self._connect()) as con:
            return [dict(r) for r in con.execute("SELECT * FROM audit_log ORDER BY id")]


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
    check = src.verify()
    if not check["ok"]:
        raise MigrationRefused(f"the source log does not verify: {check['reason']}")
    if check["reason"]:
        log(f"note: {check['reason']}")
    rows = src.entries()
    want = summary(rows)
    log(f"source {src.path}: {want['entries']} entries, last hash {want['last_hash']}, "
        f"{want['copilot_records']} copilot record(s), export sha256 {want['export_sha256']}")  # fmt: skip
    with target.connect() as conn:
        _target_empty(conn)
    if dry_run:
        log("dry run: nothing was written")
        return {**want, "dry_run": True}
    with target.transaction() as conn:
        _target_empty(conn)  # again, under the append lock
        for e in rows:
            target._insert(conn, e)
        if rows:
            target._set_anchor(conn, rows[-1]["id"], rows[-1]["hash"])
        got = summary(target.query_in(conn, limit=10**9))
        if got != want:
            raise MigrationRefused(
                f"the copy differs from the source ({got} vs {want}); rolled back"
            )
        after = target.verify_in(conn, snapshot=False)  # chain, anchor and records, before commit
        if not after["ok"]:
            raise MigrationRefused(f"the copy does not verify ({after['reason']}); rolled back")
    log(f"copied {want['entries']} entries; PostgreSQL verifies: {after}")
    return {**want, "dry_run": False, "verify": after}


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
