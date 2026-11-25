"""PostgreSQL connection settings and the plain-SQL migration runner (Sprint 3 step 3.1, D231).

Migrations are the files services/db/migrations/NNNN_name.sql, numbered from 0001 without gaps. Each
runs once, in its own transaction, under an advisory lock (two runners never interleave); the table
schema_migrations records its name and sha256 (line endings normalised). An applied migration whose
file changed or disappeared, or a gap or duplicate in the numbering, stops the runner before it
touches anything: fix it with a new migration, never by editing an applied one.

    python -m vaultic.infra.db status
    python -m vaultic.infra.db migrate

The connection comes from VAULTIC_DB_URL, else from POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD
and the port VAULTIC_POSTGRES_PORT (or the older POSTGRES_PORT; default 5442) on 127.0.0.1, read from
the environment and the repository's .env. The password is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

REPO = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = REPO / "services" / "db" / "migrations"
DEFAULT_PORT = 5442  # D231 (amends D14's 55432)
FILE_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
LOCK_KEY = 2310_0001  # pg_advisory_lock key of the runner (D231)
STATE_TABLE = """CREATE TABLE IF NOT EXISTS schema_migrations (
    name       TEXT PRIMARY KEY,
    sha256     TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)"""


class MigrationError(Exception):
    """A refusal: the migrations on disk and in the database do not fit together."""


@dataclass(frozen=True)
class Migration:
    number: int
    name: str
    path: Path
    sha256: str


def file_sha256(path: Path) -> str:
    """sha256 of the file with line endings normalised, so a CRLF checkout hashes like LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """The migration files in order; refuses a badly named file, a duplicate number or a gap."""
    found = []
    for p in sorted(directory.glob("*.sql")):
        m = FILE_NAME.match(p.name)
        if not m:
            raise MigrationError(f"{p.name}: migration files are named NNNN_lowercase_name.sql")
        found.append(Migration(int(m.group(1)), p.name, p, file_sha256(p)))
    numbers = [m.number for m in found]
    if len(set(numbers)) != len(numbers):
        raise MigrationError(f"duplicate migration numbers in {directory}")
    if numbers != list(range(1, len(numbers) + 1)):
        raise MigrationError(f"migrations must be numbered 0001, 0002, ... without gaps: {numbers}")
    return found


def pending(applied: Mapping[str, str], files: list[Migration]) -> list[Migration]:
    """The files still to apply, after checking that the applied ones are an unchanged prefix."""
    names = [m.name for m in files]
    for name, sha in applied.items():
        if name not in names:
            raise MigrationError(f"{name} is applied in the database but its file is missing")
        if sha != next(m.sha256 for m in files if m.name == name):
            raise MigrationError(
                f"{name} changed after it was applied: add a new migration instead"
            )
    done = [m.name for m in files[: len(applied)]]
    if set(done) != set(applied):
        raise MigrationError("the applied migrations are not the first ones on disk, in order")
    return files[len(applied) :]


# ---- connection ----------------------------------------------------------------------------------
def env_file() -> Path:
    """The .env to read: VAULTIC_ENV_FILE if set (e.g. the main checkout's .env from a worktree,
    the same file given to `docker compose --env-file`), else the repository's own .env."""
    return Path(os.environ.get("VAULTIC_ENV_FILE") or REPO / ".env")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(env_file(), override=False)


def db_url(env: Mapping[str, str] | None = None) -> str:
    """The connection URL: VAULTIC_DB_URL, else built from the POSTGRES_* keys (D231)."""
    if env is None:
        _load_dotenv()
        env = os.environ
    if env.get("VAULTIC_DB_URL"):
        return env["VAULTIC_DB_URL"]
    password = env.get("POSTGRES_PASSWORD")
    if not password:
        raise MigrationError("set VAULTIC_DB_URL or POSTGRES_PASSWORD (in .env)")
    port = env.get("VAULTIC_POSTGRES_PORT") or env.get("POSTGRES_PORT") or str(DEFAULT_PORT)
    user, db = env.get("POSTGRES_USER") or "vaultic", env.get("POSTGRES_DB") or "vaultic"
    return f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@127.0.0.1:{int(port)}/{quote(db, safe='')}"


def redact(url: str) -> str:
    """The URL without its password, for messages."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    netloc = parts.netloc.replace(f":{parts.password}@", ":***@", 1)
    return urlunsplit(parts._replace(netloc=netloc))


def connect(url: str | None = None, timeout: int = 5):
    import psycopg

    return psycopg.connect(url or db_url(), connect_timeout=timeout, autocommit=True)


# ---- running -------------------------------------------------------------------------------------
def applied(conn) -> dict[str, str]:
    conn.execute(STATE_TABLE)
    return dict(conn.execute("SELECT name, sha256 FROM schema_migrations ORDER BY name").fetchall())


def migrate(conn, files: list[Migration] | None = None, log=print) -> list[str]:
    """Apply the pending migrations, each in its own transaction, under the runner's advisory lock."""
    files = discover() if files is None else files
    conn.execute("SELECT pg_advisory_lock(%s)", (LOCK_KEY,))
    try:
        todo = pending(applied(conn), files)
        for m in todo:
            with conn.transaction():
                conn.execute(m.path.read_text("utf-8"))
                conn.execute(
                    "INSERT INTO schema_migrations (name, sha256) VALUES (%s, %s)",
                    (m.name, m.sha256),
                )
            log(f"applied {m.name}")
        return [m.name for m in todo]
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))


def status(conn, files: list[Migration] | None = None) -> dict:
    files = discover() if files is None else files
    done = applied(conn)
    return {"applied": sorted(done), "pending": [m.name for m in pending(done, files)]}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["status", "migrate"])
    a = p.parse_args(argv)
    try:
        url = db_url()
        with connect(url) as conn:
            if a.command == "migrate":
                done = migrate(conn)
                print(f"{len(done)} migration(s) applied to {redact(url)}")
            else:
                s = status(conn)
                print(
                    f"{redact(url)}: applied {s['applied'] or 'none'}; pending {s['pending'] or 'none'}"
                )
    except MigrationError as e:
        raise SystemExit(f"refused: {e}") from None


if __name__ == "__main__":
    main()
