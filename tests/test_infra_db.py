"""The plain-SQL migration runner (vaultic.infra.db, Sprint 3 step 3.1, D231): discovery, ordering,
hashing and every refusal without a database; the real migrations on PostgreSQL in a throwaway schema
(marker 'integration', skipped when no database is reachable, e.g. in CI until step 3.6)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from vaultic.infra import db


def _write(d: Path, name: str, sql: str = "SELECT 1;") -> Path:
    p = d / name
    p.write_bytes(sql.encode())
    return p


def test_the_repository_migrations_are_well_formed():
    files = db.discover()
    assert [m.number for m in files][:1] == [1] and files[0].name == "0001_init.sql"
    sql = files[0].path.read_text("utf-8")
    for table in ("users", "cases", "decisions", "audit_log", "copilot_audit"):
        assert f"CREATE TABLE {table} (" in sql
    assert "BEFORE UPDATE OR DELETE ON audit_log" in sql and "BEFORE TRUNCATE ON audit_log" in sql


def test_discovery_orders_and_hashes_with_normalised_line_endings(tmp_path):
    _write(tmp_path, "0002_b.sql", "SELECT 2;\r\n")
    _write(tmp_path, "0001_a.sql", "SELECT 1;\n")
    files = db.discover(tmp_path)
    assert [m.name for m in files] == ["0001_a.sql", "0002_b.sql"]
    lf = tmp_path / "lf"
    lf.mkdir()
    _write(lf, "0001_x.sql", "SELECT 2;\n")
    assert files[1].sha256 == db.file_sha256(lf / "0001_x.sql")  # CRLF and LF hash alike


@pytest.mark.parametrize(("names", "message"), [
    (["0001_a.sql", "0003_c.sql"], "without gaps"),
    (["0002_a.sql"], "without gaps"),
    (["0001_a.sql", "0001_b.sql"], "duplicate"),
    (["0001_A.sql"], "NNNN_lowercase_name"),
    (["1_a.sql"], "NNNN_lowercase_name"),
])  # fmt: skip
def test_discovery_refuses_bad_names_gaps_and_duplicates(tmp_path, names, message):
    for n in names:
        _write(tmp_path, n)
    with pytest.raises(db.MigrationError, match=message):
        db.discover(tmp_path)


def test_pending_returns_the_unapplied_tail_and_refuses_any_mismatch(tmp_path):
    for n in ("0001_a.sql", "0002_b.sql", "0003_c.sql"):
        _write(tmp_path, n, f"SELECT '{n}';")
    files = db.discover(tmp_path)
    sha = {m.name: m.sha256 for m in files}
    assert [m.name for m in db.pending({}, files)] == ["0001_a.sql", "0002_b.sql", "0003_c.sql"]
    assert [m.name for m in db.pending({"0001_a.sql": sha["0001_a.sql"]}, files)] == [
        "0002_b.sql",
        "0003_c.sql",
    ]
    with pytest.raises(db.MigrationError, match="changed after it was applied"):
        db.pending({"0001_a.sql": "0" * 64}, files)
    with pytest.raises(db.MigrationError, match="file is missing"):
        db.pending({"0009_z.sql": "0" * 64}, files)
    with pytest.raises(db.MigrationError, match="not the first ones"):
        db.pending({"0002_b.sql": sha["0002_b.sql"]}, files)


def test_the_url_comes_from_the_environment_with_the_vaultic_port_first():
    base = {"POSTGRES_PASSWORD": "pw:/@x", "POSTGRES_USER": "u", "POSTGRES_DB": "d"}
    assert db.db_url(base).endswith("@127.0.0.1:5442/d")  # D231 default
    assert db.db_url({**base, "POSTGRES_PORT": "55432"}).endswith(":55432/d")  # old key still read
    assert db.db_url({**base, "POSTGRES_PORT": "55432", "VAULTIC_POSTGRES_PORT": "5442"}).endswith(
        ":5442/d"
    )
    assert db.db_url({**base, "VAULTIC_DB_URL": "postgresql://a@h/x"}) == "postgresql://a@h/x"
    url = db.db_url(base)
    assert "pw:/@x" not in url and "pw%3A%2F%40x" in url  # the password is URL-quoted
    assert "pw" not in db.redact(url) and ":***@127.0.0.1:5442" in db.redact(url)
    with pytest.raises(db.MigrationError, match="POSTGRES_PASSWORD"):
        db.db_url({"POSTGRES_USER": "u"})


# ---- on a real PostgreSQL ---------------------------------------------------------------------------
def _reachable_url() -> str | None:
    try:
        url = db.db_url()
        with db.connect(url, timeout=2):
            return url
    except Exception:  # noqa: BLE001 - no database here: the integration tests are skipped
        return None


@pytest.fixture
def schema_conn():
    url = _reachable_url()
    if url is None:
        pytest.skip("no PostgreSQL reachable (start it with `docker compose up -d`)")
    name = f"test_{uuid.uuid4().hex[:12]}"
    with db.connect(url) as conn:
        conn.execute(f"CREATE SCHEMA {name}")
        conn.execute(f"SET search_path TO {name}")
        try:
            yield conn
        finally:
            conn.execute(f"DROP SCHEMA {name} CASCADE")


@pytest.mark.integration
def test_the_migrations_apply_once_and_the_audit_log_is_append_only(schema_conn):
    import psycopg

    conn = schema_conn
    assert db.migrate(conn, log=lambda *_: None) == ["0001_init.sql"]
    assert db.migrate(conn, log=lambda *_: None) == []  # idempotent
    assert db.status(conn) == {"applied": ["0001_init.sql"], "pending": []}
    tables = {r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()")}  # fmt: skip
    assert tables >= {
        "users",
        "cases",
        "decisions",
        "audit_log",
        "copilot_audit",
        "schema_migrations",
    }
    conn.execute(
        """INSERT INTO audit_log (id, "time", analyst, role, action, detail, prev_hash, hash)
           VALUES (1, '2026-11-25T00:00:00+00:00', 'a', 'analyst', 'x', '', '0', 'h')"""
    )
    for sql in (
        "UPDATE audit_log SET analyst = 'b'",
        "DELETE FROM audit_log",
        "TRUNCATE audit_log",
    ):
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            conn.execute(sql)
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES ('u', 'h', 'root')")


@pytest.mark.integration
def test_an_edited_applied_migration_stops_the_runner(schema_conn, tmp_path):
    p = _write(tmp_path, "0001_a.sql", "CREATE TABLE t (x INT);")
    db.migrate(schema_conn, db.discover(tmp_path), log=lambda *_: None)
    p.write_bytes(b"CREATE TABLE t (x BIGINT);")
    with pytest.raises(db.MigrationError, match="changed after it was applied"):
        db.migrate(schema_conn, db.discover(tmp_path), log=lambda *_: None)


@pytest.mark.integration
def test_a_failing_migration_leaves_nothing_behind(schema_conn, tmp_path):
    import psycopg

    _write(tmp_path, "0001_a.sql", "CREATE TABLE ok (x INT);")
    _write(tmp_path, "0002_b.sql", "CREATE TABLE half (x INT); SELECT no_such_function();")
    with pytest.raises(psycopg.errors.UndefinedFunction):
        db.migrate(schema_conn, db.discover(tmp_path), log=lambda *_: None)
    assert db.status(schema_conn, db.discover(tmp_path)) == {
        "applied": ["0001_a.sql"],
        "pending": ["0002_b.sql"],
    }
    assert schema_conn.execute("SELECT to_regclass('half')").fetchone()[0] is None  # rolled back


def test_the_env_file_can_be_pointed_at_another_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("VAULTIC_ENV_FILE", raising=False)
    assert db.env_file() == db.REPO / ".env"
    monkeypatch.setenv("VAULTIC_ENV_FILE", str(tmp_path / "main.env"))
    assert db.env_file() == tmp_path / "main.env"
