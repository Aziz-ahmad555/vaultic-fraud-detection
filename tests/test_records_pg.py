"""Sprint 3 step 3.3 (D240): the PostgreSQL audit store, labels, decisions, copilot records (Q13) and the
migration from the SQLite log.

- Without a database: the store switch, the derivation of copilot records and labels, the source checks
  of the migration, and the label route of the platform server with the SQLite log.
- On a real PostgreSQL (marker 'integration', skipped when none is reachable), each test in a throwaway
  schema: the audit-store behaviour on BOTH stores, the same hashes on both, tamper detection, the
  append-only record tables, concurrent appends, labels across a restart, the copilot's one entry plus
  one record, the decision store, and the migration of a 500-entry log.
"""

import json
import sqlite3
import threading
import uuid
import warnings
from types import SimpleNamespace

import pytest
from test_auth import _app_with, _platform
from test_copilot_api import AS_OF, WHY, decision_answer
from test_demo_server import _app, _bundle, _load_module

from vaultic.audit.pg_store import (
    COPILOT_ACTIONS,
    PostgresAuditStore,
    StoreSetupError,
    copilot_record,
    label_of,
    record_sha256,
)
from vaultic.audit.store import GENESIS, SQLiteAuditStore, default_store, entry_hash
from vaultic.infra import migrate_audit as M
from vaultic.infra.stores import DecisionStore, LabelStore, store_mode

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
    from fastapi.testclient import TestClient

T0 = "2026-11-27T10:00:00+00:00"


# ---- without a database -------------------------------------------------------------------------
def test_the_store_switch_fails_closed():
    assert store_mode({}) == "files" and store_mode({"VAULTIC_STORE": ""}) == "files"
    assert store_mode({"VAULTIC_STORE": " Postgres "}) == "postgres"
    for bad in ("sqlite", "pg", "postgresql", "1"):
        with pytest.raises(RuntimeError, match="VAULTIC_STORE"):
            store_mode({"VAULTIC_STORE": bad})


def test_tests_use_the_files_store_whatever_env_says():
    assert isinstance(default_store(), SQLiteAuditStore)  # conftest sets VAULTIC_STORE=""


def test_the_copilot_actions_are_the_frozen_copilots_own():
    from vaultic.copilot.service import ACTIONS

    assert {v: k for k, v in COPILOT_ACTIONS.items()} == ACTIONS


def test_a_copilot_record_is_derived_only_from_its_entry():
    e = {"id": 7, "action": "Copilot question", "target": "tx:5", "detail": "why?", "analyst": "A",
         "role": "analyst", "time": T0, "after": json.dumps({"route": "llm", "kept": 1}),
         "prev_hash": "x", "hash": "y", "before": None}  # fmt: skip
    rec = copilot_record(e)
    assert rec == {"audit_id": 7, "kind": "question", "target": "tx:5", "question": "why?",
                   "analyst": "A", "role": "analyst", "time": T0,
                   "result": {"route": "llm", "kept": 1}}  # fmt: skip
    assert record_sha256(rec) == record_sha256(dict(reversed(list(rec.items()))))  # key order
    assert copilot_record({**e, "action": "Marked transaction 5 as fraud"}) is None
    assert label_of({"action": "Marked transaction 5 as fraud"}) == (5, "fraud")
    assert label_of({"action": "Marked transaction 5 as maybe"}) is None


def test_the_migration_refuses_a_missing_or_broken_source(tmp_path):
    target = PostgresAuditStore("postgresql://nobody@127.0.0.1:1/none", check=False)
    with pytest.raises(M.MigrationRefused, match="does not exist"):
        M.migrate(tmp_path / "nope.sqlite", target, log=lambda *_: None)
    assert not (tmp_path / "nope.sqlite").exists()  # never created
    src = SQLiteAuditStore(tmp_path / "a.sqlite")
    for i in range(3):
        src.append("A", "analyst", f"act {i}", time=T0)
    con = sqlite3.connect(src.path)
    con.execute("DROP TRIGGER audit_no_update")
    con.execute("UPDATE audit_log SET detail = 'edited' WHERE id = 2")
    con.commit()
    con.close()
    with pytest.raises(M.MigrationRefused, match="does not verify"):
        M.migrate(src.path, target, log=lambda *_: None)


def test_with_the_sqlite_log_labels_stay_in_memory_and_the_read_says_so(tmp_path):
    app = _app_with(_load_module(), tmp_path, None)
    c = TestClient(app)
    assert c.post("/api/feedback/1000", json={"label": "fraud"}).status_code == 200
    r = c.get("/api/feedback/1000")
    assert r.status_code == 501 and "VAULTIC_STORE=postgres" in r.json()["detail"]


# ---- on a real PostgreSQL ------------------------------------------------------------------------
@pytest.fixture
def pg():
    """(admin connection, URL scoped to a throwaway schema with every migration applied)."""
    from vaultic.infra import db

    try:
        url = db.db_url()
        conn = db.connect(url, timeout=2)
    except Exception:  # noqa: BLE001 - no database here
        pytest.skip("no PostgreSQL reachable (start it with `docker compose up -d`)")
    schema = f"test_{uuid.uuid4().hex[:12]}"
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path TO {schema}")
    db.migrate(conn, log=lambda *_: None)
    sep = "&" if "?" in url else "?"
    try:
        yield conn, f"{url}{sep}options=-csearch_path%3D{schema}"
    finally:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.close()


def _pg_store(pg):
    return PostgresAuditStore(pg[1])


@pytest.fixture(params=["sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "sqlite":
        return SimpleNamespace(
            s=SQLiteAuditStore(tmp_path / "audit.sqlite"), kind="sqlite", pg=None
        )
    pg = request.getfixturevalue("pg")
    return SimpleNamespace(s=_pg_store(pg), kind="postgres", pg=pg)


def _raw(st, sql):
    """Run SQL past the append-only triggers (what someone with file or database rights can do)."""
    if st.kind == "sqlite":
        con = sqlite3.connect(st.s.path)
        con.executescript("DROP TRIGGER IF EXISTS audit_no_update; DROP TRIGGER IF EXISTS "
                          f"audit_no_delete; {sql}")  # fmt: skip
        con.close()
    else:
        conn = st.pg[0]
        conn.execute("ALTER TABLE audit_log DISABLE TRIGGER USER")
        conn.execute(sql)
        conn.execute("ALTER TABLE audit_log ENABLE TRIGGER USER")


def _fill(s, n=5):
    return [s.append("Aziz" if i % 2 else "Sara", "analyst", "label" if i < 3 else "attack run",
                     target=f"tx:{i}", after={"i": i}, time=f"2026-10-18T10:0{i}:00+00:00")
            for i in range(n)]  # fmt: skip


@pytest.mark.integration
def test_both_stores_chain_filter_and_export_alike(store):
    s = store.s
    a = _fill(s)
    assert a[0]["prev_hash"] == GENESIS and a[1]["hash"] == entry_hash(a[0]["hash"], a[1])
    assert len(s.query(analyst="Sara")) == 3 and len(s.query(action="ATTACK")) == 2
    assert [r["id"] for r in s.query(since="2026-10-18T10:03:00+00:00")] == [5, 4]
    assert [r["id"] for r in s.query(limit=2, offset=1)] == [4, 3]
    assert s.count() == 5 and s.query()[0]["after"] == '{"i": 4}'
    entry, rows = s.export("Aziz", "admin")
    assert entry["detail"] == "6 entries (CSV)" and len(rows) == 6 and rows[0]["id"] == 6
    v = s.verify()
    assert v["ok"] and v["entries"] == 6 and v["reason"] is None


@pytest.mark.integration
def test_the_same_entries_get_the_same_hashes_on_both_stores(pg, tmp_path):
    lite, post = SQLiteAuditStore(tmp_path / "a.sqlite"), _pg_store(pg)
    a, b = _fill(lite), _fill(post)
    assert [e["hash"] for e in a] == [e["hash"] for e in b]
    assert lite.query(limit=100) == post.query(limit=100)


@pytest.mark.integration
@pytest.mark.parametrize("tamper,first_bad", [
    ("UPDATE audit_log SET detail = 'edited' WHERE id = 3", 3),
    ("DELETE FROM audit_log WHERE id = 2", 2),
    ("UPDATE audit_log SET id = id + 100 WHERE id IN (2, 3); "
     "UPDATE audit_log SET id = 3 WHERE id = 102; UPDATE audit_log SET id = 2 WHERE id = 103", 2),
    ("DELETE FROM audit_log WHERE id = 5", 5),
])  # fmt: skip
def test_verify_catches_an_edited_deleted_reordered_row_and_a_cut_tail(store, tamper, first_bad):
    _fill(store.s)
    if store.kind == "postgres":
        for stmt in tamper.split("; "):
            _raw(store, stmt)
    else:
        _raw(store, tamper)
    v = store.s.verify()
    assert not v["ok"] and v["first_bad"] == first_bad, v


@pytest.mark.integration
def test_postgres_refuses_changes_to_the_log_and_the_record_tables(pg):
    import psycopg

    s = _pg_store(pg)
    s.append("A", "analyst", "Copilot question", target="tx:1", after={"route": "llm"}, detail="q")
    LabelStore(s).label("A", "analyst", 1, "fraud")
    conn = pg[0]
    for sql in ("UPDATE audit_log SET detail = 'x'", "DELETE FROM audit_log", "TRUNCATE audit_log CASCADE",
                "UPDATE copilot_audit SET record_sha256 = repeat('0', 64)", "DELETE FROM copilot_audit",
                "TRUNCATE copilot_audit", "UPDATE feedback_labels SET label = 'legit'",
                "DELETE FROM feedback_labels", "TRUNCATE feedback_labels"):  # fmt: skip
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            conn.execute(sql)
    assert s.verify()["ok"]


@pytest.mark.integration
def test_the_anchor_row_catches_a_cut_tail_and_a_rewritten_last_entry(pg):
    s = _pg_store(pg)
    _fill(s)
    assert s.read_anchor() == {"entries": 5, "hash": s.query(limit=1)[0]["hash"]}
    st = SimpleNamespace(kind="postgres", pg=pg)
    _raw(st, "UPDATE audit_log SET hash = repeat('f', 64) WHERE id = 5")
    assert s.verify()["first_bad"] == 5
    pg[0].execute("DELETE FROM audit_anchor")
    pg[0].execute("ALTER TABLE audit_log DISABLE TRIGGER USER")
    pg[0].execute("DELETE FROM audit_log WHERE id = 5")
    v = s.verify()
    assert not v["ok"] and "anchor row is missing" in v["reason"]


@pytest.mark.integration
def test_documented_limitation_a_full_rewrite_of_log_and_anchor_is_not_detected(pg):
    s = _pg_store(pg)
    _fill(s, 3)
    conn = pg[0]
    conn.execute("ALTER TABLE audit_log DISABLE TRIGGER USER")
    conn.execute("DELETE FROM audit_log WHERE id = 3")
    last = s.query(limit=1)[0]
    conn.execute("UPDATE audit_anchor SET entries = 2, hash = %s", (last["hash"],))
    assert s.verify()["ok"]  # tamper-evident, not tamper-proof (D240)


@pytest.mark.integration
def test_concurrent_appends_from_two_stores_keep_one_chain(pg):
    stores = [_pg_store(pg), _pg_store(pg)]

    def work(k):
        for i in range(10):
            stores[k % 2].append(f"w{k}", "analyst", f"act {k}.{i}")

    threads = [threading.Thread(target=work, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    v = stores[0].verify()
    assert v["ok"] and v["entries"] == 60


@pytest.mark.integration
def test_copilot_records_are_linked_and_covered_by_their_entry(pg):
    s = _pg_store(pg)
    s.append("A", "analyst", "something else")
    e = s.append("A", "analyst", "Copilot summary", target="tx:9", detail="",
                 after={"route": "cached", "seconds": 1.5, "removed_facts": [{"why": "x: 9"}]})  # fmt: skip
    conn = pg[0]
    row = conn.execute("SELECT audit_id, record, record_sha256 FROM copilot_audit").fetchall()
    assert len(row) == 1 and row[0][0] == e["id"] and row[0][1] == copilot_record(e)
    assert s.verify()["records"] == {"copilot": 1, "labels": 0}
    conn.execute("ALTER TABLE copilot_audit DISABLE TRIGGER USER")
    conn.execute("UPDATE copilot_audit SET record = jsonb_set(record, '{kind}', '\"question\"')")
    assert "does not match its entry" in s.verify()["reason"]
    conn.execute("DELETE FROM copilot_audit")
    assert "has no copilot record" in s.verify()["reason"]


@pytest.mark.integration
def test_the_copilot_writes_one_entry_and_one_linked_record(pg, tmp_path):
    from test_copilot_api import ask

    from vaultic.copilot.llm import ScriptedLLM
    from vaultic.copilot.service import Copilot

    mod = _load_module()
    bundle = _bundle(tmp_path)
    _app(mod, bundle)  # the test-data marker and the final run
    s = _pg_store(pg)
    app = mod.create_app(bundle, runs_dir=bundle.parent / "runs", audit_store=s)
    import pandas as pd

    df = pd.read_parquet(bundle)
    alerts = df[(df["day"] == 151) & df["action"].isin(["step_up", "hold", "block"])
                & (df["TransactionDT"] <= AS_OF)].sort_values("TransactionDT")  # fmt: skip
    tid = int(alerts["TransactionID"].iloc[-1])  # as test_copilot_api's fixture
    cop = Copilot(app.state.read, ScriptedLLM(decision_answer), audit=s, testlab=lambda: None)
    mod.mount_copilot(app, cop)
    status, body = ask(TestClient(app), tid, WHY)
    assert status == 200 and body["audit"] == "ok", body
    entries = s.query()
    assert len(entries) == 1 and entries[0]["action"] == "Copilot question"
    rows = pg[0].execute("SELECT audit_id, record FROM copilot_audit").fetchall()
    assert [r[0] for r in rows] == [entries[0]["id"]]
    assert rows[0][1]["question"] == WHY and rows[0][1]["result"]["removed"] == 1
    assert s.verify() == {"ok": True, "entries": 1, "first_bad": None, "reason": None,
                          "records": {"copilot": 1, "labels": 0}}  # fmt: skip


@pytest.mark.integration
def test_labels_persist_and_unsure_never_reaches_training(pg):
    s = _pg_store(pg)
    labels = LabelStore(s)
    labels.label("A", "analyst", 10, "fraud")
    labels.label("B", "admin", 10, "legit")
    labels.label("A", "analyst", 11, "unsure")
    labels.label("A", "analyst", 12, "fraud")
    labels.label("A", "analyst", 12, "unsure")  # the latest is unsure: left out entirely
    with pytest.raises(ValueError):
        labels.label("A", "analyst", 13, "maybe")
    again = LabelStore(PostgresAuditStore(pg[1]))  # a restart
    assert again.current(10)["label"] == "legit" and again.current(10)["analyst"] == "B"
    assert [h["label"] for h in again.history(10)] == ["fraud", "legit"]
    assert again.training_labels() == {10: "legit"}
    assert s.query()[-1]["detail"] == "first label" and s.query(limit=1)[0]["before"] == '"fraud"'
    v = s.verify()
    assert v["ok"] and v["records"] == {"copilot": 0, "labels": 5} and v["entries"] == 5
    pg[0].execute("ALTER TABLE feedback_labels DISABLE TRIGGER USER")
    pg[0].execute("UPDATE feedback_labels SET label = 'fraud' WHERE transaction_id = 11")
    assert "does not match its entry" in s.verify()["reason"]


def _pg_app(pg, tmp_path, auth=None):
    mod = _load_module()
    bundle = _bundle(tmp_path)
    _app(mod, bundle)
    demo = mod.create_app(bundle, runs_dir=bundle.parent / "runs", audit_store=_pg_store(pg))
    return _platform().with_login(demo, auth, announce=lambda *_: None)


@pytest.mark.integration
def test_the_label_api_persists_across_a_restart_and_is_fail_closed(pg, tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    c = TestClient(_pg_app(pg, tmp_path / "a"))
    h = {"X-Vaultic-Analyst": "Sara", "X-Vaultic-Role": "analyst"}
    assert c.post("/api/feedback/1000", json={"label": "fraud"}, headers=h).status_code == 200
    assert c.post("/api/feedback/1000", json={"label": "maybe"}, headers=h).status_code == 422
    c2 = TestClient(_pg_app(pg, tmp_path / "b"))  # a new API process on the same database
    r = c2.get("/api/feedback/1000").json()
    assert r["current"]["label"] == "fraud" and r["current"]["analyst"] == "Sara"
    store = c2.app.state.audit
    assert store.count() == 1  # the refused label left no entry
    monkeypatch.setattr(PostgresAuditStore, "_set_anchor",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("disk full")))  # fmt: skip
    r = c2.post("/api/feedback/1000", json={"label": "legit"}, headers=h)
    assert r.status_code == 503 and "not saved" in r.json()["detail"]
    monkeypatch.undo()
    assert store.count() == 1 and LabelStore(store).current(1000)["label"] == "fraud"  # neither
    assert store.verify()["ok"]


@pytest.mark.integration
def test_a_store_without_migration_0004_refuses_to_start(pg):
    from vaultic.infra import db

    url = db.db_url()
    schema = f"test_{uuid.uuid4().hex[:12]}"
    pg[0].execute(f"CREATE SCHEMA {schema}")
    try:
        with pytest.raises(StoreSetupError, match="0004"):
            PostgresAuditStore(f"{url}{'&' if '?' in url else '?'}options=-csearch_path%3D{schema}")
    finally:
        pg[0].execute(f"DROP SCHEMA {schema} CASCADE")
    with pytest.raises(StoreSetupError, match="not reachable"):
        PostgresAuditStore("postgresql://nobody:x@127.0.0.1:1/none")


@pytest.mark.integration
def test_only_stream_decisions_are_written(pg):
    d = DecisionStore(pg[1])
    i = d.record(1000, "hold", 0.91, "exp200-final")
    with pytest.raises(ValueError, match="frozen E12"):
        d.record(1000, "block", 0.99, "e12", source="frozen_e12")
    with pytest.raises(ValueError, match="action"):
        d.record(1000, "deny", 0.5, "v1")
    rows = d.for_transaction(1000)
    assert [(r["id"], r["source"], r["action"]) for r in rows] == [(i, "stream", "hold")]


def _synthetic_log(path, n=500):
    s = SQLiteAuditStore(path)
    for i in range(n):
        if i % 50 == 7:
            s.append("Sara", "analyst", "Copilot question", target=f"tx:{i}", detail=f"why {i}?",
                     after={"route": "llm", "kept": 2, "removed": 1, "seconds": 3.2},
                     time=f"2026-11-{1 + i % 27:02d}T10:00:00+00:00")  # fmt: skip
        else:
            s.append("Zoë" if i % 3 else "Aziz", "admin" if i % 4 else "analyst",
                     f"Marked transaction {i} as fraud" if i % 5 == 0 else f"action {i}",
                     target=f"tx:{i}", before={"a": i} if i % 2 else None, after=[i, "x"],
                     detail=f"d{i}", time=f"2026-11-{1 + i % 27:02d}T10:00:00+00:00")  # fmt: skip
    return s


@pytest.mark.integration
def test_the_migration_copies_500_entries_identically_and_refuses_a_second_run(pg, tmp_path):
    src = _synthetic_log(tmp_path / "audit.sqlite")
    before = (src.path.read_bytes(), src.anchor_path.read_bytes())
    target = _pg_store(pg)
    dry = M.migrate(src.path, target, dry_run=True, log=lambda *_: None)
    assert dry["dry_run"] and target.count() == 0 and target.read_anchor() is None
    out = M.migrate(src.path, target, log=lambda *_: None)
    assert out["entries"] == 500 and out["copilot_records"] == 10
    assert target.query(limit=1)[0]["hash"] == src.query(limit=1)[0]["hash"] == out["last_hash"]
    assert M.digest(target.query(limit=10**9)) == M.digest(src.query(limit=10**9))
    v = target.verify()
    assert v["ok"] and v["entries"] == 500 and v["records"]["copilot"] == 10
    assert (src.path.read_bytes(), src.anchor_path.read_bytes()) == before  # source unchanged
    with pytest.raises(M.MigrationRefused, match="not empty"):
        M.migrate(src.path, target, log=lambda *_: None)
    target.append("A", "analyst", "after the migration")  # the chain continues from the copy
    assert target.verify()["entries"] == 501


@pytest.mark.integration
def test_a_failed_copy_leaves_the_target_empty(pg, tmp_path, monkeypatch):
    src = _synthetic_log(tmp_path / "audit.sqlite", n=20)
    target = _pg_store(pg)
    real = M.summary
    monkeypatch.setattr(M, "summary", lambda rows: {**real(rows), "salt": id(rows)})  # never equal
    with pytest.raises(M.MigrationRefused, match="rolled back"):
        M.migrate(src.path, target, log=lambda *_: None)
    assert target.count() == 0 and target.read_anchor() is None
    assert pg[0].execute("SELECT COUNT(*) FROM copilot_audit").fetchone()[0] == 0
