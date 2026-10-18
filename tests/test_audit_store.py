"""Audit store (D208): append-only, hash chain, tamper detection, filters, persistence, threads."""

import csv
import io
import os
import sqlite3
import threading

import pytest

from vaultic.audit.store import (
    GENESIS,
    UNKNOWN,
    SQLiteAuditStore,
    clean_actor,
    default_path,
    entry_hash,
    to_csv,
)


@pytest.fixture
def store(tmp_path):
    return SQLiteAuditStore(tmp_path / "audit.sqlite")


def test_append_builds_a_chain_and_query_returns_newest_first(store):
    a = store.append("Aziz", "analyst", "label", target="tx:1", before=None, after="fraud",
                     detail="first label", time="2026-10-18T10:00:00+00:00")  # fmt: skip
    b = store.append("Aziz", "analyst", "label", target="tx:1", before="fraud", after="legit",
                     detail="was fraud", time="2026-10-18T10:01:00+00:00")  # fmt: skip
    assert (a["id"], b["id"]) == (1, 2)
    assert a["prev_hash"] == GENESIS and b["prev_hash"] == a["hash"]
    assert b["hash"] == entry_hash(a["hash"], b)
    rows = store.query()
    assert [r["id"] for r in rows] == [2, 1]
    assert rows[0]["before"] == '"fraud"' and rows[0]["after"] == '"legit"'
    assert store.verify() == {"ok": True, "entries": 2, "first_bad": None, "reason": None}


def test_filters_limit_and_offset(store):
    for i in range(5):
        store.append("Aziz" if i % 2 else "Sara", "analyst", "label" if i < 3 else "attack run",
                     time=f"2026-10-18T10:0{i}:00+00:00")  # fmt: skip
    assert len(store.query(analyst="Sara")) == 3
    assert len(store.query(action="attack")) == 2
    assert [r["id"] for r in store.query(since="2026-10-18T10:03:00+00:00")] == [5, 4]
    assert [r["id"] for r in store.query(limit=2, offset=1)] == [4, 3]
    assert store.count() == 5


def test_update_and_delete_are_refused(store):
    store.append("Aziz", "admin", "label")
    con = sqlite3.connect(store.path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE audit_log SET action = 'x' WHERE id = 1")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM audit_log WHERE id = 1")
    con.close()
    assert store.query()[0]["action"] == "label"


def _tamper(store, sql):
    """Bypass the triggers (as someone with file access could) and run sql."""
    con = sqlite3.connect(store.path)
    con.execute("DROP TRIGGER audit_no_update")
    con.execute("DROP TRIGGER audit_no_delete")
    con.execute(sql)
    con.commit()
    con.close()


def test_verify_finds_an_altered_entry(store):
    for i in range(4):
        store.append("Aziz", "analyst", f"action {i}")
    _tamper(store, "UPDATE audit_log SET detail = 'changed' WHERE id = 3")
    out = store.verify()
    assert out["ok"] is False and out["first_bad"] == 3 and out["entries"] == 2


def test_verify_finds_a_removed_entry(store):
    for i in range(4):
        store.append("Aziz", "analyst", f"action {i}")
    _tamper(store, "DELETE FROM audit_log WHERE id = 2")
    out = store.verify()
    assert out["ok"] is False and out["first_bad"] == 2 and "missing" in out["reason"]


def test_entries_survive_reopening(tmp_path):
    p = tmp_path / "audit.sqlite"
    SQLiteAuditStore(p).append("Aziz", "analyst", "label")
    again = SQLiteAuditStore(p)
    again.append("Aziz", "analyst", "label 2")
    assert again.count() == 2 and again.verify()["ok"]


def test_threads_keep_the_chain_whole(store):
    def work(k):
        for i in range(25):
            store.append(f"analyst {k}", "analyst", f"action {k}-{i}")

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.count() == 200
    assert store.verify() == {"ok": True, "entries": 200, "first_bad": None, "reason": None}


def test_actor_cleaning():
    assert clean_actor("  Aziz   Ahmad ", "admin") == ("Aziz Ahmad", "admin")
    assert clean_actor(None, None) == (UNKNOWN, UNKNOWN)
    assert clean_actor("x\u0000y", "root") == (UNKNOWN, UNKNOWN)
    assert len(clean_actor("a" * 200, "analyst")[0]) == 60


def test_empty_action_is_refused(store):
    with pytest.raises(ValueError):
        store.append("Aziz", "analyst", "")


def test_csv_export_has_every_field(store):
    store.append("Aziz", "analyst", "label", target="tx:1", after="fraud")
    rows = list(csv.DictReader(io.StringIO(to_csv(store.query()))))
    assert rows[0]["analyst"] == "Aziz" and rows[0]["target"] == "tx:1" and rows[0]["hash"]


def test_tests_never_use_the_real_log():
    assert "pytest" in str(default_path()).lower() or os.environ["VAULTIC_AUDIT_DB"]
    assert not str(default_path()).replace("\\", "/").startswith("E:/dev-cache")
