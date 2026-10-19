"""Audit store (D208, D209): append-only, hash chain + anchor (tamper-EVIDENT, not tamper-proof),
filters, persistence, threads, and the documented limitation."""

import csv
import io
import os
import shutil
import sqlite3
import threading
from pathlib import Path

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


def test_tests_never_use_the_real_log(tmp_path_factory):
    """conftest.py points VAULTIC_AUDIT_DB at pytest's own temporary directory for the session."""
    path = default_path()
    assert path == Path(os.environ["VAULTIC_AUDIT_DB"])
    assert path.resolve().is_relative_to(tmp_path_factory.getbasetemp().resolve())


def test_verify_finds_reordered_entries(store):
    for i in range(4):
        store.append("Aziz", "analyst", f"action {i}")
    # swap the contents of entries 2 and 3 (ids stay contiguous): the chain breaks at 2
    _tamper(
        store,
        "UPDATE audit_log SET action = CASE id WHEN 2 THEN 'action 2' WHEN 3 THEN 'action 1' "
        "END WHERE id IN (2, 3)",
    )
    out = store.verify()
    assert out["ok"] is False and out["first_bad"] == 2


def test_anchor_detects_a_cut_off_tail(store):
    for i in range(5):
        store.append("Aziz", "analyst", f"action {i}")
    assert store.read_anchor()["entries"] == 5
    _tamper(store, "DELETE FROM audit_log WHERE id >= 4")  # the chain of 1-3 is still valid
    out = store.verify()
    assert out["ok"] is False and "removed from the end" in out["reason"] and out["entries"] == 3


def test_anchor_detects_a_rewritten_last_entry(store):
    for i in range(3):
        store.append("Aziz", "analyst", f"action {i}")
    _tamper(store, "DELETE FROM audit_log WHERE id = 3")
    # a forged third entry written straight into the database with a valid chain hash; the
    # anchor (a separate file) is not touched, so it still holds the true third entry's hash
    prev = store.query(limit=1)[0]["hash"]
    forged = {"id": 3, "time": "2026-10-18T12:00:00+00:00", "analyst": "Mallory", "role": "analyst",
              "action": "a different third entry", "target": None, "before": None, "after": None,
              "detail": "", "prev_hash": prev}  # fmt: skip
    forged["hash"] = entry_hash(prev, forged)
    con = sqlite3.connect(store.path)
    con.execute(f"INSERT INTO audit_log ({', '.join(forged)}) VALUES ({', '.join('?' * len(forged))})",
                list(forged.values()))  # fmt: skip
    con.commit()
    con.close()
    out = store.verify()
    assert out["ok"] is False and out["first_bad"] == 3 and "anchor" in out["reason"]


def test_a_missing_anchor_is_reported(store):
    store.append("Aziz", "analyst", "label")
    store.anchor_path.unlink()
    assert store.verify()["reason"] == "the anchor file is missing"


def test_an_interrupted_anchor_write_is_not_called_tampering(store):
    store.append("Aziz", "analyst", "one")
    old = store.anchor_path.read_text("utf-8")
    store.append("Aziz", "analyst", "two")
    store.anchor_path.write_text(old, "utf-8")  # as if the process died between commit and anchor
    out = store.verify()
    assert out["ok"] is True and "interrupted" in out["reason"]


def test_documented_limitation_a_full_rewrite_of_both_files_is_not_detected(tmp_path):
    """D209: tamper-evident, not tamper-proof. Someone with file access who rewrites the whole
    chain AND the anchor is not detected. This test pins that limitation so it stays documented."""
    real = SQLiteAuditStore(tmp_path / "real" / "audit.sqlite")
    for i in range(3):
        real.append("Aziz", "analyst", f"true action {i}")
    forged = SQLiteAuditStore(tmp_path / "forged" / "audit.sqlite")
    forged.append("Aziz", "analyst", "invented history")
    for suffix in ("", "-wal", "-shm"):
        for f in real.path.parent.glob(real.path.name + suffix):
            f.unlink()
    shutil.copy(forged.path, real.path)
    shutil.copy(forged.anchor_path, real.anchor_path)
    for suffix in ("-wal", "-shm"):
        src = forged.path.with_name(forged.path.name + suffix)
        if src.exists():
            shutil.copy(src, real.path.with_name(real.path.name + suffix))
    out = SQLiteAuditStore(real.path).verify()
    assert out["ok"] is True and out["entries"] == 1  # NOT detected: the documented limitation


def test_export_records_itself_with_the_exact_count(store):
    for i in range(3):
        store.append("Aziz", "analyst", f"action {i}")
    entry, rows = store.export("Aziz", "admin")
    assert entry["detail"] == "4 entries (CSV)" and len(rows) == 4 and rows[0]["id"] == entry["id"]
