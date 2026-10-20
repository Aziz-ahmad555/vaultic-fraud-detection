"""EXP-310-final tooling (D213): settings, guard order, parity, resume per completed delay and the
hypothesis read-out. No real data and no test-period row is read here."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

import vaultic.learning.feedback_final as ff
from vaultic.paths import CONFIG_DIR


@pytest.fixture(autouse=True)
def _runs_in_tmp(monkeypatch, tmp_path):
    """The O_EXCL lock and run folders go to a temporary runs dir, never the real one."""
    monkeypatch.setattr(ff, "RUNS_DIR", tmp_path / "runs")


def cfg():
    return yaml.safe_load((CONFIG_DIR / "EXP-310-final.yaml").read_text("utf-8"))


def test_frozen_config_matches_the_preregistration_and_a_change_is_refused():
    ff.check_settings(cfg())
    bad = cfg()
    bad["delays"] = [0, 30]
    with pytest.raises(ValueError, match="delays"):
        ff.check_settings(bad)
    bad = cfg()
    bad["champion"]["calibration_days"] = [151, 154]
    with pytest.raises(ValueError, match="calibration_days"):
        ff.check_settings(bad)


def test_the_guards_run_before_any_data_is_read(monkeypatch):
    calls = []

    def guard_fails(c, reason):
        calls.append("start")
        raise RuntimeError("guard refused")

    monkeypatch.setattr(ff, "start", guard_fails)
    import vaultic.learning.feedback_inputs as fi

    monkeypatch.setattr(fi, "load", lambda *a, **k: calls.append("load"))
    with pytest.raises(RuntimeError, match="guard refused"):
        ff.main([])
    assert calls == ["start"]  # no load: FINAL-STARTED comes before any test row


def test_final_data_needs_this_process_token_and_the_start_row(monkeypatch, tmp_path):
    """D215: load('final') needs the token start() / resume() issued in this process and the run's
    start row (its origin check is tested against a real repo below)."""
    from vaultic.learning.feedback_inputs import load, require_final_started

    with pytest.raises(RuntimeError, match="go-ahead"):
        load("final")
    run = tmp_path / "EXP-310-final" / "20261021-000000-000001"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"status": "started"}))
    log = tmp_path / "experiment_log.md"
    log.write_text("| EXP-310-final | 2026-10-21 | FINAL-STARTED: x | run `other` | - |\n")
    with pytest.raises(RuntimeError, match="did not pass in this process"):
        require_final_started(run, None, log)
    token = ff._grant(run)
    with pytest.raises(RuntimeError, match="did not pass in this process"):
        require_final_started(run, "forged" + token, log)
    with pytest.raises(RuntimeError, match="no FINAL-STARTED row"):
        require_final_started(run, token, log)
    monkeypatch.setattr(ff, "latest_row_on_origin", lambda r, log=None: "row")
    require_final_started(run, token, log)
    (run / "metrics.json").write_text(json.dumps({"status": "aborted"}))
    with pytest.raises(RuntimeError, match="aborted"):
        require_final_started(run, token, log)
    ff.release_running(run)
    (run / "metrics.json").write_text(json.dumps({"status": "started"}))
    with pytest.raises(RuntimeError, match="did not pass in this process"):  # token gone
        require_final_started(run, token, log)


def test_start_refuses_without_the_freeze_record(monkeypatch, tmp_path):
    monkeypatch.setattr(ff, "FROZEN", tmp_path / "frozen_feedback.md")
    monkeypatch.setattr("vaultic.fusion.final_run.check_package_location", lambda: None)
    with pytest.raises(RuntimeError, match="missing"):
        ff.start(cfg(), None)


class Fixed:
    def __init__(self, p):
        self.p = np.asarray(p, float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.p, self.p])


def test_parity_passes_within_the_tolerance_and_aborts_otherwise():
    ids = np.array([10, 11, 12])
    X = pd.DataFrame({"a": [0, 0, 0]})
    models = [Fixed([0.1, 0.2, 0.3]), Fixed([0.3, 0.4, 0.5])]  # mean 0.2, 0.3, 0.4
    stored = pd.Series([0.2, 0.3, 0.4], index=ids)
    assert ff.parity(models, X, ids, stored)["passed"]
    with pytest.raises(RuntimeError, match="parity failed"):
        ff.parity(models, X, ids, stored + 1e-5)
    with pytest.raises(RuntimeError, match="parity failed"):  # a stored row the run did not score
        ff.parity(models, X, ids, pd.concat([stored, pd.Series([0.5], index=[13])]))


def _fake_run_one(calls, fail_at=None):
    def run_one(L):
        calls.append(L)
        if L == fail_at:
            raise KeyboardInterrupt("shutdown")
        return pd.DataFrame({"row": [1], "p": [0.5]}), [{"day": 121}], {"L": L}

    return run_one


def test_resume_skips_completed_delays_and_never_reuses_a_partial_one(tmp_path):
    calls = []
    with pytest.raises(KeyboardInterrupt):
        ff.run_delays(tmp_path, [0, 7, 30], _fake_run_one(calls, fail_at=7), log=lambda m: None)
    assert (tmp_path / "delay_L0" / "DONE").exists() and not (
        tmp_path / "delay_L7" / "DONE"
    ).exists()
    (tmp_path / "delay_L7" / "stale.txt").write_text("partial")  # left by the interrupted delay
    calls.clear()
    out = ff.run_delays(tmp_path, [0, 7, 30], _fake_run_one(calls), log=lambda m: None)
    assert calls == [7, 30]  # L=0 not recomputed
    assert not (tmp_path / "delay_L7" / "stale.txt").exists()
    assert out == {0: {"L": 0}, 7: {"L": 7}, 30: {"L": 30}}


def test_check_resume_accepts_only_a_started_unfinished_run(tmp_path):
    run = tmp_path / "runs" / "EXP-310-final" / "20261021-000000-000000"
    with pytest.raises(RuntimeError, match="not a started"):
        ff.check_resume(run)
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"mode": "final", "status": "completed"}))
    with pytest.raises(RuntimeError, match="already completed"):
        ff.check_resume(run)
    (run / "metrics.json").write_text(json.dumps({"mode": "final", "status": "started"}))
    assert ff.check_resume(run)["status"] == "started"
    (run / "metrics.json").write_text(json.dumps({"status": "started"}))  # D215: mode final
    with pytest.raises(RuntimeError, match="not a started"):
        ff.check_resume(run)
    # D215: the resolved parent must be RUNS_DIR/EXP-310-final, not any folder of that name
    for other in (tmp_path / "EXP-999" / "x", tmp_path / "elsewhere" / "EXP-310-final" / "x"):
        other.mkdir(parents=True)
        (other / "metrics.json").write_text(json.dumps({"mode": "final", "status": "started"}))
        with pytest.raises(RuntimeError, match="not a started"):
            ff.check_resume(other)


def test_resume_refuses_when_a_frozen_file_changed(monkeypatch, tmp_path):
    monkeypatch.setattr("vaultic.reports.phase2_summary.check_freeze", lambda p: ["src/x.py"])
    with pytest.raises(RuntimeError, match="frozen file changed"):
        ff.main(["--resume", str(tmp_path)])


def _metrics(diff, p, cost_ci_low, gated_minus_ungated):
    pol = {"vs_P0": {"diff": diff, "p_value": p, "ci_low": diff - 0.01, "ci_high": diff + 0.01},
           "cost_vs_P0": {"diff": -1.0, "ci_low": cost_ci_low, "ci_high": 1.0, "p_value": 0.5}}  # fmt: skip
    return {
        "evaluation": {g: pol for g in ("P1", "P2", "P3")},
        "gated_vs_ungated": {"P1": {"cost_vs_P1_ungated": {"diff": gated_minus_ungated}}},
    }


def test_hypotheses_read_out():
    res = {0: _metrics(0.06, 0.001, -5, -1), 7: _metrics(0.02, 0.001, -5, -1),
           30: _metrics(0.002, 0.01, -5, -1), 60: _metrics(0.0, 1.0, -5, -1)}  # fmt: skip
    h = ff.hypotheses(res)
    assert h["H-FB1"]["supported"] and h["H-FB2"]["supported"] and h["H-FB3"]["supported"]
    assert h["H-FB4"]["supported"]
    res[30] = _metrics(0.002, 0.04, 3, +2)  # Holm over 3 lifts p = 0.04 above 0.05; cost rises
    h = ff.hypotheses(res)
    assert (
        not h["H-FB1"]["supported"] and not h["H-FB2"]["supported"] and not h["H-FB4"]["supported"]
    )
    res[60] = _metrics(0.01, 0.5, -5, -1)  # gain at L=60 above L=30: not monotone
    assert not ff.hypotheses(res)["H-FB3"]["supported"]


# ---- D214: resume guards, FINAL-RESUMED, abort, champion hashes, input cache ------------------
def _started_run(tmp_path, status="started"):
    run = tmp_path / "runs" / "EXP-310-final" / "20261021-000000-000002"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"mode": "final", "status": status,
                                                  "git_commit": "a" * 40, "data_hashes": {}}))  # fmt: skip
    return run


def _guards_ok(monkeypatch, calls, hashes=None):
    def ok(c):
        calls.append("guards")
        return "record", "b" * 40, "feedback", hashes or {}

    monkeypatch.setattr(ff, "guards", ok)
    monkeypatch.setattr(ff, "aborted_on_record", lambda name, branch: [])


def test_resume_runs_the_same_guards_as_a_start(monkeypatch, tmp_path):
    run = _started_run(tmp_path)
    calls = []

    def refuse(c):
        calls.append("guards")
        raise RuntimeError("guard refused")

    monkeypatch.setattr(ff, "guards", refuse)
    monkeypatch.setattr(ff, "record_event", lambda *a: calls.append("event"))
    with pytest.raises(RuntimeError, match="guard refused"):
        ff.resume(cfg(), run)
    assert calls == ["guards"]  # nothing logged, nothing loaded


def test_resume_refuses_without_the_start_row_on_origin_or_with_other_changes(
    monkeypatch, tmp_path
):
    run = _started_run(tmp_path)
    calls = []
    _guards_ok(monkeypatch, calls)
    monkeypatch.setattr(
        ff, "check_recorded_commit", lambda rec, head: calls.append(("commit", rec, head))
    )
    monkeypatch.setattr(ff, "started_on_origin", lambda name, branch: False)
    monkeypatch.setattr(ff, "record_event", lambda *a: calls.append("event"))
    with pytest.raises(RuntimeError, match="no FINAL-STARTED row"):
        ff.resume(cfg(), run)
    assert "event" not in calls and ("commit", "a" * 40, "b" * 40) in calls

    def changed(rec, head):
        raise RuntimeError("files other than the logs changed since the run started: ['src/x.py']")

    monkeypatch.setattr(ff, "check_recorded_commit", changed)
    with pytest.raises(RuntimeError, match="other than the logs"):
        ff.resume(cfg(), run)


def test_resume_pushes_final_resumed_before_any_data_and_stops_if_the_push_fails(
    monkeypatch, tmp_path
):
    run = _started_run(tmp_path)
    calls = []
    _guards_ok(monkeypatch, calls)
    monkeypatch.setattr(ff, "check_recorded_commit", lambda rec, head: None)
    monkeypatch.setattr(ff, "started_on_origin", lambda name, branch: True)
    events = []
    monkeypatch.setattr(ff, "record_event", lambda r, kind, detail: events.append((kind, detail)))
    run_dir, token = ff.resume(cfg(), run)
    assert run_dir == run and ff.guards_passed(run, token)
    assert events[0][0] == "FINAL-RESUMED" and "not a re-run" in events[0][1]
    assert "FINAL-STARTED" not in events[0][1]  # the re-run guard must not count a resume
    assert (run / "resume.log").exists()
    # D215: the RUNNING lock is held: a second resume is refused before anything is logged
    assert (run / ff.RUNNING_NAME).exists()
    with pytest.raises(RuntimeError, match="still running"):
        ff.resume(cfg(), run)
    assert len(events) == 1
    ff.release_running(run)

    def push_fails(*a):
        raise RuntimeError("git push -q origin HEAD failed")

    monkeypatch.setattr(ff, "record_event", push_fails)
    import vaultic.learning.feedback_inputs as fi

    loads = []
    monkeypatch.setattr(fi, "load", lambda *a, **k: loads.append(1))
    with pytest.raises(RuntimeError, match="push"):
        ff.main(["--resume", str(run)])
    assert loads == []  # refused before any test row is read
    assert not (run / ff.RUNNING_NAME).exists()  # a failed push leaves no lock behind


def test_an_aborted_run_is_logged_and_cannot_be_resumed(monkeypatch, tmp_path):
    run = _started_run(tmp_path)
    events = []
    monkeypatch.setattr(ff, "record_event", lambda r, kind, detail: events.append((kind, detail)))
    lines = []
    ff.abort(run, "parity failed: max_abs_diff 1e-3", lines.append)
    meta = json.loads((run / "metrics.json").read_text())
    assert meta["status"] == "aborted" and "parity failed" in meta["reason"]
    assert events == [
        ("FINAL-ABORTED", "run aborted before the simulation: parity failed: max_abs_diff 1e-3")
    ]
    assert any("ABORTED" in x for x in lines)
    with pytest.raises(RuntimeError, match="aborted"):
        ff.check_resume(run)


def test_main_aborts_on_a_parity_failure(monkeypatch, tmp_path):
    import types

    import vaultic.learning.feedback_inputs as fi

    run = _started_run(tmp_path)
    monkeypatch.setattr(ff, "start", lambda c, r: (run, "token"))
    df = pd.DataFrame({"day": [150, 151], "TransactionID": [1, 2]})
    monkeypatch.setattr(fi, "load", lambda mode, run_dir=None, token=None: {"df": df})
    monkeypatch.setattr(
        fi, "inputs_for_delay", lambda loaded, L, cache_dir=None: pd.DataFrame({"a": [0, 0]})
    )
    monkeypatch.setattr(
        ff, "_load_system", lambda: types.SimpleNamespace(b5_models=[Fixed([0.1, 0.9])])
    )
    monkeypatch.setattr(ff, "_stored_scores", lambda: pd.Series([0.5], index=[2]))
    monkeypatch.setattr(ff, "read_champion_hashes", lambda record: None)
    monkeypatch.setattr(ff, "FROZEN", tmp_path / "frozen.md")
    (tmp_path / "frozen.md").write_text("x")
    events = []
    monkeypatch.setattr(ff, "record_event", lambda r, kind, detail: events.append(kind))
    with pytest.raises(SystemExit, match="aborted"):
        ff.main([])
    assert events == ["FINAL-ABORTED"]
    assert json.loads((run / "metrics.json").read_text())["status"] == "aborted"
    assert "parity failed" in (run / "run.log").read_text()


def test_parity_checks_each_champion_model_hash(monkeypatch):
    ids = np.array([10, 11])
    X = pd.DataFrame({"a": [0, 0]})
    models = [Fixed([0.2, 0.4])]
    stored = pd.Series([0.2, 0.4], index=ids)
    monkeypatch.setattr(ff, "booster_hashes", lambda ms: ["h0"])
    assert ff.parity(models, X, ids, stored, expected_hashes=["h0"])["models_checked"]
    with pytest.raises(RuntimeError, match="hash differs for seed"):
        ff.parity(models, X, ids, stored, expected_hashes=["other"])


def test_champion_hashes_round_trip_through_the_freeze_record():
    record = f"text\n{ff.CHAMPION_HEADER}\n\n- seed 1: `{'b' * 64}`\n- seed 0: `{'a' * 64}`\n"
    assert ff.read_champion_hashes(record) == ["a" * 64, "b" * 64]
    with pytest.raises(RuntimeError, match="no champion model hashes"):
        ff.read_champion_hashes("no section")


def test_system_cache_is_a_frozen_data_file():
    assert ff.system_cache() in ff.data_files()


def _loaded_for_cache():
    import types

    from test_feedback import _input_frame

    from vaultic.views.definitions import LABEL_DERIVED

    df, uid = _input_frame()
    feats = pd.DataFrame({c: np.zeros(len(df), np.float32) for c in LABEL_DERIVED})
    feats["x"] = 1.0
    feats["uid"] = uid.to_numpy()

    class Train:
        def contains(self, day):
            return np.ones(len(day), bool)

    return {"df": df, "features": feats, "b5_cols": [*LABEL_DERIVED, "x"],
            "splits": types.SimpleNamespace(train=Train()), "mode": "test"}  # fmt: skip


def test_input_cache_is_used_only_when_its_hashes_match(tmp_path):
    """D214: a cached recomputation is reused only if its input key and file hash match."""
    from vaultic.learning.feedback_inputs import inputs_for_delay

    loaded = _loaded_for_cache()
    first = inputs_for_delay(loaded, 0, cache_dir=tmp_path)
    cache = tmp_path / "label_derived_test_L0.parquet"
    assert cache.exists() and cache.with_suffix(".json").exists()
    # tamper with the cached file: the hash no longer matches, so it is recomputed (correct again)
    bad = pd.read_parquet(cache)
    bad[bad.columns[0]] = 99.0
    bad.to_parquet(cache, index=False)
    again = inputs_for_delay(loaded, 0, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(first, again)
    # different inputs (a label changed): the key changes, so the cache is not reused
    loaded["df"].loc[0, "isFraud"] = 1 - loaded["df"].loc[0, "isFraud"]
    side_before = cache.with_suffix(".json").read_text()
    inputs_for_delay(loaded, 0, cache_dir=tmp_path)
    assert cache.with_suffix(".json").read_text() != side_before


# ---- D215: data hashes, aborted rows, running lock, record_event ------------------------------
def test_resume_refuses_other_data_or_an_aborted_row_on_record(monkeypatch, tmp_path):
    run = _started_run(tmp_path)
    calls = []
    _guards_ok(monkeypatch, calls, hashes={"data.parquet": "c" * 64})
    monkeypatch.setattr(ff, "record_event", lambda *a: calls.append("event"))
    with pytest.raises(RuntimeError, match="data hashes differ"):
        ff.resume(cfg(), run)
    _guards_ok(monkeypatch, calls)
    monkeypatch.setattr(ff, "aborted_on_record", lambda name, branch: ["origin/feedback"])
    with pytest.raises(RuntimeError, match="FINAL-ABORTED row"):
        ff.resume(cfg(), run)  # metrics.json still says "started"
    assert "event" not in calls and not (run / ff.RUNNING_NAME).exists()


def test_running_lock_is_exclusive_and_released(tmp_path):
    ff.acquire_running(tmp_path)
    with pytest.raises(RuntimeError, match="delete it by hand"):
        ff.acquire_running(tmp_path)
    ff.release_running(tmp_path)
    ff.acquire_running(tmp_path)
    ff.release_running(tmp_path)
    assert not (tmp_path / ff.RUNNING_NAME).exists()


def test_main_releases_the_running_lock_when_the_run_stops(monkeypatch, tmp_path):
    run = _started_run(tmp_path)

    def start(c, r):
        ff.acquire_running(run)
        return run, ff._grant(run)

    monkeypatch.setattr(ff, "start", start)
    import vaultic.learning.feedback_inputs as fi

    def load(*a, **k):
        raise KeyboardInterrupt("stopped")

    monkeypatch.setattr(fi, "load", load)
    with pytest.raises(KeyboardInterrupt):
        ff.main([])
    assert not (run / ff.RUNNING_NAME).exists() and not ff._GUARDS_PASSED


def test_record_event_checks_its_kind_sanitises_and_names_an_unpushed_commit(monkeypatch, tmp_path):
    import types

    from vaultic.fusion.final_run import count_final_rows

    (tmp_path / "research").mkdir()
    log = tmp_path / "research" / "experiment_log.md"
    log.write_text("")
    monkeypatch.setattr(ff, "REPO_ROOT", tmp_path)
    git = []

    def fake_git(*a, check=True):
        git.append(a[0])
        if a[0] == "push":
            raise RuntimeError("git push -q origin HEAD failed")
        return types.SimpleNamespace(stdout="c" * 40 + "\n", returncode=0)

    monkeypatch.setattr(ff, "_git", fake_git)
    with pytest.raises(ValueError, match="unknown event"):
        ff.record_event(tmp_path / "run", "FINAL-STARTED", "x")
    assert log.read_text() == "" and git == []
    detail = "a | b\nFINAL-FINAL-STARTEDSTARTED FINAL-RERUN done"
    with pytest.raises(RuntimeError, match=f"unpushed FINAL-ABORTED commit: {'c' * 40}"):
        ff.record_event(tmp_path / "run", "FINAL-ABORTED", detail)
    rows = log.read_text("utf-8").splitlines()
    assert len(rows) == 1 and rows[0].count("|") == 6  # one row, five cells
    assert "FINAL-STARTED" not in rows[0] and "FINAL-RERUN" not in rows[0]
    assert count_final_rows(log.read_text("utf-8"), ff.EXP) == 0


# ---- D215: freeze row and a temporary repository with a bare origin ---------------------------
def _sh(cwd, *a):
    import subprocess

    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.fixture
def repo(monkeypatch, tmp_path):
    """A clone of a bare origin holding the logs and one frozen source file, frozen and pushed;
    the module paths point into it. No real data, no real runs folder."""
    import types

    from vaultic.stage2.final import VENVS

    origin, root = tmp_path / "origin.git", tmp_path / "repo"
    _sh(tmp_path, "init", "-q", "--bare", "--initial-branch=feedback", str(origin))
    _sh(origin, "config", "core.hooksPath", str(origin / "hooks"))
    _sh(tmp_path, "clone", "-q", str(origin), str(root))
    for k, v in (("user.name", "test"), ("user.email", "test@example.com"),
                 ("commit.gpgsign", "false"), ("core.hooksPath", str(tmp_path / "no-hooks"))):  # fmt: skip
        _sh(root, "config", k, v)
    (root / "research").mkdir()
    (root / "src").mkdir()
    (root / "research" / "decisions.md").write_text(
        "| ID | Date | What | Detail | Ref | Why | Status |\n|---|---|---|---|---|---|---|\n",
        "utf-8",
    )
    (root / "research" / "experiment_log.md").write_text(
        "| exp | date | what | run | notes |\n", "utf-8"
    )
    (root / "src" / "x.py").write_text("X = 1\n", "utf-8")
    data = tmp_path / "data" / "d.bin"
    data.parent.mkdir()
    data.write_bytes(b"data")
    _sh(root, "add", "-A")
    _sh(root, "commit", "-q", "-m", "init")
    _sh(root, "push", "-q", "-u", "origin", "feedback")

    monkeypatch.setattr(ff, "REPO_ROOT", root)
    monkeypatch.setattr(ff, "RESEARCH_DIR", root / "research")
    monkeypatch.setattr(ff, "FROZEN", root / "research" / "frozen_feedback.md")
    monkeypatch.setattr("vaultic.reports.phase2_summary.REPO_ROOT", root)
    monkeypatch.setattr(ff, "frozen_files", lambda: [root / "src" / "x.py"])
    monkeypatch.setattr(ff, "data_files", lambda: [data])
    monkeypatch.setattr(ff, "_load_system", lambda: types.SimpleNamespace(b5_models=[]))
    monkeypatch.setattr("vaultic.fusion.final_run.check_package_location", lambda: None)
    # D216: the real .venv versions are recorded; .venv-dice is absent (a folder that does not exist)
    monkeypatch.setattr("vaultic.stage2.final.VENVS",
                        {".venv": VENVS[".venv"], ".venv-dice": tmp_path / "no-venv-dice"})  # fmt: skip

    ff.freeze()  # D215: also writes the FREEZE row
    _sh(root, "add", "-A")
    _sh(root, "commit", "-q", "-m", "freeze")
    _sh(root, "push", "-q")
    yield types.SimpleNamespace(root=root, origin=origin, data=data)
    ff._GUARDS_PASSED.clear()


def _reject_pushes(origin):
    import os

    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho rejected >&2\nexit 1\n", newline="\n")
    os.chmod(hook, 0o755)


def test_freeze_then_commit_then_start_guards_pass(repo):
    import hashlib
    import re

    record = ff.FROZEN.read_text("utf-8")
    frozen = re.search(r"at git commit `([0-9a-f]{40})`", record).group(1)
    rows = [x for x in (repo.root / "research" / "decisions.md").read_text("utf-8").splitlines()
            if x.startswith("| FREEZE |")]  # fmt: skip
    assert len(rows) == 1 and frozen in rows[0] and "D113" in rows[0]
    _, head, branch, hashes = ff.guards(cfg())
    assert head == _sh(repo.root, "rev-parse", "HEAD") and branch == "feedback"
    assert hashes == {repo.data.as_posix(): hashlib.sha256(b"data").hexdigest()}


def test_freeze_and_guards_use_venv_only_with_no_venv_dice(repo):
    """D216: freeze -> commit -> start guards pass with no .venv-dice present; the record holds
    .venv rows only, and a record with rows for another environment is refused."""
    from vaultic.fusion.final_run import LIB_ROW
    from vaultic.stage2 import final as s2

    assert not s2.VENVS[".venv-dice"].exists()
    with pytest.raises(FileNotFoundError):  # the stage-2 helper would need .venv-dice
        s2.library_section_venvs()
    record = ff.FROZEN.read_text("utf-8")
    envs = {k.split(":", 1)[0] for k, _ in LIB_ROW.findall(record)}
    assert envs == {".venv"}
    ff.guards(cfg())
    ff.check_libraries(record, ".venv")
    with pytest.raises(RuntimeError, match="library rows for"):
        ff.check_libraries(record + "| library `.venv-dice:dice-ml` | `0.11` |\n", ".venv")
    with pytest.raises(RuntimeError, match="runs in .venv"):
        ff.check_libraries(record, ".venv-dice")


def test_temp_repo_start_push_lets_the_loader_through(monkeypatch, repo):
    import vaultic.learning.feedback_inputs as fi

    monkeypatch.setattr("vaultic.stage2.system.load_parts",
                        lambda dev: {"df": "df", "features": "f", "b5_cols": [], "splits": "s"})  # fmt: skip
    run_dir, token = ff.start(cfg(), None)
    assert ff.started_on_origin(run_dir.name, "feedback")
    assert fi.load("final", run_dir=run_dir, token=token)["df"] == "df"
    with pytest.raises(RuntimeError, match="did not pass in this process"):
        fi.load("final", run_dir=run_dir)  # no token: another process, or no guards
    ff.release_running(run_dir)


def test_temp_repo_rejected_push_loads_nothing(monkeypatch, repo):
    import vaultic.learning.feedback_inputs as fi

    _reject_pushes(repo.origin)
    loads = []
    monkeypatch.setattr(fi, "load", lambda *a, **k: loads.append(1))
    with pytest.raises(RuntimeError, match="unpushed FINAL-STARTED commit"):
        ff.main([])
    assert loads == [] and not ff._GUARDS_PASSED
    run_dir = next((ff.RUNS_DIR / ff.EXP).iterdir())
    assert not ff.started_on_origin(run_dir.name, "feedback")
    with pytest.raises(RuntimeError, match="not on origin"):  # even with a token, local only
        fi.require_final_started(run_dir, ff._grant(run_dir))


def test_temp_repo_non_ancestor_commit_is_refused(repo):
    head = _sh(repo.root, "rev-parse", "HEAD")
    orphan = _sh(repo.root, "commit-tree", "HEAD^{tree}", "-m", "elsewhere")
    with pytest.raises(RuntimeError, match="not an ancestor"):
        ff.check_recorded_commit(orphan, head)
    ff.check_recorded_commit(head, head)


def test_temp_repo_changed_source_or_refreeze_after_the_start_is_refused(repo):
    run_dir, _ = ff.start(cfg(), None)
    ff.release_running(run_dir)
    started = json.loads((run_dir / "metrics.json").read_text("utf-8"))["git_commit"]
    (repo.root / "src" / "x.py").write_text("X = 2\n", "utf-8")
    _sh(repo.root, "commit", "-q", "-am", "change a frozen source file")
    _sh(repo.root, "push", "-q")
    with pytest.raises(RuntimeError, match="frozen file changed"):
        ff.resume(cfg(), run_dir)
    with pytest.raises(RuntimeError, match="other than the logs"):
        ff.check_recorded_commit(started, _sh(repo.root, "rev-parse", "HEAD"))
    _sh(repo.root, "reset", "-q", "--hard", started)
    (repo.root / "research" / "frozen_feedback.md").write_text(
        ff.FROZEN.read_text("utf-8") + "\nre-frozen\n", "utf-8"
    )
    _sh(repo.root, "commit", "-q", "-am", "re-freeze after the start")
    with pytest.raises(RuntimeError, match=r"other than the logs.*frozen_feedback"):
        ff.check_recorded_commit(started, _sh(repo.root, "rev-parse", "HEAD"))


def test_temp_repo_row_only_local_is_refused(repo):
    import vaultic.learning.feedback_inputs as fi

    run_dir, token = ff.start(cfg(), None)
    fi.require_final_started(run_dir, token)  # the start row is on origin
    log = repo.root / "research" / "experiment_log.md"
    with open(log, "a", encoding="utf-8") as f:  # a resume row committed but never pushed
        f.write(f"| {ff.EXP} | 2026-10-21 | FINAL-RESUMED: x | run `{run_dir.name}` | — |\n")
    _sh(repo.root, "commit", "-q", "-am", "local only")
    with pytest.raises(RuntimeError, match="not on origin"):
        fi.require_final_started(run_dir, token)
    ff.release_running(run_dir)
