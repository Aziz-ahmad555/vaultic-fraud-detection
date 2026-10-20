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


def test_final_data_needs_the_final_started_row(tmp_path):
    """D214: load('final') checks the FINAL-STARTED row naming the run folder, not a flag."""
    from vaultic.learning.feedback_inputs import load, require_final_started

    with pytest.raises(RuntimeError, match="go-ahead"):
        load("final")
    run = tmp_path / "EXP-310-final" / "20261021-000000-000001"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"status": "started"}))
    log = tmp_path / "experiment_log.md"
    log.write_text("| EXP-310-final | 2026-10-21 | FINAL-STARTED: x | run `other` | - |\n")
    with pytest.raises(RuntimeError, match="no FINAL-STARTED row"):
        require_final_started(run, log)
    log.write_text(f"| EXP-310-final | 2026-10-21 | FINAL-STARTED: x | run `{run.name}` | - |\n")
    require_final_started(run, log)
    (run / "metrics.json").write_text(json.dumps({"status": "aborted"}))
    with pytest.raises(RuntimeError, match="aborted"):
        require_final_started(run, log)


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
    run = tmp_path / "EXP-310-final" / "20261021-000000-000000"
    with pytest.raises(RuntimeError, match="not a started"):
        ff.check_resume(run)
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"status": "completed"}))
    with pytest.raises(RuntimeError, match="already completed"):
        ff.check_resume(run)
    (run / "metrics.json").write_text(json.dumps({"status": "started"}))
    assert ff.check_resume(run)["status"] == "started"
    other = tmp_path / "EXP-999" / "x"
    other.mkdir(parents=True)
    (other / "metrics.json").write_text(json.dumps({"status": "started"}))
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
    (run / "metrics.json").write_text(json.dumps({"status": status, "git_commit": "a" * 40}))
    return run


def _guards_ok(monkeypatch, calls):
    def ok(c):
        calls.append("guards")
        return "record", "b" * 40, "feedback", {}

    monkeypatch.setattr(ff, "guards", ok)


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
    assert ff.resume(cfg(), run) == run
    assert events[0][0] == "FINAL-RESUMED" and "not a re-run" in events[0][1]
    assert "FINAL-STARTED" not in events[0][1]  # the re-run guard must not count a resume
    assert (run / "resume.log").exists()

    def push_fails(*a):
        raise RuntimeError("git push -q origin HEAD failed")

    monkeypatch.setattr(ff, "record_event", push_fails)
    import vaultic.learning.feedback_inputs as fi

    loads = []
    monkeypatch.setattr(fi, "load", lambda *a, **k: loads.append(1))
    with pytest.raises(RuntimeError, match="push"):
        ff.main(["--resume", str(run)])
    assert loads == []  # refused before any test row is read


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
    monkeypatch.setattr(ff, "start", lambda c, r: (run, "b" * 40))
    df = pd.DataFrame({"day": [150, 151], "TransactionID": [1, 2]})
    monkeypatch.setattr(fi, "load", lambda mode, run_dir=None: {"df": df})
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
