"""EXP-310-final tooling (D213): settings, guard order, parity, resume per completed delay and the
hypothesis read-out. No real data and no test-period row is read here."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

import vaultic.learning.feedback_final as ff
from vaultic.paths import CONFIG_DIR


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


def test_final_data_needs_the_started_flag():
    from vaultic.learning.feedback_inputs import load

    with pytest.raises(RuntimeError, match="go-ahead"):
        load("final")


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
