"""Attack Lab service and endpoints (Phase B) with a fake engine and a stubbed scenario run."""

import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vaultic.scenarios import LABEL
from vaultic.scenarios.service import AttackInputError, AttackLab

ROOT = Path(__file__).resolve().parents[1]


class FakeTestLab:
    def __init__(self, ready=True):
        self.ready = ready
        self.engine = SimpleNamespace(
            store=object(), system=object(), stored=None, lock=threading.Lock()
        )

    def _ready(self):
        from vaultic.live.service import NotReady

        if not self.ready:
            raise NotReady("loading")
        return self.engine


def _fake_run(store, system, stored, sc, log=None):
    rows = pd.DataFrame({
        "TransactionID": [1_000_000_000, 1_000_000_001, 5, 6],
        "TransactionDT": [100, 200, 150, 50], "amount": [3.0, 4.0, 10.0, 20.0],
        "synthetic": [True, True, False, False], "label": [1, 1, 0, 0],
        "source": ["scored live", "scored live", "scored live", "stored (parity-verified)"],
        "risk": [0.1, 0.2, 0.3, 0.4], "model_action": ["step_up", "allow", "allow", "allow"],
        "rule_action": ["allow", "hold", "allow", "allow"], "rules_fired": ["", "R6", "", ""],
        "combined_action": ["step_up", "hold", "allow", "allow"]})  # fmt: skip
    return {"days": [178], "info": {"customer": np.int64(7)}, "n_synthetic": 2, "n_legit_replay": 2,
            "n_legit_stored": 1, "n_legit_live": 1, "stored_score_max_diff": 0.0, "seconds": 1.0,
            "results": {"D196 (primary)": {}}, "rows": rows,
            "traces": {1_000_000_000: [{"rule": "R1"}], 1_000_000_001: [{"rule": "R6"}]}}  # fmt: skip


@pytest.fixture
def lab(monkeypatch, tmp_path):
    import vaultic.scenarios.generators as gen
    import vaultic.scenarios.run as run

    monkeypatch.setattr(run, "run_scenario", _fake_run)
    monkeypatch.setitem(
        gen.BY_VERSION["v3"], "card_testing", lambda store, seed, intensity: SimpleNamespace()
    )
    return AttackLab(
        FakeTestLab(),
        suite_dirs={"v1": tmp_path, "v2": tmp_path, "v3": tmp_path},
        docs_dir=tmp_path,
    )


def test_run_returns_labelled_replay_without_stored_rows(lab):
    out = lab.run({"scenario": "card_testing", "intensity": "low", "seed": 3, "model": True})
    assert out["label"] == LABEL and out["model"] == "on"
    ids = [r["TransactionID"] for r in out["replay"]]
    assert 6 not in ids and set(ids) == {1_000_000_000, 1_000_000_001, 5}  # stored rows not sent
    by = {r["TransactionID"]: r for r in out["replay"]}
    assert by[1_000_000_001]["decision"] == "hold" and by[1_000_000_000]["decision"] == "step_up"
    assert by[1_000_000_000]["trace"] == [{"rule": "R1"}] and by[5]["trace"] is None
    assert out["info"]["customer"] == 7
    assert "friction" in out["step_up_note"]


def test_degraded_mode_lets_the_rules_alone_decide(lab):
    out = lab.run({"scenario": "card_testing", "intensity": "low", "seed": 0, "model": False})
    by = {r["TransactionID"]: r for r in out["replay"]}
    assert out["model"].startswith("off")
    assert by[1_000_000_000]["decision"] == "allow"  # model's step-up ignored
    assert by[1_000_000_001]["decision"] == "hold"


@pytest.mark.parametrize("spec, msg", [
    ({"scenario": "nope"}, "scenario must be"),
    ({"scenario": "card_testing", "intensity": "extreme"}, "intensity must be"),
    ({"scenario": "card_testing", "seed": -1}, "seed must be between"),
    ({"scenario": "card_testing", "seed": "x"}, "whole number"),
    ({"scenario": "card_testing", "generator": "v9"}, "generator must be"),
])  # fmt: skip
def test_bad_requests_are_refused(lab, spec, msg):
    with pytest.raises(AttackInputError, match=msg):
        lab.run(spec)


def test_scenarios_list_notes_and_sets(lab):
    out = lab.scenarios()
    names = [s["name"] for s in out["scenarios"]]
    assert names == ["card_testing", "account_takeover", "mule_ring", "bust_out", "mimicry"]
    mule = next(s for s in out["scenarios"] if s["name"] == "mule_ring")
    assert any("no true device ID" in n for n in mule["notes"])
    assert out["threshold_sets"] == [
        "D196 (primary)",
        "D195 (sensitivity)",
        "D196 without R3 (post hoc)",
    ]


def test_endpoints(lab):
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    spec = importlib.util.spec_from_file_location("attack_api", ROOT / "services/api/attack.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    app = FastAPI()
    app.include_router(mod.make_router(lab))
    c = TestClient(app)
    assert c.get("/api/attack/scenarios").status_code == 200
    assert c.post("/api/attack/run", json={"scenario": "nope"}).status_code == 400
    assert c.post("/api/attack/run", json={"scenario": "card_testing"}).json()["label"] == LABEL
    suite = c.get("/api/attack/suite").json()
    assert {k: suite[k] for k in ("label", "suite", "suite_v2", "suite_v1", "posthoc")} == {
        "label": LABEL, "suite": None, "suite_v2": None, "suite_v1": None, "posthoc": None}  # fmt: skip
    assert "review finding 1" in suite["superseded"]["v2"]
    lab.testlab.ready = False
    r = c.post("/api/attack/run", json={"scenario": "card_testing"})
    assert r.status_code == 503 and "not ready" in r.json()["detail"]
