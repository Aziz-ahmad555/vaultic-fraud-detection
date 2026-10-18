"""Test Lab endpoints (services/api/testlab.py) with the real TestLabService and a small fake
engine, served by a real uvicorn server and called over HTTP (standard library only, D182)."""

import contextlib
import importlib.util
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

ROOT = Path(__file__).resolve().parents[1]


def _router_module():
    spec = importlib.util.spec_from_file_location("testlab_api", ROOT / "services/api/testlab.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeEngine:
    def __init__(self):
        n = 40
        day = np.r_[np.full(10, 140), np.full(30, 151)]
        y = np.r_[np.zeros(30, int), np.ones(10, int)]
        self.store = SimpleNamespace(
            df=pd.DataFrame({"TransactionID": np.arange(1000, 1000 + n), "day": day,
                             "TransactionDT": day * 86400 + 60, "isFraud": y,
                             "TransactionAmt": np.full(n, 25.0), "ProductCD": ["W"] * n}),
            features=pd.DataFrame({"hist_n_past": np.r_[np.zeros(20), np.ones(20)]}),
            uid=np.arange(n), splits=SimpleNamespace(test=SimpleNamespace(first=151, last=182)),
            info={},
        )  # fmt: skip
        from vaultic.live.engine import LatencyLog

        self.latency = LatencyLog()
        self.lock = threading.Lock()

    def score_stored(self, p):
        self.latency.add(12.0)
        return {"risk": 0.5, "calibrated": 0.2, "decision": "step_up", "alert": True, "transaction_id":
                int(self.store.df["TransactionID"].iat[p]), "matches_frozen_run": True}  # fmt: skip


@contextlib.contextmanager
def serve(service):
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(_router_module().make_router(service, audit=getattr(service, "audit", None)))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        th.join(5)


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_loading_and_refused_states_answer_503_with_the_reason():
    from vaultic.live.service import TestLabService

    svc = TestLabService()
    with serve(svc) as base:
        assert call(base, "/api/live/status")[1]["state"] == "loading"
        code, body = call(base, "/api/live/sample")
        assert code == 503 and "not ready" in body["detail"]
        svc.start(build=lambda log: FakeEngine(),
                  parity=lambda e, p, log: {"passed": False, "n_failing_features": 3,
                                            "max_score_diff": {"MVAF": 0.1}, "feature_tolerance": 1e-6,
                                            "score_tolerance": 1e-12},
                  log=lambda m: None)  # fmt: skip
        st = call(base, "/api/live/status")[1]
        assert st["state"] == "refused" and "parity failed" in st["message"]
        code, body = call(base, "/api/score-live", {"transaction_id": 1015})
        assert code == 503 and "parity failed" in body["detail"]


def test_ready_sample_hides_the_label_then_reveal_and_latency():
    from vaultic.live.service import TestLabService

    svc = TestLabService()
    svc.start(build=lambda log: FakeEngine(), parity=lambda e, p, log: {"passed": True},
              log=lambda m: None)  # fmt: skip
    with serve(svc) as base:
        code, s = call(base, "/api/live/sample?filter=fraud")
        assert code == 200 and "isFraud" not in s and s["day"] >= 151
        code, r = call(base, "/api/score-live", {"transaction_id": s["transaction_id"]})
        assert code == 200 and r["matches_frozen_run"] and "isFraud" not in r
        assert r["expected_loss"] == 0.2 * 25.0  # calibrated x amount (D186)
        assert call(base, f"/api/live/reveal/{s['transaction_id']}")[1]["isFraud"] == 1
        assert call(base, "/api/live/latency")[1]["n"] == 1
        assert call(base, "/api/live/sample?filter=bogus")[0] == 400
        assert call(base, "/api/live/reveal/1000")[0] == 404  # day 140: not a test transaction
        assert call(base, "/api/score-live", {"nothing": 1})[0] == 400
        a = call(base, "/api/live/sample?filter=any&seed=7")[1]["transaction_id"]
        assert a == call(base, "/api/live/sample?filter=any&seed=7")[1]["transaction_id"]
        cold = call(base, "/api/live/sample?filter=cold_start")[1]
        assert cold["history"] == 0


def test_demo_server_mounts_the_router_only_when_asked():
    spec = importlib.util.spec_from_file_location(
        "demo_server_t", ROOT / "services/api/demo_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    from fastapi import FastAPI

    from vaultic.live.service import TestLabService

    app = FastAPI()
    svc = mod.mount_testlab(app, service=TestLabService(), start=False)
    paths = set(app.openapi()["paths"])
    assert {"/api/score-live", "/api/live/status", "/api/live/batch"} <= paths
    assert app.state.testlab is svc


def test_an_internal_error_is_a_500_not_a_404():
    from vaultic.live.service import TestLabService

    eng = FakeEngine()

    def broken(p):
        raise KeyError("no reason-code template for feature 'seq19_log_gap_next'")

    eng.score_stored = broken
    svc = TestLabService()
    svc.start(build=lambda log: eng, parity=lambda e, p, log: {"passed": True}, log=lambda m: None)
    with serve(svc) as base:
        code, body = call(base, "/api/score-live", {"transaction_id": 1015})
        assert code == 500 and "internal error" in body["detail"]
        assert call(base, "/api/score-live", {"transaction_id": 999999})[0] == 404


def test_sequence_inputs_get_honest_reason_sentences():
    from vaultic.live.reasons import describe

    assert "masked counter C13" in describe("seq03_C13", 2.0, 5.0)
    assert "window step 20 of 20" in describe("seq19_log_gap_next", 1.0, -3.0)
    assert describe("seq_cur_log_amount", 4.0, 2.0).endswith("raises the risk score")
    assert "earlier transactions" in describe("seq_n_steps", 3, -1.0)


def test_benchmark_percentiles_match_numpy():
    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "bench", ROOT / "scripts/bench_testlab_latency.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    v = [float(x) for x in np.random.default_rng(0).gamma(2.0, 200.0, 200)]
    for q in (50, 95):
        assert bench.percentile(v, q) == pytest.approx(float(np.percentile(v, q)))


def test_scores_and_reveals_are_recorded_in_the_audit_log(tmp_path):
    """D208: every live score and reveal is an audit entry with the request's analyst."""
    from vaultic.audit.store import SQLiteAuditStore
    from vaultic.live.service import TestLabService

    service = TestLabService()
    service.start(build=lambda log: FakeEngine(), parity=lambda e, p, log: {"passed": True},
                  log=lambda m: None)  # fmt: skip
    service.audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    who = {
        "X-Vaultic-Analyst": "Aziz",
        "X-Vaultic-Role": "analyst",
        "Content-Type": "application/json",
    }
    with serve(service) as base:
        req = urllib.request.Request(base + "/api/score-live", data=json.dumps({"transaction_id": 1015}).encode(),
                                     method="POST", headers=who)  # fmt: skip
        with urllib.request.urlopen(req) as r:
            assert r.status == 200
        with urllib.request.urlopen(
            urllib.request.Request(base + "/api/live/reveal/1015", headers=who)
        ) as r:
            assert r.status == 200
    log = service.audit.query()
    assert [e["action"] for e in log] == [
        "Test Lab: revealed a label",
        "Test Lab: scored a real test transaction",
    ]
    assert all(e["analyst"] == "Aziz" and e["target"] == "tx:1015" for e in log)
    assert service.audit.verify()["ok"]
