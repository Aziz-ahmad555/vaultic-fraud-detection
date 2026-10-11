"""Demo backend endpoints (services/api/demo_server.py) on a small synthetic bundle, served by a
real uvicorn server and called over HTTP (standard library only)."""

import importlib.util
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DAY = 86400
VIEWS = ("tabular", "behavioral", "temporal", "graph", "anomaly")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "demo_server", ROOT / "services/api/demo_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # pydantic resolves the models' annotations through it
    spec.loader.exec_module(mod)
    return mod


def _bundle(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 60
    day = np.repeat([151, 152, 190], [25, 25, 10])  # day 190: lets labels of 151 mature (L = 30)
    t = day * DAY + np.tile(np.arange(0, 3600 * 20, 3600 * 20 // 25)[:25], 3)[:n]
    p = rng.random(n)
    actions = np.where(
        p > 0.8, "block", np.where(p > 0.6, "hold", np.where(p > 0.4, "step_up", "allow"))
    )
    df = pd.DataFrame({
        "TransactionID": np.arange(1000, 1000 + n), "TransactionDT": t.astype(float), "day": day,
        "label": (p > 0.7).astype(int), "amount": rng.lognormal(4, 1, n), "product": "W",
        "uid": [f"u{i % 7}" for i in range(n)], "card": [f"c{i % 5}" for i in range(n)],
        "email": [None if i % 3 else "gmail.com" for i in range(n)], "address": "325",
        "device": [None] * n, "p_raw": p, "p": p, "action": actions,
        "conformal_set": np.where(p > 0.5, "uncertain", "legit"), "disagreement": rng.random(n) * 0.2,
        "reasons": [json.dumps([{"feature": "view:tabular", "view": "tabular", "points": 12.0,
                                 "text": "Tabular view raises risk"}])] * n,
        **{f"p_{v}": np.where(rng.random(n) < 0.2, np.nan, rng.random(n)) for v in VIEWS},
        **{f"w_{v}": np.full(n, 0.2) for v in VIEWS}, **{f"c_{v}": np.full(n, 0.5) for v in VIEWS},
        "ctx_log_amount": 4.0, "ctx_hist_n_past": 3.0, "ctx_has_identity": 1, "ctx_product": 5.0,
        "ctx_hour": (t % DAY) // 3600, "source": "IEEE-CIS test period, final run",
    })  # fmt: skip
    path = tmp_path / "demo_final_bundle.parquet"
    df.to_parquet(path, index=False)
    meta = {"source": "IEEE-CIS test period, final run", "run": "runs/x", "generated": "2026-10-11",
            "rows": n, "label_delay_days": 30, "reasons_note": "view level",
            "thresholds": {"t_monitor": 0.2, "t_step": 0.4, "t_hold": 0.6, "t_block": float("inf"), "t_dis": float("inf")},
            "cost_model": {"c_fp": 10, "c_rev": 5, "c_step": 0.5, "step_up_success_rate": 0.9,
                           "legit_step_up_friction": 1.0},
            "routing_lambdas": {"50": {"R3": [0.0, 0.0], "R4": [0.0, 1.0]}},
            "sensitivity": [{"c_fp": 10, "step_up_success_rate": 0.9, "cost": 1.0}],
            "lab": [{"id": "MVAF", "name": "MVAF", "kind": "fusion", "prAuc": 0.6, "ciLow": 0.5,
                     "ciHigh": 0.7, "recallAt1": 0.5, "cost": 1.0, "pr": [], "calibration": []}]}  # fmt: skip
    path.with_suffix(".json").write_text(json.dumps(meta), "utf-8")
    m = 400  # calibrate_fused rows for the what-if cost mode
    pc = rng.random(m)
    pd.DataFrame({"p": pc, "amount": rng.lognormal(4, 1, m),
                  "conformal_set": np.where(pc > 0.5, "uncertain", "legit"),
                  "disagreement": rng.random(m) * 0.2,
                  "label": (rng.random(m) < pc * 0.3).astype(int)}).to_parquet(
        path.with_name("demo_final_calibration.parquet"), index=False)  # fmt: skip
    return path


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import uvicorn

    app = _load_module().create_app(_bundle(tmp_path_factory.mktemp("bundle")))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=5)


def _get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return json.loads(r.read())


def _send(base, path, body, method):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method=method,
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


AS_OF = 151 * DAY + 12 * 3600  # noon of the first day


def test_meta_and_range_label_the_data_source(server):
    assert _get(server, "/api/meta")["source"] == "IEEE-CIS test period, final run"
    r = _get(server, "/api/range")
    assert r["first"] < r["last"]


def test_nothing_after_as_of_and_label_curtain(server):
    rows = _get(server, f"/api/stream?time={AS_OF}&start=0&end={AS_OF + 10 * DAY}")
    assert rows and all(t["row"]["TransactionDT"] <= AS_OF for t in rows)
    assert all(t["row"]["label"] is None for t in rows)  # nothing matured yet
    later = 190 * DAY + 10  # labels are judged at the window end, as in the mock
    late = _get(server, f"/api/stream?time={later}&start=0&end={later}")
    day151 = [t for t in late if t["row"]["day"] == 151]
    assert day151 and all(t["row"]["label"] in (0, 1) for t in day151)  # matured after L days
    tx = rows[0]
    assert set(tx) >= {"row", "amount", "entities", "weights", "decision", "reasons", "inputs"}
    assert tx["decision"]["action"] in ("allow", "monitor", "step_up", "hold", "block")
    assert all(tx["row"][f"m_{v}"] == (tx["row"][f"p_{v}"] is not None) for v in VIEWS)
    future_id = 1000 + 59
    assert _get(server, f"/api/transactions/{future_id}?time={AS_OF}") is None


def test_day_summary_alerts_queues_history_network(server):
    s = _get(server, f"/api/day-summary?time={AS_OF}")
    assert s["day"] == 151 and s["transactions"] == sum(s["byAction"].values())
    alerts = _get(server, f"/api/alerts?time={AS_OF}")
    ps = [a["decision"]["p"] for a in alerts]
    assert ps == sorted(ps, reverse=True) and all(
        a["decision"]["action"] != "allow" for a in alerts
    )
    q = _get(server, f"/api/queues?time={AS_OF}&k=5")
    assert [x["policy"] for x in q] == ["R1", "R2", "R3", "R4"] and all(
        len(x["cases"]) <= 5 for x in q
    )
    assert q[0]["matured"]["day"] is None  # no labelled day yet at as-of
    h = _get(server, f"/api/customers/u1/history?time={AS_OF}")
    assert all(t["entities"]["uid"] == "u1" for t in h)
    net = _get(server, f"/api/network?time={AS_OF}&focus=u1")
    assert any(n["id"] == "uid:u1" for n in net["nodes"])


def test_drift_lab_score_thresholds_feedback_audit(server):
    d = _get(server, f"/api/drift?time={AS_OF}")
    assert d["alarms"] == [] and d["retrains"] == [] and len(d["featurePsi"]) == 5
    assert _get(server, "/api/lab")["runs"][0]["id"] == "MVAF"
    code, _ = _send(server, "/api/score", {}, "POST")
    assert code == 501
    th = _get(server, "/api/thresholds")["thresholds"]
    assert th["t_dis"] is None and th["t_block"] is None  # inf on the Python side = null
    code, _ = _send(server, "/api/thresholds", {"thresholds": th, "role": "analyst"}, "PUT")
    assert code == 403
    bad = {**th, "t_monitor": 0.99}
    code, _ = _send(server, "/api/thresholds", {"thresholds": bad, "role": "admin"}, "PUT")
    assert code == 422
    new = {**th, "t_block": 0.85}
    code, _ = _send(server, "/api/thresholds", {"thresholds": new, "role": "admin"}, "PUT")
    assert code == 200 and _get(server, "/api/thresholds")["thresholds"]["t_block"] == 0.85
    code, _ = _send(server, "/api/feedback/1000", {"label": "fraud", "role": "analyst"}, "POST")
    assert code == 200
    log = _get(server, "/api/audit")
    assert log[0]["action"] == "Marked transaction 1000 as fraud" and len(log) == 2


def test_no_block_note_and_whatif_cost_mode(server):
    meta = _get(server, "/api/meta")
    assert meta["whatif_available"] is True
    assert "No Block decisions" in meta["no_block_note"]  # t_block = inf in the bundle
    w = _get(server, "/api/whatif")
    assert w["active"] is False and "not the evaluated system" in w["note"]
    assert w["cost"]["hold_success_rate"] == 1.0  # the evaluated system
    before = _get(server, f"/api/day-summary?time={AS_OF}")
    assert before["whatIf"] is False
    cost = {"c_fp": 2.0, "c_rev": 1.0, "c_step": 0.1, "step_up_success_rate": 0.9,
            "legit_step_up_friction": 0.0}  # fmt: skip
    code, w = _send(server, "/api/whatif", {"cost": cost, "role": "analyst"}, "PUT")
    assert code == 200 and w["active"] is True and w["thresholds"]["t_monitor"] is not None
    after = _get(server, f"/api/day-summary?time={AS_OF}")
    assert after["whatIf"] is True and after["transactions"] == before["transactions"]
    rows = _get(server, f"/api/stream?time={AS_OF}&start=0&end={AS_OF}")
    assert all(t["decision"]["whatIf"] is True for t in rows)
    assert _get(server, "/api/audit")[0]["action"].startswith("Turned on the what-if cost mode")
    code, w = _send(
        server,
        "/api/whatif",
        {"cost": {**cost, "hold_success_rate": 0.7}, "role": "analyst"},
        "PUT",
    )
    assert code == 200 and w["cost"]["hold_success_rate"] == 0.7  # imperfect Hold (D124)
    code, _ = _send(
        server,
        "/api/whatif",
        {"cost": {**cost, "hold_success_rate": 1.2}, "role": "analyst"},
        "PUT",
    )
    assert code == 422
    bad = {**cost, "step_up_success_rate": 1.5}
    code, _ = _send(server, "/api/whatif", {"cost": bad, "role": "analyst"}, "PUT")
    assert code == 422
    req = urllib.request.Request(server + "/api/whatif?role=analyst", method="DELETE")
    with urllib.request.urlopen(req) as r:
        assert json.loads(r.read())["active"] is False
    back = _get(server, f"/api/day-summary?time={AS_OF}")
    assert back["whatIf"] is False and back["byAction"] == before["byAction"]
