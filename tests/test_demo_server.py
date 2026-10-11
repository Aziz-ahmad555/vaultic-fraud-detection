"""Demo backend endpoints (services/api/demo_server.py) on a small synthetic bundle, served by a
real uvicorn server and called over HTTP (standard library only)."""

import contextlib
import importlib.util
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
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


@contextlib.contextmanager
def _serve(app):
    import uvicorn

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
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    with _serve(_load_module().create_app(_bundle(tmp_path_factory.mktemp("bundle")))) as base:
        yield base


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
    q = _get(server, f"/api/queues?time={AS_OF}&k=50")
    assert [x["policy"] for x in q] == ["R1", "R2", "R3", "R4"] and all(
        len(x["cases"]) <= 50 for x in q
    )
    assert q[0]["matured"]["day"] is None  # no labelled day yet at as-of
    h = _get(server, f"/api/customers/u1/history?time={AS_OF}")
    assert all(t["entities"]["uid"] == "u1" for t in h)
    net = _get(server, f"/api/network?time={AS_OF}&focus=uid:u1")
    assert any(n["id"] == "uid:u1" for n in net["nodes"])


def test_drift_lab_score_thresholds_feedback_audit(server):
    d = _get(server, f"/api/drift?time={AS_OF}")
    assert d["alarms"] == [] and d["retrains"] == []
    assert "147-150" in d["reference"]
    assert {f["feature"] for f in d["featurePsi"]} <= {
        "fused risk score",
        "amount",
        "view disagreement",
    }  # the inputs the reference has
    assert _get(server, "/api/lab")["runs"][0]["id"] == "MVAF"
    code, _ = _send(server, "/api/score", {}, "POST")
    assert code == 501
    got = _get(server, "/api/thresholds")
    th = got["thresholds"]
    assert got["frozen"] is True
    assert th["t_dis"] is None and th["t_block"] is None  # inf on the Python side = null
    assert th == {"t_monitor": 0.2, "t_step": 0.4, "t_hold": 0.6, "t_block": None, "t_dis": None}
    # read-only for everyone, admins included (D129); nothing changes
    for role in ("analyst", "admin"):
        code, body = _send(server, "/api/thresholds", {"thresholds": {**th, "t_block": 0.85},
                                                        "role": role}, "PUT")  # fmt: skip
        assert code == 405 and "read-only" in body["detail"]
    assert _get(server, "/api/thresholds")["thresholds"] == th
    code, _ = _send(server, "/api/feedback/1000", {"label": "fraud", "role": "analyst"}, "POST")
    assert code == 200
    log = _get(server, "/api/audit")
    assert log[0]["action"] == "Marked transaction 1000 as fraud" and len(log) == 1


AS_OF_ENDPOINTS = ("/api/day-summary?time={t}", "/api/stream?time={t}&start=0&end={t}",
                   "/api/transactions/1000?time={t}", "/api/alerts?time={t}",
                   "/api/customers/u1/history?time={t}", "/api/network?time={t}",
                   "/api/queues?time={t}&k=50", "/api/drift?time={t}")  # fmt: skip


def _status(base, path):
    try:
        with urllib.request.urlopen(base + path) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.mark.parametrize("bad", [-1, 1, 14, 29, 31, 90, 365])
def test_every_as_of_endpoint_refuses_a_label_delay_outside_the_offered_ones(server, bad):
    for path in AS_OF_ENDPOINTS:
        code, body = _status(server, path.format(t=AS_OF) + f"&L={bad}")
        assert code == 422, path
        assert "0, 7, 30, 60" in json.dumps(body)


@pytest.mark.parametrize("L", [0, 7, 30, 60])
def test_every_as_of_endpoint_accepts_the_offered_label_delays(server, L):
    for path in AS_OF_ENDPOINTS:
        assert _status(server, path.format(t=AS_OF) + f"&L={L}")[0] == 200, path


def test_matured_day_is_always_before_the_as_of_day():
    mod = _load_module()
    end_of_day = 152 * DAY + DAY - 1  # last second of day 152
    assert mod.matured_day(end_of_day, 0, 151) == 151  # L = 0 used to give day 152 itself
    assert mod.matured_day(152 * DAY + 3600, 0, 151) == 151
    assert mod.matured_day(152 * DAY + 3600, 30, 151) is None  # nothing matured yet
    # the matured day's last second plus L is never after the as-of time
    for t in range(151 * DAY, 200 * DAY, 7919):
        for L in (0, 7, 30, 60):
            d = mod.matured_day(t, L, 1)
            assert d < t // DAY and (d + 1) * DAY - 1 + L * DAY <= t


def test_queues_matured_day_at_the_end_of_a_day_with_no_label_delay(server):
    q = _get(server, f"/api/queues?time={152 * DAY + DAY - 1}&k=50&L=0")
    assert q[0]["matured"]["day"] == 151


def _brute_inputs(df: pd.DataFrame) -> list[dict]:
    """Reference for bundle_inputs: the definitions applied row by row (D126)."""
    start = int(df["day"].min()) * DAY
    out = []
    for _, r in df.iterrows():
        t = r["TransactionDT"]
        mine = df[(df["uid"] == r["uid"]) & (df["TransactionDT"] < t)]
        whole = len(mine) > 0 and len(mine) == r["ctx_hist_n_past"]
        on_card = df[(df["card"] == r["card"]) & (df["TransactionDT"] < t)
                     & (df["TransactionDT"] >= t - 7 * DAY)]  # fmt: skip
        dev = r["device"]
        if dev is None:
            known = None
        elif dev in set(mine["device"]):
            known = True
        else:
            known = False if len(mine) == r["ctx_hist_n_past"] else None
        out.append({
            "velocity24h": int((mine["TransactionDT"] >= t - DAY).sum()) if t - DAY >= start else None,
            "usualAmount": float(np.median(mine["amount"])) if whole else None,
            "deviceKnown": known,
            "sharedEntities": len(set(on_card["uid"]) - {r["uid"]}) if t - 7 * DAY >= start else None,
        })  # fmt: skip
    return out


def test_bundle_inputs_are_exact_where_the_bundle_supports_them_and_missing_otherwise():
    mod = _load_module()
    rng = np.random.default_rng(3)
    n = 300
    t = np.sort(151 * DAY + rng.integers(0, 20 * DAY, n)).astype(float)
    t[10] = t[9]  # a same-second pair: not each other's history
    uid = rng.choice([f"u{i}" for i in range(12)], n)
    df = pd.DataFrame({"TransactionDT": t, "day": (t // DAY).astype(int), "uid": uid,
                       "card": rng.choice(["c1", "c2", "c3"], n), "amount": rng.lognormal(4, 1, n),
                       "device": rng.choice(np.array(["d1", "d2", None], dtype=object), n)})  # fmt: skip
    # half the customers have their whole history in the bundle, the others had earlier rows
    first = df.groupby("uid").cumcount().to_numpy()
    df["ctx_hist_n_past"] = np.where(np.isin(uid, [f"u{i}" for i in range(6)]), first, first + 4)
    got = mod.bundle_inputs(df)
    want = _brute_inputs(df)
    for i, w in enumerate(want):
        g = got.iloc[i]
        for k in ("velocity24h", "usualAmount", "sharedEntities"):
            if w[k] is None:
                assert np.isnan(g[k]), (i, k)
            else:
                assert g[k] == pytest.approx(w[k]), (i, k)
        assert g["deviceKnown"] == w["deviceKnown"], i
    # every kind of value occurs, including "not available"
    assert got["velocity24h"].isna().any() and got["velocity24h"].notna().any()
    assert got["usualAmount"].isna().any() and got["usualAmount"].notna().any()
    assert got["sharedEntities"].isna().any() and got["sharedEntities"].notna().any()
    assert {True, False, None} <= set(got["deviceKnown"])


def test_served_inputs_never_invent_a_value(server):
    rows = _get(server, f"/api/stream?time={190 * DAY + 3600}&start=0&end={190 * DAY + 3600}")
    first_day = [r for r in rows if r["row"]["day"] == 151]
    assert first_day and all(r["inputs"]["velocity24h"] is None for r in first_day)
    assert all(r["inputs"]["sharedEntities"] is None for r in first_day)
    assert all(r["inputs"]["maskedC13"] is None for r in rows)  # not in the bundle
    assert all(r["inputs"]["deviceKnown"] is None for r in rows)  # the bundle has no devices
    late = [r for r in rows if r["row"]["day"] == 190]
    assert late and all(isinstance(r["inputs"]["sharedEntities"], int) for r in late)


WHATIF = {"c_fp": 2.0, "c_rev": 1.0, "c_step": 0.1, "step_up_success_rate": 0.9,
          "legit_step_up_friction": 0.0}  # fmt: skip


def _q(params: dict) -> str:
    return "&" + urllib.parse.urlencode(params)


def test_no_block_note_and_whatif_cost_mode(server):
    meta = _get(server, "/api/meta")
    assert meta["whatif_available"] is True
    assert "No Block decisions" in meta["no_block_note"]  # t_block = inf in the bundle
    w = _get(server, "/api/whatif")
    assert w["active"] is False and "not the evaluated system" in w["note"]
    assert w["cost"]["hold_success_rate"] == 1.0 and w["thresholds"] is None
    before = _get(server, f"/api/day-summary?time={AS_OF}")
    assert before["whatIf"] is False
    w = _get(server, "/api/whatif?" + urllib.parse.urlencode(WHATIF))
    assert w["active"] is True and w["thresholds"]["t_monitor"] is not None
    after = _get(server, f"/api/day-summary?time={AS_OF}" + _q(WHATIF))
    assert after["whatIf"] is True and after["transactions"] == before["transactions"]
    rows = _get(server, f"/api/stream?time={AS_OF}&start=0&end={AS_OF}" + _q(WHATIF))
    assert rows and all(t["decision"]["whatIf"] is True for t in rows)
    w = _get(server, "/api/whatif?" + urllib.parse.urlencode({**WHATIF, "hold_success_rate": 0.7}))
    assert w["cost"]["hold_success_rate"] == 0.7  # imperfect Hold (D124)
    for bad in ({**WHATIF, "hold_success_rate": 1.2}, {**WHATIF, "step_up_success_rate": 1.5},
                {**WHATIF, "c_fp": -1}, {"c_fp": 2.0}):  # fmt: skip
        code, _ = _status(server, f"/api/day-summary?time={AS_OF}" + _q(bad))
        assert code == 422, bad


def test_whatif_is_stateless_every_response_says_which_it_is(server):
    """D128: what-if lives in the request, never in the server; flags on every response."""
    plain = _get(server, f"/api/day-summary?time={AS_OF}")
    _get(server, f"/api/day-summary?time={AS_OF}" + _q(WHATIF))  # another client's what-if
    again = _get(server, f"/api/day-summary?time={AS_OF}")
    assert again == plain and again["whatIf"] is False  # it leaked nothing
    for path in AS_OF_ENDPOINTS:
        if "drift" in path:
            continue  # no decisions in drift
        for params, flag in (({}, False), (WHATIF, True)):
            body = _get(server, path.format(t=AS_OF) + (_q(params) if params else ""))
            decisions = []
            if isinstance(body, list):
                for x in body:
                    decisions += (
                        [c["decision"] for c in x["cases"]] if "cases" in x else [x["decision"]]
                    )
            elif body and "decision" in body:
                decisions = [body["decision"]]
            elif body and "whatIf" in body:
                assert body["whatIf"] is flag, path
            assert all(d["whatIf"] is flag for d in decisions), path
    net = _get(server, f"/api/network?time={AS_OF}" + _q(WHATIF))
    assert net["whatIf"] is True  # the unfocused network is picked from what-if holds and blocks
    # the old session endpoints are gone
    assert _send(server, "/api/whatif", {"cost": WHATIF, "role": "analyst"}, "PUT")[0] == 405


def test_whatif_changes_decisions_but_not_scores(server):
    plain = _get(server, f"/api/stream?time={190 * DAY + 3600}&start=0&end={190 * DAY + 3600}")
    what = _get(server, f"/api/stream?time={190 * DAY + 3600}&start=0&end={190 * DAY + 3600}"
                + _q({**WHATIF, "c_fp": 50.0, "hold_success_rate": 0.7}))  # fmt: skip
    assert [t["row"]["TransactionID"] for t in plain] == [t["row"]["TransactionID"] for t in what]
    assert [t["decision"]["p"] for t in plain] == [t["decision"]["p"] for t in what]
    assert [t["decision"]["action"] for t in plain] != [t["decision"]["action"] for t in what]


def test_drift_never_uses_anything_after_the_as_of_time(tmp_path):
    """D130: the reference is the calibration days; drift at as-of is the same whether or not
    the bundle holds later rows."""
    (tmp_path / "full").mkdir()
    full = _bundle(tmp_path / "full")
    cut = tmp_path / "cut"
    cut.mkdir()
    df = pd.read_parquet(full)
    df[df["TransactionDT"] <= AS_OF].to_parquet(cut / full.name, index=False)
    (cut / full.with_suffix(".json").name).write_text(full.with_suffix(".json").read_text("utf-8"))
    cal = full.with_name("demo_final_calibration.parquet")
    (cut / cal.name).write_bytes(cal.read_bytes())
    mod = _load_module()
    with _serve(mod.create_app(full)) as a, _serve(mod.create_app(cut / full.name)) as b:
        da, db = (_get(x, f"/api/drift?time={AS_OF}") for x in (a, b))
    assert da == db and da["scorePsi"] and da["featurePsi"]


def _two_hops(df: pd.DataFrame, kind: str, value: str) -> set[str]:
    """Reference: the mock's rule (two links from the focus; email / address hubs are not
    expanded unless they are the focus)."""
    entities = ("card", "email", "address", "device")
    keep, frontier = {(kind, value)}, [(kind, value)]
    for _ in range(2):
        nxt = []
        for k, v in frontier:
            if k in ("email", "address") and (k, v) != (kind, value):
                continue
            rows = df[df[k] == v]
            nbrs = ([(e, x) for e in entities for x in rows[e].dropna().unique()] if k == "uid"
                    else [("uid", u) for u in rows["uid"].unique()])  # fmt: skip
            for nb in nbrs:
                if nb not in keep:
                    keep.add(nb)
                    nxt.append(nb)
        frontier = nxt
    return {f"{k}:{v}" for k, v in keep}


def test_network_focus_matches_its_column_only_and_expands_two_hops(server, tmp_path):
    df = pd.read_parquet(_bundle(tmp_path))
    df = df[df["TransactionDT"] <= AS_OF]
    for focus in ("uid:u1", "card:c2", "email:gmail.com", "address:325"):
        net = _get(server, f"/api/network?time={AS_OF}&focus={urllib.parse.quote(focus)}")
        kind, value = focus.split(":", 1)
        assert {n["id"] for n in net["nodes"]} == _two_hops(df, kind, value), focus
        ids = {n["id"] for n in net["nodes"]}
        assert all(e["source"] in ids and e["target"] in ids for e in net["edges"])
        assert all(e["time"] <= AS_OF for e in net["edges"]) and net["whatIf"] is False
    # the prefix names the column: "c2" is a card, not a customer; no prefix is refused
    assert _get(server, f"/api/network?time={AS_OF}&focus=uid:c2")["nodes"] == []
    for bad in ("u1", "colour:red", "uid:"):
        assert _status(server, f"/api/network?time={AS_OF}&focus={bad}")[0] == 422, bad
    # a customer reaches other customers through a shared card (two hops) ...
    u1 = {n["id"] for n in _get(server, f"/api/network?time={AS_OF}&focus=uid:u1")["nodes"]}
    assert any(i.startswith("uid:") and i != "uid:u1" for i in u1)


def test_queues_offer_only_the_frozen_budgets(server):
    # the synthetic bundle's routing weights exist for K = 50 only
    assert _status(server, f"/api/queues?time={AS_OF}&k=50")[0] == 200
    for k in (5, 10, 25, 49, 51, 100):
        code, body = _status(server, f"/api/queues?time={AS_OF}&k={k}")
        assert code == 422 and "frozen budgets 50" in body["detail"], k


def test_concurrent_feedback_keeps_labels_and_audit_consistent(tmp_path):
    """D135: each label and its audit entry change together under the state lock, so the audit
    chain of one transaction is consistent however many requests arrive at once. The audit
    timestamp is slowed down so that, without the lock, requests overtake each other between
    reading the old label and writing the audit entry."""
    mod = _load_module()
    jitter = np.random.default_rng(0)

    class SlowClock(mod.datetime):
        @classmethod
        def now(cls, tz=None):
            time.sleep(float(jitter.random()) * 0.01)  # requests overtake each other
            return super().now(tz)

    mod.datetime = SlowClock
    tid, labels = 1001, ["fraud", "legit", "unsure"]
    errors = []
    with _serve(mod.create_app(_bundle(tmp_path))) as base:

        def post(i):
            code, _ = _send(base, f"/api/feedback/{tid}",
                            {"label": labels[i % 3], "role": "analyst"}, "POST")  # fmt: skip
            if code != 200:
                errors.append(code)

        threads = [threading.Thread(target=post, args=(i,)) for i in range(60)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        log = _get(base, "/api/audit")
    assert errors == []
    chain = [e for e in log if f"transaction {tid} " in e["action"]][::-1]
    assert len(chain) == 60 and chain[0]["detail"] == "first label"
    for prev, cur in zip(chain, chain[1:], strict=False):  # each "was" is the label before it
        assert cur["detail"] == "was " + prev["action"].rsplit(" as ", 1)[1]


@pytest.mark.parametrize("hold", [None, 0.7])
def test_day_summary_has_one_loss_definition(server, hold):
    """D136: expected loss = missed fraud = expected fraud value not stopped; stopped + loss = the
    day's expected fraud value, for the evaluated system and a what-if with an imperfect Hold."""
    extra = _q({**WHATIF, "hold_success_rate": hold}) if hold else ""
    s = _get(server, f"/api/day-summary?time={AS_OF}{extra}")
    rows = _get(server, f"/api/stream?time={AS_OF}&start={151 * DAY - 1}&end={AS_OF}{extra}")
    total = sum(t["decision"]["p"] * t["amount"] for t in rows)
    assert s["expectedLoss"] == pytest.approx(s["cost"]["missedFraud"])
    assert s["expectedFraudStopped"] + s["expectedLoss"] == pytest.approx(total)
