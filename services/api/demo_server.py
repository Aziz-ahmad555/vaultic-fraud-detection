"""Demo backend for the Evidence Room dashboard: the EXP-200-final test period, served read-only.

Run:  uvicorn services.api.demo_server:app --port 8000      (from the repository root)
      VAULTIC_DEMO_BUNDLE=<path to demo_final_bundle.parquet> to override the default location

Serves the demo bundle written by `python -m vaultic.reports.demo_export` (phase3-run): per test
transaction the calibrated view scores, MVAF gate weights, the E12 decision (action, conformal
set, disagreement) and view-level reasons, reproduced from the final run and checked against it.
Every response follows the dashboard's typed contract (dashboard/src/api/types.ts) and the same
rules as the mock: nothing after the as-of time is returned, and a label only once
its_time + L days <= as-of (label-delay curtain). All data is labelled
"IEEE-CIS test period, final run": anonymized e-commerce transactions released by Vesta for the
IEEE-CIS competition, not bank data. Threshold changes, feedback and the audit log are kept in
memory for the demo session only; nothing is written back.

POST /api/score is not available: the final-run models score recorded transactions only, so the
dashboard's what-if simulator keeps its illustrative model (the HTTP adapter says so).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DAY = 86400
VIEWS = ("tabular", "behavioral", "temporal", "graph", "anomaly")
ACTIONS = ("allow", "monitor", "step_up", "hold", "block")
POLICIES = ("R1", "R2", "R3", "R4")
DEFAULT_BUNDLE = (
    Path(os.environ.get("VAULTIC_DATA_DIR", "data")) / "interim" / "demo_final_bundle.parquet"
)


class ThresholdsIn(BaseModel):
    t_monitor: float
    t_step: float
    t_hold: float
    t_block: float
    t_dis: float | None = None


class SaveThresholds(BaseModel):
    thresholds: ThresholdsIn
    role: str


class FeedbackIn(BaseModel):
    label: str
    role: str


def _none(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


def create_app(bundle_path: Path | None = None) -> FastAPI:
    bundle_path = Path(bundle_path or os.environ.get("VAULTIC_DEMO_BUNDLE") or DEFAULT_BUNDLE)
    meta = json.loads(bundle_path.with_suffix(".json").read_text("utf-8"))
    df = pd.read_parquet(bundle_path).sort_values("TransactionDT").reset_index(drop=True)
    dt = df["TransactionDT"].to_numpy()
    by_id = {int(t): i for i, t in enumerate(df["TransactionID"].to_numpy())}
    state = {"thresholds": dict(meta["thresholds"]), "feedback": {}, "audit": []}
    cost = meta["cost_model"]

    app = FastAPI(title="Vaultic demo backend", version="0.1")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])  # fmt: skip

    def matured(t: float, time: float, L: int) -> bool:
        return t + L * DAY <= time

    def transaction(i: int, time: float, L: int) -> dict:
        r = df.iloc[i]
        views = {v: _none(float(r[f"p_{v}"])) for v in VIEWS}
        row = {"TransactionID": int(r["TransactionID"]), "TransactionDT": float(r["TransactionDT"]),
               "day": int(r["day"]), "fold": "fixed", "role": "test", "split": "test",
               "label": int(r["label"]) if matured(r["TransactionDT"], time, L) else None,
               **{f"p_{v}": views[v] for v in VIEWS},
               **{f"m_{v}": int(views[v] is not None) for v in VIEWS},
               **{f"c_{v}": float(np.nan_to_num(r[f"c_{v}"])) for v in VIEWS},
               "disagreement": float(r["disagreement"]),
               "ctx_log_amount": float(r["ctx_log_amount"]),
               "ctx_hist_n_past": float(r["ctx_hist_n_past"]),
               "ctx_has_identity": int(r["ctx_has_identity"]),
               "ctx_product": float(r["ctx_product"]), "ctx_hour": int(r["ctx_hour"])}  # fmt: skip
        amount, p = float(r["amount"]), float(r["p"])
        return {
            "row": row, "amount": amount, "product": str(r["product"]),
            "entities": {"uid": str(r["uid"]), "card": str(r["card"]), "email": _none(r["email"]),
                         "address": _none(r["address"]), "device": _none(r["device"])},
            "weights": {v: float(np.nan_to_num(r[f"w_{v}"])) for v in VIEWS},
            "decision": {"action": str(r["action"]), "p": p, "expected_loss": p * amount,
                         "conformal_set": str(r["conformal_set"]),
                         "disagreement": float(r["disagreement"])},
            "reasons": json.loads(r["reasons"]),
            "inputs": {"amount": amount, "usualAmount": None, "deviceKnown": _none(r["device"]) is not None,
                       "velocity24h": 0, "sharedEntities": 0,
                       "historyCount": int(r["ctx_hist_n_past"]), "maskedC13": 0,
                       "available": {v: views[v] is not None for v in VIEWS}},
            "source": meta["source"],
        }  # fmt: skip

    def up_to(time: float) -> int:
        return int(np.searchsorted(dt, time, side="right"))

    def day_rows(time: float) -> np.ndarray:
        day = int(time // DAY)
        n = up_to(time)
        return np.flatnonzero(df["day"].to_numpy()[:n] == day)

    @app.get("/api/meta")
    def get_meta():
        return {k: meta[k] for k in ("source", "run", "generated", "rows", "label_delay_days",
                                     "reasons_note")}  # fmt: skip

    @app.get("/api/range")
    def get_range():
        return {"first": float(dt.min()), "last": float(dt.max())}

    @app.get("/api/day-summary")
    def day_summary(time: float, L: int = 30):
        idx = day_rows(time)
        sub = df.iloc[idx]
        by_action = {a: int((sub["action"] == a).sum()) for a in ACTIONS}
        p, amt, act = sub["p"].to_numpy(), sub["amount"].to_numpy(), sub["action"].to_numpy()
        el = p * amt
        low = np.isin(act, ["allow", "monitor"])
        step = act == "step_up"
        rate = cost["step_up_success_rate"]
        stopped = float(np.sum(np.where(step, rate * el, el)[~low]))
        costs = {"missedFraud": float(el[low].sum() + ((1 - rate) * el[step]).sum()),
                 "falsePositives": float(((1 - p) * cost["c_fp"])[act == "block"].sum()),
                 "reviews": float(cost["c_rev"] * np.isin(act, ["hold", "block"]).sum()),
                 "stepUps": float((cost["c_step"] + (1 - p) * cost["legit_step_up_friction"])[step].sum())}  # fmt: skip
        hours = []
        for h in range(24):
            m = sub["ctx_hour"].to_numpy() == h
            hours.append(
                {
                    "hour": h,
                    "count": int(m.sum()),
                    "meanRisk": float(p[m].mean()) if m.any() else 0.0,
                }
            )
        n = max(len(sub), 1)
        return {"day": int(time // DAY), "transactions": int(len(sub)),
                "alerts": by_action["step_up"] + by_action["hold"] + by_action["block"],
                "queue": by_action["step_up"] + by_action["hold"], "byAction": by_action,
                "expectedFraudStopped": stopped, "expectedLoss": float(el[low].sum()),
                "cost": costs, "riskByHour": hours,
                "viewAvailability": {v: float(sub[f"p_{v}"].notna().sum() / n) for v in VIEWS}}  # fmt: skip

    @app.get("/api/stream")
    def stream(time: float, start: float, end: float, L: int = 30):
        stop = min(end, time)
        lo, hi = int(np.searchsorted(dt, start, side="right")), up_to(stop)
        return [transaction(i, stop, L) for i in range(lo, hi)]

    @app.get("/api/transactions/{tid}")
    def get_transaction(tid: int, time: float, L: int = 30):
        i = by_id.get(tid)
        if i is None or dt[i] > time:
            return None
        return transaction(i, time, L)

    @app.get("/api/alerts")
    def alerts(time: float, L: int = 30):
        idx = [i for i in day_rows(time) if df.at[i, "action"] in ("step_up", "hold", "block")]
        idx.sort(key=lambda i: -df.at[i, "p"])
        return [transaction(i, time, L) for i in idx]

    @app.get("/api/customers/{uid}/history")
    def history(uid: str, time: float, L: int = 30):
        n = up_to(time)
        idx = np.flatnonzero(df["uid"].to_numpy()[:n] == uid)
        return [transaction(int(i), time, L) for i in idx]

    @app.get("/api/network")
    def network(time: float, L: int = 30, focus: str | None = None):
        n = up_to(time)
        sub = df.iloc[:n]
        if focus:
            hit = np.zeros(n, bool)
            for col in ("uid", "card", "email", "address", "device"):
                hit |= sub[col].astype(str).to_numpy() == focus
            sub = sub[hit]
        else:
            sub = sub.iloc[day_rows(time)]
            sub = sub[sub["action"].isin(["hold", "block"])].nlargest(40, "p")
        nodes, edges = {}, {}
        for _, r in sub.iterrows():
            known = int(r["label"] == 1 and matured(r["TransactionDT"], time, L))
            ids = []
            for kind in ("uid", "card", "email", "address", "device"):
                val = _none(r[kind])
                if val is None:
                    continue
                key = f"{kind}:{val}"
                node = nodes.setdefault(key, {"id": key, "type": kind, "label": str(val),
                                              "knownFraud": 0, "transactions": 0, "ring": None})  # fmt: skip
                node["transactions"] += 1
                node["knownFraud"] += known
                ids.append(key)
            for other in ids[1:]:
                eid = f"{ids[0]}|{other}"
                edges.setdefault(eid, {"id": eid, "source": ids[0], "target": other,
                                       "time": float(r["TransactionDT"])})  # fmt: skip
        return {"nodes": list(nodes.values()), "edges": list(edges.values()), "rings": []}

    def lambdas(k: int) -> dict:
        keys = sorted(int(x) for x in meta["routing_lambdas"])
        best = min(keys, key=lambda x: abs(x - k))
        return meta["routing_lambdas"][str(best)]

    def priority(policy: str, sub: pd.DataFrame, k: int) -> np.ndarray:
        p, amt = sub["p"].to_numpy(), sub["amount"].to_numpy()
        u = (sub["conformal_set"] == "uncertain").to_numpy(float)
        d = sub["disagreement"].to_numpy()
        lam = lambdas(k)
        if policy == "R1":
            return p
        el = p * amt
        if policy == "R2":
            return el
        lu, ld = lam["R3"] if policy == "R3" else lam["R4"]
        return el + lu * u * amt + (ld * d * amt if policy == "R4" else 0)

    @app.get("/api/queues")
    def queues(time: float, k: int = 50, L: int = 30):
        today = df.iloc[day_rows(time)]
        last_known = int((time - L * DAY + 1) // DAY) - 1
        first_day = int(df["day"].min())
        matured_day = last_known if last_known >= first_day else None
        past = df[df["day"] == matured_day] if matured_day is not None else df.iloc[0:0]
        out = []
        for policy in POLICIES:
            top = today.iloc[np.argsort(-priority(policy, today, k), kind="stable")[:k]]
            rev = past.iloc[np.argsort(-priority(policy, past, k), kind="stable")[:k]]
            caught = rev[rev["label"] == 1]
            out.append({"policy": policy, "day": int(time // DAY),
                        "cases": [transaction(by_id[int(t)], time, L) for t in top["TransactionID"]],
                        "matured": {"day": matured_day,
                                    "fraudValueCaught": float(caught["amount"].sum()),
                                    "fraudCaught": int(len(caught)), "reviews": int(len(rev))}})  # fmt: skip
        return out

    @app.get("/api/drift")
    def drift(time: float, L: int = 30):
        sub = df.iloc[: up_to(time)]
        first = int(df["day"].min())
        ref = df[df["day"] < first + 7]

        def psi(a, b):
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 10 or len(b) < 10:
                return 0.0
            edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
            edges[0], edges[-1] = -np.inf, np.inf
            pa = np.histogram(a, edges)[0] / len(a) + 1e-4
            pb = np.histogram(b, edges)[0] / len(b) + 1e-4
            return float(np.sum((pb - pa) * np.log(pb / pa)))

        days = sorted(sub["day"].unique())
        score_psi = [{"day": int(dd), "value": psi(ref["p_raw"].to_numpy(),
                                                   sub.loc[sub["day"] == dd, "p_raw"].to_numpy())}
                     for dd in days]  # fmt: skip
        dis = [{"day": int(dd), "value": float(sub.loc[sub["day"] == dd, "disagreement"].mean())}
               for dd in days]  # fmt: skip
        today = sub[sub["day"] == int(time // DAY)]
        feats = [{"feature": f"view score: {v}", "psi": psi(ref[f"p_{v}"].to_numpy(),
                                                            today[f"p_{v}"].to_numpy())}
                 for v in VIEWS]  # fmt: skip
        # no drift detectors, injected drifts or retraining ran on the final run (Phase 10)
        return {
            "featurePsi": feats,
            "scorePsi": score_psi,
            "disagreement": dis,
            "alarms": [],
            "injected": [],
            "retrains": [],
        }

    @app.get("/api/lab")
    def lab():
        return {"runs": meta["lab"], "ablation": [], "sensitivity": meta["sensitivity"]}

    @app.post("/api/score")
    def score():
        raise HTTPException(501, "Scoring new inputs is not available: the final-run models are "
                                 "served for recorded transactions only.")  # fmt: skip

    @app.get("/api/thresholds")
    def get_thresholds():
        return {"thresholds": state["thresholds"], "cost": cost}

    def audit(role: str, action: str, detail: str):
        state["audit"].insert(0, {"time": datetime.now(UTC).isoformat(timespec="seconds"),
                                  "actor": role, "action": action, "detail": detail})  # fmt: skip

    @app.put("/api/thresholds")
    def save_thresholds(body: SaveThresholds):
        t = body.thresholds
        if body.role != "admin":
            raise HTTPException(
                403, "Only admins can change thresholds. Switch the role to admin in Settings."
            )
        if not (t.t_monitor <= t.t_step <= t.t_hold <= t.t_block):
            raise HTTPException(
                422, "Thresholds must rise from Monitor to Block. Fix the order and save again."
            )
        before = json.dumps(state["thresholds"])
        state["thresholds"] = t.model_dump()
        audit(body.role, "Changed decision thresholds (demo session only)",
              f"{before} to {json.dumps(state['thresholds'])}")  # fmt: skip
        return {"ok": True}

    @app.post("/api/feedback/{tid}")
    def feedback(tid: int, body: FeedbackIn):
        before = state["feedback"].get(tid)
        state["feedback"][tid] = body.label
        audit(body.role, f"Marked transaction {tid} as {body.label}",
              f"was {before}" if before else "first label")  # fmt: skip
        return {"ok": True}

    @app.get("/api/audit")
    def audit_log():
        return state["audit"]

    return app


app = create_app() if (os.environ.get("VAULTIC_DEMO_BUNDLE") or DEFAULT_BUNDLE.exists()) else None
