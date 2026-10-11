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
dashboard's what-if simulator keeps its illustrative model (labelled "illustrative model, not the
evaluated system").

What-if cost mode (PUT / DELETE /api/whatif, D122): for cost settings chosen in the dashboard, the
decision thresholds are re-chosen on the calibrate_fused rows (days 147-150, from
demo_final_calibration.parquet) with the research pipeline's own choose_thresholds, and applied to
the test period; every decision and day summary served while it is on carries whatIf = true. It is
illustrative, not the evaluated system (the evaluated system is the frozen E12 decisions).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DAY = 86400
# the label delays the dashboard offers (D125); any other L is refused on every endpoint
LABEL_DELAYS = (0, 7, 30, 60)
VIEWS = ("tabular", "behavioral", "temporal", "graph", "anomaly")
ACTIONS = ("allow", "monitor", "step_up", "hold", "block")
POLICIES = ("R1", "R2", "R3", "R4")
NO_BLOCK_NOTE = (
    "No Block decisions, by the cost model (D43): a Block costs an analyst review ($5) plus $10 "
    "for a legitimate customer wrongly blocked, while a Hold costs the same review, also stops "
    "the fraud and releases a legitimate customer at no extra cost. A Block can therefore never "
    "cost less than a Hold, so the cost-optimal Block threshold is 'never' and high-risk cases "
    "are held or stepped up instead."
)
WHATIF_NOTE = "What-if cost mode: illustrative, not the evaluated system"
DEFAULT_BUNDLE = (
    Path(os.environ.get("VAULTIC_DATA_DIR", "data")) / "interim" / "demo_final_bundle.parquet"
)


class ThresholdsIn(BaseModel):
    t_monitor: float
    t_step: float
    t_hold: float
    t_block: float | None = None  # None = never block (inf on the Python side)
    t_dis: float | None = None


class SaveThresholds(BaseModel):
    thresholds: ThresholdsIn
    role: str


class CostIn(BaseModel):
    c_fp: float
    c_rev: float
    c_step: float
    step_up_success_rate: float
    legit_step_up_friction: float
    hold_success_rate: float = 1.0  # share of held fraud stopped (D124); 1.0 = the evaluated system


class WhatIfIn(BaseModel):
    cost: CostIn
    role: str


class FeedbackIn(BaseModel):
    label: str
    role: str


def _none(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


def _int_or_none(x) -> int | None:
    return None if x is None or not np.isfinite(x) else int(x)


def label_delay(L: int = 30) -> int:
    """The L query parameter of every as-of endpoint: one of LABEL_DELAYS, else 422 (D125)."""
    if L not in LABEL_DELAYS:
        raise HTTPException(
            422, f"Label delay must be one of {', '.join(map(str, LABEL_DELAYS))} days, not {L}."
        )
    return L


def matured_day(time: float, L: int, first_day: int) -> int | None:
    """The latest day ALL of whose labels are known at `time` (every transaction of day D has
    its_time + L days <= time, i.e. (D + 1) * DAY - 1 + L * DAY <= time), and always before the
    as-of day itself (D125); None when no such day is in the data."""
    last = min(int((time - L * DAY + 1) // DAY) - 1, int(time // DAY) - 1)
    return last if last >= first_day else None


SHARED_WINDOW_DAYS = 7


def bundle_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """The case inputs the bundle can support, computed from its rows (D126), each ONLY where the
    bundle makes it exact and NaN / None otherwise ("not available", never a made-up value).

    The bundle holds every test-period transaction (from day `first` on) and nothing earlier, so
    for a transaction at time t (earlier = strictly before t; same-second rows are not history):
      velocity24h     the customer's transactions in [t - 24 h, t); exact when t - 24 h >= start
      usualAmount     median of the customer's earlier amounts; exact when the bundle holds the
                      customer's whole history (its earlier rows = ctx_hist_n_past > 0)
      deviceKnown     the device was seen on an earlier transaction of the customer: True when
                      found; False only with the whole history in the bundle; None otherwise
                      (and None when the transaction has no device)
      sharedEntities  other customers on the same card in [t - 7 days, t); exact when
                      t - 7 days >= start. Devices are left out: the bundle's device is the
                      DeviceInfo string ("Windows"), shared by thousands (a hub)
    `df` must be sorted by TransactionDT.
    """
    t = df["TransactionDT"].to_numpy(dtype=float)
    if np.any(np.diff(t) < 0):
        raise ValueError("bundle_inputs needs rows sorted by TransactionDT")
    start = float(int(df["day"].min()) * DAY)
    n = len(df)
    amount = df["amount"].to_numpy(dtype=float)
    hist = df["ctx_hist_n_past"].to_numpy(dtype=float)
    device = df["device"].to_numpy(dtype=object)
    velocity = np.full(n, np.nan)
    usual = np.full(n, np.nan)
    known = np.full(n, None, dtype=object)
    for idx in df.groupby("uid", sort=False).indices.values():  # positions, in time order
        tt = t[idx]
        before = np.searchsorted(tt, tt, side="left")  # earlier rows of this customer
        recent = before - np.searchsorted(tt, tt - DAY, side="left")
        velocity[idx] = np.where(tt - DAY >= start, recent, np.nan)
        for j, b in enumerate(before):
            i = idx[j]
            whole = b > 0 and b == hist[i]
            if whole:
                usual[i] = float(np.median(amount[idx[:b]]))
            if device[i] is not None and not (isinstance(device[i], float) and np.isnan(device[i])):
                seen = device[i] in set(device[idx[:b]])
                known[i] = True if seen else (False if b == hist[i] else None)
    shared = np.full(n, np.nan)
    window = SHARED_WINDOW_DAYS * DAY
    uid = df["uid"].to_numpy(dtype=object)
    for idx in df.groupby("card", sort=False).indices.values():
        counts: dict = {}
        lo = hi = 0
        for j, i in enumerate(idx):
            while hi < j and t[idx[hi]] < t[i]:  # earlier rows enter the window
                counts[uid[idx[hi]]] = counts.get(uid[idx[hi]], 0) + 1
                hi += 1
            while lo < hi and t[idx[lo]] < t[i] - window:  # rows older than 7 days leave it
                u = uid[idx[lo]]
                counts[u] -= 1
                if counts[u] == 0:
                    del counts[u]
                lo += 1
            if t[i] - window >= start:
                shared[i] = len(counts) - (1 if uid[i] in counts else 0)
    return pd.DataFrame({"velocity24h": velocity, "usualAmount": usual, "deviceKnown": known,
                         "sharedEntities": shared}, index=df.index)  # fmt: skip


def create_app(bundle_path: Path | None = None) -> FastAPI:
    bundle_path = Path(bundle_path or os.environ.get("VAULTIC_DEMO_BUNDLE") or DEFAULT_BUNDLE)
    meta = json.loads(bundle_path.with_suffix(".json").read_text("utf-8"))
    df = pd.read_parquet(bundle_path).sort_values("TransactionDT").reset_index(drop=True)
    dt = df["TransactionDT"].to_numpy()
    by_id = {int(t): i for i, t in enumerate(df["TransactionID"].to_numpy())}
    inputs = bundle_inputs(df)  # exact where the bundle supports it, else not available (D126)
    # t_dis = inf (disagreement rule off) on the Python side is null in the contract
    thresholds = {k: _none(float(v)) if v is not None else None
                  for k, v in meta["thresholds"].items()}  # fmt: skip
    state = {"thresholds": thresholds, "feedback": {}, "audit": [], "whatif": None}
    base_actions = df["action"].to_numpy().astype(object)
    actions = base_actions.copy()  # the served actions: the frozen E12 ones unless what-if is on
    cal_path = bundle_path.with_name("demo_final_calibration.parquet")
    cal = pd.read_parquet(cal_path) if cal_path.exists() else None
    cost = {
        **meta["cost_model"],
        "hold_success_rate": meta["cost_model"].get("hold_success_rate", 1.0),
    }

    app = FastAPI(title="Vaultic demo backend", version="0.1")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])  # fmt: skip

    def matured(t: float, time: float, L: int) -> bool:
        return t + L * DAY <= time

    def transaction(i: int, time: float, L: int) -> dict:
        r, x = df.iloc[i], inputs.iloc[i]
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
            "decision": {"action": str(actions[i]), "p": p, "expected_loss": p * amount,
                         "conformal_set": str(r["conformal_set"]),
                         "disagreement": float(r["disagreement"]),
                         "whatIf": state["whatif"] is not None},
            "reasons": json.loads(r["reasons"]),
            "inputs": {"amount": amount, "usualAmount": _none(float(x["usualAmount"])),
                       "deviceKnown": x["deviceKnown"],
                       "velocity24h": _int_or_none(x["velocity24h"]),
                       "sharedEntities": _int_or_none(x["sharedEntities"]),
                       "historyCount": int(r["ctx_hist_n_past"]),
                       "maskedC13": None,  # C13 is not in the bundle
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
        out = {k: meta[k] for k in ("source", "run", "generated", "rows", "label_delay_days",
                                    "reasons_note")}  # fmt: skip
        out["no_block_note"] = (
            NO_BLOCK_NOTE if meta["thresholds"].get("t_block") in (None, float("inf")) else None
        )
        out["whatif_available"] = cal is not None
        return out

    @app.get("/api/range")
    def get_range():
        return {"first": float(dt.min()), "last": float(dt.max())}

    @app.get("/api/day-summary")
    def day_summary(time: float, L: int = Depends(label_delay)):
        idx = day_rows(time)
        sub = df.iloc[idx]
        act = actions[idx]
        by_action = {a: int((act == a).sum()) for a in ACTIONS}
        p, amt = sub["p"].to_numpy(), sub["amount"].to_numpy()
        cost = state["whatif"]["cost"] if state["whatif"] else meta["cost_model"]
        hold_rate = cost.get("hold_success_rate", 1.0)
        held = act == "hold"
        el = p * amt
        low = np.isin(act, ["allow", "monitor"])
        step = act == "step_up"
        rate = cost["step_up_success_rate"]
        stopped = float(np.sum(np.where(step, rate * el, np.where(held, hold_rate * el, el))[~low]))
        costs = {"missedFraud": float(el[low].sum() + ((1 - rate) * el[step]).sum()
                                      + ((1 - hold_rate) * el[held]).sum()),
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
                "viewAvailability": {v: float(sub[f"p_{v}"].notna().sum() / n) for v in VIEWS},
                "whatIf": state["whatif"] is not None}  # fmt: skip

    @app.get("/api/stream")
    def stream(time: float, start: float, end: float, L: int = Depends(label_delay)):
        stop = min(end, time)
        lo, hi = int(np.searchsorted(dt, start, side="right")), up_to(stop)
        return [transaction(i, stop, L) for i in range(lo, hi)]

    @app.get("/api/transactions/{tid}")
    def get_transaction(tid: int, time: float, L: int = Depends(label_delay)):
        i = by_id.get(tid)
        if i is None or dt[i] > time:
            return None
        return transaction(i, time, L)

    @app.get("/api/alerts")
    def alerts(time: float, L: int = Depends(label_delay)):
        idx = [i for i in day_rows(time) if actions[i] in ("step_up", "hold", "block")]
        idx.sort(key=lambda i: -df.at[i, "p"])
        return [transaction(i, time, L) for i in idx]

    @app.get("/api/customers/{uid}/history")
    def history(uid: str, time: float, L: int = Depends(label_delay)):
        n = up_to(time)
        idx = np.flatnonzero(df["uid"].to_numpy()[:n] == uid)
        return [transaction(int(i), time, L) for i in idx]

    @app.get("/api/network")
    def network(time: float, L: int = Depends(label_delay), focus: str | None = None):
        n = up_to(time)
        sub = df.iloc[:n]
        if focus:
            hit = np.zeros(n, bool)
            for col in ("uid", "card", "email", "address", "device"):
                hit |= sub[col].astype(str).to_numpy() == focus
            sub = sub[hit]
        else:
            rows = day_rows(time)
            sub = sub.iloc[rows][np.isin(actions[rows], ["hold", "block"])].nlargest(40, "p")
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
    def queues(time: float, k: int = 50, L: int = Depends(label_delay)):
        today = df.iloc[day_rows(time)]
        known_day = matured_day(time, L, int(df["day"].min()))
        past = df[df["day"] == known_day] if known_day is not None else df.iloc[0:0]
        # every label of the matured day is known at as-of; checked per row all the same
        if not all(matured(t, time, L) for t in past["TransactionDT"]):
            raise HTTPException(500, "Internal error: an outcome label is not known yet.")
        out = []
        for policy in POLICIES:
            top = today.iloc[np.argsort(-priority(policy, today, k), kind="stable")[:k]]
            rev = past.iloc[np.argsort(-priority(policy, past, k), kind="stable")[:k]]
            caught = rev[rev["label"] == 1]
            out.append({"policy": policy, "day": int(time // DAY),
                        "cases": [transaction(by_id[int(t)], time, L) for t in top["TransactionID"]],
                        "matured": {"day": known_day,
                                    "fraudValueCaught": float(caught["amount"].sum()),
                                    "fraudCaught": int(len(caught)), "reviews": int(len(rev))}})  # fmt: skip
        return out

    @app.get("/api/drift")
    def drift(time: float, L: int = Depends(label_delay)):
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
        block = float("inf") if t.t_block is None else t.t_block
        if not (t.t_monitor <= t.t_step <= t.t_hold <= block):
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

    def whatif_status():
        w = state["whatif"]
        return {"active": w is not None, "available": cal is not None,
                "cost": w["cost"] if w else cost,
                "thresholds": w["thresholds"] if w else None, "note": WHATIF_NOTE}  # fmt: skip

    @app.get("/api/whatif")
    def get_whatif():
        return whatif_status()

    @app.put("/api/whatif")
    def set_whatif(body: WhatIfIn):
        nonlocal actions
        if cal is None:
            raise HTTPException(501, "The what-if cost mode needs demo_final_calibration.parquet.")
        c = body.cost
        if not (0 <= c.step_up_success_rate <= 1) or not (0 <= c.hold_success_rate <= 1) or min(c.c_fp, c.c_rev, c.c_step,
                                                         c.legit_step_up_friction) < 0:  # fmt: skip
            raise HTTPException(
                422, "Costs must be at least 0 and the success rates between 0 and 1."
            )
        src = Path(__file__).resolve().parents[2] / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from vaultic.trust.decision import ACTIONS as CODES
        from vaultic.trust.decision import CostModel, choose_thresholds, decide

        cm = CostModel(c.c_fp, c.c_rev, c.c_step, c.step_up_success_rate, c.legit_step_up_friction,
                       hold_success_rate=c.hold_success_rate)  # fmt: skip
        th, _ = choose_thresholds(cal["p"].to_numpy(), cal["amount"].to_numpy(),
                                  cal["label"].to_numpy(), cal["conformal_set"].to_numpy(object),
                                  cal["disagreement"].to_numpy(), cost=cm)  # fmt: skip
        codes = decide(df["p"].to_numpy(), df["amount"].to_numpy(), th,
                       df["conformal_set"].to_numpy(object), df["disagreement"].to_numpy())  # fmt: skip
        actions = np.asarray([CODES[k] for k in codes], dtype=object)
        state["whatif"] = {"cost": c.model_dump(),
                           "thresholds": {k: _none(float(v)) for k, v in th.__dict__.items()}}  # fmt: skip
        audit(
            body.role, "Turned on the what-if cost mode (illustrative)", json.dumps(c.model_dump())
        )
        return whatif_status()

    @app.delete("/api/whatif")
    def clear_whatif(role: str = "analyst"):
        nonlocal actions
        actions = base_actions.copy()
        if state["whatif"] is not None:
            audit(role, "Turned off the what-if cost mode", "back to the evaluated system")
        state["whatif"] = None
        return whatif_status()

    @app.get("/api/audit")
    def audit_log():
        return state["audit"]

    return app


app = create_app() if (os.environ.get("VAULTIC_DEMO_BUNDLE") or DEFAULT_BUNDLE.exists()) else None
