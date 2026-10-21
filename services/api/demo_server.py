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
IEEE-CIS competition, not bank data. The decision thresholds are the frozen E12 ones and are
read-only (D129); feedback labels are kept in memory for the demo session only and nothing is
written back to the data. Every analyst action is recorded in the persistent audit log (D208,
D209; vaultic.audit: SQLite on drive E, append-only, hash-chained with a truncation anchor:
tamper-evident, not tamper-proof). GET /api/audit reads it, /api/audit/verify checks the chain,
/api/audit/export.csv exports it (the export is recorded too). Labels are fail-closed (no audit
entry, no label); everything else is fail-open (audit = "failed" in the response).

POST /api/score is not available: the final-run models score recorded transactions only, so the
dashboard's what-if simulator keeps its illustrative model (labelled "illustrative model, not the
evaluated system"). With VAULTIC_TESTLAB=1 the Test Lab's endpoints are mounted as well
(services/api/testlab.py): the frozen stage-2 system scoring one transaction live (D182).

What-if cost mode (D122, stateless since D128): a request that carries the what-if cost parameters
(c_fp, c_rev, c_step, step_up_success_rate, legit_step_up_friction and optionally
hold_success_rate, as query parameters) is answered with decisions under thresholds re-chosen for
those costs on the calibrate_fused rows (days 147-150, demo_final_calibration.parquet) with the
research pipeline's own choose_thresholds; every decision, day summary and network in such a
response carries whatIf = true. A request without them gets the frozen E12 decisions. Nothing is
stored between requests: two dashboards can never see each other's what-if settings. It is
illustrative, not the evaluated system.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
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
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:  # the research package (vaultic) for the audit log and the Test Lab
    sys.path.insert(0, str(SRC))
REPO = SRC.parent
EXP310_EVENTS = REPO / "research" / "tables" / "feedback" / "exp310_events.json"  # D220
LADDER_DIR = REPO / "research" / "tables" / "stage2"  # D220: the stage-2 ablation ladder
RUNG_NAMES = {"rung 1": "Rung 1: tabular (F4)", "rung 2": "Rung 2: + behavioral (E3)",
              "rung 3": "Rung 3: + temporal (E4)", "rung 4": "Rung 4: + graph (E5)",
              "rung 5": "Rung 5: + anomaly, all five views (E8)"}  # fmt: skip
STEP_OF = {"rung 2": "E3 (+behavioral)", "rung 3": "E4 (+temporal)", "rung 4": "E5 (+graph)",
           "rung 5": "E8 (+anomaly)"}  # fmt: skip


def ladder_ablation(ladder_dir: Path = LADDER_DIR) -> tuple[list[dict], list[dict]]:
    """Model-lab ablation rows from the committed stage-2 ladder (D121, D220): the five rungs as
    runs (no curves stored) and one row per rung with its step vs the previous rung (PR-AUC,
    paired bootstrap, Holm across the four steps). Empty when the files are not there."""
    metrics_f, steps_f = ladder_dir / "ladder_metrics.json", ladder_dir / "ladder_steps.json"
    if not (metrics_f.exists() and steps_f.exists()):
        return [], []
    metrics = {m["method"]: m for m in json.loads(metrics_f.read_text("utf-8"))}
    steps = {
        s_["step"]: s_ for s_ in json.loads(steps_f.read_text("utf-8")) if s_["metric"] == "pr_auc"
    }
    runs, ablation = [], []
    for key, name in RUNG_NAMES.items():
        m = metrics.get(key)
        if m is None:
            continue
        run_id = "ladder-" + key.replace(" ", "-")
        runs.append({"id": run_id, "name": name, "kind": "ablation", "prAuc": m["pr_auc"],
                     "ciLow": m["pr_auc_ci_low"], "ciHigh": m["pr_auc_ci_high"],
                     "recallAt1": m["recall_at_1pct_fpr"], "cost": m["cost"], "pr": [],
                     "calibration": []})  # fmt: skip
        st = steps.get(STEP_OF.get(key, ""))
        row = {"step": name, "runId": run_id,
               "source": "research/tables/stage2/ladder_metrics.json"}  # fmt: skip
        if st is not None:
            row.update(stepDiff=st["diff"], stepCiLow=st["ci_low"], stepCiHigh=st["ci_high"],
                       stepPHolm=st["p_holm"], stepSupported=bool(st["h1_supported"]),
                       stepSource="research/tables/stage2/ladder_steps.json")  # fmt: skip
        ablation.append(row)
    return runs, ablation


def exp310_drift(events: dict | None, delay: int, as_of_day: int, seed: int = 0) -> dict:
    """EXP-310 alarms and retraining attempts for the Drift monitor (D220): one seed at the
    chosen label delay, P1 (weekly schedule) and P3 (ADWIN alarm), never after the as-of day."""
    if not events or str(delay) not in events.get("events", {}):
        return {"alarms": [], "retrains": [], "simulation": None}
    rows = [
        e for e in events["events"][str(delay)] if e.get("seed") == seed and e["day"] <= as_of_day
    ]
    alarms = [{"day": e["day"], "detector": e.get("detector", "ADWIN"), "policy": e["policy"],
               "seed": seed} for e in rows if e.get("kind") == "alarm" and e["policy"] == "P3"]  # fmt: skip
    retrains = []
    for e in rows:
        if e.get("kind") != "attempt" or e["policy"] not in ("P1", "P3"):
            continue
        retrains.append({"day": e["day"], "trigger": "alarm" if e["policy"] == "P3" else "schedule",
                         "policy": e["policy"], "seed": seed, "accepted": bool(e.get("accepted")),
                         "reason": e.get("reason"), "gain": e.get("gain"),
                         "gainCiLow": e.get("ci_low"), "gainCiHigh": e.get("ci_high"),
                         "costChampion": e.get("cost_champion"),
                         "costChallenger": e.get("cost_challenger"),
                         "prChampion": None, "prChallenger": None})  # fmt: skip
    return {"alarms": alarms, "retrains": retrains,
            "simulation": {"label": events["label"], "sourceRun": events["source_run_id"],
                           "sourceFile": "research/tables/feedback/exp310_events.json",
                           "seed": seed, "delay": delay,
                           "policies": "P1: retrain every week (schedule); P3: retrain after an ADWIN alarm"}}  # fmt: skip


DEFAULT_BUNDLE = (
    Path(os.environ.get("VAULTIC_DATA_DIR", "data")) / "interim" / "demo_final_bundle.parquet"
)


class CostIn(BaseModel):
    c_fp: float
    c_rev: float
    c_step: float
    step_up_success_rate: float
    legit_step_up_friction: float
    hold_success_rate: float = 1.0  # share of held fraud stopped (D124); 1.0 = the evaluated system


class FeedbackIn(BaseModel):
    label: str
    role: str | None = None  # older clients; the X-Vaultic-Role header wins (D208)


class AuditEventIn(BaseModel):
    action: str  # one of vaultic.audit.http.CLIENT_EVENTS
    detail: str = ""


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


LabelDelay = Annotated[int, Depends(label_delay)]


def matured_day(time: float, L: int, first_day: int) -> int | None:
    """The latest day ALL of whose labels are known at `time` (every transaction of day D has
    its_time + L days <= time, i.e. (D + 1) * DAY - 1 + L * DAY <= time), and always before the
    as-of day itself (D125); None when no such day is in the data."""
    last = min(int((time - L * DAY + 1) // DAY) - 1, int(time // DAY) - 1)
    return last if last >= first_day else None


def whatif_params(
    c_fp: float | None = None,
    c_rev: float | None = None,
    c_step: float | None = None,
    step_up_success_rate: float | None = None,
    legit_step_up_friction: float | None = None,
    hold_success_rate: float | None = None,
) -> dict | None:
    """The what-if cost parameters a request carries (D128), or None (the evaluated system).
    All five costs are needed; hold_success_rate is optional (1.0 = the evaluated system)."""
    given = {"c_fp": c_fp, "c_rev": c_rev, "c_step": c_step,
             "step_up_success_rate": step_up_success_rate,
             "legit_step_up_friction": legit_step_up_friction}  # fmt: skip
    if all(v is None for v in given.values()) and hold_success_rate is None:
        return None
    if any(v is None for v in given.values()):
        missing = ", ".join(k for k, v in given.items() if v is None)
        raise HTTPException(422, f"What-if cost parameters are incomplete: {missing} missing.")
    c = {**given, "hold_success_rate": 1.0 if hold_success_rate is None else hold_success_rate}
    rates = (c["step_up_success_rate"], c["hold_success_rate"])
    money = (c["c_fp"], c["c_rev"], c["c_step"], c["legit_step_up_friction"])
    ok = all(np.isfinite(list(c.values()))) and all(0 <= r <= 1 for r in rates)
    if not ok or min(money) < 0:
        raise HTTPException(422, "Costs must be at least 0 and the success rates between 0 and 1.")
    return c


WhatIfDep = Annotated[dict | None, Depends(whatif_params)]


SHARED_WINDOW_DAYS = 7
ENTITY_KINDS = ("card", "email", "address", "device")
HUB_KINDS = ("email", "address")  # one domain / region links most customers: never expanded
MAX_NODES = (
    160  # a focused network stops growing here (the mock stops after the hop that passes it)
)


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


RUNS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "runs"


def check_bundle_marker(bundle_path: Path, runs_dir: Path = RUNS_DIR) -> dict:
    """D154: the bundle holds test-period rows, so it is served only if the guarded export
    (vaultic.reports.demo_export, which runs the post-hoc guard, D144) marked it: the sidecar
    <bundle>.TEST_DATA.json names the final run it comes from. When the run folders are present,
    that run must also be a completed final run. Returns the marker."""
    side = bundle_path.with_name(bundle_path.name + ".TEST_DATA.json")
    if not side.exists():
        raise RuntimeError(f"{bundle_path} has no test-data marker ({side.name}); write the bundle "
                           "with the guarded export (python -m vaultic.reports.demo_export, D144)")  # fmt: skip
    marker = json.loads(side.read_text("utf-8"))
    if marker.get("contains_test_period") is not True or not marker.get("final_run"):
        raise RuntimeError(f"{side.name} is not a test-data marker written by the guarded export")
    if runs_dir.exists():
        metrics = (
            runs_dir
            / marker.get("experiment", "EXP-200-final")
            / marker["final_run"]
            / "metrics.json"
        )
        meta = json.loads(metrics.read_text("utf-8")) if metrics.exists() else {}
        if meta.get("mode") != "final" or meta.get("status") != "completed":
            raise RuntimeError(f"the bundle comes from final run {marker['final_run']}, which is not a "
                               "completed final run in experiments/runs (D154)")  # fmt: skip
    return marker


def create_app(
    bundle_path: Path | None = None, runs_dir: Path = RUNS_DIR, audit_store=None,
    events_path: Path = EXP310_EVENTS, ladder_dir: Path = LADDER_DIR,
) -> FastAPI:  # fmt: skip
    bundle_path = Path(bundle_path or os.environ.get("VAULTIC_DEMO_BUNDLE") or DEFAULT_BUNDLE)
    meta = json.loads(bundle_path.with_suffix(".json").read_text("utf-8"))
    check_bundle_marker(bundle_path, runs_dir)  # D154: before any test-period row is read
    df = pd.read_parquet(bundle_path).sort_values("TransactionDT").reset_index(drop=True)
    dt = df["TransactionDT"].to_numpy()
    by_id = {int(t): i for i, t in enumerate(df["TransactionID"].to_numpy())}
    inputs = bundle_inputs(df)  # exact where the bundle supports it, else not available (D126)
    # row positions (time order) of every customer and entity, for the focused network (D131)
    rows_of = {k: {v: idx for v, idx in df.groupby(k, sort=False).indices.items()}
               for k in ("uid", *ENTITY_KINDS)}  # fmt: skip
    # t_dis = inf (disagreement rule off) on the Python side is null in the contract
    thresholds = {k: _none(float(v)) if v is not None else None
                  for k, v in meta["thresholds"].items()}  # fmt: skip
    # the frozen E12 thresholds of the final run: read-only (D129); others only via what-if
    from vaultic.audit.http import AUDIT_FAILED, CLIENT_EVENTS, actor_of, public, record_open
    from vaultic.audit.store import default_store, to_csv

    audit_log_store = audit_store if audit_store is not None else default_store()  # D208
    state = {"feedback": {}}
    # FastAPI runs these handlers in a thread pool: every read and write of `state` holds this
    # lock, so a label and its audit entry change together and a reader never sees half (D135)
    state_lock = threading.Lock()
    base_actions = df["action"].to_numpy().astype(object)  # the frozen E12 decisions
    whatif_cache: dict = {}  # cost settings -> (actions, thresholds); a cache, not session state
    cache_lock = threading.Lock()
    cal_path = bundle_path.with_name("demo_final_calibration.parquet")
    cal = pd.read_parquet(cal_path) if cal_path.exists() else None
    exp310_events = json.loads(events_path.read_text("utf-8")) if events_path.exists() else None
    ladder_runs, ladder_rows = ladder_ablation(ladder_dir)
    cost = {
        **meta["cost_model"],
        "hold_success_rate": meta["cost_model"].get("hold_success_rate", 1.0),
    }

    app = FastAPI(title="Vaultic demo backend", version="0.1")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])  # fmt: skip

    def matured(t: float, time: float, L: int) -> bool:
        return t + L * DAY <= time

    def decisions(w: dict | None) -> tuple[np.ndarray, dict | None]:
        """The actions served for a request: the frozen E12 ones, or those of its what-if costs."""
        if w is None:
            return base_actions, None
        if cal is None:
            raise HTTPException(501, "The what-if cost mode needs demo_final_calibration.parquet.")
        key = tuple(sorted(w.items()))
        with cache_lock:
            if key in whatif_cache:
                return whatif_cache[key]
        src = Path(__file__).resolve().parents[2] / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from vaultic.trust.decision import ACTIONS as CODES
        from vaultic.trust.decision import CostModel, choose_thresholds, decide

        cm = CostModel(w["c_fp"], w["c_rev"], w["c_step"], w["step_up_success_rate"],
                       w["legit_step_up_friction"], hold_success_rate=w["hold_success_rate"])  # fmt: skip
        th, _ = choose_thresholds(cal["p"].to_numpy(), cal["amount"].to_numpy(),
                                  cal["label"].to_numpy(), cal["conformal_set"].to_numpy(object),
                                  cal["disagreement"].to_numpy(), cost=cm)  # fmt: skip
        codes = decide(df["p"].to_numpy(), df["amount"].to_numpy(), th,
                       df["conformal_set"].to_numpy(object), df["disagreement"].to_numpy())  # fmt: skip
        out = (np.asarray([CODES[k] for k in codes], dtype=object),
               {k: _none(float(v)) for k, v in th.__dict__.items()})  # fmt: skip
        with cache_lock:
            if len(whatif_cache) >= 32:
                whatif_cache.pop(next(iter(whatif_cache)))
            whatif_cache[key] = out
        return out

    def transaction(i: int, time: float, L: int, actions: np.ndarray, whatif: dict | None) -> dict:
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
                         "whatIf": whatif is not None},
            "reasons": json.loads(r["reasons"]),
            # D155: the what-if costs that produced this decision (None: the evaluated system)
            "whatIfCost": whatif,
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
    def day_summary(time: float, L: LabelDelay, w: WhatIfDep):
        idx = day_rows(time)
        sub = df.iloc[idx]
        act = decisions(w)[0][idx]
        by_action = {a: int((act == a).sum()) for a in ACTIONS}
        p, amt = sub["p"].to_numpy(), sub["amount"].to_numpy()
        cost = w if w is not None else meta["cost_model"]
        hold_rate = cost.get("hold_success_rate", 1.0)
        rate = cost["step_up_success_rate"]
        el = p * amt  # expected fraud value of each transaction (risk x amount)
        low = np.isin(act, ["allow", "monitor"])
        step, held, block = act == "step_up", act == "hold", act == "block"
        # one definition (D136): the expected fraud value a decision stops, and the rest missed;
        # stopped + missed = the day's expected fraud value
        stop_share = np.select([low, step, held, block], [0.0, rate, hold_rate, 1.0], 0.0)
        stopped = float((stop_share * el).sum())
        missed = float(((1 - stop_share) * el).sum())
        costs = {"missedFraud": missed,
                 "falsePositives": float(((1 - p) * cost["c_fp"])[block].sum()),
                 "reviews": float(cost["c_rev"] * (held | block).sum()),
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
                "expectedFraudStopped": stopped, "expectedLoss": missed,
                "cost": costs, "riskByHour": hours,
                "viewAvailability": {v: float(sub[f"p_{v}"].notna().sum() / n) for v in VIEWS},
                "whatIf": w is not None,
                # D155: the costs and thresholds that produced these figures
                "whatIfCost": w, "whatIfThresholds": decisions(w)[1]}  # fmt: skip

    @app.get("/api/stream")
    def stream(time: float, start: float, end: float, L: LabelDelay,
               w: WhatIfDep):  # fmt: skip
        stop = min(end, time)
        lo, hi = int(np.searchsorted(dt, start, side="right")), up_to(stop)
        act = decisions(w)[0]
        return [transaction(i, stop, L, act, w) for i in range(lo, hi)]

    @app.get("/api/transactions/{tid}")
    def get_transaction(tid: int, time: float, L: LabelDelay,
                        w: WhatIfDep):  # fmt: skip
        i = by_id.get(tid)
        if i is None or dt[i] > time:
            return None
        return transaction(i, time, L, decisions(w)[0], w)

    @app.get("/api/alerts")
    def alerts(time: float, L: LabelDelay, w: WhatIfDep):
        act = decisions(w)[0]
        idx = [i for i in day_rows(time) if act[i] in ("step_up", "hold", "block")]
        idx.sort(key=lambda i: -df.at[i, "p"])
        return [transaction(i, time, L, act, w) for i in idx]

    @app.get("/api/customers/{uid}/history")
    def history(uid: str, time: float, L: LabelDelay, w: WhatIfDep):
        n = up_to(time)
        idx = np.flatnonzero(df["uid"].to_numpy()[:n] == uid)
        act = decisions(w)[0]
        return [transaction(int(i), time, L, act, w) for i in idx]

    def node_rows(kind: str, value, n: int) -> np.ndarray:
        idx = rows_of[kind].get(value)
        return np.empty(0, int) if idx is None else idx[idx < n]

    def focused_network(focus: str, n: int, time: float, L: int) -> dict:
        """The entities within two links of the focus, as the mock (D131): the focus's
        `kind:` prefix names the one column it is matched in; email domains and billing regions
        (hubs) are never expanded unless they are the focus; at most MAX_NODES entities."""
        kind, _, value = focus.partition(":")
        if kind not in ("uid", *ENTITY_KINDS) or not value:
            raise HTTPException(
                422, "focus must be kind:value with kind uid, card, email, address or device."
            )
        start = (kind, value)
        if len(node_rows(kind, value, n)) == 0:
            return {"nodes": [], "edges": [], "rings": []}
        keep, frontier = {start}, [start]
        for _hop in range(2):
            nxt = []
            for k, v in frontier:
                if k in HUB_KINDS and (k, v) != start:
                    continue
                idx = node_rows(k, v, n)
                if k == "uid":
                    nbrs = [(e, x) for e in ENTITY_KINDS for x in pd.unique(df[e].to_numpy()[idx])
                            if _none(x) is not None]  # fmt: skip
                else:
                    nbrs = [("uid", u) for u in pd.unique(df["uid"].to_numpy()[idx])]
                for nb in nbrs:
                    if nb not in keep and len(keep) < MAX_NODES:
                        keep.add(nb)
                        nxt.append(nb)
            frontier = nxt
        label, tx_time = df["label"].to_numpy(), df["TransactionDT"].to_numpy()
        nodes = []
        for k, v in keep:
            idx = node_rows(k, v, n)
            known = (label[idx] == 1) & (tx_time[idx] + L * DAY <= time)
            nodes.append({"id": f"{k}:{v}", "type": k, "label": str(v), "ring": None,
                          "knownFraud": int(known.sum()), "transactions": int(len(idx))})  # fmt: skip
        edges = {}
        for k, u in keep:
            if k != "uid":
                continue
            for i in node_rows("uid", u, n):  # time order: the first row of a link sets its time
                for e in ENTITY_KINDS:
                    x = _none(df[e].iat[i])
                    if x is not None and (e, x) in keep:
                        eid = f"uid:{u}|{e}:{x}"
                        edges.setdefault(eid, {"id": eid, "source": f"uid:{u}", "target": f"{e}:{x}",
                                               "time": float(tx_time[i])})  # fmt: skip
        return {"nodes": nodes, "edges": list(edges.values()), "rings": []}

    @app.get("/api/network")
    def network(time: float, L: LabelDelay, w: WhatIfDep, focus: str | None = None):
        n = up_to(time)
        if focus:
            return {**focused_network(focus, n, time, L), "whatIf": False}
        actions = decisions(w)[0]
        rows = day_rows(time)
        sub = df.iloc[:n].iloc[rows][np.isin(actions[rows], ["hold", "block"])].nlargest(40, "p")
        nodes, edges = {}, {}
        for _, r in sub.iterrows():
            known = int(r["label"] == 1 and matured(r["TransactionDT"], time, L))
            ids = []
            for kind in ("uid", *ENTITY_KINDS):
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
        # without a focus the entities come from today's holds and blocks: what-if decisions
        return {"nodes": list(nodes.values()), "edges": list(edges.values()), "rings": [],
                "whatIf": w is not None}  # fmt: skip

    budgets = sorted(int(x) for x in meta["routing_lambdas"])  # the frozen K (D133)

    def lambdas(k: int) -> dict:
        return meta["routing_lambdas"][str(k)]

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
    def queues(time: float, L: LabelDelay, w: WhatIfDep, k: int = 50):
        if k not in budgets:
            raise HTTPException(
                422,
                f"Cases per day (K) must be one of the frozen budgets {', '.join(map(str, budgets))}"
                f", not {k}: the routing weights were chosen for those only.",
            )
        act = decisions(w)[0]
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
                        "cases": [transaction(by_id[int(t)], time, L, act, w)
                                  for t in top["TransactionID"]],
                        "matured": {"day": known_day,
                                    "fraudValueCaught": float(caught["amount"].sum()),
                                    "fraudCaught": int(len(caught)), "reviews": int(len(rev))}})  # fmt: skip
        return out

    @app.get("/api/drift")
    def drift(time: float, L: LabelDelay):
        """PSI of the test period up to the as-of time against the calibration days 147-150
        (demo_final_calibration.parquet), which end before the test period starts: the reference
        never contains anything after the as-of time (D130). Only inputs present in both are
        compared: the fused (calibrated) risk score, the amount and the view disagreement, plus
        each view score if the calibration file carries it. A day or input with fewer than 10
        values is left out, never shown as 0."""
        sub = df.iloc[: up_to(time)]
        if cal is None:
            raise HTTPException(501, "Drift needs demo_final_calibration.parquet as its reference.")

        def psi(a, b):
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 10 or len(b) < 10:
                return None
            edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
            edges[0], edges[-1] = -np.inf, np.inf
            pa = np.histogram(a, edges)[0] / len(a) + 1e-4
            pb = np.histogram(b, edges)[0] / len(b) + 1e-4
            return float(np.sum((pb - pa) * np.log(pb / pa)))

        days = sorted(sub["day"].unique())
        ref_p = cal["p"].to_numpy(dtype=float)
        score_psi = [{"day": int(dd), "value": v} for dd in days
                     if (v := psi(ref_p, sub.loc[sub["day"] == dd, "p"].to_numpy(dtype=float)))
                     is not None]  # fmt: skip
        dis = [{"day": int(dd), "value": float(sub.loc[sub["day"] == dd, "disagreement"].mean())}
               for dd in days]  # fmt: skip
        today = sub[sub["day"] == int(time // DAY)]
        inputs_ = [("fused risk score", "p"), ("amount", "amount"),
                   ("view disagreement", "disagreement")]  # fmt: skip
        inputs_ += [(f"view score: {v}", f"p_{v}") for v in VIEWS if f"p_{v}" in cal]
        feats = [{"feature": name, "psi": v} for name, col in inputs_
                 if (v := psi(cal[col].to_numpy(dtype=float), today[col].to_numpy(dtype=float)))
                 is not None]  # fmt: skip
        # no drift detectors or injected drifts ran on the final run (E13 not run); alarms and
        # retraining come from the EXP-310 simulation on the test period (D220)
        sim = exp310_drift(exp310_events, L, int(time // DAY))
        return {
            "reference": "calibration days 147-150 (before the test period)",
            "featurePsi": feats,
            "scorePsi": score_psi,
            "disagreement": dis,
            "alarms": sim["alarms"],
            "injected": [],
            "retrains": sim["retrains"],
            "simulation": sim["simulation"],
        }

    @app.get("/api/lab")
    def lab():
        return {"runs": meta["lab"] + ladder_runs, "ablation": ladder_rows,
                "sensitivity": meta["sensitivity"]}  # fmt: skip

    @app.post("/api/score")
    def score():
        raise HTTPException(501, "Scoring new inputs is not available: the final-run models are "
                                 "served for recorded transactions only.")  # fmt: skip

    @app.get("/api/thresholds")
    def get_thresholds():
        return {"thresholds": thresholds, "cost": cost, "frozen": True}

    def audit(request, action: str, target=None, before=None, after=None, detail: str = "",
              fallback_role: str | None = None) -> dict:  # fmt: skip
        """Append an entry to the persistent audit log (D208) for the request's analyst."""
        analyst, role = actor_of(request, fallback_role)
        return audit_log_store.append(analyst, role, action, target=target, before=before,
                                      after=after, detail=detail,
                                      time=datetime.now(UTC).isoformat(timespec="seconds"))  # fmt: skip

    @app.put("/api/thresholds")
    def save_thresholds(request: Request):
        record_open(
            audit_log_store,
            request,
            "Refused: change the frozen decision thresholds",
            target="thresholds",
            detail="read-only (D129)",
        )  # fmt: skip  (fail-open, D209)
        raise HTTPException(
            405,
            "The decision thresholds are the frozen ones of the final run and are read-only. "
            "Use the what-if cost mode in Settings to see decisions under other costs.",
        )

    @app.post("/api/feedback/{tid}")
    def feedback(tid: int, body: FeedbackIn, request: Request):
        with state_lock:  # the read, the audit entry and the write as one step
            before = state["feedback"].get(tid)
            try:  # fail-closed (D209): the audit entry first, the label only after it succeeded
                audit(request, f"Marked transaction {tid} as {body.label}", target=f"tx:{tid}",
                      before=before, after=body.label,
                      detail=f"was {before}" if before else "first label",
                      fallback_role=body.role)  # fmt: skip
            except Exception as e:
                raise HTTPException(503, "The label was not saved: the audit log could not be "
                                         f"written ({type(e).__name__}). Try again.") from e  # fmt: skip
            state["feedback"][tid] = body.label
        return {"ok": True}

    @app.get("/api/whatif")
    def get_whatif(w: WhatIfDep):
        """Status of the what-if cost mode for the parameters sent (none: the evaluated system)."""
        return {"active": w is not None, "available": cal is not None,
                "cost": w if w is not None else cost, "thresholds": decisions(w)[1],
                "note": WHATIF_NOTE}  # fmt: skip

    @app.get("/api/audit")
    def audit_log(analyst: str | None = None, action: str | None = None, since: str | None = None,
                  until: str | None = None, limit: int = 500, offset: int = 0):  # fmt: skip
        """Entries, newest first (D208); filters by analyst, action text and time."""
        if not 1 <= limit <= 5000 or offset < 0:
            raise HTTPException(422, "limit must be 1-5000 and offset >= 0")
        rows = audit_log_store.query(analyst=analyst, action=action, since=since, until=until,
                                     limit=limit, offset=offset)  # fmt: skip
        return [public(r) for r in rows]

    @app.get("/api/audit/verify")
    def audit_verify():
        return audit_log_store.verify()

    @app.get("/api/audit/export.csv", response_class=PlainTextResponse)
    def audit_export(request: Request):
        """The whole log as CSV. The export is recorded in the same locked step as the snapshot
        (the count in its entry is exact); fail-open (D209): if recording fails the CSV is still
        returned, with the header X-Vaultic-Audit: failed."""
        headers = {"Content-Disposition": "attachment; filename=vaultic_audit_log.csv"}
        try:
            _, rows = audit_log_store.export(*actor_of(request))
        except Exception:
            import logging

            logging.getLogger("vaultic.audit").exception("audit write failed for the export")
            rows, headers["X-Vaultic-Audit"] = audit_log_store.query(limit=10**9), AUDIT_FAILED
        return PlainTextResponse(to_csv(rows), media_type="text/csv", headers=headers)

    @app.post("/api/audit/event")
    def audit_event(body: AuditEventIn, request: Request):
        """Client-side actions the server cannot see (D208): only the listed ones are accepted."""
        if body.action not in CLIENT_EVENTS:
            raise HTTPException(422, f"action must be one of {sorted(CLIENT_EVENTS)}")
        ok = record_open(
            audit_log_store,
            request,
            CLIENT_EVENTS[body.action],
            target=body.action,
            detail=body.detail[:500],
        )  # fmt: skip  (fail-open, D209)
        return {"ok": ok} if ok else {"ok": False, "audit": AUDIT_FAILED}

    app.state.audit = audit_log_store

    if os.environ.get("VAULTIC_TESTLAB") == "1":  # D182: the Test Lab's live scoring endpoints
        mount_testlab(app)
    return app


def mount_testlab(app: FastAPI, service=None, start: bool = True):
    """Add the Test Lab router (services/api/testlab.py); its engine loads in the background."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "testlab_api", Path(__file__).with_name("testlab.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if service is None:
        from vaultic.live.service import TestLabService

        service = TestLabService()
        if start:
            service.start_in_background()
    app.include_router(mod.make_router(service, audit=getattr(app.state, "audit", None)))
    app.state.testlab = service
    # Phase B: the Attack Lab runs synthetic scenarios through the same engine (services/api/attack.py)
    aspec = importlib.util.spec_from_file_location(
        "attack_api", Path(__file__).with_name("attack.py")
    )
    amod = importlib.util.module_from_spec(aspec)
    aspec.loader.exec_module(amod)
    from vaultic.scenarios.service import AttackLab

    app.state.attack = AttackLab(service)
    app.include_router(amod.make_router(app.state.attack, audit=getattr(app.state, "audit", None)))
    return service


app = create_app() if (os.environ.get("VAULTIC_DEMO_BUNDLE") or DEFAULT_BUNDLE.exists()) else None
