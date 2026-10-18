"""The live scoring engine of the Test Lab (D182-D184).

Loads the frozen stage-2 system (stage2/system.FrozenSystem: fixed-fold view models,
calibrators, MVAF gate seeds 0-4, E12 layer, B5, anomaly models) from its cache on drive E (fits
and caches it if absent; the refits reproduce the final run exactly, D179) and the history store
(live/store.py), then scores ONE transaction at a time:

  risk        MVAF's 5-seed mean score (the score E10 evaluated)
  decision    the frozen E12 decision (allow / monitor / step-up / hold / block) from the fused
              calibration, the 90% Mondrian conformal set and the view disagreement; an alert is
              step-up or higher (D159), and the E10 cost threshold (0.881) is shown alongside
  braid       per view: raw and calibrated score (or missing) and the gate weight
  reasons     view contributions (gate weight x calibrated view logit) and the top feature
              contributions (TreeSHAP of the view models, weighted by the gate, as in E15) with
              explain/reason_codes sentences
  latency     wall-clock of features + scoring, per request; p50 / p95 over the session
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vaultic.live.parity import SCORE_TOLERANCE
from vaultic.live.store import CACHE_DIR

SYSTEM_CACHE = CACHE_DIR / "system.joblib"
STAGE2_DEV_SYSTEM = Path("E:/dev-cache/stage2/system_dev.joblib")
ACTIONS = ("allow", "monitor", "step_up", "hold", "block")
VIEWS = ("tabular", "behavioral", "temporal", "graph", "anomaly")


def load_system(log=print):
    """The frozen system from the cache on E (or the stage-2 development refit, which
    reproduces the final run exactly; D179), else fitted and cached."""
    import joblib

    for path in (SYSTEM_CACHE, STAGE2_DEV_SYSTEM):
        if path.exists():
            log(f"system from {path}")
            return joblib.load(path)
    from vaultic.stage2.system import fit_system, load_parts

    log("fitting the frozen system (about 25 minutes, once)")
    system = fit_system(load_parts(dev=True), log=log)
    SYSTEM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(system, SYSTEM_CACHE)
    return system


class LatencyLog:
    def __init__(self):
        self.values: list[float] = []
        self.lock = threading.Lock()

    def add(self, ms: float) -> None:
        with self.lock:
            self.values.append(float(ms))

    def summary(self) -> dict:
        with self.lock:
            v = np.array(self.values)
        if not len(v):
            return {"n": 0, "p50_ms": None, "p95_ms": None}
        return {"n": int(len(v)), "p50_ms": float(np.percentile(v, 50)),
                "p95_ms": float(np.percentile(v, 95)), "max_ms": float(v.max())}  # fmt: skip


class LiveEngine:
    def __init__(self, store, system, stored_scores: pd.DataFrame | None = None,
                 e10_threshold: float | None = None):  # fmt: skip
        self.store, self.system = store, system
        self.stored = stored_scores  # EXP-200-final predictions by TransactionID (real rows)
        self.e10_threshold = e10_threshold
        self.latency = LatencyLog()
        self.lock = threading.Lock()  # one score at a time (the store's new-uid counter)

    # ---- scoring -----------------------------------------------------------------------------
    def score(self, query: pd.DataFrame, uid_code: int, explain: bool = True,
              record_latency: bool = True) -> dict:  # fmt: skip
        from vaultic.explain.reason_codes import to_points, top_reasons
        from vaultic.explain.view_contrib import view_contributions
        from vaultic.fusion.mvaf import view_inputs
        from vaultic.live.features import online_features
        from vaultic.live.reasons import describe
        from vaultic.stage2.e15 import MvafExplained
        from vaultic.trust.conformal import set_labels
        from vaultic.trust.decision import STEP_UP, decide

        started = time.perf_counter()
        frame = online_features(self.store, self.system, query, uid_code)
        sysm = self.system
        ctx = sysm.context(query.reset_index(drop=True), frame)
        avail = sysm.availability(frame)
        s = sysm.score(frame, ctx, avail)
        risk = float(s["risk"][0])
        pe = sysm.fused_calibrator.predict(s["risk"])
        conf = set_labels(sysm.conformal.predict_sets(pe))
        dis = view_inputs(s["p"])[3]
        amount = query["TransactionAmt"].to_numpy(float)
        action = int(decide(pe, amount, sysm.thresholds, conf, dis)[0])
        b5 = float(sysm.b5(frame)[0][0])
        ms = (time.perf_counter() - started) * 1000
        if record_latency:
            self.latency.add(ms)
        out = {
            "risk": risk, "calibrated": float(pe[0]), "conformal_set": str(conf[0]),
            "disagreement": float(dis[0]), "decision": ACTIONS[action],
            "alert": bool(action >= STEP_UP),
            "flagged_e10": bool(self.e10_threshold is not None and risk >= self.e10_threshold),
            "e10_threshold": self.e10_threshold, "b5_score": b5,
            "views": [{"view": v, "available": bool(avail[0, j]),
                       "raw": _num(s["raw"][0, j]), "p": _num(s["p"][0, j]),
                       "weight": _num(s["weights"][0, j])} for j, v in enumerate(VIEWS)],
            "latency_ms": ms, "uid": int(uid_code),
        }  # fmt: skip
        if explain:
            vc = view_contributions(s["weights"], s["p"], list(VIEWS), s["risk"])
            out["view_reasons"] = sorted(
                [{"view": v, "points": float(to_points(vc.contribution[0, j]))}
                 for j, v in enumerate(VIEWS) if np.isfinite(vc.contribution[0, j])],
                key=lambda r: -abs(r["points"]))  # fmt: skip
            expl = MvafExplained(sysm)
            attr = expl.attributions(frame, ctx, avail)[0]
            shap = dict(zip(expl.columns, attr.tolist(), strict=True))
            values = frame.iloc[0].to_dict()
            out["reasons"] = [{"feature": r.feature, "text": r.text, "points": r.points}
                              for r in top_reasons(shap, values, k=5, describe_fn=describe)]  # fmt: skip
            out["explain_ms"] = (time.perf_counter() - started) * 1000 - ms
        return out

    def score_stored(self, position: int, **kw) -> dict:
        """Score a stored (real test) transaction live, and compare with the final run."""
        q = self.store.df.iloc[[position]]
        out = self.score(q, int(self.store.uid[position]), **kw)
        tid = int(q["TransactionID"].iloc[0])
        out["transaction_id"] = tid
        if self.stored is not None and tid in self.stored.index:
            ref = float(self.stored.at[tid, "score_MVAF"])
            out["stored_risk"] = ref
            out["matches_frozen_run"] = bool(abs(ref - out["risk"]) <= SCORE_TOLERANCE)  # D189
        return out


def _num(x) -> float | None:
    x = float(x)
    return None if not np.isfinite(x) else x


def build_engine(log=print):
    """System + store + the final run's stored test scores and E10 threshold."""
    from vaultic.live.store import build_store
    from vaultic.paths import REPO_ROOT, RESEARCH_DIR
    from vaultic.stage2.final import E10_RUN

    system = load_system(log)
    store = build_store(log)
    stored = pd.read_parquet(REPO_ROOT / E10_RUN / "predictions.parquet").set_index("TransactionID")
    final = json.loads((RESEARCH_DIR / "tables" / "final_results.json").read_text("utf-8"))
    return LiveEngine(store, system, stored, float(final["meta"]["thresholds"]["MVAF"]))
