"""The Test Lab service behind the API (D182-D186): engine state, parity gate, and the three modes.

States: "loading" (system, store and parity check starting up), "ready", "refused" (parity failed:
the reason and the failing features are reported, nothing is scored), "error". Every scoring call
needs "ready".

  real test transaction   sample(filter) picks a random held-out test transaction (fraud /
                          legit / cold start / any) and returns it WITHOUT its label;
                          score_real scores it live and compares with the stored final-run score;
                          reveal returns the label afterwards
  build your own          score_custom (live/custom.py): "custom input - no ground truth"
  batch CSV               template, score_batch (live/batch.py)
"""

from __future__ import annotations

import io
import threading
import traceback

import numpy as np
import pandas as pd

FILTERS = ("any", "fraud", "legit", "cold_start")


def _with_expected_loss(out: dict, amount: float) -> dict:
    """E12 decides on the expected loss = fused-calibrated probability x amount (D43): shown so a
    high risk on a small amount (allowed) is not mistaken for an error."""
    out["amount"] = amount
    out["expected_loss"] = out["calibrated"] * amount
    return out


class NotReady(RuntimeError):
    pass


class NotFound(LookupError):
    """A transaction the request names that is not a test-period transaction (404)."""


class TestLabService:
    def __init__(self):
        self.state, self.message, self.parity = "loading", "starting up", None
        self.engine = None
        self.lock = threading.Lock()

    # ---- startup -------------------------------------------------------------------------------
    def start(self, build=None, parity=None, log=print) -> None:
        """Build the engine and run (or read) the parity gate; never raises."""
        try:
            from vaultic.live.engine import STAGE2_DEV_SYSTEM, SYSTEM_CACHE, build_engine
            from vaultic.live.parity import cached_parity

            self.message = "loading the frozen system and the history store"
            engine = (build or build_engine)(log)
            self.message = "running the parity check (first start only)"
            path = SYSTEM_CACHE if SYSTEM_CACHE.exists() else STAGE2_DEV_SYSTEM
            result = (parity or cached_parity)(engine, path, log)
            self.parity = result
            if not result["passed"] and result.get("reason"):
                self.state, self.message = "refused", f"parity failed: {result['reason']}"
                return
            if not result["passed"]:
                self.state = "refused"
                self.message = (f"parity failed: {result['n_failing_features']} feature columns "
                                f"and scores {result['max_score_diff']} differ from the frozen run "
                                f"(tolerances: features {result.get('feature_tolerance')}, scores "
                                f"{result.get('score_tolerance')}); the Test Lab will not score")  # fmt: skip
                return
            self.engine = engine
            self.state, self.message = "ready", "parity passed; scoring live"
        except Exception as e:  # reported on the page, not raised into the server
            self.state, self.message = "error", f"{type(e).__name__}: {e}"
            log(traceback.format_exc())

    def start_in_background(self, **kw) -> threading.Thread:
        th = threading.Thread(target=self.start, kwargs=kw, daemon=True)
        th.start()
        return th

    def _ready(self):
        if self.state != "ready":
            raise NotReady(self.message)
        return self.engine

    def status(self) -> dict:
        out = {"state": self.state, "message": self.message, "parity": self.parity}
        if self.engine is not None:
            out["latency"] = self.engine.latency.summary()
        return out

    # ---- real test transactions --------------------------------------------------------------
    def _test_positions(self, flt: str) -> np.ndarray:
        e = self._ready()
        df = e.store.df
        test = df["day"].to_numpy() >= e.store.splits.test.first
        y = df["isFraud"].to_numpy()
        if flt == "fraud":
            test &= y == 1
        elif flt == "legit":
            test &= y == 0
        elif flt == "cold_start":
            test &= np.nan_to_num(e.store.features["hist_n_past"].to_numpy(float)) == 0
        elif flt != "any":
            raise ValueError(f"filter must be one of {FILTERS}")
        return np.flatnonzero(test)

    def sample(self, flt: str = "any", seed: int | None = None) -> dict:
        pos = self._test_positions(flt)
        p = int(np.random.default_rng(seed).choice(pos))
        return self.describe(p) | {"filter": flt}

    def describe(self, p: int) -> dict:
        """The transaction as shown before scoring: no label."""
        e = self._ready()
        r = e.store.df.iloc[p]

        def val(c):
            v = r.get(c)
            return (
                None
                if v is None or pd.isna(v)
                else (str(v) if not isinstance(v, (int, float, np.number)) else float(v))
            )

        return {"transaction_id": int(r["TransactionID"]), "uid": int(e.store.uid[p]),
                "day": int(r["day"]), "seconds": int(r["TransactionDT"]) % 86400,
                "amount": float(r["TransactionAmt"]), "product": val("ProductCD"),
                "card4": val("card4"), "card6": val("card6"), "email": val("P_emaildomain"),
                "device_type": val("DeviceType"), "device_info": val("DeviceInfo"),
                "history": int(np.nan_to_num(e.store.features["hist_n_past"].iloc[p]))}  # fmt: skip

    def _position(self, tid: int) -> int:
        e = self._ready()
        pos = e.store.info.get("by_tid")
        if pos is None:
            ids = e.store.df["TransactionID"].to_numpy()
            pos = e.store.info["by_tid"] = {int(t): i for i, t in enumerate(ids)}
        if tid not in pos:
            raise NotFound(f"transaction {tid} is not in the data")
        p = pos[tid]
        if e.store.df["day"].iat[p] < e.store.splits.test.first:
            raise NotFound(f"transaction {tid} is not a test-period transaction")
        return p

    def score_real(self, tid: int) -> dict:
        e = self._ready()
        with e.lock:
            out = e.score_stored(self._position(tid))
        out["kind"] = "real test transaction (label hidden until revealed)"
        return _with_expected_loss(
            out, float(e.store.df["TransactionAmt"].iat[self._position(tid)])
        )

    def reveal(self, tid: int) -> dict:
        e = self._ready()
        p = self._position(tid)
        return {"transaction_id": tid, "isFraud": int(e.store.df["isFraud"].iat[p])}

    # ---- custom ----------------------------------------------------------------------------------
    def options(self) -> dict:
        from vaultic.live.custom import options

        return options(self._ready().store)

    def score_custom(self, spec: dict) -> dict:
        from vaultic.live.custom import build_custom

        e = self._ready()
        with e.lock:
            row, uid, notes = build_custom(e.store, spec)
            out = e.score(row, uid)
        out["kind"] = "custom input - no ground truth"
        out["notes"] = notes
        return _with_expected_loss(out, float(spec["amount"]))

    # ---- batch -------------------------------------------------------------------------------------
    def template(self) -> str:
        from vaultic.live.batch import template_csv

        return template_csv(self._ready().store)

    def score_batch(self, text: str) -> dict:
        from vaultic.live.batch import parse, score_batch

        e = self._ready()
        rows = parse(e.store, text)  # raises BatchError with the reason
        with e.lock:
            table, metrics = score_batch(e, rows)
        buf = io.StringIO()
        table.to_csv(buf, index=False)
        return {"kind": "user file - not the evaluation", "metrics": metrics,
                "rows": table.to_dict(orient="records"), "csv": buf.getvalue(),
                "note": "each row was scored independently against the stored history before its "
                        "own time; rows of the file do not affect each other"}  # fmt: skip
