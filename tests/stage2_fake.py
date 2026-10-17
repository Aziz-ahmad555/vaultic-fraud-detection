"""A small but real FrozenSystem on synthetic data for the stage-2 tests: five XGBoost views,
Platt calibrators, an MVAF gate, an E12 layer and B5, all fitted on synthetic rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaultic.fusion.mvaf import MVAF, view_inputs
from vaultic.stage2.system import FrozenSystem
from vaultic.trust.calibration import PlattCalibrator
from vaultic.trust.conformal import MondrianConformal, set_labels
from vaultic.trust.decision import choose_thresholds
from vaultic.views.orchestrate import CONTEXT, VIEWS, SupervisedView
from vaultic.views.tabular import make_model

COLUMNS = {
    "tabular": ["TransactionAmt", "uid_amt_vs_mean", "ProductCD", "x1", "x2"],
    "behavioral": ["hist_n_past", "amt_z", "amt_robust_z", "b1"],
    "temporal": ["seq_n_steps", "seq_cur_log_amount", "t1"],
    "graph": ["g_shared_nonhub", "g1"],
    "anomaly": ["anomaly_if", "anomaly_ae"],
}
XGB = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.3}


def synthetic_frame(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    f = pd.DataFrame({c: rng.normal(size=n) for cols in COLUMNS.values() for c in cols})
    f["x1"] += 2.5 * y  # one strong tabular signal
    f["b1"] += 1.0 * y
    f["TransactionAmt"] = np.round(rng.lognormal(3, 1, n), 2)
    f["ProductCD"] = rng.integers(0, 3, n).astype(float)
    f["hist_n_past"] = rng.integers(0, 5, n).astype(float)
    f["seq_n_steps"] = np.minimum(f["hist_n_past"], 3)
    f["g_shared_nonhub"] = rng.integers(0, 2, n).astype(float)
    f.loc[rng.random(n) < 0.2, ["anomaly_if", "anomaly_ae"]] = np.nan
    f["uid"] = rng.integers(0, 50, n).astype(str)
    f["uid_n_past"] = f["hist_n_past"]
    return f.astype({c: np.float32 for c in f.columns if c != "uid"}), y


def _avail(view):
    from vaultic.views.definitions import available

    return None if view == "tabular" else (lambda fr, v=view: available(v, fr))


def fake_system(seed: int = 0) -> tuple[FrozenSystem, pd.DataFrame, np.ndarray, np.ndarray]:
    train, y = synthetic_frame(3000, seed)
    views = {v: SupervisedView(COLUMNS[v], "xgboost", XGB, available=_avail(v)).fit(train, y)
             for v in VIEWS}  # fmt: skip
    raw = np.column_stack([views[v].predict(train) for v in VIEWS])
    calibrators = {}
    for j, v in enumerate(VIEWS):
        ok = ~np.isnan(raw[:, j])
        calibrators[v] = PlattCalibrator().fit(raw[ok, j], y[ok])
    p = np.column_stack([np.where(np.isnan(raw[:, j]), np.nan,
                                  calibrators[v].predict(np.nan_to_num(raw[:, j], nan=0.5)))
                         for j, v in enumerate(VIEWS)])  # fmt: skip
    rng = np.random.default_rng(seed)
    ctx = rng.random((len(y), len(CONTEXT)))
    ctx[:, CONTEXT.index("ctx_log_amount")] = np.log1p(train["TransactionAmt"].to_numpy(float))
    gates = [MVAF(hidden=(8,), epochs=3, seed=s).fit(p, y, ctx) for s in range(5)]
    risk = np.mean([g.predict_proba(p, ctx) for g in gates], axis=0)
    fused_cal = PlattCalibrator().fit(risk, y)
    pc = fused_cal.predict(risk)
    conformal = MondrianConformal(alpha=0.10).fit(y, pc)
    amount = train["TransactionAmt"].to_numpy(float)
    th, _ = choose_thresholds(pc, amount, y, set_labels(conformal.predict_sets(pc)),
                              view_inputs(p)[3])  # fmt: skip
    b5_cols = COLUMNS["tabular"] + ["hist_n_past", "b1"]
    b5 = [make_model("xgboost", XGB, s).fit(train[b5_cols], y) for s in range(5)]

    class _Anomaly:  # deterministic stand-in: scores move with the amount
        def transform(self, X, uid, n_past):
            a = np.log1p(np.abs(X[:, 0]))
            return pd.DataFrame({"anomaly_if": a / 10, "anomaly_ae": a / 20,
                                 "anomaly_uid_if": np.full(len(a), np.nan)})  # fmt: skip

    numeric = train.drop(columns=["uid"])
    system = FrozenSystem(
        views=views, calibrators=calibrators, gates=gates, fused_calibrator=fused_cal,
        conformal=conformal, thresholds=th, b5_models=b5, b5_columns=b5_cols, b5_threshold=0.5,
        anomaly=_Anomaly(), anomaly_columns=["TransactionAmt"], encoder=None,
        medians=numeric.median(),
    )  # fmt: skip
    return system, train, y, ctx
