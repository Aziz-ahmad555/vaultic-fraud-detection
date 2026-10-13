"""The frozen EXP-200-final system, refitted for stage 2 (D158 (4)).

The final run saved scores (predictions.parquet), not fitted models. E15 needs the models to
explain and rescore transactions, so they are refitted here with the frozen settings, exactly
as the final run fitted them:

  view models   the fixed fold's five XGBoost views (fusion/dev_compare.d95_views, tuned
                EXP-V-<view> settings), trained on days 1-120 where each view is available
  calibrators   the fixed fold's per-view calibrators (calibrate_views, days 144-146)
  gate          MVAF with the D95-tuned settings (fusion_d95.json), seeds 0-4, on the gate rows
                (fit + tune) of the D95 view table
  E12 layer     MVAF's fused calibrator, 90% Mondrian conformal sets and decision thresholds,
                fitted on calibrate_fused (147-150) as in trust/final_trust
  B5            EXP-009's frozen XGBoost, seeds 0-4, on days 1-120, and its cost threshold
                chosen on calibrate_fused
  anomaly       the anomaly models that scored every row after the training period
                (views/anomaly_build, last forward-chained cut), for rescoring perturbed rows

Before anything is explained, `check_validation` compares the refits with what was stored, on
validation rows only: raw and calibrated view scores against the D95 view table, the anomaly
scores against the anomaly feature file, B5 against EXP-009's stored validation predictions,
and the MVAF and B5 thresholds against the final run's (validation-chosen) thresholds. The
final runs repeat the check on the test rows against the stored final-run scores
(`check_test`), and stop on any difference above TOLERANCE.

A view's availability belongs to the transaction, not to a feature value (rule 11): when
features are replaced or perturbed, every view keeps the availability of the original row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from vaultic.paths import INTERIM_DIR, RESEARCH_DIR

SEEDS = (0, 1, 2, 3, 4)
TOLERANCE = 1e-6
VIEW_TABLE = INTERIM_DIR / "view_table_d95.parquet"  # D95 validation table, no test rows
FUSION_PARAMS = RESEARCH_DIR / "tables" / "fusion_d95.json"
FINAL_RESULTS = RESEARCH_DIR / "tables" / "final_results.json"
# anomaly_build's settings for the scores in data/features/anomaly_<uid>.parquet (D68)
ANOMALY = {"block_days": 30, "max_fit_rows": 150_000, "ae_max_iter": 50, "seed": 0}


def refit_complete(rc: dict, n: int, tol: float = TOLERANCE) -> bool:
    """A refit check (fusion/final_eval.refit_check) passes only if EVERY row was compared and
    the largest difference is within tol (D173): no row missing from the stored run, none
    non-finite on either side."""
    return bool(rc["n_missing"] == 0 and rc["n_compared"] == n and rc["n_rows"] == n
                and rc["max_abs_diff"] <= tol)  # fmt: skip


def load_parts(dev: bool) -> dict:
    """Data and features as the final run loads them (build_table parts_only). dev=True drops
    every test-period row before anything else happens."""
    from vaultic.fusion.dev_compare import build_table

    parts = build_table(False, d95=True, parts_only=True)
    if dev:
        keep = parts["df"]["day"].to_numpy() <= parts["splits"].validation.last
        parts["df"] = parts["df"][keep].reset_index(drop=True)
        parts["features"] = parts["features"][keep].reset_index(drop=True)
    parts["dev"] = dev
    return parts


def require_no_test(parts: dict, day) -> None:
    if parts["dev"] and (np.asarray(day) >= parts["splits"].test.first).any():
        raise RuntimeError("a test-period row reached a development run")


def anomaly_fit_rows(day: np.ndarray, train_first: int, train_last: int, block_days: int,
                     max_fit_rows: int | None, seed: int) -> np.ndarray:  # fmt: skip
    """The rows anomaly_build.forward_chained_scores fitted its LAST cut on (the models that
    scored every row after the training period), drawing from the generator exactly as it did
    for the earlier cuts."""
    rng = np.random.default_rng(seed)
    starts = list(range(train_first, train_last + 1, block_days))
    cuts = [(s, min(s + block_days, train_last + 1)) for s in starts[1:]]
    cuts.append((train_last + 1, int(day.max()) + 1))
    fit = None
    for start, stop in cuts:
        rows = np.flatnonzero((day < start) & (day >= train_first))
        if not ((day >= start) & (day < stop)).any() or len(rows) == 0:
            continue
        if max_fit_rows is not None and len(rows) > max_fit_rows:
            rows = np.sort(rng.choice(rows, max_fit_rows, replace=False))
        fit = rows
    if fit is None:
        raise ValueError("no anomaly cut to fit")
    return fit


def per_uid_scores(per_uid, X, uid, n_past) -> np.ndarray:
    """PerUidIsolationForest.transform, visiting only the customers present (same result; the
    original loops over every fitted customer's forest)."""
    X = np.asarray(X, dtype=float)
    X = np.where(np.isnan(X), per_uid.fill_, X)
    uid, n_past = np.asarray(uid), np.asarray(n_past)
    raw = np.full(len(X), np.nan)
    for user in pd.unique(uid):
        forest = per_uid.models_.get(user)
        if forest is None:
            continue
        rows = (uid == user) & (n_past >= per_uid.min_history)
        if rows.any():
            raw[rows] = -forest.score_samples(X[rows])
    return raw if per_uid.norm_ is None else per_uid.norm_.transform(raw)


@dataclass
class FrozenSystem:
    views: dict  # name -> fitted SupervisedView (fixed fold)
    calibrators: dict  # name -> fitted per-view calibrator (fixed fold)
    gates: list  # MVAF, one per seed
    fused_calibrator: object
    conformal: object
    thresholds: object  # trust.decision.Thresholds (E12 default cost)
    b5_models: list
    b5_columns: list[str]
    b5_threshold: float
    anomaly: object  # views.anomaly.AnomalyView (training-period legit rows)
    anomaly_columns: list[str]
    encoder: object  # ProductCD encoder fitted on the training period (gate context)
    medians: pd.Series  # training-period median of every feature column (E15 reference)
    info: dict = field(default_factory=dict)

    # ---- views ------------------------------------------------------------------------------
    def availability(self, frame: pd.DataFrame) -> np.ndarray:
        """(n, 5) bool in VIEWS order: where each view exists for these rows."""
        from vaultic.views.orchestrate import VIEWS

        return np.column_stack([self.views[v]._mask(frame) for v in VIEWS])

    def raw(self, frame: pd.DataFrame, avail: np.ndarray | None = None) -> np.ndarray:
        """(n, 5) raw view probabilities, NaN where the view is missing; `avail` fixes the
        availability (that of the original rows) when `frame` holds perturbed features."""
        from vaultic.views.orchestrate import VIEWS

        avail = self.availability(frame) if avail is None else np.asarray(avail, bool)
        out = np.full((len(frame), len(VIEWS)), np.nan)
        for j, v in enumerate(VIEWS):
            view, rows = self.views[v], avail[:, j]
            if view.model_ is not None and rows.any():
                out[rows, j] = view.model_.predict_proba(frame.loc[rows, view.columns])[:, 1]
        return out

    def calibrated(self, raw: np.ndarray) -> np.ndarray:
        from vaultic.views.orchestrate import VIEWS

        out = np.full(raw.shape, np.nan)
        for j, v in enumerate(VIEWS):
            ok = ~np.isnan(raw[:, j])
            if ok.any():
                out[ok, j] = self.calibrators[v].predict(raw[ok, j])
        return out

    # ---- gate and decisions -------------------------------------------------------------------
    def fused(self, p: np.ndarray, context: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """MVAF risk: the 5-seed mean, and each seed's score."""
        per_seed = [g.predict_proba(p, context) for g in self.gates]
        return np.mean(per_seed, axis=0), per_seed

    def gate_weights(self, p: np.ndarray, context: np.ndarray, per_seed: bool = False):
        w = [g.view_weights(p, context) for g in self.gates]
        return w if per_seed else np.mean(w, axis=0)

    def decide(self, risk: np.ndarray, p: np.ndarray, amount: np.ndarray) -> np.ndarray:
        """E12 action codes (trust.decision.ACTIONS) for MVAF risk, recomputing the fused
        calibration, the conformal set and the view disagreement from the given rows."""
        from vaultic.fusion.mvaf import view_inputs
        from vaultic.trust.conformal import set_labels
        from vaultic.trust.decision import decide

        pe = self.fused_calibrator.predict(risk)
        sets = set_labels(self.conformal.predict_sets(pe))
        dis = view_inputs(p)[3]
        return decide(pe, amount, self.thresholds, sets, dis)

    def is_alert(self, risk, p, amount) -> np.ndarray:
        """D159: an MVAF alert is any action from step-up up (not allow, not monitor)."""
        from vaultic.trust.decision import STEP_UP

        return self.decide(risk, p, amount) >= STEP_UP

    def context(self, df: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
        from vaultic.views.orchestrate import CONTEXT, context

        return context(df, features, self.encoder)[list(CONTEXT)].to_numpy(dtype=float)

    def score(self, frame: pd.DataFrame, context: np.ndarray, avail=None) -> dict:
        """Everything MVAF computes for rows: raw and calibrated views, risk (seed mean and
        per seed) and the mean gate weights."""
        raw = self.raw(frame, avail)
        p = self.calibrated(raw)
        risk, per_seed = self.fused(p, context)
        return {"raw": raw, "p": p, "risk": risk, "per_seed": per_seed,
                "weights": self.gate_weights(p, context)}  # fmt: skip

    # ---- B5 -----------------------------------------------------------------------------------
    def b5(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
        X = frame[self.b5_columns]
        per_seed = [m.predict_proba(X)[:, 1] for m in self.b5_models]
        return np.mean(per_seed, axis=0), per_seed

    # ---- anomaly ------------------------------------------------------------------------------
    def anomaly_scores(self, frame: pd.DataFrame) -> pd.DataFrame:
        """anomaly_if / anomaly_ae / anomaly_uid_if for rows (any features may be perturbed),
        from the models that scored the period after training; rows whose stored score is NaN
        (no model for them) must be handled by the caller."""
        X = frame[self.anomaly_columns].to_numpy(dtype=float)
        uid, n_past = frame["uid"].to_numpy(), frame["uid_n_past"].to_numpy()
        per_uid = getattr(self.anomaly, "per_uid", None)
        if per_uid is None:
            return self.anomaly.transform(X, uid, n_past)
        out = pd.DataFrame({"anomaly_if": self.anomaly.global_if.transform(X),
                            "anomaly_ae": self.anomaly.autoencoder.transform(X)})  # fmt: skip
        out["anomaly_uid_if"] = per_uid_scores(per_uid, X, uid, n_past)
        return out


def fit_system(parts: dict, table: pd.DataFrame | None = None, seeds=SEEDS,
               log=print) -> FrozenSystem:  # fmt: skip
    """Refit every component with the frozen settings (see the module docstring)."""
    import yaml

    from vaultic.eval.metrics import choose_cost_threshold
    from vaultic.features.categories import fit_on_training_period
    from vaultic.fusion.dev_compare import d95_views
    from vaultic.fusion.final_run import _params
    from vaultic.fusion.mvaf import view_inputs
    from vaultic.fusion.tune_fusion import concat_rows, fit_seeds
    from vaultic.paths import CONFIG_DIR
    from vaultic.trust.conformal import MondrianConformal, set_labels
    from vaultic.trust.decision import choose_thresholds
    from vaultic.trust.final_trust import fit_calibrator
    from vaultic.views.anomaly import AnomalyView
    from vaultic.views.definitions import LABEL_DERIVED
    from vaultic.views.orchestrate import VIEWS, calibrate_views
    from vaultic.views.plan import crossfit_plan
    from vaultic.views.tabular import make_model

    df, features, splits = parts["df"], parts["features"], parts["splits"]
    day, time = df["day"].to_numpy(), df["TransactionDT"].to_numpy()
    y = df["isFraud"].to_numpy()
    fixed = next(f for f in crossfit_plan(splits) if f.name == "fixed")
    train = fixed.train_rows(day, time)

    log("view models (fixed fold, days 1-120)")
    views = d95_views(parts["tab_cols"], parts["cols"])
    X_train = features.loc[train].reset_index(drop=True)
    for v in VIEWS:
        views[v].fit(X_train, y[train])

    log("per-view calibrators, gate and E12 layer (D95 view table)")
    table = pd.read_parquet(VIEW_TABLE) if table is None else table
    if (table["day"] >= splits.test.first).any():
        raise RuntimeError("the D95 view table holds test-period rows")
    table, _, cals = calibrate_views(table, return_calibrators=True)
    calibrators = {v: cals[("fixed", v)] for v in VIEWS if ("fixed", v) in cals}
    from vaultic.fusion.data import fusion_split

    split = fusion_split(table)
    both = concat_rows(split.fit, split.tune)
    tuned = json.loads(FUSION_PARAMS.read_text("utf-8"))["tuned"]
    gates = fit_seeds("MVAF", _params(tuned["MVAF"]["params"]), both, seeds)
    cal = split.calibrate
    p_cal = np.mean([g.predict_proba(cal.views, cal.context) for g in gates], axis=0)
    amount = df.set_index("TransactionID")["TransactionAmt"]
    a_cal = amount.reindex(cal.ids).to_numpy(float)
    fused_calibrator, _ = fit_calibrator(cal.y, p_cal)
    pc = fused_calibrator.predict(p_cal)
    conformal = MondrianConformal(alpha=0.10).fit(cal.y, pc)
    sets_cal = set_labels(conformal.predict_sets(pc))
    thresholds, _ = choose_thresholds(pc, a_cal, cal.y, sets_cal, view_inputs(cal.views)[3])

    log("B5 (EXP-009, seeds 0-4)")
    params = yaml.safe_load((CONFIG_DIR / "EXP-009.yaml").read_text("utf-8"))["model"]["params"]
    b5_columns = list(parts["b5_cols"])
    b5_models = [make_model("xgboost", params, s).fit(features.loc[train, b5_columns], y[train])
                 for s in seeds]  # fmt: skip
    pos = pd.Index(df["TransactionID"].to_numpy()).get_indexer(cal.ids)
    s_cal = np.mean([m.predict_proba(features.loc[pos, b5_columns])[:, 1] for m in b5_models], 0)
    b5_threshold = choose_cost_threshold(cal.y, s_cal, a_cal)

    log("anomaly models (last forward-chained cut, D68)")
    anomaly_columns = [c for c in b5_columns if c not in LABEL_DERIVED]
    fit = anomaly_fit_rows(day, splits.train.first, splits.train.last, ANOMALY["block_days"],
                           ANOMALY["max_fit_rows"], ANOMALY["seed"])  # fmt: skip
    anomaly = AnomalyView(seed=ANOMALY["seed"], max_iter=ANOMALY["ae_max_iter"]).fit(
        features.loc[fit, anomaly_columns].to_numpy(dtype=float), y[fit],
        features["uid"].to_numpy()[fit],
    )  # fmt: skip

    numeric = features.drop(columns=["uid"]).select_dtypes("number")
    medians = numeric.loc[train].median(skipna=True)
    return FrozenSystem(
        views=views, calibrators=calibrators, gates=gates, fused_calibrator=fused_calibrator,
        conformal=conformal, thresholds=thresholds, b5_models=b5_models, b5_columns=b5_columns,
        b5_threshold=b5_threshold, anomaly=anomaly, anomaly_columns=anomaly_columns,
        encoder=fit_on_training_period(df, splits, columns=("ProductCD",)), medians=medians,
        info={"gate_rows": int(len(both.y)), "calibrate_fused_rows": int(len(cal.y)),
              "anomaly_fit_rows": int(len(fit))},
    )  # fmt: skip


def check_validation(system: FrozenSystem, parts: dict, table: pd.DataFrame | None = None,
                     b5_stored: pd.Series | None = None) -> dict:  # fmt: skip
    """The refits against what was stored, on validation rows (fixed fold, days 128-150)."""
    from vaultic.views.orchestrate import CONTEXT, VIEWS, calibrate_views

    df, features = parts["df"], parts["features"]
    table = pd.read_parquet(VIEW_TABLE) if table is None else table
    table, _ = calibrate_views(table)
    t = table[table["fold"] == "fixed"]
    pos = pd.Index(df["TransactionID"].to_numpy()).get_indexer(t["TransactionID"].to_numpy())
    if (pos < 0).any():
        raise ValueError("view-table rows missing from the data")
    require_no_test(parts, df["day"].to_numpy()[pos])
    frame = features.loc[pos].reset_index(drop=True)
    raw = system.raw(frame)
    p = system.calibrated(raw)
    out = {"rows": int(len(t))}

    def diff(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            return np.inf  # availability differs
        ok = ~np.isnan(a)
        return float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0

    for j, v in enumerate(VIEWS):
        out[f"raw_{v}"] = diff(raw[:, j], t[f"raw_{v}"])
        out[f"p_{v}"] = diff(p[:, j], t[f"p_{v}"])
    ctx = system.context(df.loc[pos].reset_index(drop=True), frame)
    out["context"] = diff(ctx, t[list(CONTEXT)].to_numpy(float))

    sc = system.anomaly_scores(frame)
    for c in sc.columns:
        stored = frame[c].to_numpy(float)
        new = np.where(np.isnan(stored), np.nan, sc[c].to_numpy(float))  # NaN = no model
        out[c] = diff(new, stored)

    final = json.loads(FINAL_RESULTS.read_text("utf-8"))
    want = final["e12"]["MVAF"]["decision"]["default"]["thresholds"]
    got = {k: float(getattr(system.thresholds, k)) for k in want}
    out["thresholds_mvaf"] = max(
        0.0 if (np.isinf(want[k]) and np.isinf(got[k])) else abs(got[k] - want[k]) for k in want
    )
    out["threshold_b5"] = abs(system.b5_threshold - final["meta"]["thresholds"]["B5"])
    if b5_stored is not None:
        from vaultic.fusion.final_eval import refit_check

        val = df.loc[parts["splits"].validation.contains(df["day"].to_numpy())]
        s, _ = system.b5(features.loc[val.index].reset_index(drop=True))
        rc = refit_check(b5_stored, val["TransactionID"].to_numpy(), s)
        out["b5_validation"] = rc["max_abs_diff"]
        out["b5_validation_rows"] = rc["n_compared"]
        out["b5_validation_missing"] = rc["n_missing"]
        complete = refit_complete(rc, len(val))
    else:
        complete = True
    out["passed"] = complete and all(
        v <= TOLERANCE for k, v in out.items()
        if k not in ("rows", "b5_validation_rows", "b5_validation_missing")
    )  # fmt: skip
    return out


DEV_DIR = Path("E:/dev-cache/stage2")  # development outputs, on drive E, never in the repository


def dev_system_path() -> Path:
    return DEV_DIR / "system_dev.joblib"


def main() -> None:
    """Development refit and validation check: python -m vaultic.stage2.system"""
    import time

    import joblib
    import yaml

    from vaultic.fusion.final_run import CONFIG, _stored

    started = time.perf_counter()
    cfg = yaml.safe_load(CONFIG.read_text("utf-8"))
    parts = load_parts(dev=True)
    system = fit_system(parts, log=lambda m: print(f"[{time.perf_counter() - started:6.0f}s] {m}",
                                                   flush=True))  # fmt: skip
    report = check_validation(system, parts, b5_stored=_stored(cfg["b5_run"], "validation"))
    report["seconds"] = round(time.perf_counter() - started)
    out = dev_system_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "system_check.json").write_text(json.dumps(report, indent=1), "utf-8")
    print(json.dumps(report, indent=1), flush=True)
    joblib.dump(system, out)
    if not report["passed"]:
        raise SystemExit("the refitted system does not reproduce the stored validation scores")


if __name__ == "__main__":
    # run the importable module's main, so pickled objects name vaultic.stage2.system, not
    # __main__ (other processes load them)
    from vaultic.stage2 import system as _system

    _system.main()


def check_test(system: FrozenSystem, parts: dict, stored: pd.DataFrame) -> dict:
    """Final runs only: the refitted MVAF and B5 on the test rows against the stored
    EXP-200-final scores, and the rebuilt E12 decisions against E12's default-cost total."""
    from vaultic.trust.decision import total_cost

    if parts["dev"]:
        raise RuntimeError("check_test reads the test period; final runs only")
    df, features = parts["df"], parts["features"]
    pos = np.flatnonzero(df["day"].to_numpy() >= parts["splits"].test.first)
    frame = features.loc[pos].reset_index(drop=True)
    rows = df.loc[pos].reset_index(drop=True)
    ids = rows["TransactionID"].to_numpy()
    sc = system.score(frame, system.context(rows, frame))
    b5, _ = system.b5(frame)
    amount, y = rows["TransactionAmt"].to_numpy(float), rows["isFraud"].to_numpy()
    cost = total_cost(system.decide(sc["risk"], sc["p"], amount), y, amount)
    final = json.loads(FINAL_RESULTS.read_text("utf-8"))
    want = final["e12"]["MVAF"]["decision"]["default"]["evaluation_cost"]
    from vaultic.fusion.final_eval import refit_check

    rc = {"mvaf": refit_check(stored["score_MVAF"], ids, sc["risk"]),
          "b5": refit_check(stored["score_B5"], ids, b5)}  # fmt: skip
    out = {"rows": int(len(pos)), "mvaf": rc["mvaf"]["max_abs_diff"], "b5": rc["b5"]["max_abs_diff"],
           "mvaf_missing": rc["mvaf"]["n_missing"], "b5_missing": rc["b5"]["n_missing"],
           "e12_cost": abs(cost - want)}  # fmt: skip
    out["passed"] = bool(all(refit_complete(r, len(ids)) for r in rc.values())
                         and out["e12_cost"] <= 0.01)  # fmt: skip
    return out
