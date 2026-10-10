"""E12 (D101, D102): the trust layer of one method, fitted on calibration rows and applied once to
evaluation rows. Pure functions over arrays; used unchanged by fusion/final_run.py.

  calibration   fused calibrator chosen like the per-view ones (choose_calibrator by out-of-fold
                ECE with >= 100 frauds, Platt otherwise, D83); ECE (15 bins) and Brier before /
                after; reliability tables
  conformal     Mondrian conformal at each target (90%, 95%) on calibrated scores: coverage
                overall and per class, share of uncertain sets
  adaptive      adaptive conformal (D55) with the label delay in days, Mondrian, fitted on the
                calibration rows and run over the evaluation days in order; coverage per 7-day
                block
  decision      thresholds chosen on calibration rows (calibrated p, the 90% Mondrian sets,
                view disagreement where the method has it), total cost on evaluation rows with
                the default cost model and for every setting of the D46 sensitivity sweep
                (thresholds re-chosen on calibration rows for each setting)
  routing       R1-R4 at K reviews per day; lambda_u / lambda_d for R3 / R4 chosen on calibration
                rows per K (grid) by fraud value caught; uncertainty u = 1 for an uncertain 90%
                set; R4 only for methods with view disagreement. H5 (R3 / R4 catch more fraud
                value than R1 at equal K) with a day-block bootstrap of the difference
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from vaultic.trust.calibration import (
    PlattCalibrator,
    calibration_report,
    choose_calibrator,
    reliability_table,
)
from vaultic.trust.conformal import (
    AdaptiveConformal,
    MondrianConformal,
    coverage,
    coverage_by_block,
    set_labels,
)
from vaultic.trust.decision import DEFAULT_COST, CostModel, choose_thresholds, decide, total_cost
from vaultic.trust.routing import evaluate_policy

MIN_ISOTONIC_FRAUDS = 100
TARGETS = (0.90, 0.95)
KS = (50, 100, 200, 500)
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
GATE_TOLERANCE = 0.02


def fit_calibrator(y, p):
    y = np.asarray(y).astype(int)
    if y.sum() >= MIN_ISOTONIC_FRAUDS:
        cal, ece = choose_calibrator(y, p)
        return cal, {"method": type(cal).__name__, "fallback": False, "ece_cv": ece}
    return PlattCalibrator().fit(p, y), {"method": "PlattCalibrator", "fallback": True}


def conformal_section(y_cal, p_cal, y_ev, p_ev, targets=TARGETS) -> tuple[dict, dict]:
    """Mondrian coverage per target, and the 90% evaluation sets (for decisions / routing)."""
    out, sets90 = {}, None
    for t in targets:
        cp = MondrianConformal(alpha=1 - t).fit(y_cal, p_cal)
        sets = cp.predict_sets(p_ev)
        cov = coverage(y_ev, sets)
        cov["target"] = t
        cov["within_tolerance"] = bool(abs(cov["coverage"] - t) <= GATE_TOLERANCE)
        out[f"{t:.2f}"] = cov
        if abs(t - 0.90) < 1e-9:
            sets90 = (cp, sets)
    return out, sets90


def adaptive_section(y_cal, p_cal, day_ev, y_ev, p_ev, label_delay_days: int,
                     targets=TARGETS, gamma: float = 0.05, block_days: int = 7) -> dict:  # fmt: skip
    order = np.argsort(day_ev, kind="stable")
    out = {}
    for t in targets:
        ac = AdaptiveConformal(alpha=1 - t, gamma=gamma, mondrian=True,
                               label_delay_days=label_delay_days).fit(y_cal, p_cal)  # fmt: skip
        sets, trace = ac.run(day_ev[order], y_ev[order], p_ev[order])
        blocks = coverage_by_block(day_ev[order], y_ev[order], sets, block_days=block_days)
        out[f"{t:.2f}"] = {"overall": coverage(y_ev[order], sets),
                           "blocks": blocks.to_dict("records"), "gamma": gamma,
                           "label_delay_days": label_delay_days}  # fmt: skip
    return out


def decision_section(y_cal, p_cal, amt_cal, sets_cal, d_cal, y_ev, p_ev, amt_ev, sets_ev, d_ev,
                     sweep: bool = True) -> dict:  # fmt: skip
    th, info = choose_thresholds(p_cal, amt_cal, y_cal, sets_cal, d_cal)
    action = decide(p_ev, amt_ev, th, sets_ev, d_ev)
    out = {"default": {"thresholds": asdict(th), "calibration_cost": info["validation_cost"],
                       "evaluation_cost": total_cost(action, y_ev, amt_ev),
                       "action_shares": {a: float((action == i).mean()) for i, a in
                                         enumerate(("allow", "monitor", "step_up", "hold", "block"))}}}  # fmt: skip
    if sweep:
        rows = []
        for c_fp in (2, 5, 10, 20, 50):
            for rate in (0.7, 0.9, 1.0):
                for friction in (0.0, 1.0, 5.0):
                    for c_step in (0.10, 0.50, 2.00):
                        cost = CostModel(c_fp, DEFAULT_COST.c_rev, c_step, rate, friction)
                        t2, _ = choose_thresholds(p_cal, amt_cal, y_cal, sets_cal, d_cal, cost=cost)
                        a2 = decide(p_ev, amt_ev, t2, sets_ev, d_ev)
                        rows.append({"c_fp": c_fp, "step_up_success_rate": rate,
                                     "legit_step_up_friction": friction, "c_step": c_step,
                                     "evaluation_cost": total_cost(a2, y_ev, amt_ev, cost)})  # fmt: skip
        out["sensitivity"] = rows
    return out


def _uncertain(sets) -> np.ndarray:
    return (set_labels(sets) == "uncertain").astype(float)


def routing_section(day_cal, y_cal, amt_cal, p_cal, u_cal, d_cal, day_ev, y_ev, amt_ev, p_ev,
                    u_ev, d_ev, ks=KS, grid=LAMBDA_GRID, n_boot: int = 1000, seed: int = 0) -> dict:  # fmt: skip
    policies = ["R1", "R2", "R3"] + (["R4"] if d_ev is not None else [])
    rows, lambdas, h5 = [], {}, []
    days = np.unique(day_ev)
    rng = np.random.default_rng(seed)
    boot_days = [rng.choice(days, len(days), replace=True) for _ in range(n_boot)]
    for k in ks:
        best = {}
        for pol in ("R3", "R4"):
            if pol == "R4" and d_ev is None:
                continue
            cands = [(lu, ld) for lu in grid for ld in (grid if pol == "R4" else (0.0,))]
            vals = [evaluate_policy(day_cal, y_cal, amt_cal, p_cal, pol, k, u_cal, d_cal, lu, ld)
                    ["fraud_value_caught"] for lu, ld in cands]  # fmt: skip
            best[pol] = cands[int(np.argmax(vals))]
        lambdas[k] = best
        caught_by_day = {}
        for pol in policies:
            lu, ld = best.get(pol, (1.0, 1.0))
            r = evaluate_policy(day_ev, y_ev, amt_ev, p_ev, pol, k, u_ev, d_ev, lu, ld)
            rows.append({**r, "lambda_u": lu, "lambda_d": ld})
            caught_by_day[pol] = _value_by_day(
                day_ev, y_ev, amt_ev, p_ev, pol, k, u_ev, d_ev, lu, ld
            )
        for pol in [p for p in ("R3", "R4") if p in policies]:
            diff_day = caught_by_day[pol] - caught_by_day["R1"]
            d = np.array([diff_day.reindex(b).sum() for b in boot_days])
            p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
            h5.append({"K": k, "comparison": f"{pol} - R1", "fraud_value_diff": float(diff_day.sum()),
                       "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
                       "p_value": p})  # fmt: skip
    return {"policies": rows, "lambdas": {str(k): v for k, v in lambdas.items()}, "h5": h5}


def _value_by_day(day, y, amount, p, policy, k, u, d, lu, ld) -> pd.Series:
    from vaultic.trust.routing import priority, select_daily

    reviewed = select_daily(day, priority(policy, p, amount, u, d, lu, ld), k)
    caught = reviewed & (np.asarray(y) == 1)
    return pd.Series(np.where(caught, amount, 0.0)).groupby(np.asarray(day)).sum()


def trust_report(cal: dict, ev: dict, label_delay_days: int, sweep: bool = True,
                 n_boot: int = 1000) -> dict:  # fmt: skip
    """cal / ev: dicts with y, p (the method's fused score), amount, day and d (view
    disagreement or None)."""
    calibrator, cal_info = fit_calibrator(cal["y"], cal["p"])
    pc, pe = calibrator.predict(cal["p"]), calibrator.predict(ev["p"])
    out = {"calibrator": cal_info,
           "calibration": calibration_report(ev["y"], ev["p"], pe),
           "reliability_before": reliability_table(ev["y"], ev["p"]).to_dict("records"),
           "reliability_after": reliability_table(ev["y"], pe).to_dict("records")}  # fmt: skip
    conf, (cp90, sets_ev) = conformal_section(cal["y"], pc, ev["y"], pe)
    sets_cal = cp90.predict_sets(pc)
    out["conformal"] = conf
    out["adaptive"] = adaptive_section(cal["y"], pc, ev["day"], ev["y"], pe, label_delay_days)
    out["decision"] = decision_section(cal["y"], pc, cal["amount"], set_labels(sets_cal), cal["d"],
                                       ev["y"], pe, ev["amount"], set_labels(sets_ev), ev["d"], sweep)  # fmt: skip
    out["routing"] = routing_section(cal["day"], cal["y"], cal["amount"], pc, _uncertain(sets_cal),
                                     cal["d"], ev["day"], ev["y"], ev["amount"], pe,
                                     _uncertain(sets_ev), ev["d"], n_boot=n_boot)  # fmt: skip
    return out


def phase8_gate(reports: dict[str, dict], tolerance: float = GATE_TOLERANCE) -> dict:
    """Mondrian coverage within `tolerance` of the target on the evaluation rows, per method
    and target (roadmap Phase 8 exit gate)."""
    rows = {m: {t: bool(abs(r["conformal"][t]["coverage"] - float(t)) <= tolerance)
                for t in r["conformal"]} for m, r in reports.items()}  # fmt: skip
    return {"per_method": rows, "passed": all(all(v.values()) for v in rows.values())}
