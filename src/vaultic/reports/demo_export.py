"""Demo bundle of the EXP-200-final test period for the dashboard backend (services/api on the
dashboard branch). Not an experiment and not a reported result.

Run:  python -m vaultic.reports.demo_export     (needs the inputs cached by fusion/f3_diagnosis)

Rebuilds, from the cached final-run fusion inputs (D117), what the final run computed for MVAF
but did not save per row, and checks it against what it did save:
  - MVAF refitted exactly as in the final run (D95-tuned settings, seeds 0-4, gate rows); its
    test scores must equal the stored score_MVAF of run 20261011-002400-202994 (else it stops)
  - gate weights per view (mean over seeds) and the signed view contributions (explain/
    view_contrib.py), turned into view-level reasons (points as in reason_codes, PDO 20)
  - the E12 trust layer for MVAF: fused calibrator, 90% Mondrian conformal sets, decision
    thresholds chosen on calibrate_fused, actions on the test rows; the default-cost total must
    equal E12's (else it stops)
plus amount, product, entities and the uid from the data files. Writes
<data>/interim/demo_final_bundle.parquet, demo_final_bundle.json and demo_final_calibration.parquet
(the calibrate_fused rows, for the what-if cost mode; data stay on drive E, not committed). Every row is labelled "IEEE-CIS test period, final run".
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd

from vaultic.paths import INTERIM_DIR, MERGED_PATH, RESEARCH_DIR

BUNDLE = INTERIM_DIR / "demo_final_bundle.parquet"
META = INTERIM_DIR / "demo_final_bundle.json"
CALIBRATION = INTERIM_DIR / "demo_final_calibration.parquet"  # for the what-if cost mode
SOURCE_LABEL = "IEEE-CIS test period, final run"
ACTIONS = ("allow", "monitor", "step_up", "hold", "block")


def lab_runs(final: dict, stored: pd.DataFrame, ids, y) -> list[dict]:
    """Model-lab entries for every E10 method: the final-run metrics and, from the stored test
    scores, a PR curve (50 points) and a 10-bin calibration table of the raw scores."""
    from sklearn.metrics import precision_recall_curve

    runs = []
    for m in final["e10_methods"]:
        s = stored[f"score_{m['method']}"].reindex(ids).to_numpy(float)
        prec, rec, _ = precision_recall_curve(y, s)
        keep = np.unique(np.linspace(0, len(rec) - 1, 50).astype(int))
        bins = np.minimum((np.clip(s, 0, 1) * 10).astype(int), 9)
        cal = [{"predicted": float(s[bins == b].mean()), "observed": float(y[bins == b].mean()),
                "count": int((bins == b).sum())} for b in range(10) if (bins == b).any()]  # fmt: skip
        runs.append({"id": m["method"], "name": m["method"],
                     "kind": "baseline" if m["method"] in ("B5", "F0") else "fusion",
                     "prAuc": m["pr_auc"], "ciLow": m["pr_auc_ci_low"], "ciHigh": m["pr_auc_ci_high"],
                     "recallAt1": m["recall_at_1pct_fpr"], "cost": m["cost"],
                     "pr": [{"recall": float(rec[i]), "precision": float(prec[i])} for i in keep],
                     "calibration": cal})  # fmt: skip
    return runs


def main() -> None:
    from vaultic.data.splits import load_splits
    from vaultic.data.uid import UID_PATH
    from vaultic.explain.reason_codes import to_points
    from vaultic.explain.view_contrib import view_contributions
    from vaultic.fusion.f3_diagnosis import CACHE, CONTEXT, FINAL_RUN, VIEWS
    from vaultic.fusion.final_run import _params
    from vaultic.fusion.mvaf import view_inputs
    from vaultic.fusion.tune_fusion import fit_seeds
    from vaultic.trust.conformal import MondrianConformal, set_labels
    from vaultic.trust.decision import DEFAULT_COST, choose_thresholds, decide, total_cost
    from vaultic.trust.final_trust import fit_calibrator

    d = dict(np.load(CACHE))
    tuned = json.loads((RESEARCH_DIR / "tables" / "fusion_d95.json").read_text("utf-8"))["tuned"]

    class _Rows:
        views, context, y = d["gate_views"], d["gate_context"], d["gate_y"]

    models = fit_seeds("MVAF", _params(tuned["MVAF"]["params"]), _Rows, (0, 1, 2, 3, 4))
    tv, tc, ids = d["test_views"], d["test_context"], d["test_ids"]
    p_test = np.mean([m.predict_proba(tv, tc) for m in models], axis=0)
    p_cal = np.mean([m.predict_proba(d["cal_views"], d["cal_context"]) for m in models], axis=0)
    stored = pd.read_parquet(FINAL_RUN / "predictions.parquet").set_index("TransactionID")
    ref = stored["score_MVAF"].reindex(ids).to_numpy()
    diff = float(np.max(np.abs(ref - p_test)))
    print(f"MVAF reproduction: max |diff| {diff:.2e}", flush=True)
    if diff > 1e-6:
        raise RuntimeError("the refitted MVAF does not reproduce the stored final-run scores")
    weights = np.mean([m.view_weights(tv, tc) for m in models], axis=0)
    vc = view_contributions(weights, tv, list(VIEWS), p_test)

    merged = pd.read_parquet(MERGED_PATH).set_index("TransactionID")
    variant = load_splits().uid_variant
    uid = pd.read_parquet(UID_PATH, columns=["TransactionID", variant])
    uid = uid.set_index("TransactionID")[variant]
    amt_test = merged["TransactionAmt"].reindex(ids).to_numpy(float)
    amt_cal = merged["TransactionAmt"].reindex(d["cal_ids"]).to_numpy(float)

    # E12 trust layer for MVAF, as in trust/final_trust.trust_report
    calibrator, cal_info = fit_calibrator(d["cal_y"], p_cal)
    pc, pe = calibrator.predict(p_cal), calibrator.predict(p_test)
    cp = MondrianConformal(alpha=0.10).fit(d["cal_y"], pc)
    sets_cal, sets_test = set_labels(cp.predict_sets(pc)), set_labels(cp.predict_sets(pe))
    dis_cal = view_inputs(d["cal_views"])[3]
    _, _, conf_test, dis_test = view_inputs(tv)
    th, _ = choose_thresholds(pc, amt_cal, d["cal_y"], sets_cal, dis_cal)
    action = decide(pe, amt_test, th, sets_test, dis_test)
    cost = total_cost(action, d["test_y"], amt_test)
    e12 = json.loads((RESEARCH_DIR / "tables" / "final_results.json").read_text("utf-8"))["e12"]
    want = e12["MVAF"]["decision"]["default"]["evaluation_cost"]
    print(f"E12 default cost: rebuilt {cost:,.2f}, final run {want:,.2f}", flush=True)
    if abs(cost - want) > 0.01:
        raise RuntimeError("the rebuilt decisions do not reproduce E12's default cost")

    m = merged.reindex(ids)

    def ent(col):
        return [None if pd.isna(x) else str(x) for x in m[col]] if col in m else [None] * len(m)

    card = (
        m[["card1", "card2", "card3", "card4", "card5", "card6"]].astype(str).agg("-".join, axis=1)
    )
    reasons = []
    for i in range(len(ids)):
        row = []
        for j, v in enumerate(VIEWS):
            c = vc.contribution[i, j]
            if np.isfinite(c):
                effect = "raises" if c > 0 else "lowers"
                row.append({"feature": f"view:{v}", "view": v, "points": float(to_points(c)),
                            "text": f"{v.capitalize()} view {effect} risk (gate weight "
                                    f"{weights[i, j]:.2f}, view probability {tv[i, j]:.3f})"})  # fmt: skip
        reasons.append(json.dumps(sorted(row, key=lambda r: -abs(r["points"]))))
    out = pd.DataFrame({
        "TransactionID": ids, "TransactionDT": m["TransactionDT"].to_numpy(), "day": d["test_day"],
        "label": d["test_y"], "amount": amt_test, "product": m["ProductCD"].astype(str).to_numpy(),
        "uid": uid.reindex(ids).astype(str).to_numpy(), "card": card.to_numpy(),
        "email": ent("P_emaildomain"), "address": ent("addr1"), "device": ent("DeviceInfo"),
        "p_raw": p_test, "p": pe, "action": [ACTIONS[a] for a in action],
        "conformal_set": sets_test, "disagreement": dis_test, "reasons": reasons,
        **{f"p_{v}": tv[:, j] for j, v in enumerate(VIEWS)},
        **{f"w_{v}": weights[:, j] for j, v in enumerate(VIEWS)},
        **{f"c_{v}": conf_test[:, j] for j, v in enumerate(VIEWS)},
        **{c: tc[:, k] for k, c in enumerate(CONTEXT)},
        "source": SOURCE_LABEL,
    })  # fmt: skip
    out.sort_values("TransactionDT").to_parquet(BUNDLE, index=False)
    # calibrate_fused rows (147-150): the demo's what-if cost mode re-chooses thresholds on them
    pd.DataFrame({"p": pc, "amount": amt_cal, "conformal_set": sets_cal,
                  "disagreement": dis_cal, "label": d["cal_y"]}).to_parquet(CALIBRATION, index=False)  # fmt: skip
    final = json.loads((RESEARCH_DIR / "tables" / "final_results.json").read_text("utf-8"))
    META.write_text(json.dumps({
        "source": SOURCE_LABEL, "run": "experiments/runs/EXP-200-final/20261011-002400-202994",
        "generated": date.today().isoformat(), "rows": int(len(out)),
        "label_delay_days": 30, "thresholds": asdict(th), "cost_model": asdict(DEFAULT_COST),
        "calibrator": cal_info, "mvaf_reproduction_max_abs_diff": diff,
        "e12_default_cost": cost, "e10_methods": final["e10_methods"],
        "lab": lab_runs(final, stored, ids, d["test_y"]),
        "routing_lambdas": e12["MVAF"]["routing"]["lambdas"],
        "sensitivity": [{"c_fp": r["c_fp"], "step_up_success_rate": r["step_up_success_rate"],
                         "cost": r["evaluation_cost"]}
                        for r in e12["MVAF"]["decision"]["sensitivity"]
                        if r["legit_step_up_friction"] == DEFAULT_COST.legit_step_up_friction
                        and r["c_step"] == DEFAULT_COST.c_step],
        "reasons_note": "view-level reasons from MVAF's gate (weight x view logit), not "
                        "feature-level SHAP",
    }, indent=1, default=str), encoding="utf-8")  # fmt: skip
    print(f"wrote {BUNDLE} ({len(out)} rows) and {META}")


if __name__ == "__main__":
    main()
