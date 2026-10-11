"""POST-HOC / EXPLORATORY: Block decisions when a Hold can fail (D124; motivated by D123).

Run:  python -m vaultic.trust.hold_sensitivity     (needs the demo bundle from reports/demo_export)

In the D43 cost model a Hold costs one review and always stops the fraud, so it weakly dominates
Block and Block is never chosen (D123). Here a held fraud is stopped only with probability
hold_success_rate (CostModel, default 1.0 = D43). For hold_success_rate in {1.0, 0.95, 0.9, 0.8,
0.7} crossed with the D46 grid (135 settings), MVAF's decision thresholds are re-chosen on the
calibrate_fused rows (days 147-150) and applied to the test period, using the verified
reconstruction of the final run's MVAF inputs (calibrated scores, 90% Mondrian sets, view
disagreement; reports/demo_export.py checked them against the stored scores and E12's cost).
Reported per hold_success_rate: share of settings where Block becomes reachable (finite
t_block), share of test transactions blocked, and the test total cost against the same setting
with hold_success_rate = 1.0 (both evaluated under the setting's own cost model). No
pre-registered number changes. Writes research/hold_sensitivity.md and
research/tables/hold_sensitivity.json.
"""

from __future__ import annotations

import json
from datetime import date
from itertools import product

import numpy as np
import pandas as pd

from vaultic.paths import INTERIM_DIR, RESEARCH_DIR

BUNDLE = INTERIM_DIR / "demo_final_bundle.parquet"
CALIBRATION = INTERIM_DIR / "demo_final_calibration.parquet"
OUT_MD = RESEARCH_DIR / "hold_sensitivity.md"
OUT_JSON = RESEARCH_DIR / "tables" / "hold_sensitivity.json"
HOLD_RATES = (1.0, 0.95, 0.9, 0.8, 0.7)
GRID = {"c_fp": (2, 5, 10, 20, 50), "step_up_success_rate": (0.7, 0.9, 1.0),
        "legit_step_up_friction": (0.0, 1.0, 5.0), "c_step": (0.10, 0.50, 2.00)}  # fmt: skip
E12_DEFAULT_COST = None  # read from research/tables/final_results.json


def sweep(cal: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    from vaultic.trust.decision import (
        BLOCK,
        DEFAULT_COST,
        CostModel,
        choose_thresholds,
        decide,
        total_cost,
    )

    rows = []
    for h, c_fp, rate, friction, c_step in product(HOLD_RATES, GRID["c_fp"], GRID["step_up_success_rate"],
                                                   GRID["legit_step_up_friction"], GRID["c_step"]):  # fmt: skip
        cost = CostModel(c_fp, DEFAULT_COST.c_rev, c_step, rate, friction, hold_success_rate=h)
        th, _ = choose_thresholds(cal["p"].to_numpy(), cal["amount"].to_numpy(), cal["label"].to_numpy(),
                                  cal["conformal_set"].to_numpy(object), cal["disagreement"].to_numpy(),
                                  cost=cost)  # fmt: skip
        action = decide(test["p"].to_numpy(), test["amount"].to_numpy(), th,
                        test["conformal_set"].to_numpy(object), test["disagreement"].to_numpy())  # fmt: skip
        blocked = action == BLOCK
        rows.append({"hold_success_rate": h, "c_fp": c_fp, "step_up_success_rate": rate,
                     "legit_step_up_friction": friction, "c_step": c_step,
                     "block_reachable": bool(np.isfinite(th.t_block)),
                     "t_block": float(th.t_block), "share_blocked": float(blocked.mean()),
                     "n_blocked": int(blocked.sum()),
                     "frauds_blocked": int((blocked & (test["label"].to_numpy() == 1)).sum()),
                     "test_cost": total_cost(action, test["label"].to_numpy(), test["amount"].to_numpy(), cost),
                     "test_cost_if_never_block": None})  # fmt: skip
    # the same setting with Block switched off, to price what Block adds under each model
    return rows


def main() -> None:
    from vaultic.trust.decision import (
        DEFAULT_COST,
        CostModel,
        Thresholds,
        choose_thresholds,
        decide,
        total_cost,
    )

    cal = pd.read_parquet(CALIBRATION)
    test = pd.read_parquet(BUNDLE)
    final = json.loads((RESEARCH_DIR / "tables" / "final_results.json").read_text("utf-8"))
    want = final["e12"]["MVAF"]["decision"]["default"]["evaluation_cost"]
    th, _ = choose_thresholds(cal["p"].to_numpy(), cal["amount"].to_numpy(), cal["label"].to_numpy(),
                              cal["conformal_set"].to_numpy(object), cal["disagreement"].to_numpy())  # fmt: skip
    got = total_cost(decide(test["p"].to_numpy(), test["amount"].to_numpy(), th,
                            test["conformal_set"].to_numpy(object), test["disagreement"].to_numpy()),
                     test["label"].to_numpy(), test["amount"].to_numpy())  # fmt: skip
    print(f"reproduction: E12 default cost {want:,.2f}, rebuilt {got:,.2f}", flush=True)
    if abs(got - want) > 0.01:
        raise RuntimeError("the inputs do not reproduce E12's default cost")

    rows = sweep(cal, test)
    base = {(r["c_fp"], r["step_up_success_rate"], r["legit_step_up_friction"], r["c_step"]): r
            for r in rows if r["hold_success_rate"] == 1.0}  # fmt: skip
    # cost under each hold_success_rate if Block were forbidden (thresholds re-chosen without it)
    for r in rows:
        key = (r["c_fp"], r["step_up_success_rate"], r["legit_step_up_friction"], r["c_step"])
        r["cost_change_vs_hold_1"] = r["test_cost"] - base[key]["test_cost"]
        if r["block_reachable"]:
            cost = CostModel(r["c_fp"], DEFAULT_COST.c_rev, r["c_step"], r["step_up_success_rate"],
                             r["legit_step_up_friction"], hold_success_rate=r["hold_success_rate"])  # fmt: skip
            t2, _ = choose_thresholds(cal["p"].to_numpy(), cal["amount"].to_numpy(), cal["label"].to_numpy(),
                                      cal["conformal_set"].to_numpy(object), cal["disagreement"].to_numpy(),
                                      cost=cost)  # fmt: skip
            no_block = Thresholds(t2.t_monitor, t2.t_step, t2.t_hold, float("inf"), t2.t_dis)
            a = decide(test["p"].to_numpy(), test["amount"].to_numpy(), no_block,
                       test["conformal_set"].to_numpy(object), test["disagreement"].to_numpy())  # fmt: skip
            r["test_cost_if_never_block"] = total_cost(
                a, test["label"].to_numpy(), test["amount"].to_numpy(), cost
            )
        else:
            r["test_cost_if_never_block"] = r["test_cost"]

    summary = []
    for h in HOLD_RATES:
        sub = [r for r in rows if r["hold_success_rate"] == h]
        reach = [r for r in sub if r["block_reachable"]]
        summary.append({"hold_success_rate": h, "settings": len(sub),
                        "settings_block_reachable": len(reach),
                        "mean_share_blocked": float(np.mean([r["share_blocked"] for r in sub])),
                        "max_share_blocked": float(np.max([r["share_blocked"] for r in sub])),
                        "mean_cost_change_vs_hold_1": float(np.mean([r["cost_change_vs_hold_1"] for r in sub])),
                        "mean_saving_from_block": float(np.mean([r["test_cost_if_never_block"] - r["test_cost"]
                                                                 for r in sub]))})  # fmt: skip
    default = [r for r in rows if r["c_fp"] == DEFAULT_COST.c_fp and r["c_step"] == DEFAULT_COST.c_step
               and r["step_up_success_rate"] == DEFAULT_COST.step_up_success_rate
               and r["legit_step_up_friction"] == DEFAULT_COST.legit_step_up_friction]  # fmt: skip
    result = {"generated": date.today().isoformat(), "post_hoc": True, "e12_default_cost": want,
              "rows": int(len(test)), "frauds": int(test["label"].sum()), "summary": summary,
              "default_costs": default, "grid": rows}  # fmt: skip
    OUT_JSON.write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_md(result)
    print(f"wrote {OUT_MD} and {OUT_JSON}")


def write_md(r: dict) -> None:
    lines = ["# Block decisions when a Hold can fail (POST-HOC / EXPLORATORY)", "",
             "**POST-HOC / EXPLORATORY (D124), motivated by D123.** Not pre-registered; E12 and every "
             "pre-registered number stand as reported. In the D43 cost model a Hold always stops the "
             "fraud, so it weakly dominates Block and Block is never chosen. Here a held fraud is "
             "stopped only with probability hold_success_rate (1.0 = D43). "
             f"Generated {r['generated']} by `python -m vaultic.trust.hold_sensitivity`.", "",
             f"MVAF on the test period ({r['rows']:,} rows, {r['frauds']:,} frauds); thresholds re-chosen "
             "on the calibration days (147-150) for each of the 135 D46 cost settings and each "
             f"hold_success_rate. Inputs reproduce E12's default cost (${r['e12_default_cost']:,.2f}). "
             "Cost change: test total cost minus the same setting with hold_success_rate = 1.0, each "
             "under its own cost model (a leakier Hold makes every held fraud dearer, so costs rise). "
             "Saving from Block: test cost with Block forbidden (thresholds otherwise the same) minus "
             "the cost with the chosen thresholds.", "",
             "| hold_success_rate | settings where Block is reachable | mean share of test rows blocked | max share blocked | mean cost change vs 1.0 ($) | mean saving from Block ($) |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for s in r["summary"]:
        lines.append(f"| {s['hold_success_rate']:.2f} | {s['settings_block_reachable']} of {s['settings']} | "
                     f"{s['mean_share_blocked']:.2%} | {s['max_share_blocked']:.2%} | "
                     f"{s['mean_cost_change_vs_hold_1']:+,.0f} | {s['mean_saving_from_block']:,.0f} |")  # fmt: skip
    lines += ["", "## Default costs (C_FP $10, review $5, OTP $0.50, step-up 90%, friction $1)", "",
              "| hold_success_rate | Block reachable | t_block (expected loss, $) | test rows blocked | frauds among them | test cost ($) | change vs 1.0 ($) |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for d in r["default_costs"]:
        tb = "never" if not np.isfinite(d["t_block"]) else f"{d['t_block']:,.2f}"
        lines.append(f"| {d['hold_success_rate']:.2f} | {'yes' if d['block_reachable'] else 'no'} | {tb} | "
                     f"{d['n_blocked']:,} | {d['frauds_blocked']:,} | {d['test_cost']:,.0f} | "
                     f"{d['cost_change_vs_hold_1']:+,.0f} |")  # fmt: skip
    lines += ["", "## Reading (added by hand, not generated)", "", "_To be written._", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
