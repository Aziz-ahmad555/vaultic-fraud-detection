"""Written report of the feedback-loop final run EXP-310-final (D210, D218).

Run:  python -m vaultic.reports.feedback_report [--final-json ...] [--run ...] [--dev-run ...]

Every number comes from the committed final results (research/tables/feedback/exp310_final.json,
written by vaultic.learning.feedback_final), the final run folder's per-delay events.json and the
design run's metrics.json / events.json; nothing is recomputed from data. Writes
research/tables/feedback/exp310_report.md and exp310_gain_by_delay.json (chart data: P1 minus P0
PR-AUC by label delay, final and design run).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from vaultic.paths import REPO_ROOT, RESEARCH_DIR

OUT_DIR = RESEARCH_DIR / "tables" / "feedback"
FINAL_JSON = OUT_DIR / "exp310_final.json"
FINAL_RUN = REPO_ROOT / "experiments" / "runs" / "EXP-310-final" / "20261020-174032-191246"
REFUSED_RUN = "20261020-171045-498585"
DEV_RUN = REPO_ROOT / "experiments" / "runs" / "EXP-310-dev" / "20261020-004133"
DELAYS = (0, 7, 30, 60)
GATED = ("P1", "P3", "P1_monthly")
REASONS = ("accepted", "PR-AUC gain below the margin or CI not above 0", "cost higher",
           "insufficient evidence")  # fmt: skip


def f(x: float, nd: int = 4) -> str:
    return f"{x:+.{nd}f}"


def ci(d: dict, nd: int = 4) -> str:
    return f"{d['ci_low']:+.{nd}f} to {d['ci_high']:+.{nd}f}"


def event_summary(events: list[dict]) -> dict:
    """Per policy: attempts, outcome counts by reason, alarms; plus attempts whose gate window
    overlaps the champion's threshold slice (D214)."""
    out: dict = {}
    for e in events:
        pol = out.setdefault(e["policy"], {"attempts": 0, "reasons": Counter(), "alarms": 0,
                                           "threshold_overlap": 0, "accepted_with_overlap": 0})  # fmt: skip
        if e.get("kind") == "alarm":
            pol["alarms"] += 1
        elif e.get("kind") == "attempt":
            pol["attempts"] += 1
            pol["reasons"][e.get("reason")] += 1
            if e.get("champion_threshold_rows_in_gate", 0) > 0:
                pol["threshold_overlap"] += 1
                pol["accepted_with_overlap"] += bool(e.get("accepted"))
    return out


def same_days(events: list[dict], a: str = "P3", b: str = "P1") -> int:
    """Seeds where policy a attempted on exactly the days policy b did."""
    seeds = sorted({e["seed"] for e in events})
    days = {(p, s): [e["day"] for e in events if e["policy"] == p and e["seed"] == s
                     and e.get("kind") == "attempt"] for p in (a, b) for s in seeds}  # fmt: skip
    return sum(days[(a, s)] == days[(b, s)] for s in seeds)


def alarm_days(events: list[dict], policy: str) -> str:
    return ", ".join(str(d) for d in sorted({e["day"] for e in events
                                             if e["policy"] == policy and e.get("kind") == "alarm"}))  # fmt: skip


def gain_by_delay(per_delay: dict, dev: dict | None) -> list[dict]:
    rows = []
    for L in DELAYS:
        fin = per_delay[str(L)]["evaluation"]["P1"]["vs_P0"]
        row = {"delay": L, "final": {k: fin[k] for k in ("diff", "ci_low", "ci_high", "p_value")}}
        if dev is not None:
            d = dev[f"L{L}"]["evaluation"]["P1"]["vs_P0"]
            row["design"] = {k: d[k] for k in ("diff", "ci_low", "ci_high", "p_value")}
        rows.append(row)
    return rows


def hypothesis_lines(h: dict) -> list[str]:
    h1, h2, h4 = h["H-FB1"]["per_policy"], h["H-FB2"]["per_policy"], h["H-FB4"]
    yes = {True: "supported", False: "not supported"}
    lines = [
        "| Hypothesis | Policy | Estimate | 95% CI | p | Holm p | Read-out |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in ("P1", "P2", "P3"):
        lines.append(f"| H-FB1: PR-AUC gain over P0 at L=30 (Holm over P1/P2/P3) | {g} | "
                     f"{f(h1[g]['diff'], 5)} | {ci(h1[g], 5)} | {h1[g]['p_value']:.3f} | "
                     f"{h1[g]['p_holm']:.3f} | {yes[h1[g]['supported']]} |")  # fmt: skip
    lines.append(f"| H-FB1 overall | | | | | | **{yes[h['H-FB1']['supported']]}** |")
    for g in ("P1", "P2", "P3"):
        lines.append(f"| H-FB2: cost change vs P0 at L=30 (E12 cost) | {g} | {h2[g]['diff']:+,.0f} | "
                     f"{h2[g]['ci_low']:+,.0f} to {h2[g]['ci_high']:+,.0f} | {h2[g]['p_value']:.3f} | | "
                     f"{'cost increase' if h2[g]['cost_increase'] else 'no cost increase'} |")  # fmt: skip
    lines.append(f"| H-FB2 overall (no cost increase; not a cost reduction) | | | | | | "
                 f"**{yes[h['H-FB2']['supported']]}** |")  # fmt: skip
    gains = h["H-FB3"]["gains_by_delay"]
    lines.append("| H-FB3: gain does not grow with delay (descriptive, no test) | P1 | "
                 + " / ".join(f"L={L}: {f(gains[str(L)])}" for L in DELAYS)
                 + f" | | | | **{yes[h['H-FB3']['supported']]}** |")  # fmt: skip
    d4 = h4["gated_minus_ungated_cost"]
    lines.append(f"| H-FB4: gated minus ungated P1 cost at L=30 (decided on the point estimate) | P1 | "
                 f"{d4['diff']:+,.0f} | {d4['ci_low']:+,.0f} to {d4['ci_high']:+,.0f} | "
                 f"{d4['p_value']:.3f} | | **{yes[h4['supported']]}** |")  # fmt: skip
    return lines


def render(final: dict, events: dict[int, list], dev: dict | None, dev_events: list | None) -> str:
    h, per = final["hypotheses"], final["per_delay"]
    summ = {L: event_summary(events[L]) for L in DELAYS}
    gains = gain_by_delay(per, dev)
    out = ["# EXP-310-final: feedback loop (champion-challenger) final run", "",
           "Anonymized e-commerce transactions released by Vesta for the IEEE-CIS competition. Champion: the "
           "frozen B5 configuration (five seeds); challengers: the same configuration retrained on matured "
           "labels. Deployment from day 121, warm-up 121-150, evaluation on test days 151-182 (92,427 rows, "
           "3,213 frauds), label delays L = 0 / 7 / 30 / 60, 5 seeds, 1,000-sample bootstrap. Pre-registered "
           "in D210 with amendments D211, D214, D215 (tooling), D216 (tooling); results recorded as-is (D218).",
           "", f"Run `experiments/runs/EXP-310-final/{FINAL_RUN.name}`; parity before the simulation: the five "
           "champion models reproduced the stored EXP-200-final score_B5 on every test row (max absolute "
           "difference 0.0, tolerance 1e-6) and matched their frozen booster hashes.", "",
           "## Hypotheses", "", *hypothesis_lines(h), "",
           "H-FB2 is supported in the pre-registered sense: no policy's cost-difference CI lies above 0, so "
           "there is **no cost increase** versus P0. It does **not** show a cost reduction: P1's CI reaches "
           f"{h['H-FB2']['per_policy']['P1']['ci_high']:+,.0f} and P3's crosses 0. H-FB4 is supported on "
           "the point estimate only; its CI includes 0, and ungated P1 has a slightly higher PR-AUC at L=30 "
           f"({per['30']['evaluation']['P1_ungated']['pr_auc']:.4f} vs {per['30']['evaluation']['P1']['pr_auc']:.4f}).",
           "", "## Gain by delay (P1 minus P0 PR-AUC, test days 151-182)", "",
           "| L | P0 PR-AUC | P1 PR-AUC | Gain | 95% CI | p |", "|---|---|---|---|---|---|"]  # fmt: skip
    for g in gains:
        L, e = g["delay"], per[str(g["delay"])]["evaluation"]
        out.append(f"| {L} | {e['P0']['pr_auc']:.4f} | {e['P1']['pr_auc']:.4f} | {f(g['final']['diff'])} | "
                   f"{ci(g['final'])} | {g['final']['p_value']:.3f} |")  # fmt: skip
    out += ["", "P0's PR-AUC is not comparable across delays (D212 point 4): the champion's three "
            "label-derived inputs follow each arm's delay. Policies are compared within a delay only.", "",
            "## All policies by delay (PR-AUC / E12 cost, seed mean)", "",
            "| L | " + " | ".join(per["0"]["evaluation"]) + " |", "|---|" + "---|" * len(per["0"]["evaluation"])]  # fmt: skip
    for L in DELAYS:
        e = per[str(L)]["evaluation"]
        out.append(
            f"| {L} | "
            + " | ".join(f"{e[p]['pr_auc']:.4f} / {e[p]['cost']:,.0f}" for p in e)
            + " |"
        )
    out += ["", "## Promotions and rejections (sum over 5 seeds)", "",
            "| L | Policy | Attempts | Accepted | Gain below margin / CI | Cost higher | Insufficient evidence | Alarms |",
            "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for L in DELAYS:
        for p in (*GATED, "P1_ungated", "P2"):
            s = summ[L].get(p, {"attempts": 0, "reasons": Counter(), "alarms": 0})
            r = s["reasons"]
            acc = r["accepted"] + r["ungated"] if p == "P1_ungated" else r["accepted"]
            out.append(f"| {L} | {p} | {s['attempts']} | {acc} | {r[REASONS[1]]} | {r[REASONS[2]]} | "
                       f"{r[REASONS[3]]} | {s['alarms']} |")  # fmt: skip
    p3_l0 = per["0"]["evaluation"]["P3"]
    out += ["", "P1_ungated promotes every challenger it can fit (its 'accepted' column counts those).", "",
            "## Notes", "",
            "- **P2 = P0 at every delay.** The PSI monitor never exceeded 0.2, so P2 never attempted a "
            "challenger and served the champion throughout, exactly as the design-run diagnostic predicted "
            "(D212 point 6).",
            f"- **P3 = P1 at L=0** (PR-AUC {p3_l0['pr_auc']:.4f}, cost {p3_l0['cost']:,.0f} for both): an ADWIN "
            "alarm was pending at every weekly slot (alarms fired soon after each promotion, and a pending "
            "alarm is retried at each slot until a challenger is accepted, at most 6 times), so P3 attempted "
            f"on exactly P1's days in {same_days(events[0])} of 5 seeds and reached the same decisions. At "
            "L=7 and L=30 the two differ.",
            "- **L=60 is a structural null** (D212 point 3): every attempt was 'insufficient evidence' "
            "(fewer than 50 matured frauds in the gate window), so every policy served the champion. P3's "
            f"alarms at L=60 came on day(s) {alarm_days(events[60], 'P3')}, after the last weekly slot "
            f"({max(e['day'] for e in events[60] if e['policy'] == 'P1')}), so P3 never attempted.",
            "- **In-sample champion cost in early gates (D214).** The champion's E12 thresholds were chosen "
            "on days 147-150. Attempts whose gate window includes those days compare the challenger with a "
            "champion whose cost there is in-sample: "
            + "; ".join(f"L={L}: {summ[L].get('P1', {}).get('threshold_overlap', 0)} of "
                        f"{summ[L].get('P1', {}).get('attempts', 0)} P1 attempts "
                        f"({summ[L].get('P1', {}).get('accepted_with_overlap', 0)} accepted)" for L in DELAYS)
            + ". Reported as a limitation.",
            f"- **Refused first start (D217).** Run `{REFUSED_RUN}` pushed its start row and was then refused "
            "by the loader's token check (a `python -m` double-import bug) before any test-period row was "
            "read: no parity, no simulation, no results. It is marked aborted on record (FINAL-ABORTED). "
            "After the fix and a logged re-freeze, the run above was started with a logged re-run reason "
            "naming D217."]  # fmt: skip
    if dev is not None:
        out += ["", "## Design run (EXP-310-dev) vs final", "",
                "The design run used training days only (champion fitted on days 1-56, evaluated on 61-120), "
                "the old label curtain (its_time + L <= t) and a 28-day monthly cadence; its numbers are not "
                "hypothesis evidence (D212).", "",
                "| L | Design P0 | Design gain (P1 - P0) | Design CI | Final P0 | Final gain | Final CI | "
                "Design P1 promotions | Final P1 promotions |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        dev_summ = {
            L: event_summary([e for e in (dev_events or []) if e.get("delay") == L]) for L in DELAYS
        }
        for g in gains:
            L = g["delay"]
            out.append(f"| {L} | {dev[f'L{L}']['evaluation']['P0']['pr_auc']:.4f} | {f(g['design']['diff'])} | "
                       f"{ci(g['design'])} | {per[str(L)]['evaluation']['P0']['pr_auc']:.4f} | "
                       f"{f(g['final']['diff'])} | {ci(g['final'])} | "
                       f"{dev_summ[L].get('P1', {'reasons': Counter()})['reasons']['accepted']} | "
                       f"{summ[L].get('P1', {'reasons': Counter()})['reasons']['accepted']} |")  # fmt: skip
        out += [
            "",
            "The pattern carries over: a clear gain at L=0 and L=7, a small one at L=30 (smaller in "
            "the final run, CI including 0) and none at L=60.",
        ]
    return "\n".join(out) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="EXP-310-final written report")
    p.add_argument("--final-json", type=Path, default=FINAL_JSON)
    p.add_argument("--run", type=Path, default=FINAL_RUN)
    p.add_argument("--dev-run", type=Path, default=DEV_RUN)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)
    final = json.loads(a.final_json.read_text("utf-8"))
    events = {
        L: json.loads((a.run / f"delay_L{L}" / "events.json").read_text("utf-8")) for L in DELAYS
    }
    dev = dev_events = None
    if (a.dev_run / "metrics.json").exists():
        dev = json.loads((a.dev_run / "metrics.json").read_text("utf-8"))
        dev_events = json.loads((a.dev_run / "events.json").read_text("utf-8"))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "exp310_report.md").write_text(render(final, events, dev, dev_events), "utf-8")
    chart = {"title": "EXP-310: P1 minus P0 PR-AUC by label delay",
             "x": "label delay L (days)", "y": "PR-AUC gain of P1 over P0 (seed mean, 95% CI)",
             "final_days": [151, 182], "design_days": [61, 120],
             "points": gain_by_delay(final["per_delay"], dev)}  # fmt: skip
    (a.out_dir / "exp310_gain_by_delay.json").write_text(json.dumps(chart, indent=1), "utf-8")
    print(f"wrote {a.out_dir / 'exp310_report.md'} and exp310_gain_by_delay.json")


if __name__ == "__main__":
    main()
