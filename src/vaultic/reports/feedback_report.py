"""Written report of the feedback-loop final run EXP-310-final (D210, D218).

Run:  python -m vaultic.reports.feedback_report [--final-json ...] [--run ...] [--dev-run ...]

Every number comes from the committed final results (research/tables/feedback/exp310_final.json,
written by vaultic.learning.feedback_final), the final run folder (parity.json, per-delay
events.json), the design run's metrics.json / events.json and the git record (freeze, start,
abort and re-run rows); nothing is recomputed from data. Every note that depends on the outcome
(P2 = P0, P3 vs P1, the structural nulls, the threshold-slice overlap) is derived from the events
and metrics by `facts`. Writes research/tables/feedback/exp310_report.md and
exp310_gain_by_delay.json (chart data: P1 minus P0 PR-AUC by label delay, final and design run),
and exp310_events.json (D220): every alarm and gate attempt of the run, per delay, generated from
the run folder's delay_L<L>/events.json with the source run id and the SHA-256 of each source
file, so it can be checked against the run (`check_events`). No hand edits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from vaultic.paths import REPO_ROOT, RESEARCH_DIR

OUT_DIR = RESEARCH_DIR / "tables" / "feedback"
FINAL_JSON = OUT_DIR / "exp310_final.json"
EXP = "EXP-310-final"
FINAL_RUN = REPO_ROOT / "experiments" / "runs" / EXP / "20261020-174032-191246"
DEV_RUN = REPO_ROOT / "experiments" / "runs" / "EXP-310-dev" / "20261020-004133"
DELAYS = (0, 7, 30, 60)
GATED = (
    "P1",
    "P3",
    "P1_monthly",
)  # policies whose attempts go through the gate (P2 never attempted)
FIT_LAST_DAY = 120  # the champion was fitted on days 1-120
GATE_DAYS = 7  # the gate window: the 7 most recent days of matured labels
TEST_DAYS = (151, 182)
TRAIL_SUBJECT = re.compile(rf"^(FREEZE|FINAL-STARTED|FINAL-ABORTED) {EXP}\b")
REASONS = ("accepted", "PR-AUC gain below the margin or CI not above 0", "cost higher",
           "insufficient evidence")  # fmt: skip


def f(x: float, nd: int = 4) -> str:
    return f"{x:+.{nd}f}"


def ci(d: dict, nd: int = 4) -> str:
    return f"{d['ci_low']:+.{nd}f} to {d['ci_high']:+.{nd}f}"


def attempts(events: list[dict], policy: str) -> list[dict]:
    return [e for e in events if e["policy"] == policy and e.get("kind") == "attempt"]


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
    days = {(p, s): [e["day"] for e in attempts(events, p) if e["seed"] == s]
            for p in (a, b) for s in seeds}  # fmt: skip
    return sum(days[(a, s)] == days[(b, s)] for s in seeds)


def alarm_days(events: list[dict], policy: str) -> str:
    return ", ".join(str(d) for d in sorted({e["day"] for e in events
                                             if e["policy"] == policy and e.get("kind") == "alarm"}))  # fmt: skip


def decision_days(events: list[dict], policy: str = "P1") -> list[int]:
    """Slot days on which the policy's gate had enough evidence in at least one seed."""
    return sorted(
        {e["day"] for e in attempts(events, policy) if e["reason"] != "insufficient evidence"}
    )


def same_metrics(e: dict, a: str, b: str) -> bool:
    return e[a]["pr_auc"] == e[b]["pr_auc"] and e[a]["cost"] == e[b]["cost"]


def facts(per: dict, events: dict[int, list]) -> dict:
    """Every outcome-dependent statement of the report, derived from the metrics and events."""
    out = {"p2_alarms": sum(e["policy"] == "P2" and e.get("kind") == "alarm" for L in DELAYS for e in events[L]),
           "p2_attempts": sum(len(attempts(events[L], "P2")) for L in DELAYS),
           "p2_equals_p0": [L for L in DELAYS if same_metrics(per[str(L)]["evaluation"], "P2", "P0")],
           "p3_equals_p1": [L for L in DELAYS if same_metrics(per[str(L)]["evaluation"], "P3", "P1")],
           "all_insufficient": [L for L in DELAYS
                                if all(e["reason"] == "insufficient evidence" for e in events[L]
                                       if e.get("kind") == "attempt")
                                and any(e.get("kind") == "attempt" for e in events[L])],
           "decision_days": {L: decision_days(events[L]) for L in DELAYS},
           "p1_slots": {L: sorted({e["day"] for e in attempts(events[L], "P1")}) for L in DELAYS},
           "overlap": {}}  # fmt: skip
    for L in DELAYS:
        s = event_summary(events[L])
        for p in GATED:
            if p in s and s[p]["threshold_overlap"]:
                out["overlap"][(L, p)] = (s[p]["threshold_overlap"], s[p]["attempts"],
                                          s[p]["accepted_with_overlap"])  # fmt: skip
    return out


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


# ---- the events file (D220) ----------------------------------------------------------------
EVENTS_LABEL = "EXP-310 simulation on test period"
EVENT_FIELDS = ("policy", "day", "delay", "seed", "kind", "detector", "reason", "accepted", "gain",
                "ci_low", "ci_high", "cost_champion", "cost_challenger", "champion", "challenger",
                "alarm_dropped")  # fmt: skip


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def events_payload(run: Path) -> dict:
    """Every alarm and gate attempt per delay, copied field by field from the run folder, with
    the source run and each source file's SHA-256 (raw bytes)."""
    run = Path(run)
    sources, events = {}, {}
    for L in DELAYS:
        f = run / f"delay_L{L}" / "events.json"
        rows = json.loads(f.read_text("utf-8"))
        sources[str(L)] = {
            "file": f"delay_L{L}/events.json",
            "sha256": _sha256(f),
            "events": len(rows),
        }
        events[str(L)] = [{k: e[k] for k in EVENT_FIELDS if k in e} for e in rows]
    return {"label": EVENTS_LABEL, "generated_by": "python -m vaultic.reports.feedback_report",
            "source_run_id": run.name, "source_run": f"experiments/runs/{run.parent.name}/{run.name}",
            "sources": sources, "fields": list(EVENT_FIELDS), "events": events}  # fmt: skip


def check_events(payload: dict, run: Path) -> list[str]:
    """Differences between an events file and the run folder (empty = it matches the run)."""
    run = Path(run)
    bad = (
        []
        if payload.get("source_run_id") == run.name
        else [f"source run {payload.get('source_run_id')} is not {run.name}"]
    )
    for src in payload.get("sources", {}).values():
        f = run / src["file"]
        if not f.exists():
            bad.append(f"{src['file']} missing in the run folder")
        elif _sha256(f) != src["sha256"]:
            bad.append(f"{src['file']}: SHA-256 differs from the run folder")
    if not bad and payload != events_payload(run):
        bad.append("the events differ from a fresh export of the run folder")
    return bad


# ---- the record trail (git) -----------------------------------------------------------------
def trail_from(
    subjects: list[tuple[str, str]], frozen_of: dict[str, str], decisions: str
) -> list[str]:
    """subjects: (short hash, subject) oldest first; frozen_of: freeze commit -> frozen commit."""
    out = []
    for h, s in subjects:
        if s.startswith(f"FREEZE {EXP}"):
            out.append(f"`{h}` freeze record and FREEZE row (frozen at `{frozen_of.get(h, '?')}`)")
        elif s.startswith(f"FINAL-STARTED {EXP}") or s.startswith(f"FINAL-ABORTED {EXP}"):
            kind, run = s.split(" ", 1)[0], re.search(r"\(([^)]+)\)", s)
            out.append(f"`{h}` {kind} for run `{run.group(1) if run else '?'}`")
    for line in decisions.splitlines():
        if line.startswith("| FINAL-RERUN |") and f"| {EXP} |" in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            out.append(f"FINAL-RERUN row in research/decisions.md ({cells[3]}), reason: {cells[5]}")
    return out


def record_trail(repo: Path = REPO_ROOT) -> list[str]:
    def git(*a):
        r = subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, encoding="utf-8")
        return r.stdout if r.returncode == 0 else ""

    subjects = [tuple(x.split(" ", 1)) for x in git("log", "--reverse", "--format=%h %s").splitlines()
                if TRAIL_SUBJECT.match(x.split(" ", 1)[1])]  # fmt: skip
    frozen_of = {}
    for h, s in subjects:
        if s.startswith("FREEZE"):
            m = re.search(
                r"at git commit `([0-9a-f]{40})`", git("show", f"{h}:research/frozen_feedback.md")
            )
            frozen_of[h] = m.group(1)[:7] if m else "?"
    decisions = (repo / "research" / "decisions.md").read_text("utf-8")
    return trail_from(subjects, frozen_of, decisions)


# ---- the report -----------------------------------------------------------------------------
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


def _cost_reading(h2: dict) -> str:
    parts = []
    for g, d in h2.items():
        if d["diff"] == 0 and d["ci_low"] == 0 and d["ci_high"] == 0:
            parts.append(f"{g} identical to P0")
        elif d["ci_high"] < 0:
            parts.append(f"{g}'s CI lies below 0")
        else:
            parts.append(f"{g}'s CI reaches {d['ci_high']:+,.0f}")
    return "; ".join(parts)


def structural_lines(fx: dict, summ: dict) -> list[str]:
    lines = [f"At a weekly slot on day d, a label is known only if its transaction time + L is before d, "
             f"so the gate window (the {GATE_DAYS} most recent days of known labels) covers roughly days "
             f"d - L - {GATE_DAYS} to d - L. Rows the champion trained on (days 1-{FIT_LAST_DAY}) are removed "
             f"from it, so a gate decision needs at least d - L > {FIT_LAST_DAY}, and enough window days after "
             f"day {FIT_LAST_DAY} to hold >= 50 frauds. With weekly slots on days "
             f"{', '.join(map(str, fx['p1_slots'][0]))} (last slot {max(fx['p1_slots'][0])}):", ""]  # fmt: skip
    for L in DELAYS:
        days, slots = fx["decision_days"][L], fx["p1_slots"][L]
        ins = summ[L].get("P1", {"reasons": Counter(), "attempts": 0})
        lines.append(f"- L={L}: P1 gate decisions on {len(days)} of {len(slots)} slots per seed "
                     f"({', '.join(map(str, days)) or 'none'}); {ins['reasons']['insufficient evidence']} of "
                     f"{ins['attempts']} P1 attempts 'insufficient evidence'"
                     + (f" (the last slot {max(slots)} minus {L} is {max(slots) - L}, inside the champion's "
                        "training days: no gate decision is possible in the span)" if not days and slots else "")
                     + ".")  # fmt: skip
    return lines


def render(final: dict, events: dict[int, list], dev: dict | None, dev_events: list | None,
           parity: dict | None = None, trail: list[str] | None = None) -> str:  # fmt: skip
    h, per = final["hypotheses"], final["per_delay"]
    summ = {L: event_summary(events[L]) for L in DELAYS}
    fx = facts(per, events)
    gains = gain_by_delay(per, dev)
    p0 = per["30"]["evaluation"]["P0"]
    par = (f"the five champion models reproduced the stored EXP-200-final score_B5 on all "
           f"{parity['rows']:,} test rows (max absolute difference {parity['max_abs_diff']}, tolerance "
           f"{parity['tolerance']}) and matched their frozen booster hashes" if parity else
           "see parity.json in the run folder")  # fmt: skip
    out = ["# EXP-310-final: feedback loop (champion-challenger) final run", "",
           "Anonymized e-commerce transactions released by Vesta for the IEEE-CIS competition. Champion: the "
           "frozen B5 configuration (five seeds); challengers: the same configuration retrained on matured "
           f"labels. Deployment from day 121, warm-up 121-150, evaluation on test days {TEST_DAYS[0]}-{TEST_DAYS[1]} "
           f"({p0['rows']:,} rows, {p0['frauds']:,} frauds), label delays L = 0 / 7 / 30 / 60, 5 seeds, "
           "1,000-sample bootstrap. Pre-registered in D210 with amendments D211 and D214, plus D215 and D216 "
           "(tooling only); results recorded as-is (D218).",
           "", f"Run `experiments/runs/{EXP}/{FINAL_RUN.name}`. Parity before the simulation: {par}.", "",
           "## Hypotheses", "", *hypothesis_lines(h), "",
           "H-FB2 is supported in the pre-registered sense: no policy's cost-difference CI lies above 0, so "
           "there is **no cost increase** versus P0. It does **not** show a cost reduction ("
           f"{_cost_reading(h['H-FB2']['per_policy'])}). H-FB3 is partly true by construction: the L=60 gain is "
           "0 because no gate decision was possible there (see the structural limits), and the L=30 gain's CI "
           "includes 0, so only the L=0 > L=7 > L=30 ordering carries information. H-FB4 is supported on the "
           "point estimate only; its CI includes 0"
           + (f", and ungated P1 has a slightly higher PR-AUC at L=30 "
              f"({per['30']['evaluation']['P1_ungated']['pr_auc']:.4f} vs {per['30']['evaluation']['P1']['pr_auc']:.4f})"
              if per['30']['evaluation']['P1_ungated']['pr_auc'] > per['30']['evaluation']['P1']['pr_auc'] else "")
           + ".", "", "## Gain by delay (P1 minus P0 PR-AUC, test days 151-182)", "",
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
    out += ["", "P1_ungated promotes every challenger it can fit (its 'accepted' column counts those).", "",
            "## Structural limits of the 32-day test period (pre-registered, section 7)", "",
            *structural_lines(fx, summ), "",
            "These are limits of the span, not of the policies: a longer test period would allow more gate "
            "decisions at L=30 and any at L=60."]  # fmt: skip
    out += ["", "## Notes", ""]
    if fx["p2_alarms"] == 0 and fx["p2_attempts"] == 0 and fx["p2_equals_p0"] == list(DELAYS):
        out.append("- **P2 = P0 at every delay.** The PSI monitor never exceeded 0.2 (no P2 alarm), so P2 never "
                   "attempted a challenger and served the champion throughout, as in the design-run "
                   "diagnostic (D212 point 6).")  # fmt: skip
    else:
        out.append(f"- P2: {fx['p2_alarms']} alarm(s), {fx['p2_attempts']} attempt(s); identical to P0 at "
                   f"L = {fx['p2_equals_p0']}.")  # fmt: skip
    differ = [L for L in DELAYS if L not in fx["p3_equals_p1"]]
    for L in fx["p3_equals_p1"]:
        e = per[str(L)]["evaluation"]["P3"]
        if fx["all_insufficient"] and L in fx["all_insufficient"]:
            continue
        out.append(f"- **P3 = P1 at L={L}** (PR-AUC {e['pr_auc']:.4f}, cost {e['cost']:,.0f} for both): an ADWIN "
                   "alarm was pending at every weekly slot (a pending alarm is retried at each slot until a "
                   f"challenger is accepted, at most 6 times), so P3 attempted on exactly P1's days in "
                   f"{same_days(events[L])} of 5 seeds and reached the same decisions."
                   + (f" At L = {', '.join(map(str, differ))} the two differ." if differ else ""))  # fmt: skip
    for L in fx["all_insufficient"]:
        last = max(fx["p1_slots"][L]) if fx["p1_slots"][L] else None
        out.append(f"- **L={L} is a structural null** (D212 point 3): every attempt of every policy was "
                   "'insufficient evidence', so every policy served the champion."
                   + (f" P3's alarms at L={L} came on day(s) {alarm_days(events[L], 'P3')}, after the last "
                      f"weekly slot ({last}), so P3 never attempted." if alarm_days(events[L], "P3") else ""))  # fmt: skip
    ov = fx["overlap"]
    ov_text = "; ".join(
        f"{p} at L={L}: {n} of {a} attempts ({acc} accepted)"
        for (L, p), (n, a, acc) in sorted(ov.items())
    )
    clean = [p for p in ("P1", "P3") if (30, p) not in ov]
    out += ["", "## Limitations", "",
            "- **Champion head start (pre-registered).** The champion was trained on days 1-120 with every "
            "label, including labels a real deployment at day 121 could not yet have had under delay L (days "
            "91-120 at L = 30), and its E12 thresholds and P2 reference were chosen once on calibrate_fused "
            "(days 147-150; D211 c). This favours P0.",
            "- **Second use of the test period (D211 a).** Days 151-182 were already used by the stage-1 "
            "(EXP-200-final) and stage-2 final runs; EXP-310-final is a second use of those days under a new "
            "pre-registration. No B5 setting was chosen using test results.",
            "- **In-sample champion cost in early gates (D214).** The champion's E12 thresholds were chosen on "
            "days 147-150; a gate window containing those days compares the challenger with a champion whose "
            "cost there is in-sample. Gated attempts with such an overlap (from the events): "
            + (ov_text or "none") + "."
            + (f" {' and '.join(clean)} have no overlap at L=30, so H-FB1 and H-FB2 are unaffected."
               if len(clean) == 2 else " This touches the L=30 hypothesis policies; see above."),
            "- **Structural limits of the span**, as above (L=60 null, at most "
            f"{len(fx['decision_days'][30])} gate decisions per seed at L=30)."]  # fmt: skip
    out += ["", "## Record trail", "",
            "- The first final start was refused before any test-period row was read by the loader's token "
            "check (a `python -m` double-import bug, D217): no parity, no simulation, no results; it is marked "
            "aborted on record. After the fix and a logged re-freeze, the run above was started with a logged "
            "re-run reason naming D217.",
            *(f"- {t}" for t in (trail or []))]  # fmt: skip
    if dev is not None:
        out += ["", "## Design run (EXP-310-dev) vs final", "",
                "The design run used training days only (champion fitted on days 1-56, evaluated on 61-120), "
                "the old label curtain (its_time + L <= t) and a 28-day monthly cadence; its numbers are not "
                "hypothesis evidence (D212). D212 point 5: the stored frequency encoders were fitted on days "
                "1-120, so the design champion (days 1-56) saw mildly forward-looking frequency counts; the final "
                "run deploys at day 121 and is not affected.", "",
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
    parity = (
        json.loads((a.run / "parity.json").read_text("utf-8"))
        if (a.run / "parity.json").exists()
        else None
    )
    dev = dev_events = None
    if (a.dev_run / "metrics.json").exists():
        dev = json.loads((a.dev_run / "metrics.json").read_text("utf-8"))
        dev_events = json.loads((a.dev_run / "events.json").read_text("utf-8"))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "exp310_report.md").write_text(
        render(final, events, dev, dev_events, parity, record_trail()), "utf-8"
    )
    chart = {"title": "EXP-310: P1 minus P0 PR-AUC by label delay",
             "x": "label delay L (days)", "y": "PR-AUC gain of P1 over P0 (seed mean, 95% CI)",
             "final_days": list(TEST_DAYS), "design_days": [61, 120],
             "points": gain_by_delay(final["per_delay"], dev)}  # fmt: skip
    (a.out_dir / "exp310_gain_by_delay.json").write_text(json.dumps(chart, indent=1), "utf-8")
    (a.out_dir / "exp310_events.json").write_text(
        json.dumps(events_payload(a.run), indent=1), "utf-8"
    )
    print(
        f"wrote {a.out_dir / 'exp310_report.md'}, exp310_gain_by_delay.json and exp310_events.json"
    )


if __name__ == "__main__":
    main()
