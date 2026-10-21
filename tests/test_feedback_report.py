"""EXP-310 report helpers (D218) on hand-made events with known counts."""

from vaultic.reports.feedback_report import alarm_days, event_summary, same_days


def _ev(policy, day, seed, kind="attempt", reason=None, overlap=0):
    return {"policy": policy, "day": day, "seed": seed, "kind": kind, "reason": reason,
            "accepted": reason in ("accepted", "ungated"),
            "champion_threshold_rows_in_gate": overlap}  # fmt: skip


EVENTS = [
    _ev("P1", 128, 0, reason="accepted", overlap=5),
    _ev("P1", 135, 0, reason="cost higher"),
    _ev("P1", 128, 1, reason="insufficient evidence"),
    _ev("P1", 135, 1, reason="PR-AUC gain below the margin or CI not above 0", overlap=2),
    _ev("P3", 124, 0, kind="alarm"),
    _ev("P3", 128, 0, reason="accepted"),
    _ev("P3", 135, 0, reason="cost higher"),
    _ev("P3", 182, 1, kind="alarm"),
    _ev("P1_ungated", 128, 0, reason="ungated"),
]


def test_event_summary_counts_attempts_reasons_alarms_and_threshold_overlap():
    s = event_summary(EVENTS)
    assert s["P1"]["attempts"] == 4 and s["P1"]["alarms"] == 0
    assert s["P1"]["reasons"]["accepted"] == 1 and s["P1"]["reasons"]["insufficient evidence"] == 1
    assert s["P1"]["threshold_overlap"] == 2 and s["P1"]["accepted_with_overlap"] == 1
    assert s["P3"]["attempts"] == 2 and s["P3"]["alarms"] == 2
    assert s["P1_ungated"]["reasons"]["ungated"] == 1


def test_same_days_and_alarm_days():
    assert same_days(EVENTS) == 1  # seed 0: P3 on P1's days; seed 1: P3 never attempted
    assert alarm_days(EVENTS, "P3") == "124, 182"
    assert alarm_days(EVENTS, "P1") == ""


# ---- D218 review: outcome-dependent notes are derived from the events and metrics -------------
def _metrics(p0, p1, p2, p3):
    return {"evaluation": {p: {"pr_auc": v[0], "cost": v[1]}
                           for p, v in (("P0", p0), ("P1", p1), ("P2", p2), ("P3", p3))}}  # fmt: skip


def _delay_events(decide_days, slots=(128, 135, 142), seeds=(0, 1), p3_days=None, p2_alarm=False):
    ev = []
    for s in seeds:
        for d in slots:
            reason = "accepted" if d in decide_days else "insufficient evidence"
            ev.append(_ev("P1", d, s, reason=reason, overlap=3 if d == 142 else 0))
            if p3_days is None or d in p3_days:
                ev.append(_ev("P3", d, s, reason=reason))
    if p2_alarm:
        ev.append(_ev("P2", 130, 0, kind="alarm"))
    return ev


def test_facts_derive_p2_p3_structural_nulls_and_overlap():
    from vaultic.reports.feedback_report import DELAYS, facts

    same, other = (0.8, 100.0), (0.7, 120.0)
    per = {"0": _metrics(other, same, other, same), "7": _metrics(other, same, other, (0.75, 110.0)),
           "30": _metrics(other, same, other, (0.75, 110.0)), "60": _metrics(other, other, other, other)}  # fmt: skip
    events = {0: _delay_events({128, 135, 142}), 7: _delay_events({135, 142}, p3_days={135}),
              30: _delay_events({142}), 60: _delay_events(set())}  # fmt: skip
    fx = facts(per, events)
    assert fx["p2_alarms"] == 0 and fx["p2_attempts"] == 0 and fx["p2_equals_p0"] == list(DELAYS)
    assert fx["p3_equals_p1"] == [0, 60]  # L=7 and L=30 differ
    assert fx["all_insufficient"] == [60]
    assert fx["decision_days"] == {0: [128, 135, 142], 7: [135, 142], 30: [142], 60: []}
    assert fx["overlap"][(30, "P1")] == (2, 6, 2) and (60, "P1") in fx["overlap"]
    events[7] = _delay_events({135, 142}, p2_alarm=True)
    per["7"]["evaluation"]["P2"] = {"pr_auc": 0.71, "cost": 119.0}
    fx = facts(per, events)
    assert fx["p2_alarms"] == 1 and fx["p2_equals_p0"] == [0, 30, 60]


def test_trail_from_lists_freezes_starts_aborts_and_the_rerun_row():
    from vaultic.reports.feedback_report import TRAIL_SUBJECT, trail_from

    subjects = [("93f9730", "FREEZE EXP-310-final: freeze record"),
                ("967e8a8", "FINAL-STARTED EXP-310-final (run-a)"),
                ("9de52d0", "FINAL-ABORTED EXP-310-final (run-a)"),
                ("7a08ecc", "FREEZE EXP-310-final (re-freeze after D217): x")]  # fmt: skip
    assert all(TRAIL_SUBJECT.match(s) for _, s in subjects)
    assert not TRAIL_SUBJECT.match("FREEZE EXP-200-final: other experiment")
    decisions = ("| FINAL-RERUN | 2026-10-20 | EXP-310-final | --final re-run after `run-a` | — | D217: why | Logged |\n"
                 "| FINAL-RERUN | 2026-10-11 | EXP-200-final | other | — | other | Logged |\n")  # fmt: skip
    t = trail_from(subjects, {"93f9730": "845f51b", "7a08ecc": "c0b7bea"}, decisions)
    assert t[0] == "`93f9730` freeze record and FREEZE row (frozen at `845f51b`)"
    assert t[1] == "`967e8a8` FINAL-STARTED for run `run-a`" and t[2].startswith(
        "`9de52d0` FINAL-ABORTED"
    )
    assert "frozen at `c0b7bea`" in t[3]
    assert len(t) == 5 and t[4].endswith("reason: D217: why")


def test_structural_lines_explain_a_null_delay():
    from collections import Counter

    from vaultic.reports.feedback_report import structural_lines

    fx = {"p1_slots": {L: [128, 177] for L in (0, 7, 30, 60)},
          "decision_days": {0: [128, 177], 7: [177], 30: [177], 60: []}}  # fmt: skip
    summ = {L: {"P1": {"attempts": 10, "reasons": Counter({"insufficient evidence": n})}}
            for L, n in ((0, 0), (7, 5), (30, 5), (60, 10))}  # fmt: skip
    lines = structural_lines(fx, summ)
    assert "d - L > 120" in lines[0]
    assert "L=30: P1 gate decisions on 1 of 2 slots per seed (177); 5 of 10" in lines[4]
    assert "177 minus 60 is 117" in lines[5]


# ---- D220: the events file is generated from the run folder and checkable against it ----------
def test_events_payload_records_the_run_and_hashes_and_detects_changes(tmp_path):
    import hashlib
    import json

    from vaultic.reports.feedback_report import DELAYS, EVENTS_LABEL, check_events, events_payload

    run = tmp_path / "EXP-310-final" / "20261020-174032-191246"
    for L in DELAYS:
        (run / f"delay_L{L}").mkdir(parents=True)
        rows = [{"policy": "P3", "day": 152, "delay": L, "seed": 0, "kind": "alarm", "detector": "ADWIN",
                 "champion": "champion_s0", "gate_rows": 5}]  # fmt: skip
        (run / f"delay_L{L}" / "events.json").write_text(json.dumps(rows))
    p = events_payload(run)
    assert p["label"] == EVENTS_LABEL and p["source_run_id"] == run.name
    assert p["source_run"] == f"experiments/runs/EXP-310-final/{run.name}"
    raw = (run / "delay_L30" / "events.json").read_bytes()
    assert p["sources"]["30"] == {
        "file": "delay_L30/events.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "events": 1,
    }
    assert "gate_rows" not in p["events"]["30"][0] and p["events"]["30"][0]["detector"] == "ADWIN"
    assert check_events(p, run) == []
    (run / "delay_L7" / "events.json").write_text("[]")  # the run changed
    assert any("delay_L7" in x for x in check_events(p, run))
    assert any("source run" in x for x in check_events({**p, "source_run_id": "other"}, run))
