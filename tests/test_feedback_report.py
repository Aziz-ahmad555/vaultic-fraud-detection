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
