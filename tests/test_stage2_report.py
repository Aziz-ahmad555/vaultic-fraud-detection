"""The E15 report (stage2/report.py): primary and secondary orderings side by side with their
registered labels, and the decision-boundary note on counterfactual validity (D188)."""

from vaultic.stage2.e15 import SECONDARY_KEY, SECONDARY_LABEL
from vaultic.stage2.report import BOUNDARY_NOTE, PRIMARY_LABEL, render_e15


def _ci(m):
    return {"mean": m, "ci_low": m - 0.01, "ci_high": m + 0.01, "n": 10}


def _test(d, supported=None):
    t = {"diff": d, "ci_low": d - 0.01, "ci_high": d + 0.01, "p_value": 0.0, "p_holm": 0.0}
    if supported is not None:
        t["supported"] = supported
    return t


def _system(h6a, h6b):
    return {
        "aopc": {"attribution": _ci(0.04), "random": _ci(0.005), "permutation": _ci(0.19),
                 SECONDARY_KEY: _ci(0.28)},
        "hypotheses": {"H6a": _test(0.03, h6a), "H6b": _test(-0.15, h6b)},
        "secondary_positive_order": {"label": SECONDARY_LABEL, "deletion_curve": [0.5] * 10,
                                     "tests": {"positive - random": _test(0.27),
                                               "positive - permutation": _test(0.09)}},
        "stability": {f: {"top3_jaccard": _ci(0.95), "top1_changed": _ci(0.05)}
                      for f in ("x0.99", "x1.01", "x0.95", "x1.05")},
        "seed_consistency": _ci(0.87),
        "counterfactuals": {"validity": _ci(0.9), "any_valid": _ci(0.9), "proximity": _ci(6.0),
                            "sparsity": _ci(1.5), "loose_validity": _ci(0.9),
                            "setting": {"total_cfs": 3}},
    }  # fmt: skip


def test_report_shows_both_orderings_with_their_labels_and_the_boundary_note():
    results = {"MVAF": _system(True, False), "B5": _system(True, True),
               "phase9_gate": {"passed": True, "H6a_MVAF_supported": True, "MVAF_validity": 0.976,
                               "validity_target": 0.9}}  # fmt: skip
    md = render_e15(results)
    assert md.count(PRIMARY_LABEL) == 2 and md.count(SECONDARY_LABEL) >= 2  # MVAF and B5 rows
    assert "H6b" in md and "not supported" in md and "p < 0.001" in md  # D156 formatting
    assert BOUNDARY_NOTE in md and "not directly comparable" in md
    assert "E12 step-up threshold" in md and "frozen cost threshold" in md
    assert "Phase 9 exit gate" in md and "PASSED" in md
