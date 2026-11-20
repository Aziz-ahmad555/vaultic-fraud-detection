"""E20 final analysis (vaultic.eval.e20_analysis) on a small hand-made example with known answers:
kappa, the pre-set 0.6 rule, pooled per-claim rates, the case-level paired bootstrap, the one-sided
p-values with Holm, coverage, refusals, and the refusals to run (hash mismatch, no FINAL-STARTED row).
Mocked data only: no real label, and the real sealed key is never opened."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from vaultic.eval import e20_analysis as an

CASES = [1, 2, 3, 4]


def _data(human_flip: bool = False) -> dict:
    """Per case: A shows 1 fact (claims S, U) and removed 1 (claim W) on Q1; B has 2 sentences on Q1
    (claims S | S, W); C has one sentence (claim S). Pass 2: A unit accurate, B units Y and N."""
    items, key, judge1, pass1, n = [], {}, {}, [], 0

    def unit(case, system, status, item, claims, labels):
        nonlocal n
        n += 1
        uid = f"U{n:04d}"
        key[uid] = {"system": system, "status": status, "item": item, "index": 0, "question": "Q1"}
        judge1[uid] = {"unit": uid, "case": case, "claims": claims, "labels": labels, "lines": [[]] * len(claims),
                       "note": ""}  # fmt: skip
        pass1.append({"unit id": uid, "case": str(case)})
        return uid

    for c in CASES:
        for slot in ("Q1", "Q2", "Q3", "Q4"):
            shown = [{"text": "f", "citations": ["E1"]}] if slot == "Q1" else []
            items.append({"key": f"c{c:02d}-{slot}-A", "case_no": c, "slot": slot, "system": "A", "status": "ok",
                          "refused": None, "shown": shown, "seconds": 10.0 * c})  # fmt: skip
            items.append({"key": f"c{c:02d}-{slot}-B", "case_no": c, "slot": slot, "system": "B", "status": "ok",
                          "units": ["s1 [1]", "s2"] if slot == "Q1" else ["s"], "seconds": 20.0})  # fmt: skip
        items.append({"key": f"c{c:02d}-ADV-A", "case_no": c, "slot": "ADV", "system": "A", "status": "refused",
                      "refused": "change" if c != 4 else "ignore_rules", "seconds": 0.03})  # fmt: skip
        items.append({"key": f"c{c:02d}-ADV-B", "case_no": c, "slot": "ADV", "system": "B", "status": "ok",
                      "units": ["x"], "seconds": 5.0})  # fmt: skip
        items.append({"key": f"c{c:02d}-C", "case_no": c, "slot": "C", "system": "C", "status": "ok",
                      "units": ["narrative"], "seconds": 0.0})  # fmt: skip
        unit(c, "A", "shown", f"c{c:02d}-Q1-A", ["a1", "a2"], ["S", "U"])
        unit(c, "A", "removed", f"c{c:02d}-Q1-A", ["r1"], ["W"])
        unit(c, "B", "shown", f"c{c:02d}-Q1-B", ["b1"], ["S"])
        unit(c, "B", "shown", f"c{c:02d}-Q1-B", ["b2", "b3"], ["S", "W"])
        unit(c, "C", "shown", f"c{c:02d}-C", ["n1"], ["S"])
    sample_units = {
        str(c): [u for u, k in key.items() if pass1[int(u[1:]) - 1]["case"] == str(c)]
        for c in CASES
    }
    human1 = []
    for c in CASES:
        for uid in sample_units[str(c)]:
            for i, lab in enumerate(judge1[uid]["labels"], 1):
                h = {"S": "U", "U": "S", "W": "S"}[lab] if human_flip else lab
                human1.append({"unit id": uid, "case": str(c), "unit text": "t", "claim no": str(i), "claim": "x",
                               "label": h, "split problem": "", "notes": ""})  # fmt: skip
    pass2, judge2 = [], {}
    for c in CASES:
        for system, acc in (("A", "Y"), ("B", "Y"), ("B", "N")):
            pid = f"P{len(pass2) + 1:04d}"
            pass2.append({"unit id": pid, "item": f"c{c:02d}-Q1-{system}", "system": system, "case": str(c),
                          "question": "Q1", "unit text with citations": "u", "cited lines": "l",
                          "accurate": ""})  # fmt: skip
            judge2[pid] = {"unit": pid, "case": c, "accurate": acc, "note": ""}
    human2 = [{**r, "accurate": judge2[r["unit id"]]["accurate"]} for r in pass2]
    adv = [{"k (= case)": str(c), "prompt": "p", "A output": "o", "A refusal kind": "", "A label": "refused",
            "B output": "o", "B label": "complied" if c == 1 else "declined"} for c in CASES]  # fmt: skip
    sample = {"pass1": {"units": sample_units}, "pass2": {"units": {}}}
    return {"items": items, "pass1": pass1, "pass2": pass2, "key": key, "judge1": judge1, "judge2": judge2,
            "human1": human1, "human2": human2, "adversarial": adv, "sample": sample}  # fmt: skip


CFG = {
    "adversarial": [
        [1, "change", "p"],
        [2, "change", "p"],
        [3, "change", "p"],
        [4, "ignore rules", "p"],
    ]
}


def test_kappa_matches_sklearn_and_is_nan_for_one_category():
    a = ["S", "S", "U", "W", "S", "U", "W", "W"]
    b = ["S", "U", "U", "W", "S", "S", "W", "U"]
    assert an.cohen_kappa(a, b, an.P1_LABELS) == pytest.approx(cohen_kappa_score(a, b))
    assert an.cohen_kappa(["S", "S"], ["S", "S"], an.P1_LABELS) != an.cohen_kappa(
        ["S", "S"], ["S", "S"], an.P1_LABELS
    )


def test_pooled_rates_and_the_paired_bootstrap_use_the_same_case_draws():
    r = an.analyse(_data(), CFG)
    j = r["pass1"]["judge"]["systems"]
    assert j["A"]["faithfulness"]["value"] == pytest.approx(0.5)  # S, U per case
    assert j["A"]["unsupported_claim_rate"]["value"] == pytest.approx(0.5)
    assert j["B"]["faithfulness"]["value"] == pytest.approx(2 / 3) and j["B"]["wrong_rate"][
        "value"
    ] == pytest.approx(1 / 3)
    assert j["B"]["unit_wrong_rate"]["value"] == pytest.approx(0.5)  # the 2-claim unit has a W
    assert j["C"]["faithfulness"]["value"] == 1.0 and j["C"]["claims"] == 4
    assert j["A removed"]["wrong_rate"]["value"] == 1.0
    assert j["A draft (shown + removed)"]["faithfulness"]["value"] == pytest.approx(1 / 3)
    # identical per-case rates: every resample gives the same value, so the CI is a point
    assert (
        j["A"]["faithfulness"]["ci_low"] == j["A"]["faithfulness"]["ci_high"] == pytest.approx(0.5)
    )
    t = {x["hypothesis"]: x for x in r["primary"]["tests"]["tests"]}
    assert t["H8"]["diff_A_minus_other"] == pytest.approx(0.5 - 1 / 3)
    assert t["H8"]["p_one_sided"] == 1.0 and not t["H8"]["supported"]  # A is worse than B here
    assert t["H8c"]["diff_A_minus_other"] == pytest.approx(0.5)  # citation accuracy 1.0 vs 0.5
    assert t["H8c"]["p_one_sided"] == 0.0 and t["H8c"]["supported"]
    assert t["H8d"]["diff_A_minus_other"] == pytest.approx(-0.5) and not t["H8d"]["supported"]
    assert [x["p_holm"] for x in r["primary"]["tests"]["tests"]] == an.holm(
        [x["p_one_sided"] for x in r["primary"]["tests"]["tests"]]
    )
    ce = r["pass1"]["judge"]["checker_effect"]["A draft vs A shown"]["faithfulness"]
    assert ce["diff"] == pytest.approx(1 / 3 - 0.5)


def test_the_kappa_rule_picks_the_primary_labels_for_each_pass():
    r = an.analyse(_data(), CFG)
    assert (
        r["agreement"]["pass1"]["kappa"] == pytest.approx(1.0) and r["agreement"]["pass1"]["passed"]
    )
    assert (
        r["label_sources"]["pass1 (faithfulness, unsupported, wrong, unit rates, checker effect)"]
        == "judge"
    )
    flipped = an.analyse(_data(human_flip=True), CFG)
    p1 = flipped["agreement"]["pass1"]
    assert p1["kappa"] < 0.6 and not p1["passed"] and p1["primary"] == "human (sample only)"
    assert (
        flipped["label_sources"][
            "pass1 (faithfulness, unsupported, wrong, unit rates, checker effect)"
        ]
        == "human"
    )
    assert flipped["label_sources"]["pass2 (citation accuracy)"] == "judge"  # each pass on its own
    # the primary tests then use Aziz's labels: A shown claims S, U flipped to U, S -> faithfulness 0.5
    assert flipped["pass1"]["human"]["systems"]["B"]["faithfulness"]["value"] == pytest.approx(
        1 / 3
    )


def test_kappa_ci_comes_from_a_case_level_bootstrap_with_the_preregistered_seed():
    pairs = [(c, x, y) for c in CASES for x, y in (("S", "S"), ("U", "S"), ("W", "W"))]
    k = an.kappa_with_ci(pairs, CASES, an.P1_LABELS, an.SEED_KAPPA_P1)
    assert k["seed"] == 2253 and k["bootstrap_resamples"] == 1000 and k["n"] == 12
    assert k["ci_low"] <= k["kappa"] <= k["ci_high"]
    assert an.kappa_with_ci(pairs, CASES, an.P1_LABELS, an.SEED_KAPPA_P1) == k  # deterministic


def test_coverage_refusals_and_latency():
    r = an.analyse(_data(), CFG)
    cov = r["coverage"]
    assert cov["A"]["items"] == 16 and cov["A"]["share_items_with_a_shown_unit"][
        "value"
    ] == pytest.approx(0.25)
    assert cov["C"]["items"] == 8  # C answers Q1 and Q4
    assert cov["B"]["shown_units_per_item"]["value"] == pytest.approx(5 / 4)
    f = r["refusals"]
    assert f["A_refusal_correctness"]["refused"] == 4 and f["A_false_refusals"]["refused"] == 0
    assert (
        f["per_type"]["change"]["A_kind_matches_type"] == 3
        and f["per_type"]["ignore rules"]["A_kind_matches_type"] == 1
    )
    assert f["B_complied"]["complied"] == 1
    assert r["latency"]["A"]["120 normal items"]["n"] == 16 and r["latency"]["C"].startswith(
        "not applicable"
    )


def test_the_analysis_refuses_without_final_started_or_with_a_changed_input(tmp_path):
    with pytest.raises(an.AnalysisRefused, match="FINAL-STARTED"):
        an.check_final_started("| D1 | ... |", "abc123")
    an.check_final_started("| D9 | E20 FINAL-STARTED code abc123 |", "abc123")
    p = tmp_path / "x.csv"
    p.write_text("a", "utf-8")
    with pytest.raises(an.AnalysisRefused, match="does not match"):
        an._require(p, "0" * 64, "x")
    with pytest.raises(SystemExit):
        an.main([])  # --final is required


def test_load_inputs_checks_every_hash_and_the_sealed_key_last(tmp_path, monkeypatch):
    run = tmp_path / "run"
    tables = tmp_path / "tables"
    (run / "sheets").mkdir(parents=True)
    (run / "judge").mkdir()
    (run / "human").mkdir()
    tables.mkdir()

    def write_csv(path, cols, rows):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["header"])
            w.writerow([])
            w.writerow(cols)
            w.writerows(rows)

    (run / "items.jsonl").write_text('{"x": 1}\n', "utf-8")
    write_csv(run / "sheets/pass1_blind_claims.csv", an.P1_COLUMNS, [])
    write_csv(run / "sheets/pass2_citations.csv", an.P2_COLUMNS, [])
    (run / "sheets/SEALED_pass1_key.json").write_text("{}", "utf-8")
    (run / "judge/judge_pass1.jsonl").write_text("", "utf-8")
    (run / "judge/judge_pass2.jsonl").write_text("", "utf-8")
    write_csv(run / "human/human_pass1.csv", an.HUMAN_P1_COLUMNS, [])
    write_csv(run / "human/human_pass2.csv", an.P2_COLUMNS, [])
    write_csv(run / "human/adversarial_labels.csv", an.ADV_COLUMNS, [])
    sha = an.file_sha256
    sheets = {
        n: sha(run / "sheets" / n)
        for n in ("pass1_blind_claims.csv", "pass2_citations.csv", "SEALED_pass1_key.json")
    }
    (run / "sheets/manifest.json").write_text(json.dumps(sheets), "utf-8")
    (tables / "run_manifest.json").write_text(
        json.dumps({"sha256": {"items.jsonl": sha(run / "items.jsonl"), "sheets": sheets}}), "utf-8"
    )
    jm = {"pass1": {"check": {"joined_sha256": sha(run / "judge/judge_pass1.jsonl")},
                    "human_labels": {"sha256": sha(run / "human/human_pass1.csv")}},
          "pass2": {"check": {"joined_sha256": sha(run / "judge/judge_pass2.jsonl")},
                    "human_labels": {"sha256": sha(run / "human/human_pass2.csv")}},
          "adversarial": {"human_labels": {"sha256": sha(run / "human/adversarial_labels.csv")}}}  # fmt: skip
    (tables / "judge_manifest.json").write_text(json.dumps(jm), "utf-8")
    (tables / "human_sample.json").write_text(
        json.dumps(
            {
                "pass1": {"sheet_sha256": sheets["pass1_blind_claims.csv"]},
                "pass2": {"sheet_sha256": sheets["pass2_citations.csv"]},
            }
        ),
        "utf-8",
    )
    d = an.load_inputs(run, tables)
    assert d["key"] == {} and set(d["hashes"]) >= {
        "sealed_key",
        "items",
        "human_pass1",
        "adversarial",
    }
    opened = []
    real_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *a, **k: (opened.append(self.name), real_read(self, *a, **k))[1],
    )
    (run / "human/human_pass2.csv").write_text("changed", "utf-8")
    with pytest.raises(an.AnalysisRefused, match="human_pass2"):
        an.load_inputs(run, tables)
    assert "SEALED_pass1_key.json" not in opened  # a bad input stops the run before the key is read


def test_the_holm_family_and_ci_helpers():
    assert an.ci(np.array([np.nan, 1.0, 1.0]))[2] == 1
    idx = an.case_resamples(4, 2251)
    assert idx.shape == (1000, 4) and (an.case_resamples(4, 2251) == idx).all()


@pytest.mark.parametrize("flip", [False, True])
def test_the_markdown_tables_render_for_either_primary_label_set(flip):
    md = an.markdown_tables(an.analyse(_data(human_flip=flip), CFG))
    for heading in ("## Agreement", "## Pass-1 metrics, judge labels", "## Pass-1 metrics, human labels",
                    "## Citation accuracy", "## Pre-registered tests", "## Coverage", "## Refusals", "## Latency"):  # fmt: skip
        assert heading in md
    primary = "human" if flip else "judge"
    assert f"## Pass-1 metrics, {primary} labels (PRIMARY)" in md
    assert md.count("(PRIMARY)") == 1 + 2  # one pass-1 table, two pass-2 rows (A and B)
