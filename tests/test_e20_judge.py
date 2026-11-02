"""tools/e20_judge.py (D225e, D225f) on a small hand-made run: the seeded samples, the isolated judge
folders with their deny rule, the check of the judge's case files and transcripts, and the human sheets
that carry the judge's claim split but none of its labels. Mocked data only: no real label anywhere.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from vaultic.eval import e20

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e20_judge", ROOT / "tools/e20_judge.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)

GUIDE = [
    "## 5. Labelling (guide fixed now)",
    "",
    "**Claims (D225a):** split.",
    "**Labeller:** Aziz.",
]
UNITS_PER_CASE = {1: 7, 2: 3, 3: 6}  # case 2 has fewer than 5 units: all of them are taken


def _manifest(sheets: Path) -> None:
    m = {f.name: e20.file_sha256(f) for f in sorted(sheets.iterdir()) if f.name != "manifest.json"}
    (sheets / "manifest.json").write_text(json.dumps(m), "utf-8")


@pytest.fixture
def run(tmp_path):
    run_dir = tmp_path / "e20" / "20261026-000000-test"
    sheets = run_dir / "sheets"
    sheets.mkdir(parents=True)
    rng = np.random.default_rng(0)
    units = [(c, f"case {c} statement {i}") for c, n in UNITS_PER_CASE.items() for i in range(n)]
    order = rng.permutation(len(units))
    p1 = [[f"U{k:04d}", units[j][0], units[j][1], "", "", ""] for k, j in enumerate(order, 1)]
    e20._write_csv(sheets / "pass1_blind_claims.csv", ["E20 pass 1 (blind). Split ...", ""] + GUIDE,
                   judge.P1_COLUMNS, p1)  # fmt: skip
    p2 = [[f"P{k:04d}", f"c{c:02d}-Q1-A", "AB"[k % 2], c, "Q1", f"fact {k} [1]", "[1] line one", ""]
          for k, c in enumerate([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3], 1)]  # fmt: skip
    e20._write_csv(sheets / "pass2_citations.csv", ["E20 pass 2 (citations, not blind).", ""] + GUIDE,
                   judge.P2_COLUMNS, p2)  # fmt: skip
    ev = [[c, n, f"evidence {c}.{n}"] for c in UNITS_PER_CASE for n in range(1, 6)]
    e20._write_csv(
        sheets / "reference_evidence.csv", ["E20 reference evidence."], ["case", "line", "text"], ev
    )
    (sheets / "SEALED_pass1_key.json").write_text('{"U0001": {"system": "A"}}', "utf-8")
    _manifest(sheets)
    return run_dir


def _judge_out(run_dir, folder, name, bad=None):
    """Mocked judge case files (fake labels), optionally broken by `bad(case, records)`."""
    spec_ = judge.PASSES[name]
    _, rows = judge.read_sheet(run_dir / "sheets" / spec_["sheet"], spec_["columns"])
    out = folder / "out"
    out.mkdir(parents=True, exist_ok=True)
    for case in sorted({int(r["case"]) for r in rows}):
        recs = []
        for r in sorted((r for r in rows if int(r["case"]) == case), key=lambda r: r["unit id"]):
            if name == "pass1":
                recs.append({"unit": r["unit id"], "case": case, "claims": [r["unit text"], "a second claim"],
                             "labels": ["S", "U"], "lines": [[1], []], "note": "SECRET-JUDGE-NOTE"})  # fmt: skip
            else:
                recs.append({"unit": r["unit id"], "case": case, "accurate": "Y", "note": ""})
        if bad:
            recs = bad(case, recs)
        if recs is not None:
            (out / f"case_{case:02d}.jsonl").write_text(
                "".join(json.dumps(x) + "\n" for x in recs), "utf-8"
            )


def _transcript(tdir, calls, model="claude-opus-5-5"):
    tdir.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "assistant", "message": {"role": "assistant", "model": model, "content": [
        {"type": "tool_use", "name": n, "input": i} for n, i in calls]}}]  # fmt: skip
    (tdir / "session.jsonl").write_text("\n".join(json.dumps(x) for x in lines), "utf-8")


def _sample(run_dir, tmp_path):
    out = tmp_path / "human_sample.json"
    return judge.sample(run_dir, out), out


def test_the_sample_is_seeded_stratified_and_fixed(run, tmp_path):
    s, out = _sample(run, tmp_path)
    p1 = s["pass1"]
    assert p1["seed"] == 2252 and p1["n"] == 5 + 3 + 5 and set(p1["units"]) == {"1", "2", "3"}
    assert len(p1["units"]["2"]) == 3  # fewer than 5: all of them, no top-up
    assert s["pass2"]["seed"] == 2254 and [len(v) for v in s["pass2"]["units"].values()] == [
        3,
        3,
        3,
    ]
    assert p1["sheet_sha256"] == e20.file_sha256(run / "sheets/pass1_blind_claims.csv")
    # the rule, recomputed by hand: one rng, cases in order, ids in id order
    _, rows = judge.read_sheet(run / "sheets/pass1_blind_claims.csv", judge.P1_COLUMNS)
    rng = np.random.default_rng(2252)
    for case in (1, 2, 3):
        ids = sorted(r["unit id"] for r in rows if int(r["case"]) == case)
        want = sorted(
            ids[int(i)] for i in rng.choice(len(ids), size=min(5, len(ids)), replace=False)
        )
        assert p1["units"][str(case)] == want
    assert judge.sample(run, out) == s  # re-running gives the same sample
    data = json.loads(out.read_text("utf-8"))
    data["pass1"]["units"]["1"] = data["pass1"]["units"]["1"][:1]
    out.write_text(json.dumps(data), "utf-8")
    with pytest.raises(judge.JudgeToolError, match="different sample"):
        judge.sample(run, out)


def test_a_sheet_that_does_not_match_the_manifest_is_refused(run, tmp_path):
    with open(run / "sheets/pass1_blind_claims.csv", "a", encoding="utf-8") as f:
        f.write("U9999,1,tampered,,,\n")
    with pytest.raises(judge.JudgeToolError, match="does not match"):
        judge.sample(run, tmp_path / "s.json")


def test_prepare_builds_an_isolated_folder_with_the_deny_rule(run, tmp_path):
    _, sample_file = _sample(run, tmp_path)
    root = run.parent / "judge"
    rec = judge.prepare(
        "pass1", run, root, sample_file, repo=ROOT, drives=["D", "Z"], require_pushed=False
    )
    folder = root / "pass1"
    names = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    assert names == [".claude/settings.json", "PROMPT.md", "inputs/guide.md", "inputs/pass1_blind_claims.csv",
                     "inputs/reference_evidence.csv"]  # fmt: skip
    assert list((folder / "out").iterdir()) == []
    for f in ("pass1_blind_claims.csv", "reference_evidence.csv"):  # byte copies, checked
        assert e20.file_sha256(folder / "inputs" / f) == e20.file_sha256(run / "sheets" / f)
    assert (folder / "inputs/guide.md").read_text("utf-8").splitlines()[2:] == GUIDE
    prompt = (folder / "PROMPT.md").read_text("utf-8")
    assert prompt == judge.prompt_from_prereg(judge.PREREG.read_text("utf-8"), "pass1")
    assert "out/case_01.jsonl" in prompt and "SEALED" not in prompt
    perm = json.loads((folder / ".claude/settings.json").read_text("utf-8"))["permissions"]
    deny = perm["deny"]
    for tool in (
        "Bash",
        "PowerShell",
        "WebFetch",
        "WebSearch",
        "Agent",
        "mcp__Claude_Browser",
        "mcp__terminal",
    ):
        assert tool in deny
    assert (
        f"Read({judge.rule_path(run)}/**)" in deny
    )  # the run folder (sealed key, items) is blocked
    assert f"Read({judge.rule_path(root / 'pass2')}/**)" in deny  # the other pass's folder
    assert f"Read({judge.rule_path(ROOT)}/**)" in deny  # the repository
    assert "Read(//d/**)" in deny and "Read(//z/**)" in deny  # every other drive
    own_drive = judge.rule_path(folder).split("/")[2]
    assert f"Read(//{own_drive}/**)" not in deny  # never the judge folder's own drive as a whole
    own = judge.rule_path(folder)
    assert not any(
        r in (f"Read({own})", f"Read({own}/**)") for r in deny
    )  # the folder itself stays readable
    assert not any(r.startswith(f"Read({p})") for r in deny for p in [judge.rule_path(x) for x in folder.parents]
                   if r == f"Read({p}/**)")  # no ancestor is denied as a whole  # fmt: skip
    assert "Edit(./out/**)" in perm["allow"] and "Edit(./inputs/**)" in deny
    assert perm["blockReadsOutsideWorkingDirectories"] is True
    assert rec["sha256"]["PROMPT.md"] == e20.file_sha256(folder / "PROMPT.md")
    (folder / "out/case_01.jsonl").write_text("{}", "utf-8")
    with pytest.raises(judge.JudgeToolError, match="not empty"):
        judge.prepare("pass1", run, root, sample_file, repo=ROOT, drives=[], require_pushed=False)


def test_prepare_pass2_hides_the_item_system_and_question_columns(run, tmp_path):
    _, sample_file = _sample(run, tmp_path)
    root = run.parent / "judge"
    judge.prepare("pass2", run, root, sample_file, repo=ROOT, drives=[], require_pushed=False)
    with open(root / "pass2/inputs/pass2_for_judge.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == judge.P2_JUDGE_COLUMNS and len(rows) == 12
    assert not any("Q1-A" in c or c in ("A", "B") for r in rows for c in r)
    assert not (root / "pass2/inputs/reference_evidence.csv").exists()
    assert "accurate" in (root / "pass2/PROMPT.md").read_text("utf-8")


def test_prepare_refuses_before_the_sample_is_pushed(run, tmp_path):
    with pytest.raises(judge.JudgeToolError, match="committed and pushed"):
        judge.prepare(
            "pass1", run, run.parent / "judge", tmp_path / "none.json", repo=ROOT, drives=[]
        )
    _, sample_file = _sample(run, tmp_path)  # outside the repo: never "pushed"
    with pytest.raises((judge.JudgeToolError, ValueError)):
        judge.prepare("pass1", run, run.parent / "judge", sample_file, repo=ROOT, drives=[])


def test_rule_paths_use_the_posix_form_and_escape_specials():
    assert judge.rule_path(Path("E:/old work/fyp 11")) == "//e/old work/fyp 11"
    assert judge.rule_path(Path("E:/$RECYCLE.BIN")) == "//e/$RECYCLE.BIN"
    assert judge.rule_path(Path("E:/a[1]")) == "//e/a\\[1\\]"


def test_both_prompts_come_from_the_preregistration():
    text = judge.PREREG.read_text("utf-8")
    p1, p2 = judge.prompt_from_prereg(text, "pass1"), judge.prompt_from_prereg(text, "pass2")
    assert (
        "inputs/pass1_blind_claims.csv" in p1 and '"labels": ["S", "U"]' in p1 and "no shell" in p1
    )
    assert (
        "inputs/pass2_for_judge.csv" in p2 and '"accurate": "Y"' in p2 and "(no line number)" in p2
    )
    assert p1 != p2


def test_check_accepts_well_formed_output_and_joins_it(run, tmp_path):
    root = run.parent / "judge"
    folder = root / "pass1"
    _judge_out(run, folder, "pass1")
    tdir = tmp_path / "t"
    _transcript(tdir, [("Read", {"file_path": str(folder / "inputs/reference_evidence.csv")}),
                       ("Grep", {"pattern": "^U", "path": "inputs"}), ("Glob", {"pattern": "out/*"}),
                       ("Write", {"file_path": str(folder / "out/case_01.jsonl")})])  # fmt: skip
    rec = judge.check("pass1", run, root, tdir)
    assert rec["units"] == 16 and rec["claims"] == 32 and rec["violations"] == 0
    assert rec["audit"]["models"] == ["claude-opus-5-5"] and rec["audit"]["tool_calls"] == 4
    joined = run / "judge/judge_pass1.jsonl"
    assert rec["joined_sha256"] == e20.file_sha256(joined)
    cases = [json.loads(x)["case"] for x in joined.read_text("utf-8").splitlines()]
    assert cases == sorted(cases)  # joined in case order


@pytest.mark.parametrize(("bad", "message"), [
    (lambda c, r: None if c == 2 else r, "missing case_02.jsonl"),
    (lambda c, r: r[:-1] if c == 1 else r, "missing"),
    (lambda c, r: r + [r[0]] if c == 1 else r, "appears twice"),
    (lambda c, r: [{**r[0], "labels": ["S", "X"]}] + r[1:] if c == 1 else r, "S, U or W"),
    (lambda c, r: [{**r[0], "labels": ["S"]}] + r[1:] if c == 1 else r, "same length"),
    (lambda c, r: [{**r[0], "lines": [[99], []]}] + r[1:] if c == 1 else r, "not an evidence line"),
    (lambda c, r: [{**r[0], "case": 3}] + r[1:] if c == 1 else r, "not a unit of case 1"),
    (lambda c, r: [{**r[0], "extra": 1}] + r[1:] if c == 1 else r, "fields must be"),
    (lambda c, r: [{**r[0], "claims": []}] + r[1:] if c == 1 else r, "non-empty"),
])  # fmt: skip
def test_check_refuses_malformed_output(run, tmp_path, bad, message):
    root = run.parent / "judge"
    _judge_out(run, root / "pass1", "pass1", bad)
    _transcript(tmp_path / "t", [])
    with pytest.raises(judge.JudgeToolError, match=message):
        judge.check("pass1", run, root, tmp_path / "t")
    assert not (run / "judge/judge_pass1.jsonl").exists()


@pytest.mark.parametrize(("calls", "message"), [
    ([("Bash", {"command": "cat x"})], "tool Bash used"),
    ([("mcp__Claude_Browser__navigate", {"url": "file:///E:/x"})], "tool mcp__Claude_Browser"),
    ([("Read", {"file_path": "E:/dev-cache/copilot/e20/run/sheets/SEALED_pass1_key.json"})], "outside the judge"),
    ([("Read", {"file_path": "../../20261026-000000-test/items.jsonl"})], "outside the judge"),
    ([("Grep", {"pattern": "x", "path": "C:/"})], "outside the judge"),
    ([("Write", {"file_path": "inputs/guide.md"})], "outside out/"),
    (None, "no session transcript"),
])  # fmt: skip
def test_the_transcript_audit_flags_any_tool_or_path_outside_the_folder(
    run, tmp_path, calls, message
):
    root = run.parent / "judge"
    _judge_out(run, root / "pass1", "pass1")
    if calls is not None:
        _transcript(tmp_path / "t", calls)
    with pytest.raises(judge.JudgeToolError, match=message):
        judge.check("pass1", run, root, tmp_path / "t")


def test_check_pass2(run, tmp_path):
    root = run.parent / "judge"
    _judge_out(run, root / "pass2", "pass2")
    _transcript(tmp_path / "t", [])
    assert judge.check("pass2", run, root, tmp_path / "t")["units"] == 11
    _judge_out(run, root / "pass2", "pass2", lambda c, r: [{**r[0], "accurate": "maybe"}] + r[1:])
    with pytest.raises(judge.JudgeToolError, match="Y or N"):
        judge.check("pass2", run, root, tmp_path / "t")


def test_the_human_pass1_sheet_has_the_judge_split_but_none_of_its_labels(run, tmp_path):
    s, sample_file = _sample(run, tmp_path)
    root = run.parent / "judge"
    with pytest.raises(judge.JudgeToolError, match="check pass1"):
        judge.human_sheet("pass1", run, sample_file, {})
    _judge_out(run, root / "pass1", "pass1")
    _transcript(tmp_path / "t", [])
    manifest = {"pass1": {"check": judge.check("pass1", run, root, tmp_path / "t")}}
    rec = judge.human_sheet("pass1", run, sample_file, manifest)
    path = run / "human/human_pass1.csv"
    text = path.read_text("utf-8-sig")
    assert rec["units"] == s["pass1"]["n"] and rec["rows"] == 2 * s["pass1"]["n"]
    assert "SECRET-JUDGE-NOTE" not in text and "a second claim" in text
    header, rows = judge.read_sheet(path, judge.HUMAN_P1_COLUMNS)
    assert header[0] == judge.HUMAN_P1_INSTRUCTION and header[2:] == GUIDE
    assert all(
        r["label"] == r["split problem"] == r["notes"] == "" for r in rows
    )  # no judge labels
    assert {r["unit id"] for r in rows} == set(judge.sample_units(sample_file, "pass1"))
    with pytest.raises(judge.JudgeToolError, match="never overwritten"):
        judge.human_sheet("pass1", run, sample_file, manifest)
    joined = run / "judge/judge_pass1.jsonl"  # a joined file changed after the check is refused
    joined.write_text(joined.read_text("utf-8") + "\n", "utf-8")
    path.unlink()
    with pytest.raises(judge.JudgeToolError, match="check pass1"):
        judge.human_sheet("pass1", run, sample_file, manifest)


def test_the_human_pass2_sheet_is_the_sampled_rows_unchanged(run, tmp_path):
    _, sample_file = _sample(run, tmp_path)
    rec = judge.human_sheet("pass2", run, sample_file, {})
    _, rows = judge.read_sheet(run / "human/human_pass2.csv", judge.P2_COLUMNS)
    _, src = judge.read_sheet(run / "sheets/pass2_citations.csv", judge.P2_COLUMNS)
    wanted = set(judge.sample_units(sample_file, "pass2"))
    assert rows == [r for r in src if r["unit id"] in wanted] and rec["units"] == 9


def test_the_tool_never_reads_the_sealed_key_or_the_item_records():
    src = (ROOT / "tools/e20_judge.py").read_text("utf-8")
    assert "SEALED" not in src and "items.jsonl" not in src and "load_records" not in src


def test_the_audit_counts_each_real_call_once_and_lists_the_d225g_deviations(run, tmp_path):
    root = run.parent / "judge"
    folder = root / "pass1"
    _judge_out(run, folder, "pass1")
    tdir = tmp_path / "t"
    tdir.mkdir()
    write = {"type": "tool_use", "id": "toolu_1", "name": "Write",
             "input": {"file_path": str(folder / "out/case_01.jsonl")}}  # fmt: skip
    plan = {
        "type": "tool_use",
        "id": "toolu_2",
        "name": "mcp__ccd_turn__update_plan",
        "input": {"items": []},
    }
    entries = [
        {"type": "assistant", "message": {"role": "assistant", "model": "claude-opus-5-5", "content": [write]}},
        {"type": "assistant", "message": {"role": "assistant", "model": "claude-opus-5-5", "content": [write]}},
        {"type": "api-request-blob", "message": {"role": "assistant", "content": [write, plan]}},
        {"type": "assistant", "message": {"role": "assistant", "model": "claude-opus-5-5", "content": [plan]}},
        {"type": "api-request-blob", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_9", "name": "Bash", "input": {"command": "x"}}]}},
    ]  # fmt: skip
    (tdir / "s.jsonl").write_text("\n".join(json.dumps(e) for e in entries), "utf-8")
    audit = judge.audit_transcripts(folder, tdir, judge.DECLARED["pass1"])
    assert (
        audit["tool_calls"] == 2 and audit["violations"] == []
    )  # a repeat record is not a new call
    assert [d["tool"] for d in audit["declared_deviations"]] == ["mcp__ccd_turn__update_plan"]
    assert (
        judge.check("pass1", run, root, tdir)["audit"]["declared_deviations"][0]["decision"]
        == "D225g"
    )
    # the same call is a violation for pass 2, where the tool is denied
    assert any(
        "mcp__ccd_turn__update_plan" in v
        for v in judge.audit_transcripts(folder, tdir)["violations"]
    )
    assert "mcp__ccd_turn" in judge.DENIED_TOOLS


def _labelled_run(run, tmp_path):
    """A run whose blank human pass-1 sheet is built, as after 'human-sheet pass1'."""
    _, sample_file = _sample(run, tmp_path)
    root = run.parent / "judge"
    _judge_out(run, root / "pass1", "pass1")
    _transcript(tmp_path / "t", [])
    manifest = {"pass1": {"check": judge.check("pass1", run, root, tmp_path / "t")}}
    manifest["pass1"]["human_sheet"] = judge.human_sheet("pass1", run, sample_file, manifest)
    return manifest


def _fill(xlsx, fill):
    from openpyxl import load_workbook

    wb = load_workbook(xlsx)
    ws = wb["Claims"]
    for row in ws.iter_rows(min_row=judge.FIRST_ROW):
        fill(row)
    wb.save(xlsx)


def test_the_labelling_workbook_is_protected_validated_and_has_no_connections(run, tmp_path):
    import zipfile

    from openpyxl import load_workbook

    manifest = _labelled_run(run, tmp_path)
    rec = judge.export_xlsx(run, manifest)
    xlsx = Path(rec["file"])
    assert (
        xlsx.name == "human_pass1_labelling.xlsx" and (run / "human/human_pass1_blank.csv").exists()
    )
    names = zipfile.ZipFile(xlsx).namelist()
    assert not any(
        k in n for n in names for k in ("connection", "queryTable", "externalLink", "vbaProject")
    )
    wb = load_workbook(xlsx)
    assert wb.sheetnames == ["Guide", "Claims", "Evidence"]
    guide = [c.value for c in wb["Guide"]["A"]]
    header, rows = judge.read_sheet(run / "human/human_pass1.csv", judge.HUMAN_P1_COLUMNS)
    assert [g or "" for g in guide] == header  # word for word
    ws = wb["Claims"]
    n = len(rows)
    assert (
        ws["A1"].value.startswith('="labelled "&(COUNTIF(F3:F') and f'" of {n}"' in ws["A1"].value
    )
    assert [c.value for c in ws[2]] == judge.HUMAN_P1_COLUMNS and ws.freeze_panes == "A3"
    got = [(ws.cell(i, 2).value, ws.cell(i, 1).value) for i in range(3, 3 + n)]
    assert got == sorted(got) and ws.cell(3 + n, 1).value is None  # by case, then unit id
    assert ws.protection.sheet and wb["Evidence"].protection.sheet and wb["Guide"].protection.sheet
    for i in (3, 2 + n):
        assert all(ws.cell(i, c).protection.locked for c in range(1, 6))  # ids, case, texts locked
        assert not any(ws.cell(i, c).protection.locked for c in range(6, 9))  # label, split, notes
    dv = {v.formula1: str(v.sqref) for v in ws.data_validations.dataValidation}
    assert dv == {'"S,U,W"': f"F3:F{2 + n}", '"Y"': f"G3:G{2 + n}"}
    rules = [r for cf in ws.conditional_formatting for r in cf.rules]
    assert rules and rules[0].formula == ['$F3=""']
    ev = wb["Evidence"]
    assert (
        ev.auto_filter.ref == f"A1:C{1 + 5 * len(UNITS_PER_CASE)}"
        and ev.protection.autoFilter is False
    )
    assert [c.value for c in ev[1]] == ["case", "line", "text"] and ev["C2"].value == "evidence 1.1"
    with pytest.raises(judge.JudgeToolError, match="never overwritten"):
        judge.export_xlsx(run, manifest)


def test_import_xlsx_writes_the_labels_back_with_the_same_checks(run, tmp_path):
    manifest = _labelled_run(run, tmp_path)
    blank_sha = manifest["pass1"]["human_sheet"]["sha256"]
    xlsx = Path(judge.export_xlsx(run, manifest)["file"])

    def label_all(row):
        row[5].value = "S" if row[3].value == 1 else "U"  # mocked labels
        row[6].value = "Y" if row[0].value == "U0001" else None
        row[7].value = "note"

    _fill(xlsx, label_all)
    rec = judge.import_xlsx(run, manifest)
    header, rows = judge.read_sheet(run / "human/human_pass1.csv", judge.HUMAN_P1_COLUMNS)
    _, blank = judge.read_sheet(run / "human/human_pass1_blank.csv", judge.HUMAN_P1_COLUMNS)
    assert [(r["unit id"], r["claim no"], r["claim"]) for r in rows] == [
        (r["unit id"], r["claim no"], r["claim"]) for r in blank
    ]  # the blank's order and text
    assert all(r["label"] in ("S", "U") and r["notes"] == "note" for r in rows)
    assert rec["rows"] == len(blank) and rec["sha256"] != blank_sha
    assert rec["split_problems"] == sum(1 for r in blank if r["unit id"] == "U0001")
    manifest["pass1"]["human_labels"] = rec
    assert (
        judge.import_xlsx(run, manifest)["sha256"] == rec["sha256"]
    )  # re-import of the same workbook


@pytest.mark.parametrize(("change", "message"), [
    (lambda row: None, "label must be S, U or W"),  # nothing labelled
    (lambda row: setattr(row[5], "value", "X"), "label must be S, U or W"),
    (lambda row: (setattr(row[5], "value", "S"), setattr(row[6], "value", "N")), "split problem must be Y"),
    (lambda row: (setattr(row[5], "value", "S"), setattr(row[4], "value", row[4].value + " edited")),
     "claim changed"),
    (lambda row: (setattr(row[5], "value", "S"), setattr(row[2], "value", "other text")), "unit text changed"),
    (lambda row: (setattr(row[5], "value", "S"), setattr(row[0], "value", "U9999")), "is missing"),
])  # fmt: skip
def test_import_xlsx_refuses_any_other_change(run, tmp_path, change, message):
    manifest = _labelled_run(run, tmp_path)
    xlsx = Path(judge.export_xlsx(run, manifest)["file"])
    before = judge.file_sha256(run / "human/human_pass1.csv")
    _fill(xlsx, change)
    with pytest.raises(judge.JudgeToolError, match=message):
        judge.import_xlsx(run, manifest)
    assert judge.file_sha256(run / "human/human_pass1.csv") == before  # nothing written


def test_export_xlsx_refuses_a_sheet_that_is_no_longer_the_blank(run, tmp_path):
    manifest = _labelled_run(run, tmp_path)
    with open(run / "human/human_pass1.csv", "a", encoding="utf-8") as f:
        f.write("U0001,1,x,1,x,S,,\n")
    with pytest.raises(judge.JudgeToolError, match="not the recorded blank"):
        judge.export_xlsx(run, manifest)


def test_a_text_starting_with_an_equals_sign_stays_text(tmp_path):
    from openpyxl import Workbook, load_workbook

    wb = Workbook()
    judge._text(wb.active["A1"], "=1+1 is not a formula")
    wb.save(tmp_path / "x.xlsx")
    assert load_workbook(tmp_path / "x.xlsx").active["A1"].value == "=1+1 is not a formula"
