"""E20 judge tooling (D225e, D225f): the seeded human samples, the isolated judge folders, the check of
the judge's output (with a transcript audit) and the human sample sheets. It reads the sheets of the run,
their manifest and the judge's output only: never the sealed pass-1 key or the run's item records. It
changes no frozen file and computes no metric (no kappa, no rates).

Run from the repository root:
    python tools/e20_judge.py sample
    python tools/e20_judge.py prepare pass1|pass2
    python tools/e20_judge.py check pass1|pass2 [--transcripts DIR]
    python tools/e20_judge.py human-sheet pass1|pass2
    python tools/e20_judge.py export-xlsx pass1   (D225h: the labelling workbook)
    python tools/e20_judge.py import-xlsx pass1   (D225h: its labels back into human_pass1.csv)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import string
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "research/tables/e20"
PREREG = REPO / "research/preregistration_e20.md"
E20_ROOT = Path("E:/dev-cache/copilot/e20")
CLAUDE_PROJECTS = Path.home() / ".claude/projects"

P1_COLUMNS = ["unit id", "case", "unit text", "claim split", "claim labels", "notes"]
P2_COLUMNS = ["unit id", "item", "system", "case", "question", "unit text with citations", "cited lines",
              "accurate"]  # fmt: skip
P2_JUDGE_COLUMNS = ["unit id", "case", "unit text with citations", "cited lines"]
HUMAN_P1_COLUMNS = [
    "unit id",
    "case",
    "unit text",
    "claim no",
    "claim",
    "label",
    "split problem",
    "notes",
]
PASSES = {
    "pass1": {"sheet": "pass1_blind_claims.csv", "columns": P1_COLUMNS, "per_case": 5, "seed": 2252,
              "prompt": "**Pass-1 prompt**"},
    "pass2": {"sheet": "pass2_citations.csv", "columns": P2_COLUMNS, "per_case": 3, "seed": 2254,
              "prompt": "**Pass-2 prompt**"},
}  # fmt: skip

# D225f: the judge has no shell, web, subagents, messaging or MCP tools (native Windows has no sandbox)
DENIED_TOOLS = [
    "Bash", "PowerShell", "WebFetch", "WebSearch", "Agent", "NotebookEdit", "SendMessage", "ListAgents",
    "RemoteTrigger", "mcp__Claude_Browser", "mcp__claude-in-chrome", "mcp__computer-use", "mcp__terminal",
    "mcp__ccd_session", "mcp__ccd_session_mgmt", "mcp__ccd_host", "mcp__ccd_directory", "mcp__ccd_view",
    "mcp__ccd_window", "mcp__ccd_sidebar", "mcp__ccd_settings", "mcp__ccd_pr", "mcp__ccd_connectors",
    "mcp__scheduled-tasks", "mcp__mcp-registry", "mcp__remote-devices", "mcp__visualize",
    "mcp__1a59c906-04da-521d-bda7-7f71b9f9e01c", "mcp__ccd_turn",  # mcp__ccd_turn added by D225g
]  # fmt: skip
# D225g: two desktop-app interface calls of the pass-1 judge (no file access, no labels), accepted as a
# declared deviation; listed in the audit record, not counted as violations, and denied for pass 2
DECLARED = {"pass1": {"mcp__ccd_turn__update_plan", "mcp__ccd_turn__wrap_up"}}
AUDIT_TOOLS = {"Read", "Grep", "Glob", "Write", "Edit", "MultiEdit", "TodoWrite", "TaskCreate", "TaskUpdate",
               "TaskList", "TaskGet", "ToolSearch"}  # fmt: skip
WRITE_TOOLS = {"Write", "Edit", "MultiEdit"}
PATH_KEYS = ("file_path", "path", "notebook_path")
LABELS = {"pass1": {"S", "U", "W"}, "pass2": {"Y", "N"}}
P1_KEYS = {"unit", "case", "claims", "labels", "lines", "note"}
P2_KEYS = {"unit", "case", "accurate", "note"}
HUMAN_P1_INSTRUCTION = (
    "E20 pass 1, human validation sample (D225e). Each row is one claim of a unit, as split by the judge. "
    "Label each claim in 'label' (S = supported, U = unsupported, W = wrong) against the case's reference "
    "evidence (reference_evidence.csv), by the guide below. If the split seems wrong, put Y in 'split "
    "problem' and still label the claim as written. Do not open the judge's output before this sheet is "
    "finished."
)


class JudgeToolError(Exception):
    """A refusal or a failed check; the message lists every problem."""


# ---- small helpers -------------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", "utf-8", newline="\n")
    tmp.replace(path)


def read_sheet(path: Path, columns: list[str]) -> tuple[list[str], list[dict]]:
    """A sheet written by e20._write_csv: header lines (one cell each), a blank row, the columns, rows."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if columns not in rows:
        raise JudgeToolError(f"{path.name}: column row {columns} not found")
    at = rows.index(columns)
    header = rows[:at]
    while header and header[-1] == []:
        header.pop()
    lines = [r[0] if r else "" for r in header]
    return lines, [dict(zip(columns, r, strict=True)) for r in rows[at + 1 :] if r]


def guide_lines(header: list[str]) -> list[str]:
    """The guide under a sheet's own instruction line (the line and the blank after it removed)."""
    rest = header[1:]
    return rest[1:] if rest and rest[0] == "" else rest


def read_evidence(path: Path) -> dict[int, set[int]]:
    _, rows = read_sheet(path, ["case", "line", "text"])
    lines: dict[int, set[int]] = {}
    for r in rows:
        lines.setdefault(int(r["case"]), set()).add(int(r["line"]))
    return lines


def default_run_dir() -> Path:
    run_id = json.loads((TABLES / "run_manifest.json").read_text("utf-8"))["run_id"]
    return E20_ROOT / run_id


def sheet_manifest(run_dir: Path) -> dict[str, str]:
    return json.loads((run_dir / "sheets/manifest.json").read_text("utf-8"))


def checked_sheet(run_dir: Path, name: str) -> Path:
    """A sheet whose bytes match sheets/manifest.json (the sheets are fixed at generation)."""
    path = run_dir / "sheets" / name
    want = sheet_manifest(run_dir).get(name)
    if want is None or file_sha256(path) != want:
        raise JudgeToolError(f"{name} does not match sheets/manifest.json")
    return path


# ---- sample --------------------------------------------------------------------------------------
def draw(ids_by_case: dict[int, list[str]], per_case: int, seed: int) -> dict[int, list[str]]:
    """D225e: cases in order, each case's ids in id order, rng.choice(n, min(k, n), replace=False), one rng."""
    rng = np.random.default_rng(seed)
    out = {}
    for case in sorted(ids_by_case):
        ids = sorted(ids_by_case[case])
        idx = rng.choice(len(ids), size=min(per_case, len(ids)), replace=False)
        out[case] = sorted(ids[int(i)] for i in idx)
    return out


def sample(run_dir: Path, out: Path) -> dict:
    result = {"rule": "D225e / D225f: drawn from the unit id and case columns only, before any judging",
              "run_id": run_dir.name}  # fmt: skip
    for name, spec in PASSES.items():
        path = checked_sheet(run_dir, spec["sheet"])
        _, rows = read_sheet(path, spec["columns"])
        by_case: dict[int, list[str]] = {}
        for r in rows:
            by_case.setdefault(int(r["case"]), []).append(r["unit id"])
        picked = draw(by_case, spec["per_case"], spec["seed"])
        result[name] = {"sheet": spec["sheet"], "sheet_sha256": file_sha256(path), "seed": spec["seed"],
                        "per_case": spec["per_case"], "units_in_sheet": len(rows),
                        "cases": len(by_case), "n": sum(len(v) for v in picked.values()),
                        "units": {str(k): v for k, v in picked.items()}}  # fmt: skip
    if out.exists():
        old = json.loads(out.read_text("utf-8"))
        if old != result:
            raise JudgeToolError(
                f"{out} exists with a different sample; the sample is fixed (D225e)"
            )
        return old
    write_json(out, result)
    return result


def sample_units(sample_file: Path, name: str) -> list[str]:
    s = json.loads(sample_file.read_text("utf-8"))[name]
    return sorted(u for ids in s["units"].values() for u in ids)


def pushed(path: Path, repo: Path = REPO) -> bool:
    """True if the file is tracked and identical to its copy on the pushed upstream branch."""
    rel = path.resolve().relative_to(repo.resolve()).as_posix()

    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)

    up = git("rev-parse", f"@{{u}}:{rel}")
    local = git("hash-object", rel)
    return (
        up.returncode == 0 and local.returncode == 0 and up.stdout.strip() == local.stdout.strip()
    )


# ---- prepare -------------------------------------------------------------------------------------
def prompt_from_prereg(text: str, name: str) -> str:
    """The block-quoted prompt under its label in section 11.2 (one source for PROMPT.md)."""
    label = PASSES[name]["prompt"]
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(label))
    quoted = []
    for ln in lines[start + 1 :]:
        if ln.startswith(">"):
            quoted.append(ln[2:] if ln.startswith("> ") else "")
        elif quoted:
            break
    if not quoted:
        raise JudgeToolError(f"no prompt under {label} in the pre-registration")
    return "\n".join(quoted).strip() + "\n"


def rule_path(path: Path) -> str:
    """A Windows path in the POSIX form of permission rules (E:\\a b\\c -> //e/a b/c), specials escaped."""
    p = Path(path).resolve()
    parts = [p.drive.rstrip(":").lower()] + list(p.parts[1:])
    escaped = [re.sub(r"([\[\]*?!\\#])", r"\\\1", x) for x in parts]
    return "//" + "/".join(escaped)


def read_denies(folder: Path, extra: list[Path], drives: list[str] | None = None) -> list[str]:
    """Read denies for every sibling of every ancestor of the judge folder, the extra paths, and every
    other drive (D225f). Deny always wins over allow, so the folder itself is never denied."""
    folder = folder.resolve()
    chain = [*reversed(folder.parents), folder]
    rules: list[str] = []

    def deny(entry: Path) -> None:
        rp = rule_path(entry)
        rules.extend([f"Read({rp})", f"Read({rp}/**)"] if entry.is_dir() or not entry.exists()
                     else [f"Read({rp})"])  # fmt: skip

    for parent, keep in zip(chain, chain[1:], strict=False):
        try:
            entries = sorted(parent.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.resolve() != keep:
                deny(entry)
    for p in extra:
        if Path(p).resolve() != folder:
            deny(Path(p))
    own = folder.drive.rstrip(":").upper()
    if drives is None:
        drives = [d for d in string.ascii_uppercase if d != own and Path(f"{d}:/").exists()]
    rules += [f"Read(//{d.lower()}/**)" for d in drives if d.upper() != own]
    return list(dict.fromkeys(rules))


def judge_settings(folder: Path, extra: list[Path], drives: list[str] | None = None) -> dict:
    deny = DENIED_TOOLS + read_denies(folder, extra, drives)
    deny += ["Edit(./inputs/**)", "Edit(./PROMPT.md)", "Edit(./.claude/**)"]
    return {"permissions": {"deny": deny,
                            "allow": ["Read(./PROMPT.md)", "Read(./inputs/**)", "Read(./out/**)",
                                      "Edit(./out/**)"],
                            "blockReadsOutsideWorkingDirectories": True}}  # fmt: skip


def prepare(name: str, run_dir: Path, judge_root: Path, sample_file: Path, prereg: Path = PREREG,
            repo: Path = REPO, drives: list[str] | None = None, require_pushed: bool = True) -> dict:  # fmt: skip
    if not sample_file.exists() or (require_pushed and not pushed(sample_file, repo)):
        raise JudgeToolError(
            "the human sample must be committed and pushed before any judge folder (D225f)"
        )
    folder = judge_root / name
    out = folder / "out"
    if out.exists() and any(out.iterdir()):
        raise JudgeToolError(f"{out} is not empty: a judge has already written here")
    spec = PASSES[name]
    sheet = checked_sheet(run_dir, spec["sheet"])
    header, rows = read_sheet(sheet, spec["columns"])
    inputs = folder / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)
    if name == "pass1":
        for f in (spec["sheet"], "reference_evidence.csv"):
            src = checked_sheet(run_dir, f)
            shutil.copyfile(src, inputs / f)
            if file_sha256(inputs / f) != file_sha256(src):
                raise JudgeToolError(f"copy of {f} differs from the sheet")
    else:
        with open(inputs / "pass2_for_judge.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(P2_JUDGE_COLUMNS)
            w.writerows([[r[c] for c in P2_JUDGE_COLUMNS] for r in rows])
    (inputs / "guide.md").write_text("\n".join(header) + "\n", "utf-8", newline="\n")
    (folder / "PROMPT.md").write_text(prompt_from_prereg(prereg.read_text("utf-8"), name), "utf-8",
                                      newline="\n")  # fmt: skip
    others = [judge_root / n for n in PASSES if n != name]
    settings = judge_settings(folder, [run_dir, repo, *others], drives)
    (folder / ".claude").mkdir(exist_ok=True)
    write_json(folder / ".claude/settings.json", settings)
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    record = {"rule": "D225e / D225f", "pass": name, "folder": str(folder), "run_id": run_dir.name,
              "source_sheet_sha256": file_sha256(sheet), "units": len(rows),
              "prepared": datetime.now(UTC).isoformat(timespec="seconds"),
              "sha256": {p.relative_to(folder).as_posix(): file_sha256(p) for p in files}}  # fmt: skip
    return record


# ---- check ---------------------------------------------------------------------------------------
def transcript_dir(folder: Path) -> Path:
    return CLAUDE_PROJECTS / re.sub(r"[^A-Za-z0-9]", "-", str(folder.resolve()))


def _inside(path_text: str, root: Path, cwd: Path) -> bool:
    """Is the path (relative ones resolve against the session's folder) inside root?"""
    p = Path(path_text)
    p = (cwd / p) if not p.is_absolute() else p
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def audit_transcripts(folder: Path, tdir: Path, declared: set[str] = frozenset()) -> dict:
    """Every tool call of the judge's sessions: allowed tool, paths inside the folder, writes in out/.
    Only real assistant entries count, each tool_use id once: the transcript also holds request records
    (api-request-blob and others) that repeat earlier calls (D225g)."""
    files = sorted(tdir.glob("*.jsonl")) if tdir.exists() else []
    violations, models, calls, seen, deviations = [], set(), 0, set(), []
    for f in files:
        for raw in f.read_text("utf-8").splitlines():
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg = entry.get("message") or {}
            if entry.get("type") != "assistant" or msg.get("role") != "assistant":
                continue
            if msg.get("model"):
                models.add(msg["model"])
            content = msg.get("content")
            for item in content if isinstance(content, list) else []:
                if not isinstance(item, dict) or item.get("type") != "tool_use":
                    continue
                if item.get("id") is not None:
                    if item["id"] in seen:
                        continue
                    seen.add(item["id"])
                calls += 1
                tool, args = item.get("name", ""), item.get("input") or {}
                if tool in declared:
                    deviations.append({"tool": tool, "input": args, "decision": "D225g"})
                    continue
                if tool not in AUDIT_TOOLS:
                    violations.append(f"{f.name}: tool {tool} used")
                    continue
                paths = [args[k] for k in PATH_KEYS if isinstance(args.get(k), str)]
                if tool == "Glob" and isinstance(args.get("pattern"), str):
                    paths.append(args["pattern"].split("*")[0] or ".")
                for p in paths:
                    if not _inside(p, folder, folder):
                        violations.append(f"{f.name}: {tool} outside the judge folder: {p}")
                    elif tool in WRITE_TOOLS and not _inside(p, folder / "out", folder):
                        violations.append(f"{f.name}: {tool} outside out/: {p}")
    if not files:
        violations.append(f"no session transcript found in {tdir}")
    return {"transcripts": [f.name for f in files], "tool_calls": calls, "models": sorted(models),
            "declared_deviations": deviations, "violations": violations}  # fmt: skip


def check_output(name: str, folder: Path, sheet_rows: list[dict], evidence: dict[int, set[int]] | None
                 ) -> tuple[list[str], list[dict]]:  # fmt: skip
    """Problems with the judge's case files, and the records joined in case order."""
    problems, joined = [], []
    case_of = {r["unit id"]: int(r["case"]) for r in sheet_rows}
    seen: set[str] = set()
    keys = P1_KEYS if name == "pass1" else P2_KEYS
    for case in sorted(set(case_of.values())):
        path = folder / "out" / f"case_{case:02d}.jsonl"
        if not path.exists():
            problems.append(f"missing {path.name}")
            continue
        want = sorted(u for u, c in case_of.items() if c == case)
        got = []
        for n, raw in enumerate(path.read_text("utf-8-sig").splitlines(), 1):
            if not raw.strip():
                continue
            where = f"{path.name}:{n}"
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                problems.append(f"{where}: not JSON")
                continue
            if not isinstance(rec, dict) or set(rec) != keys:
                problems.append(f"{where}: fields must be {sorted(keys)}")
                continue
            unit = rec["unit"]
            if unit in seen:
                problems.append(f"{where}: {unit} appears twice")
            seen.add(unit)
            got.append(unit)
            if case_of.get(unit) != case or rec["case"] != case:
                problems.append(f"{where}: {unit} is not a unit of case {case}")
            if not isinstance(rec["note"], str):
                problems.append(f"{where}: note must be text")
            if name == "pass1":
                claims, labels, lines = rec["claims"], rec["labels"], rec["lines"]
                if not (isinstance(claims, list) and claims and all(isinstance(c, str) and c.strip()
                                                                     for c in claims)):  # fmt: skip
                    problems.append(f"{where}: claims must be a non-empty list of texts")
                    continue
                if not (isinstance(labels, list) and isinstance(lines, list)
                        and len(labels) == len(claims) == len(lines)):  # fmt: skip
                    problems.append(f"{where}: claims, labels and lines must have the same length")
                    continue
                if any(lab not in LABELS["pass1"] for lab in labels):
                    problems.append(f"{where}: labels must be S, U or W")
                for ls in lines:
                    if not isinstance(ls, list) or not all(isinstance(x, int) for x in ls):
                        problems.append(f"{where}: lines must be lists of line numbers")
                    elif evidence is not None and any(
                        x not in evidence.get(case, set()) for x in ls
                    ):
                        problems.append(
                            f"{where}: a line number is not an evidence line of case {case}"
                        )
            elif rec["accurate"] not in LABELS["pass2"]:
                problems.append(f"{where}: accurate must be Y or N")
            joined.append(rec)
        if sorted(got) != want and not any(p.startswith(path.name) for p in problems):
            missing, extra = set(want) - set(got), set(got) - set(want)
            problems.append(f"{path.name}: missing {sorted(missing)[:5]}... ({len(missing)}), "
                            f"extra {sorted(extra)[:5]} ({len(extra)})")  # fmt: skip
    strays = sorted(p.name for p in (folder / "out").glob("*")
                    if not re.fullmatch(r"case_\d\d\.jsonl", p.name))  # fmt: skip
    if strays:
        problems.append(f"unexpected files in out/: {strays}")
    return problems, joined


def check(name: str, run_dir: Path, judge_root: Path, tdir: Path | None = None) -> dict:
    folder = judge_root / name
    spec = PASSES[name]
    _, rows = read_sheet(checked_sheet(run_dir, spec["sheet"]), spec["columns"])
    evidence = (
        read_evidence(checked_sheet(run_dir, "reference_evidence.csv")) if name == "pass1" else None
    )
    problems, joined = check_output(name, folder, rows, evidence)
    audit = audit_transcripts(folder, tdir or transcript_dir(folder), DECLARED.get(name, set()))
    problems += audit["violations"]
    if problems:
        raise JudgeToolError(f"check {name} failed:\n- " + "\n- ".join(problems))
    dest = run_dir / "judge" / f"judge_{name}.jsonl"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in joined), "utf-8",
                    newline="\n")  # fmt: skip
    case_files = sorted((folder / "out").glob("case_*.jsonl"))
    return {"checked": datetime.now(UTC).isoformat(timespec="seconds"), "units": len(joined),
            "claims": sum(len(r["claims"]) for r in joined) if name == "pass1" else None,
            "joined_file": str(dest), "joined_sha256": file_sha256(dest),
            "case_files_sha256": {p.name: file_sha256(p) for p in case_files},
            "audit": {k: v for k, v in audit.items() if k != "violations"}, "violations": 0}  # fmt: skip


# ---- human sheets --------------------------------------------------------------------------------
def write_sheet(path: Path, header: list[str], columns: list[str], rows: list[list]) -> None:
    if path.exists():
        raise JudgeToolError(f"{path} exists; a human sheet is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        for line in header:
            w.writerow([line])
        w.writerow([])
        w.writerow(columns)
        w.writerows(rows)


def human_sheet(name: str, run_dir: Path, sample_file: Path, manifest: dict) -> dict:
    spec = PASSES[name]
    header, rows = read_sheet(checked_sheet(run_dir, spec["sheet"]), spec["columns"])
    wanted = set(sample_units(sample_file, name))
    path = run_dir / "human" / f"human_{name}.csv"
    if name == "pass1":
        done = manifest.get("pass1", {}).get("check")
        joined = run_dir / "judge/judge_pass1.jsonl"
        if not done or not joined.exists() or file_sha256(joined) != done["joined_sha256"]:
            raise JudgeToolError(
                "run 'check pass1' first: the human sheet uses the checked judge split"
            )
        split = {}
        for raw in joined.read_text("utf-8").splitlines():
            rec = json.loads(raw)
            split[rec["unit"]] = rec["claims"]  # claim texts only: no labels, lines or notes
        out_rows = []
        for r in rows:
            if r["unit id"] in wanted:
                for n, claim in enumerate(split[r["unit id"]], 1):
                    out_rows.append([r["unit id"], r["case"], r["unit text"], n, claim, "", "", ""])
        write_sheet(
            path, [HUMAN_P1_INSTRUCTION, ""] + guide_lines(header), HUMAN_P1_COLUMNS, out_rows
        )
        units = len({x[0] for x in out_rows})
    else:
        out_rows = [[r[c] for c in P2_COLUMNS] for r in rows if r["unit id"] in wanted]
        write_sheet(path, header, P2_COLUMNS, out_rows)
        units = len(out_rows)
    return {"file": str(path), "sha256": file_sha256(path), "units": units, "rows": len(out_rows),
            "built": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip


# ---- labelling workbook (D225h) ------------------------------------------------------------------
XLSX_NAME, BLANK_NAME = "human_pass1_labelling.xlsx", "human_pass1_blank.csv"
EDITABLE = ("label", "split problem", "notes")
FIRST_ROW = 3  # row 1: the counter, row 2: the column names


def _text(cell, value) -> None:
    """A text cell that stays text (openpyxl would turn a leading '=' into a formula)."""
    cell.value = value
    cell.data_type = "s"


def _blank(run_dir: Path, manifest: dict) -> Path:
    """The untouched pass-1 sheet: human_pass1.csv while it is still the recorded blank, else its copy."""
    want = manifest.get("pass1", {}).get("human_sheet", {}).get("sha256")
    human = run_dir / "human"
    copy = human / BLANK_NAME
    if copy.exists():
        if file_sha256(copy) != want:
            raise JudgeToolError(f"{copy} is not the recorded blank sheet")
        return copy
    if not want or file_sha256(human / "human_pass1.csv") != want:
        raise JudgeToolError(
            "human_pass1.csv is not the recorded blank sheet; nothing is built from it"
        )
    shutil.copyfile(human / "human_pass1.csv", copy)
    return copy


def export_xlsx(run_dir: Path, manifest: dict) -> dict:
    """D225h: a protected workbook over the blank pass-1 sheet; no connections, nothing can refresh."""
    from openpyxl import Workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Font, PatternFill, Protection
    from openpyxl.worksheet.datavalidation import DataValidation

    out = run_dir / "human" / XLSX_NAME
    if out.exists():
        raise JudgeToolError(f"{out} exists; it may hold labels and is never overwritten")
    blank = _blank(run_dir, manifest)
    header, rows = read_sheet(blank, HUMAN_P1_COLUMNS)
    _, evidence = read_sheet(
        checked_sheet(run_dir, "reference_evidence.csv"), ["case", "line", "text"]
    )
    rows = sorted(rows, key=lambda r: (int(r["case"]), r["unit id"], int(r["claim no"])))
    wrap, bold = Alignment(wrap_text=True, vertical="top"), Font(bold=True)
    wb = Workbook()
    wb.calculation.fullCalcOnLoad = True

    guide = wb.active
    guide.title = "Guide"
    guide.column_dimensions["A"].width = 130
    for i, line in enumerate(header, 1):
        _text(guide.cell(i, 1), line)
        guide.cell(i, 1).alignment = wrap
    guide.protection.sheet = True

    ws = wb.create_sheet("Claims")
    n, last = len(rows), FIRST_ROW + len(rows) - 1
    f = f"F{FIRST_ROW}:F{last}"
    ws["A1"] = f'="labelled "&(COUNTIF({f},"S")+COUNTIF({f},"U")+COUNTIF({f},"W"))&" of {n}"'
    ws["A1"].font = Font(bold=True, size=13)
    for c, name in enumerate(HUMAN_P1_COLUMNS, 1):
        _text(ws.cell(2, c), name)
        ws.cell(2, c).font = bold
    for i, r in enumerate(rows, FIRST_ROW):
        for c, name in enumerate(HUMAN_P1_COLUMNS, 1):
            cell = ws.cell(i, c)
            if name in ("case", "claim no"):
                cell.value = int(r[name])
            elif name in EDITABLE:
                cell.protection = Protection(locked=False)
            else:
                _text(cell, r[name])
            cell.alignment = wrap
    for col, width in zip("ABCDEFGH", (9, 6, 60, 7, 50, 7, 9, 30), strict=True):
        ws.column_dimensions[col].width = width
    ws.freeze_panes = f"A{FIRST_ROW}"
    label = DataValidation(type="list", formula1='"S,U,W"', allow_blank=True, showErrorMessage=True,
                           errorTitle="label", error="Only S, U or W.")  # fmt: skip
    split = DataValidation(type="list", formula1='"Y"', allow_blank=True, showErrorMessage=True,
                           errorTitle="split problem", error="Only Y, or leave it blank.")  # fmt: skip
    label.add(f)
    split.add(f"G{FIRST_ROW}:G{last}")
    ws.add_data_validation(label)
    ws.add_data_validation(split)
    ws.conditional_formatting.add(
        f"A{FIRST_ROW}:H{last}",
        FormulaRule(formula=[f'$F{FIRST_ROW}=""'], fill=PatternFill("solid", fgColor="FFF2CC")),
    )
    ws.protection.sheet = True
    ws.protection.formatColumns = ws.protection.formatRows = False  # widths may be changed

    ev = wb.create_sheet("Evidence")
    for c, name in enumerate(("case", "line", "text"), 1):
        _text(ev.cell(1, c), name)
        ev.cell(1, c).font = bold
    for i, r in enumerate(evidence, 2):
        ev.cell(i, 1).value, ev.cell(i, 2).value = int(r["case"]), int(r["line"])
        _text(ev.cell(i, 3), r["text"])
        ev.cell(i, 3).alignment = wrap
    ev.column_dimensions["C"].width = 120
    ev.freeze_panes = "A2"
    ev.auto_filter.ref = f"A1:C{len(evidence) + 1}"
    ev.protection.sheet = True
    ev.protection.autoFilter = False  # filtering stays allowed on the protected sheet
    ev.protection.formatColumns = ev.protection.formatRows = False

    wb.active = wb.index(ws)
    wb.save(out)
    return {"file": str(out), "sha256_at_build": file_sha256(out), "blank": str(blank),
            "blank_sha256": file_sha256(blank), "rows": n,
            "built": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip


def _cell_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v)


def import_xlsx(run_dir: Path, manifest: dict) -> dict:
    """D225h: the workbook's labels back into human_pass1.csv, refused unless nothing but label /
    split problem / notes changed and every label is S, U or W."""
    from openpyxl import load_workbook

    blank = _blank(run_dir, manifest)
    header, rows = read_sheet(blank, HUMAN_P1_COLUMNS)
    ws = load_workbook(run_dir / "human" / XLSX_NAME)["Claims"]
    if [_cell_text(c.value) for c in ws[2][: len(HUMAN_P1_COLUMNS)]] != HUMAN_P1_COLUMNS:
        raise JudgeToolError("Claims: the column row has changed")
    got, problems = {}, []
    for i, cells in enumerate(
        ws.iter_rows(min_row=FIRST_ROW, max_col=len(HUMAN_P1_COLUMNS)), FIRST_ROW
    ):
        r = dict(zip(HUMAN_P1_COLUMNS, (_cell_text(c.value) for c in cells), strict=True))
        if not any(r.values()):
            continue
        key = (r["unit id"], r["claim no"])
        if key in got:
            problems.append(f"row {i}: {key} appears twice")
        got[key] = (i, r)
    want = {(r["unit id"], r["claim no"]): r for r in rows}
    for key in sorted(set(want) - set(got)):
        problems.append(f"{key} is missing")
    for key in sorted(set(got) - set(want)):
        problems.append(f"row {got[key][0]}: {key} is not in the blank sheet")
    for key in sorted(set(want) & set(got)):
        i, r = got[key]
        for name in ("case", "unit text", "claim"):
            if r[name] != want[key][name]:
                problems.append(f"row {i}: {name} changed")
        if r["label"].strip() not in ("S", "U", "W"):
            problems.append(f"row {i}: label must be S, U or W (is {r['label'] or 'empty'!r})")
        if r["split problem"].strip() not in ("", "Y"):
            problems.append(f"row {i}: split problem must be Y or blank")
    if problems:
        shown = problems[:40] + (
            [f"... and {len(problems) - 40} more"] if len(problems) > 40 else []
        )
        raise JudgeToolError("import-xlsx pass1 refused:\n- " + "\n- ".join(shown))
    path = run_dir / "human/human_pass1.csv"
    allowed = {
        manifest["pass1"]["human_sheet"]["sha256"],
        manifest["pass1"].get("human_labels", {}).get("sha256"),
    }
    if file_sha256(path) not in allowed:
        raise JudgeToolError(f"{path} is neither the blank nor the last import; not overwritten")
    out_rows = []
    for r in rows:
        g = got[(r["unit id"], r["claim no"])][1]
        out_rows.append([r["unit id"], r["case"], r["unit text"], r["claim no"], r["claim"],
                         g["label"].strip(), g["split problem"].strip(), g["notes"]])  # fmt: skip
    tmp = path.with_suffix(".csv.tmp")
    tmp.unlink(missing_ok=True)
    write_sheet(tmp, header, HUMAN_P1_COLUMNS, out_rows)
    tmp.replace(path)
    return {"file": str(path), "sha256": file_sha256(path), "rows": len(out_rows),
            "split_problems": sum(1 for x in out_rows if x[6] == "Y"),
            "imported": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip


# ---- command line --------------------------------------------------------------------------------
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "command",
        choices=["sample", "prepare", "check", "human-sheet", "export-xlsx", "import-xlsx"],
    )
    p.add_argument("which", nargs="?", choices=list(PASSES))
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--judge-root", type=Path, default=E20_ROOT / "judge")
    p.add_argument("--transcripts", type=Path)
    a = p.parse_args(argv)
    run_dir = a.run_dir or default_run_dir()
    sample_file, manifest_file = TABLES / "human_sample.json", TABLES / "judge_manifest.json"
    manifest = (
        json.loads(manifest_file.read_text("utf-8"))
        if manifest_file.exists()
        else {"rule": "D225e / D225f: hashes and counts only, never labels", "run_id": run_dir.name}
    )
    if a.command != "sample" and not a.which:
        p.error(f"{a.command} needs pass1 or pass2")
    try:
        if a.command == "sample":
            s = sample(run_dir, sample_file)
            print(
                f"sample: pass1 {s['pass1']['n']} units, pass2 {s['pass2']['n']} units -> {sample_file}"
            )
            return
        if a.command == "prepare":
            manifest.setdefault(a.which, {})["prepare"] = prepare(
                a.which, run_dir, a.judge_root, sample_file
            )
            print(f"prepared {a.judge_root / a.which}")
        elif a.command == "check":
            manifest.setdefault(a.which, {})["check"] = check(
                a.which, run_dir, a.judge_root, a.transcripts
            )
            print(f"check {a.which}: passed ({manifest[a.which]['check']['units']} units)")
        elif a.command in ("export-xlsx", "import-xlsx"):
            if a.which != "pass1":
                p.error(f"{a.command} is for pass1 only")
            if a.command == "export-xlsx":
                rec = manifest["pass1"]["labelling_xlsx"] = export_xlsx(run_dir, manifest)
                print(f"workbook: {rec['file']} ({rec['rows']} claims)")
            else:
                rec = manifest["pass1"]["human_labels"] = import_xlsx(run_dir, manifest)
                print(f"imported {rec['rows']} labelled claims into {rec['file']}")
        else:
            manifest.setdefault(a.which, {})["human_sheet"] = human_sheet(a.which, run_dir, sample_file,
                                                                          manifest)  # fmt: skip
            print(f"human sheet: {manifest[a.which]['human_sheet']['file']}")
    except JudgeToolError as e:
        raise SystemExit(str(e)) from None
    write_json(manifest_file, manifest)


if __name__ == "__main__":
    main()
