"""Freeze record and results summary for Phase 2.

Run:  python -m vaultic.reports.phase2_summary freeze    (before any --final run)
      python -m vaultic.reports.phase2_summary summary   (after the final runs)

`freeze` writes research/frozen_configs.md: the SHA-256 of every Table 1 config plus the
split and V-column files, and the git commit. `summary` assembles research/phase2_results.md
from the generated tuning reports, Table 1, the exit gate, E2 and SHAP outputs, and re-hashes
the frozen files to show whether anything changed after freezing.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from vaultic.paths import CONFIG_DIR, REPO_ROOT, RESEARCH_DIR
from vaultic.reports.table1 import SPEC_PATH

FROZEN_PATH = RESEARCH_DIR / "frozen_configs.md"
SUMMARY_PATH = RESEARCH_DIR / "phase2_results.md"


def frozen_files(spec_path: Path = SPEC_PATH) -> list[Path]:
    spec = yaml.safe_load(spec_path.read_text("utf-8"))
    files = [CONFIG_DIR / f"{r['experiment']}.yaml" for r in spec["rows"]]
    return files + [CONFIG_DIR / "splits.yaml", CONFIG_DIR / "v_columns.yaml", spec_path]


def sha256(path: Path) -> str:
    # normalise line endings so a git checkout on another OS gives the same hash
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_freeze(files: list[Path], out: Path = FROZEN_PATH) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    lines = [
        "# Frozen configs for the Phase 2 final runs",
        "",
        f"Frozen {datetime.now().isoformat(timespec='seconds')} at git commit `{commit}`, before "
        "any --final run. Final runs are made once per baseline and never repeated after seeing "
        "results (a forced re-run is logged in research/decisions.md as FINAL-RERUN).",
        "",
        "| file | sha256 |",
        "|---|---|",
        *[f"| `{f.relative_to(REPO_ROOT).as_posix()}` | `{sha256(f)}` |" for f in files],
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def check_freeze(frozen: Path = FROZEN_PATH) -> list[str]:
    """Files whose current hash differs from the freeze record."""
    changed = []
    for name, digest in re.findall(
        r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", frozen.read_text("utf-8")
    ):
        path = REPO_ROOT / name
        if not path.exists() or sha256(path) != digest:
            changed.append(name)
    return changed


def _section(title: str, path: Path) -> list[str]:
    if not path.exists():
        return [
            f"## {title}",
            "",
            f"_Missing: `{path.relative_to(REPO_ROOT).as_posix()}` was not produced._",
            "",
        ]
    body = path.read_text("utf-8").splitlines()
    if body and body[0].startswith("# "):
        body = body[1:]
    return [f"## {title}", "", f"From `{path.relative_to(REPO_ROOT).as_posix()}`:", *body, ""]


def write_summary(out: Path = SUMMARY_PATH) -> None:
    changed = check_freeze() if FROZEN_PATH.exists() else None
    if changed is None:
        freeze_line = "No freeze record found."
    elif changed:
        freeze_line = f"**Changed after freezing:** {', '.join(changed)}."
    else:
        freeze_line = "All frozen configs are unchanged since the freeze."
    lines = [
        "# Phase 2 results",
        "",
        f"Assembled {datetime.now().isoformat(timespec='seconds')} by "
        "`python -m vaultic.reports.phase2_summary summary`. " + freeze_line,
        "",
        *_section("Phase 2 exit gate", RESEARCH_DIR / "phase2_exit_gate.md"),
        *_section("Table 1 (test period, final runs)", RESEARCH_DIR / "tables" / "table1_final.md"),
        *_section(
            "E2: FYP-1 original vs temporal", RESEARCH_DIR / "tables" / "e2_fyp1_comparison.md"
        ),
        *[
            _section(f"Tuning {b}", RESEARCH_DIR / f"tuning_{b}.md")
            for b in ("B5", "B3", "B4", "B1")
        ],
        *_section("B5 SHAP top 20", RESEARCH_DIR / "tables" / "b5_shap_top20.md"),
        "Figure: `research/figures/b5_shap_summary.png`.",
        "",
    ]
    flat = []
    for item in lines:
        flat.extend(item if isinstance(item, list) else [item])
    out.write_text("\n".join(flat), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["freeze", "summary"])
    args = parser.parse_args()
    if args.command == "freeze":
        write_freeze(frozen_files())
        print(f"wrote {FROZEN_PATH}")
    else:
        write_summary()
        print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
