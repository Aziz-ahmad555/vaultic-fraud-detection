"""Regenerate every paper table and figure from existing harness runs (roadmap Phase 13:
`make paper-results`). Nothing here trains a model: each step only reads experiments/runs/.

Run:  python -m vaultic.reports.paper_results          (all steps)
      python -m vaultic.reports.paper_results --list   (show the steps)
Also: `make paper-results` (Makefile) or `powershell -File tools/paper_results.ps1`.

Every step runs even if an earlier one fails; failures are listed at the end and the exit code
is non-zero. Add new tables and figures to STEPS as they are built.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from vaultic.paths import REPO_ROOT


@dataclass(frozen=True)
class Step:
    name: str
    args: tuple[str, ...]  # arguments to the Python interpreter
    output: str  # what it writes, for the listing


STEPS: tuple[Step, ...] = (
    Step(
        "Table 1: baselines (final runs, test period)",
        ("-m", "vaultic.reports.table1", "--mode", "final"),
        "research/tables/table1_final.md, .csv",
    ),
    Step(
        "E2: FYP-1 original vs temporal split",
        ("-m", "vaultic.reports.e2_fyp1"),
        "research/tables/e2_fyp1_comparison.md, .csv",
    ),
)


def run_all(steps=None) -> list[dict]:
    # read the registry at call time, so steps added to STEPS later are always included
    steps = STEPS if steps is None else steps
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    results = []
    for step in steps:
        started = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, *step.args], cwd=REPO_ROOT, env=env, capture_output=True, text=True
        )
        results.append(
            {
                "step": step.name,
                "ok": proc.returncode == 0,
                "seconds": round(time.perf_counter() - started, 1),
                "error": "" if proc.returncode == 0 else (proc.stderr or proc.stdout)[-1500:],
            }
        )
        print(f"{'ok  ' if proc.returncode == 0 else 'FAIL'} {step.name}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    args = parser.parse_args()
    if args.list:
        for step in STEPS:
            print(f"- {step.name}  ->  {step.output}")
        return
    results = run_all()
    failed = [r for r in results if not r["ok"]]
    for r in failed:
        print(f"\n--- {r['step']} failed:\n{r['error']}")
    print(f"\n{len(results) - len(failed)} of {len(results)} steps succeeded")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
