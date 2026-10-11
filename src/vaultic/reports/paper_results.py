"""Regenerate every paper table and figure from existing harness runs (roadmap Phase 13:
`make paper-results`). Nothing here trains a model: each step only reads experiments/runs/.

Run:  python -m vaultic.reports.paper_results          (all steps)
      python -m vaultic.reports.paper_results --list   (show the steps)
Also: `make paper-results` (Makefile) or `powershell -File tools/paper_results.ps1`.

In a worktree without the git-ignored FYP-1 artifacts, set VAULTIC_FYP1_DIR to a checkout's
legacy/fyp1 for the E2 step; VAULTIC_DATA_DIR as for every run.

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


B5 = "experiments/runs/EXP-009/20261008-211052-888539"


def _gain(name: str, base: str, new: str, title: str, *extra: str) -> Step:
    return Step(f"gain {name}: {title}",
                ("-m", "vaultic.reports.gain", base, new, "--name", name, "--title", title, *extra),
                f"research/tables/gain_{name}.md, .json")  # fmt: skip


def _render(report: str, title: str, output: str) -> Step:
    return Step(title, ("-m", "vaultic.reports.render_saved", report), output)


STEPS: tuple[Step, ...] = (
    Step(
        "Table 1: baselines incl. the B1 (original) row (final runs, test period)",
        ("-m", "vaultic.reports.table1", "--mode", "final"),
        "research/tables/table1_final.md, .csv",
    ),
    Step(
        "E2: FYP-1 original vs temporal split",
        ("-m", "vaultic.reports.e2_fyp1"),
        "research/tables/e2_fyp1_comparison.md, .csv",
    ),
    # development gains over B5 (validation period only)
    _gain("b5_drop_labels", B5, "experiments/runs/EXP-009-drop-labels/20261009-101431-746940",
          "Ablation: B5 without the label-derived group (uid_fraud_known, uid_fraud_rate_known)"),
    _gain("b5_behavioral", B5, "experiments/runs/EXP-103-frozen/20261009-105027-836010",
          "Phase 3: B5 + behavioral features over B5 (frozen B5 hyperparameters)"),
    _gain("b5_graph", B5, "experiments/runs/EXP-105-frozen/20261009-123145-610994",
          "Phase 4: B5 + graph features (setting C) over B5 (frozen B5 hyperparameters)"),
    _gain("b5_anomaly", B5, "experiments/runs/EXP-108-frozen/20261009-133329-203930",
          "Phase 6: B5 + anomaly scores over B5 (frozen B5 hyperparameters)"),
    _gain("e3_tuned", B5, "experiments/runs/EXP-103-tuned/20261010-151421-193228",
          "E3 (B5 + behavioral features, tuned with B5's budget) over B5", "--groups", "full"),
    _gain("e5_tuned", B5, "experiments/runs/EXP-105-tuned/20261010-164859-235447",
          "E5 (B5 + graph features, tuned with B5's budget) over B5", "--groups", "full"),
    _gain("e8_tuned", B5, "experiments/runs/EXP-108-tuned/20261010-191717-729992",
          "E8 (B5 + anomaly features, tuned with B5's budget) over B5", "--groups", "full"),
    *[_gain(f"e25_L{L}", B5, f"experiments/runs/EXP-125-L{L}/{run}",
            f"E25: B5 trained only on labels mature {L} days before validation, over B5",
            "--groups", "full")
      for L, run in ((7, "20261010-194606-655896"), (30, "20261010-193050-481642"),
                     (60, "20261010-195707-522990"))],
    # emerging fraud (Phase 6 exit gate)
    Step("Emerging fraud: hold-out groups vs random", ("-m", "vaultic.reports.emerging", "report"),
         "research/tables/emerging_fraud.md, .json"),
    _gain("holdout_R_b5", B5, "experiments/runs/EXP-009-holdout-R/20261009-112357-306004",
          "Emerging fraud: B5 trained without ProductCD R frauds vs full B5, on ProductCD R "
          "validation rows", "--subset", "ProductCD=R"),
    _gain("holdout_R_anomaly", "experiments/runs/EXP-009-holdout-R/20261009-112357-306004",
          "experiments/runs/EXP-108-holdout-R/20261009-142427-959567",
          "Emerging fraud: B5 + anomaly vs B5, both trained without ProductCD R frauds, on "
          "ProductCD R validation rows", "--subset", "ProductCD=R"),
    # fusion (rendered from the saved evaluation JSON; nothing is refitted)
    _render("d95", "D95: MVAF vs F0-F7, B5 on days 147-150 (validation)",
            "research/tables/fusion_d95.md"),
    _render("final", "E10-E12: Paper 2 final results (test period, EXP-200-final)",
            "research/tables/final_e10.md, final_e11.md, final_e12.md, "
            "research/figures/final_e12_reliability_*.png"),
    _render("f3-diagnosis", "F3 collapse diagnosis (post-hoc, D117/D118)",
            "research/f3_collapse_diagnosis.md"),
    _render("f3-sensitivity", "F3-log sensitivity (post-hoc, D119)",
            "research/f3_log_sensitivity.md"),
    _render("e5-drift", "E5 graph-feature drift (post-hoc, validation only, D120)",
            "research/e5_graph_drift.md"),
    _render("hold-sensitivity", "Block with an imperfect Hold (post-hoc, D124)",
            "research/hold_sensitivity.md"),
)  # fmt: skip


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
