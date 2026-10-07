"""Phase 2 exit gate: does B5 beat B3 with non-overlapping 95% CIs on the test period?

Run:  python -m vaultic.reports.phase2_gate

Reads the latest --final runs of the B3 and B5 experiments named in
experiments/configs/table1.yaml, checks the PR-AUC CIs for overlap, adds a paired bootstrap
of the difference on identical test resamples, and writes research/phase2_exit_gate.md.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import yaml

from vaultic.eval.compare import compare_runs
from vaultic.paths import RESEARCH_DIR, RUNS_DIR
from vaultic.reports.table1 import SPEC_PATH, latest_run


def gate(b3: dict, b5: dict) -> bool:
    """Passes when B5's lower PR-AUC bound is above B3's upper bound."""
    return b5["ci_low"] > b3["ci_high"]


def check(runs_dir: Path = RUNS_DIR, spec_path: Path = SPEC_PATH) -> tuple[bool, str]:
    spec = {r["baseline"]: r for r in yaml.safe_load(spec_path.read_text("utf-8"))["rows"]}
    runs = {b: latest_run(spec[b]["experiment"], "final", runs_dir) for b in ("B3", "B5")}
    missing = [b for b, r in runs.items() if r is None]
    if missing:
        raise FileNotFoundError(f"no --final run for {missing}; run them before the gate")
    pr = {
        b: json.loads((r / "metrics.json").read_text("utf-8"))["test"]["pr_auc"]
        for b, r in runs.items()
    }
    passed = gate(pr["B3"], pr["B5"])
    paired = compare_runs(runs["B5"], runs["B3"], split="test")

    def fmt(e):
        return f"{e['mean']:.4f} ± {e['std']:.4f} (95% CI {e['ci_low']:.4f}–{e['ci_high']:.4f})"

    lines = [
        "# Phase 2 exit gate",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.reports.phase2_gate` from "
        "--final harness runs (test period).",
        "",
        "Gate: B5 beats B3 with non-overlapping 95% CIs on test PR-AUC.",
        "",
        "| baseline | experiment | run | test PR-AUC |",
        "|---|---|---|---|",
        *[
            f"| {b} | {spec[b]['experiment']} | `{runs[b].name}` | {fmt(pr[b])} |"
            for b in ("B3", "B5")
        ],
        "",
        f"Paired bootstrap B5 − B3 (same test resamples): {paired['diff']:+.4f} "
        f"(95% CI {paired['ci_low']:+.4f} to {paired['ci_high']:+.4f}, p = {paired['p_value']:.3f}).",
        "",
        f"**Result: {'PASSED' if passed else 'NOT PASSED'}.**",
        "",
    ]
    if not passed:
        lines += [
            "Per the roadmap: recheck the uid and aggregation features before moving on.",
            "",
        ]
    return passed, "\n".join(lines)


def main() -> None:
    passed, report = check()
    (RESEARCH_DIR / "phase2_exit_gate.md").write_text(report, encoding="utf-8")
    print(
        f"Phase 2 exit gate: {'PASSED' if passed else 'NOT PASSED'} (research/phase2_exit_gate.md)"
    )


if __name__ == "__main__":
    main()
