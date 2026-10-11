"""Guard for post-hoc analyses of the test period (D144).

A post-hoc script may read the test period only after the pre-registered final run is on record:
experiments/runs/<experiment> has a run whose metrics.json says mode "final" and status
"completed", and research/experiment_log.md has that run's completed FINAL / FINAL-RERUN row
(not just its FINAL-STARTED row). Cached files holding test-period rows are marked as test data
with a sidecar <file>.TEST_DATA.json naming the final run they come from; open_test_data checks
both the guard and the mark before such a file is read.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from vaultic.paths import RESEARCH_DIR, RUNS_DIR

FINAL_EXPERIMENT = "EXP-200-final"
SIDECAR = ".TEST_DATA.json"


def require_final_on_record(experiment: str = FINAL_EXPERIMENT, runs_dir: Path = RUNS_DIR,
                            log: Path = RESEARCH_DIR / "experiment_log.md") -> Path:  # fmt: skip
    """The completed final run of `experiment` (its folder); raises if it is not on record."""
    root = runs_dir / experiment
    done = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        m = d / "metrics.json"
        if m.exists():
            meta = json.loads(m.read_text("utf-8"))
            if meta.get("mode") == "final" and meta.get("status") == "completed":
                done.append(d)
    if not done:
        raise RuntimeError(f"{experiment} has no completed final run in {root}: post-hoc analyses "
                           "of the test period are refused until it is on record (D144)")  # fmt: skip
    run = done[-1]
    text = log.read_text("utf-8") if log.exists() else ""
    rows = [line for line in text.splitlines() if line.startswith(f"| {experiment} |")
            and run.name in line and "**FINAL" in line]  # fmt: skip
    if not rows:
        raise RuntimeError(f"{log.name} has no completed FINAL row for {experiment} run {run.name}: "
                           "post-hoc analyses of the test period are refused (D144)")  # fmt: skip
    return run


def mark_test_data(path: Path, run: Path, what: str) -> Path:
    """Write <path>.TEST_DATA.json: the file holds test-period rows of `run`."""
    path = Path(path)
    side = path.with_name(path.name + SIDECAR)
    side.write_text(json.dumps({"contains_test_period": True, "final_run": run.name,
                                "experiment": run.parent.name, "what": what,
                                "marked": date.today().isoformat(),
                                "rule": "read only through vaultic.eval.posthoc_guard (D144)"},
                               indent=1), "utf-8")  # fmt: skip
    return side


def open_test_data(path: Path, **guard_kwargs) -> Path:
    """Check the guard and the test-data mark of a cached file before it is read."""
    run = require_final_on_record(**guard_kwargs)
    path = Path(path)
    side = path.with_name(path.name + SIDECAR)
    if not side.exists():
        raise RuntimeError(f"{path} holds test-period rows but is not marked as test data "
                           f"({side.name} missing; D144)")  # fmt: skip
    meta = json.loads(side.read_text("utf-8"))
    if meta.get("final_run") != run.name:
        raise RuntimeError(f"{path} was built from final run {meta.get('final_run')}, but the "
                           f"completed final run is {run.name} (D144)")  # fmt: skip
    return path
