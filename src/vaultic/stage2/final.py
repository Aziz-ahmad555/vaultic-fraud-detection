"""Freeze and guards for the stage-2 final runs (D121; D164).

  python -m vaultic.stage2.final --freeze                    hash configs, data, libraries
  python -m vaultic.stage2.final --exp EXP-223-final         ONE final run (test period)

The final runs use the EXP-200-final guards (fusion/final_run.py, D108-D115) with stage 2's own
freeze record (research/frozen_stage2.md): every setting in the frozen config equals its
pre-registered value (EXPECTED); the freeze record matches every frozen file, data file and the
library versions of the experiment's environment (E15 runs in .venv-dice, the others in .venv);
a clean tree descending from the frozen commit (only the logs, the freeze record and earlier
stage-2 results in research/tables/stage2/ may differ); not behind origin; one final run per
experiment (run folder or FINAL-STARTED row; a re-run needs --rerun-reason); an O_EXCL lock; and
the FINAL-STARTED row committed and pushed BEFORE any test data is read.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from vaultic.paths import CONFIG_DIR, REPO_ROOT, RESEARCH_DIR, RUNS_DIR

FROZEN = RESEARCH_DIR / "frozen_stage2.md"
FREEZE_RECORD = "research/frozen_stage2.md"
LOG_FILES = {FREEZE_RECORD, "research/experiment_log.md", "research/decisions.md"}
OUTPUT_DIR = RESEARCH_DIR / "tables" / "stage2"
OUTPUTS = ("research/tables/stage2/",)
E10_RUN = "experiments/runs/EXP-200-final/20261011-002400-202994"
EXPERIMENTS = ("EXP-223-final", "EXP-210-final", "EXP-215-final")  # run order (D121 section 5)
VENVS = {".venv": REPO_ROOT.parent.parent / ".venv" if (REPO_ROOT.parent.name == ".worktrees")
         else REPO_ROOT / ".venv", ".venv-dice": REPO_ROOT / ".venv-dice"}  # fmt: skip
WHAT = {"EXP-223-final": "E23 subgroup gaps final run",
        "EXP-210-final": "ablation ladder (E3-E5, E8) final run",
        "EXP-215-final": "E15 explanation final run"}  # fmt: skip

# D121 / D158-D163 values of every setting the stage-2 runs read from their configs
EXPECTED = {
    "EXP-223-final": {
        ("status",): "frozen-final", ("venv",): ".venv", ("e10_run",): E10_RUN,
        ("methods",): ["B5", "F0", "F4", "MVAF"], ("min_frauds",): 30, ("top_emails",): 5,
        ("products",): ["W", "C", "R", "H", "S"], ("devices",): ["desktop", "mobile"],
        ("bootstrap", "n"): 1000, ("bootstrap", "seed"): 0,
    },
    "EXP-210-final": {
        ("status",): "frozen-final", ("venv",): ".venv", ("e10_run",): E10_RUN,
        ("seeds",): [0, 1, 2, 3, 4], ("bootstrap", "n"): 1000, ("bootstrap", "seed"): 0,
        ("rungs",): {1: ["tabular"], 2: ["tabular", "behavioral"],
                     3: ["tabular", "behavioral", "temporal"],
                     4: ["tabular", "behavioral", "temporal", "graph"],
                     5: ["tabular", "behavioral", "temporal", "graph", "anomaly"]},
        ("f4_trials",): 50, ("tuned",): "research/tables/stage2_ladder_tuned.json",
        ("metrics",): ["pr_auc", "recall_at_1pct_fpr", "cost"], ("h1_alpha",): 0.05,
        ("sensitivity",): "bounded_count",
    },
    "EXP-215-final": {
        ("status",): "frozen-final", ("venv",): ".venv-dice", ("e10_run",): E10_RUN,
        ("seeds",): [0, 1, 2, 3, 4], ("alerts",): 1000, ("counterfactual_alerts",): 500,
        ("sample_seed",): 0, ("ks",): list(range(1, 11)), ("random_orders",): 20,
        ("permutation_repeats",): 5, ("perturbations",): [0.99, 1.01, 0.95, 1.05],
        ("stability_top",): 3, ("seed_top",): 5, ("bootstrap", "n"): 1000,
        ("bootstrap", "seed"): 0, ("cf_vary",): ["TransactionAmt", "ProductCD"],
        ("cf_setting",): "research/tables/stage2_cf_setting.json", ("validity_target",): 0.9,
    },
}  # fmt: skip


def load_config(exp: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{exp}.yaml").read_text("utf-8"))


def check_settings(exp: str, cfg: dict) -> None:
    """Every setting equals its pre-registered value (as final_run.check_settings)."""
    bad = []
    for path, want in EXPECTED[exp].items():
        got = cfg
        for k in path:
            got = got.get(k) if isinstance(got, dict) else None
        if got != want:
            bad.append(f"{'.'.join(map(str, path))} = {got!r} (pre-registered: {want!r})")
    if cfg.get("id") != exp:
        bad.append(f"id = {cfg.get('id')!r}")
    if bad:
        raise ValueError(f"{exp} settings differ from the pre-registration: " + "; ".join(bad))


def frozen_files() -> list[Path]:
    """Configs and parameter files the stage-2 final runs depend on."""
    files = [CONFIG_DIR / f"{e}.yaml" for e in EXPERIMENTS]
    files += [CONFIG_DIR / n for n in ("splits.yaml", "v_columns.yaml", "EXP-009.yaml",
                                       "EXP-200-final.yaml")]  # fmt: skip
    files += [CONFIG_DIR / f"EXP-V-{v}.yaml"
              for v in ("tabular", "behavioral", "temporal", "graph", "anomaly")]  # fmt: skip
    files += [RESEARCH_DIR / "tables" / n for n in ("fusion_d95.json", "final_results.json",
                                                    "stage2_ladder_tuned.json",
                                                    "stage2_cf_setting.json")]  # fmt: skip
    return files


def data_files() -> list[Path]:
    """Every data file a stage-2 final run reads (D112)."""
    from vaultic.data.uid import UID_PATH
    from vaultic.paths import FEATURES_DIR, INTERIM_DIR, MERGED_PATH

    temporal = INTERIM_DIR / "temporal_dev_v4"
    return [MERGED_PATH, UID_PATH, *sorted(FEATURES_DIR.glob("*.parquet")),
            INTERIM_DIR / "view_table_d95.parquet",
            *sorted(p for p in temporal.iterdir() if p.is_file()),
            REPO_ROOT / E10_RUN / "predictions.parquet", REPO_ROOT / E10_RUN / "metrics.json",
            REPO_ROOT / "experiments/runs/EXP-009/20261008-211052-888539/predictions.parquet"]  # fmt: skip


def venv_versions(venv: str) -> dict[str, str]:
    """library_versions() as seen by one environment's interpreter."""
    py = VENVS[venv] / "Scripts" / "python.exe"
    code = ("import json; from vaultic.fusion.final_run import library_versions; "
            "print(json.dumps(library_versions()))")  # fmt: skip
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    out = subprocess.run([str(py), "-c", code], capture_output=True, text=True, env=env,
                         check=True)  # fmt: skip
    return json.loads(out.stdout.strip().splitlines()[-1])


def library_section_venvs() -> str:
    lines = ["", "## Library versions per environment (D114)", "", "| library | version |",
             "|---|---|"]  # fmt: skip
    for venv in sorted(VENVS):
        lines += [f"| library `{venv}:{k}` | `{v}` |" for k, v in venv_versions(venv).items()]
    return "\n".join(lines) + "\n"


def check_libraries_venv(record: str, venv: str) -> dict[str, str]:
    from vaultic.fusion.final_run import LIB_ROW, library_versions

    frozen = {k.split(":", 1)[1]: v for k, v in LIB_ROW.findall(record)
              if k.startswith(f"{venv}:")}  # fmt: skip
    if not frozen:
        raise RuntimeError(f"the freeze record has no library versions for {venv}")
    now = library_versions()
    diff = {k: (frozen.get(k), now.get(k)) for k in set(frozen) | set(now)
            if frozen.get(k) != now.get(k)}  # fmt: skip
    if diff:
        raise RuntimeError(f"library versions differ from the freeze (frozen, now): {diff}; "
                           f"is this the {venv} interpreter?")  # fmt: skip
    return now


def freeze() -> None:
    from vaultic.fusion.final_run import data_section, env_values
    from vaultic.reports.phase2_summary import write_freeze

    for e in EXPERIMENTS:
        check_settings(e, load_config(e))
    write_freeze(frozen_files(), FROZEN, title="Frozen configs for the stage-2 final runs "
                 "(E23, ablation ladder, E15; D121)")  # fmt: skip
    with open(FROZEN, "a", encoding="utf-8") as f:
        f.write(data_section(data_files(), env_values()))
        f.write(library_section_venvs())
    import re

    commit = re.search(r"at git commit `([0-9a-f]{40})`", FROZEN.read_text("utf-8")).group(1)
    with open(RESEARCH_DIR / "decisions.md", "a", encoding="utf-8") as f:
        f.write(f"| FREEZE | {datetime.now().date().isoformat()} | Stage-2 final-run configs "
                f"(EXP-223/210/215-final), data files, data paths and library versions frozen at "
                f"git commit `{commit}` (research/frozen_stage2.md) | — | Written by "
                "`vaultic.stage2.final --freeze` | A stage-2 final run needs this row for its "
                "frozen commit (D113) | Default |\n")  # fmt: skip
    print(f"wrote {FROZEN}; logged the freeze of {commit[:12]}")


def start(exp: str, cfg: dict, rerun_reason: str | None) -> tuple[Path, str]:
    """All guards, then FINAL-STARTED committed and pushed. Nothing reads test data before."""
    from vaultic.eval.run import _guard_final_rerun, previous_final_runs
    from vaultic.fusion.final_run import (
        LOCK_NAME,
        check_data,
        check_git,
        check_origin,
        check_package_location,
        env_values,
        exclusive_lock,
        log_guard,
        record_start,
    )
    from vaultic.reports.phase2_summary import check_freeze

    check_settings(exp, cfg)
    check_package_location()
    if not FROZEN.exists() or check_freeze(FROZEN):
        raise RuntimeError(f"{FROZEN} is missing or a frozen file changed: "
                           f"{check_freeze(FROZEN) if FROZEN.exists() else 'no record'}")  # fmt: skip
    record = FROZEN.read_text("utf-8")
    check_libraries_venv(record, cfg["venv"])
    head = check_git(FROZEN, REPO_ROOT, log_files=LOG_FILES, freeze_record=FREEZE_RECORD,
                     outputs=OUTPUTS)  # fmt: skip
    branch = check_origin(REPO_ROOT)
    with exclusive_lock(RUNS_DIR / exp / LOCK_NAME):
        prev = previous_final_runs(exp, RUNS_DIR)
        _guard_final_rerun(exp, RUNS_DIR, rerun_reason, RESEARCH_DIR / "decisions.md")
        logged = log_guard(exp, rerun_reason, branch, REPO_ROOT)
        if any(logged.values()) and not prev:
            with open(RESEARCH_DIR / "decisions.md", "a", encoding="utf-8") as f:
                f.write(f"| FINAL-RERUN | {datetime.now().date().isoformat()} | {exp} | --final "
                        f"re-run; earlier final run on record in the logs | — | {rerun_reason} | "
                        "Logged by stage2.final |\n")  # fmt: skip
        hashes = check_data(record, data_files(), env_values())
        run_dir = RUNS_DIR / exp / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        record_start(run_dir, head, RESEARCH_DIR / "experiment_log.md",
                     extra_logs=[RESEARCH_DIR / "decisions.md"], repo=REPO_ROOT,
                     data_hashes=hashes, what=WHAT[exp], prereg="D121",
                     freeze_record=FREEZE_RECORD)  # fmt: skip
    return run_dir, head


def complete(run_dir: Path, cfg: dict, meta: dict) -> None:
    start_meta = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    (run_dir / "metrics.json").write_text(json.dumps(
        {**start_meta, "status": "completed",
         "completed": datetime.now().isoformat(timespec="seconds"), **meta},
        indent=1, default=str), "utf-8")  # fmt: skip
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--exp", choices=EXPERIMENTS)
    parser.add_argument("--rerun-reason", default=None)
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    if not args.exp:
        parser.error("--freeze or --exp")
    from vaultic.stage2 import run as stage2_run

    cfg = load_config(args.exp)
    run_dir, head = start(args.exp, cfg, args.rerun_reason)
    meta = stage2_run.FINAL[args.exp](cfg, run_dir, head)
    complete(run_dir, cfg, meta)
    print(f"{args.exp} completed; results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
