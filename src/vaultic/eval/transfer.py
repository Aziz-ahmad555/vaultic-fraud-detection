"""Move harness runs between machines (e.g. Kaggle -> laptop).

Run:  python -m vaultic.eval.transfer export runs.zip [--experiments EXP-009 EXP-011]
      python -m vaultic.eval.transfer import runs.zip

The unit of transfer is the harness run folder (config.yaml, metrics.json, run_info.json,
predictions.parquet), because every reported number comes from it. `import` copies new run
folders into experiments/runs/ (an existing run folder is never overwritten) and logs each
imported run into the local MLflow store; its run_info.json (hardware, device, git commit)
goes along as an artifact, so the origin stays visible. Imported --final runs
count towards the one-final-run-per-experiment rule like local ones.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

from vaultic.paths import RUNS_DIR

RUN_FILES = ("config.yaml", "metrics.json", "run_info.json", "predictions.parquet")


def _run_dirs(runs_dir: Path, experiments: list[str] | None) -> list[Path]:
    out = []
    for exp_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if experiments and exp_dir.name not in experiments:
            continue
        out += [d for d in sorted(exp_dir.iterdir()) if (d / "metrics.json").exists()]
    return out


def export_runs(zip_path: Path, runs_dir: Path = RUNS_DIR, experiments=None) -> int:
    runs = _run_dirs(runs_dir, experiments)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for run in runs:
            for name in RUN_FILES:
                if (run / name).exists():
                    zf.write(run / name, f"{run.parent.name}/{run.name}/{name}")
    return len(runs)


def import_runs(
    zip_path: Path, runs_dir: Path = RUNS_DIR, log_to_mlflow: bool = True
) -> tuple[list[Path], list[str]]:
    """Returns (imported run folders, skipped run ids that already existed)."""
    imported, skipped = [], []
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                parts = Path(member).parts
                if len(parts) != 3 or parts[2] not in RUN_FILES or ".." in parts:
                    raise ValueError(f"unexpected file in archive: {member}")
            zf.extractall(tmp)
        for src in _run_dirs(Path(tmp), None):
            dest = runs_dir / src.parent.name / src.name
            if dest.exists():
                skipped.append(f"{src.parent.name}/{src.name}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest)
            imported.append(dest)
    if log_to_mlflow:
        from vaultic.eval.run import _log_mlflow

        for dest in imported:
            cfg = yaml.safe_load((dest / "config.yaml").read_text(encoding="utf-8"))
            result = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
            _log_mlflow(cfg, result, dest)
    return imported, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("zip", type=Path)
    exp.add_argument("--experiments", nargs="*", default=None)
    imp = sub.add_parser("import")
    imp.add_argument("zip", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        n = export_runs(args.zip, experiments=args.experiments)
        print(f"exported {n} runs to {args.zip}")
    else:
        imported, skipped = import_runs(args.zip)
        print(f"imported {len(imported)} runs; skipped {len(skipped)} already present")
        for s in skipped:
            print(f"  skipped {s}")


if __name__ == "__main__":
    main()
