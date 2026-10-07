"""Paired comparison of two harness runs on the same rows.

Run:  python -m vaultic.eval.compare <run_dir_A> <run_dir_B> [--split validation|test]

Uses each run's predictions.parquet (per-seed scores) and a paired bootstrap of the
seed-averaged metric: both runs are scored on identical resamples. --split test needs two
--final runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vaultic.eval.bootstrap import paired_bootstrap
from vaultic.eval.metrics import RANKING_METRICS


def _seed_scores(preds: pd.DataFrame) -> list[np.ndarray]:
    return [preds[c].to_numpy() for c in preds.columns if c.startswith("score_seed")]


def compare_runs(
    run_a: Path,
    run_b: Path,
    split: str = "validation",
    metric: str = "pr_auc",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    a = pd.read_parquet(run_a / "predictions.parquet")
    b = pd.read_parquet(run_b / "predictions.parquet")
    a, b = a[a["split"] == split], b[b["split"] == split]
    if a.empty or b.empty:
        raise ValueError(f"both runs need {split} predictions (test needs --final runs)")
    if not np.array_equal(a["TransactionID"].to_numpy(), b["TransactionID"].to_numpy()):
        raise ValueError("runs were scored on different rows")
    result = paired_bootstrap(
        a["label"].to_numpy(),
        _seed_scores(a),
        _seed_scores(b),
        RANKING_METRICS[metric],
        n_boot=n_boot,
        seed=seed,
    )
    result.update({"metric": metric, "split": split, "n_boot": n_boot})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--metric", default="pr_auc", choices=sorted(RANKING_METRICS))
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.run_a, args.run_b, args.split, args.metric), indent=2))


if __name__ == "__main__":
    main()
