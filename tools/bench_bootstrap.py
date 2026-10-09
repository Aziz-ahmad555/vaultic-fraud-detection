"""Benchmark the threaded bootstrap (D48) against the original code on real predictions.

Usage: python tools/bench_bootstrap.py <run dir with predictions.parquet> [split] [metric ...]
Prints per metric: wall time old vs new, speed-up, and whether the CIs are identical.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd

from vaultic.eval.bootstrap import seed_mean_ci, seed_mean_ci_reference
from vaultic.eval.metrics import RANKING_METRICS


def main() -> None:
    run = Path(sys.argv[1])
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    names = sys.argv[3:] or ["pr_auc", "recall_at_1pct_fpr"]
    preds = pd.read_parquet(run / "predictions.parquet")
    part = preds[preds["split"] == split]
    y = part["label"].to_numpy()
    scores = [part[c].to_numpy() for c in part.columns if c.startswith("score_seed")]
    print(
        f"{run.name}: {split} rows {len(y):,}, frauds {int(y.sum()):,}, seeds {len(scores)}, cores {os.cpu_count()}"
    )
    for name in names:
        metric = RANKING_METRICS[name]
        t0 = time.perf_counter()
        old = seed_mean_ci_reference(y, scores, metric, n_boot=1000, seed=0)
        t1 = time.perf_counter()
        new = seed_mean_ci(y, scores, metric, n_boot=1000, seed=0)
        t2 = time.perf_counter()
        print(
            f"{name}: old {t1 - t0:.1f} s, new {t2 - t1:.1f} s, speed-up {(t1 - t0) / (t2 - t1):.1f}x, identical {old == new} {new}"
        )


if __name__ == "__main__":
    main()
