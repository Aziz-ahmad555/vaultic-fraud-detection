import numpy as np
import pandas as pd
import pytest

from vaultic.eval.compare import compare_runs


def _write(path, scores_per_seed, n=3000, split="validation"):
    path.mkdir(parents=True)
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.2).astype(int)
    df = pd.DataFrame({"TransactionID": np.arange(len(y)), "split": split, "label": y})
    for i, s in enumerate(scores_per_seed):
        df[f"score_seed{i}"] = s(y)
    df.to_parquet(path / "predictions.parquet")


def test_better_run_wins_on_paired_bootstrap(tmp_path):
    rng = np.random.default_rng(1)
    n = 3000
    _write(tmp_path / "a", [lambda y: y + rng.normal(0, 0.5, n)] * 2)
    _write(tmp_path / "b", [lambda y: y + rng.normal(0, 2.0, n)] * 2)
    r = compare_runs(tmp_path / "a", tmp_path / "b", n_boot=200)
    assert r["diff"] > 0 and r["ci_low"] > 0 and r["split"] == "validation"


def test_test_split_requires_final_runs(tmp_path):
    _write(tmp_path / "a", [lambda y: y * 1.0])
    _write(tmp_path / "b", [lambda y: y * 1.0])
    with pytest.raises(ValueError, match="--final"):
        compare_runs(tmp_path / "a", tmp_path / "b", split="test")
