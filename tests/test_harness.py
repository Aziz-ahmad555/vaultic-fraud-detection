import json

import numpy as np
import pandas as pd
import pytest
import yaml

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.eval.run import run
from vaultic.features.sets import design_matrix


def _data(n=6_000, seed=0):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.integers(SECONDS_PER_DAY, 183 * SECONDS_PER_DAY, n))
    signal = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(signal * 2 - 4)))).astype(np.int8)
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n, dtype=np.int32),
            "isFraud": y,
            "TransactionDT": time.astype(np.int32),
            "day": (time // SECONDS_PER_DAY).astype(np.int16),
            "TransactionAmt": rng.lognormal(4, 1, n),
            "ProductCD": pd.Categorical(rng.choice(["W", "C", "H"], n)),
            "V1": (signal + rng.normal(0, 0.5, n)).astype(np.float32),
            "C1": rng.integers(0, 5, n).astype(np.float32),
        }
    )
    base = pd.DataFrame({"TransactionID": df["TransactionID"], "uid_n_past": rng.integers(0, 9, n)})
    return df, base


def _config(tmp_path, model="xgboost", params=None, features="raw"):
    cfg = {
        "id": "EXP-TEST",
        "question": "harness test",
        "model": {"name": model, "params": params or {"n_estimators": 20, "max_depth": 3}},
        "features": features,
        "seeds": [0, 1],
        "bootstrap": {"n": 50, "seed": 0},
    }
    path = tmp_path / "EXP-TEST.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_same_config_and_seeds_give_identical_metrics(tmp_path):
    cfg = _config(tmp_path)
    a = run(cfg, runs_dir=tmp_path / "runs", experiment_log=None, data=_data())
    b = run(cfg, runs_dir=tmp_path / "runs", experiment_log=None, data=_data())
    assert a != b
    assert (a / "metrics.json").read_bytes() == (b / "metrics.json").read_bytes()


def test_run_writes_outputs_and_uses_only_test_rows_for_metrics(tmp_path):
    out = run(_config(tmp_path), runs_dir=tmp_path / "runs", experiment_log=None, data=_data())
    for name in ("config.yaml", "metrics.json", "run_info.json", "predictions.parquet"):
        assert (out / name).exists()
    result = json.loads((out / "metrics.json").read_text())
    df, _ = _data()
    assert result["rows"]["test"] == int((df["day"] >= 151).sum())
    pr = result["metrics"]["pr_auc"]
    assert pr["ci_low"] <= pr["mean"] <= pr["ci_high"]
    assert len(pr["per_seed"]) == 2
    preds = pd.read_parquet(out / "predictions.parquet")
    assert set(preds["split"]) == {"validation", "test"}
    assert preds["TransactionDT"].min() >= 128 * SECONDS_PER_DAY


@pytest.mark.parametrize(
    "model,params,features",
    [
        ("logistic_regression", {"max_iter": 200}, "raw_lr"),
        ("random_forest", {"n_estimators": 10}, "raw"),
        ("xgboost", {"n_estimators": 10}, "raw_base"),
    ],
)
def test_baseline_models_run(tmp_path, model, params, features):
    out = run(
        _config(tmp_path, model, params, features),
        runs_dir=tmp_path / "runs", experiment_log=None, data=_data(),
    )
    pr = json.loads((out / "metrics.json").read_text())["metrics"]["pr_auc"]["mean"]
    assert pr > 0.1  # the synthetic signal is learnable (base rate ~4%)


def test_feature_sets_exclude_label_id_and_time():
    df, base = _data(200)
    for name in ("raw", "raw_lr", "raw_base"):
        cols = set(design_matrix(df, base, name).columns)
        assert not cols & {"isFraud", "TransactionID", "TransactionDT", "day"}
    assert "ProductCD_W" in design_matrix(df, base, "raw_lr").columns
    assert "uid_n_past" in design_matrix(df, base, "raw_base").columns


def test_experiment_log_line_is_appended(tmp_path):
    log = tmp_path / "log.md"
    log.write_text("| ID |\n")
    run(_config(tmp_path), runs_dir=tmp_path / "runs", experiment_log=log, data=_data())
    assert log.read_text().count("| EXP-TEST |") == 1
