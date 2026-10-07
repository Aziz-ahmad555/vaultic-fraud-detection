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


def _run(tmp_path, cfg=None, final=False, log=None, data=None, runs="runs"):
    return run(
        cfg or _config(tmp_path),
        runs_dir=tmp_path / runs,
        experiment_log=log,
        data=data or _data(),
        final=final,
        decisions_log=tmp_path / "decisions.md",
    )


@pytest.mark.parametrize("final", [False, True])
def test_same_config_and_seeds_give_identical_metrics(tmp_path, final):
    # separate run folders, like two independent checkouts (a second --final run in the
    # same folder is refused by design)
    a = _run(tmp_path, final=final, runs="runs_a")
    b = _run(tmp_path, final=final, runs="runs_b")
    assert a != b
    assert (a / "metrics.json").read_bytes() == (b / "metrics.json").read_bytes()


def test_development_run_reports_validation_only(tmp_path):
    out = _run(tmp_path)
    for name in ("config.yaml", "metrics.json", "run_info.json", "predictions.parquet"):
        assert (out / name).exists()
    result = json.loads((out / "metrics.json").read_text())
    assert result["mode"] == "development"
    assert "test" not in result and "test" not in result["rows"]
    pr = result["validation"]["pr_auc"]
    assert pr["ci_low"] <= pr["mean"] <= pr["ci_high"] and len(pr["per_seed"]) == 2
    assert "f1_at_val_threshold" not in result["validation"]  # threshold chosen on this data
    preds = pd.read_parquet(out / "predictions.parquet")
    assert set(preds["split"]) == {"validation"}
    assert preds["TransactionDT"].between(128 * SECONDS_PER_DAY, 151 * SECONDS_PER_DAY - 1).all()


def test_timings_go_to_run_info_not_metrics(tmp_path):
    out = _run(tmp_path)
    info = json.loads((out / "run_info.json").read_text())
    assert len(info["per_seed_timing"]) == 2
    assert all(t["train_seconds"] > 0 for t in info["per_seed_timing"])
    assert "per_seed_timing" not in json.loads((out / "metrics.json").read_text())


def test_development_run_cannot_see_test_period(tmp_path):
    """Corrupting every test-period label and feature leaves a development run unchanged."""
    clean = _run(tmp_path)
    df, base = _data()
    test = (df["day"] >= 151).to_numpy()
    df.loc[test, "isFraud"] = 1 - df.loc[test, "isFraud"]
    df.loc[test, ["V1", "C1"]] = np.nan
    corrupted = _run(tmp_path, data=(df, base))
    assert (clean / "metrics.json").read_bytes() == (corrupted / "metrics.json").read_bytes()


def test_final_run_adds_test_metrics_with_validation_thresholds(tmp_path):
    out = _run(tmp_path, final=True)
    result = json.loads((out / "metrics.json").read_text())
    df, _ = _data()
    assert result["mode"] == "final"
    assert result["rows"]["test"] == int((df["day"] >= 151).sum())
    assert {"pr_auc", "f1_at_val_threshold", "cost_at_val_threshold"} <= set(result["test"])
    assert len(result["thresholds_chosen_on_validation"]) == 2
    preds = pd.read_parquet(out / "predictions.parquet")
    assert set(preds["split"]) == {"validation", "test"}


@pytest.mark.parametrize(
    "model,params,features",
    [
        ("logistic_regression", {"max_iter": 200}, "raw_lr"),
        ("random_forest", {"n_estimators": 10}, "raw"),
        ("xgboost", {"n_estimators": 10}, "raw_base"),
    ],
)
def test_baseline_models_run(tmp_path, model, params, features):
    out = _run(tmp_path, cfg=_config(tmp_path, model, params, features))
    pr = json.loads((out / "metrics.json").read_text())["validation"]["pr_auc"]["mean"]
    assert pr > 0.1  # the synthetic signal is learnable (base rate ~4%)


def test_feature_sets_exclude_label_id_and_time():
    df, base = _data(200)
    for name in ("raw", "raw_lr", "raw_base"):
        cols = set(design_matrix(df, base, name).columns)
        assert not cols & {"isFraud", "TransactionID", "TransactionDT", "day"}
    assert "ProductCD_W" in design_matrix(df, base, "raw_lr").columns
    assert "uid_n_past" in design_matrix(df, base, "raw_base").columns


def test_experiment_log_marks_final_runs(tmp_path):
    log = tmp_path / "log.md"
    log.write_text("| ID |\n")
    _run(tmp_path, log=log)
    _run(tmp_path, log=log, final=True)
    lines = log.read_text(encoding="utf-8").splitlines()[1:]
    assert "development run" in lines[0] and "FINAL" not in lines[0]
    assert "**FINAL** test PR-AUC" in lines[1]


def test_mlflow_logging_records_run(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    import vaultic.eval.run as harness

    monkeypatch.setattr(harness, "MLFLOW_DIR", tmp_path / "mlflow")
    out = tmp_path / "run"
    out.mkdir()
    (out / "metrics.json").write_text("{}")
    cfg = {"id": "EXP-MLFLOW-TEST", "model": {"name": "xgboost"}, "features": "raw"}
    result = {"mode": "development", "uid_variant": "uid2", "validation": {"pr_auc": {"mean": 0.5}}}
    assert harness._log_mlflow(cfg, result, out)
    runs = mlflow.search_runs(experiment_names=["EXP-MLFLOW-TEST"])
    assert len(runs) == 1 and runs["metrics.validation_pr_auc"].iloc[0] == 0.5


def test_b5_feature_set_drops_unkept_v_columns(monkeypatch):
    import vaultic.features.vreduce as vreduce

    monkeypatch.setattr(vreduce, "load_kept", lambda: ["V1"])
    df, base = _data(100)
    df["V2"] = df["V1"] * 2
    cols = set(design_matrix(df, base, "b5").columns)
    assert "V1" in cols and "V2" not in cols and "uid_n_past" in cols


def test_second_final_run_needs_a_logged_reason(tmp_path):
    decisions = tmp_path / "decisions.md"
    decisions.write_text("| # |\n", encoding="utf-8")
    kwargs = dict(runs_dir=tmp_path / "runs", experiment_log=None, decisions_log=decisions)
    run(_config(tmp_path), data=_data(), final=True, **kwargs)
    with pytest.raises(RuntimeError, match="already has a --final run"):
        run(_config(tmp_path), data=_data(), final=True, **kwargs)
    # development runs are never blocked
    run(_config(tmp_path), data=_data(), final=False, **kwargs)
    run(_config(tmp_path), data=_data(), final=True, rerun_reason="bug #1 fixed", **kwargs)
    text = decisions.read_text(encoding="utf-8")
    assert "FINAL-RERUN" in text and "bug #1 fixed" in text


def test_label_maturity_drops_training_labels_unknown_at_validation_start(tmp_path):
    """E25: with L = 30, a training row is used only if its time + 30 days <= the start of
    validation (day 128), i.e. TransactionDT <= day 98 00:00."""
    cfg = yaml.safe_load(_config(tmp_path).read_text())
    cfg["train_label_maturity_days"] = 30
    path = tmp_path / "EXP-MAT.yaml"
    path.write_text(yaml.safe_dump(cfg))
    df, base = _data()
    out = _run(tmp_path, cfg=path, data=(df, base))
    result = json.loads((out / "metrics.json").read_text())
    in_train = (df["day"] >= 1) & (df["day"] <= 120)
    expected = int((in_train & (df["TransactionDT"] <= 98 * SECONDS_PER_DAY)).sum())
    assert result["rows"]["train"] == expected < int(in_train.sum())
    assert result["train_label_maturity_days"] == 30
    plain = json.loads((_run(tmp_path, runs="plain") / "metrics.json").read_text())
    assert plain["rows"]["validation"] == result["rows"]["validation"]
    assert "train_label_maturity_days" not in plain


def test_planned_configs_are_refused_and_extends_merges(tmp_path):
    from vaultic.eval.run import load_config, read_config

    _config(tmp_path)  # EXP-TEST.yaml
    child = {"extends": "EXP-TEST.yaml", "id": "EXP-CHILD", "model": {"params": {"max_depth": 5}}}
    (tmp_path / "child.yaml").write_text(yaml.safe_dump(child))
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["id"] == "EXP-CHILD" and cfg["model"]["name"] == "xgboost"
    assert cfg["model"]["params"] == {"n_estimators": 20, "max_depth": 5}
    planned = {**child, "status": "planned", "needs": ["the temporal view"]}
    (tmp_path / "planned.yaml").write_text(yaml.safe_dump(planned))
    with pytest.raises(ValueError, match="planned experiment.*temporal view"):
        load_config(tmp_path / "planned.yaml")
    with pytest.raises(ValueError, match="planned"):
        _run(tmp_path, cfg=tmp_path / "planned.yaml")
    # a runnable child of a planned parent does not inherit the parent's status
    (tmp_path / "grandchild.yaml").write_text(yaml.safe_dump({"extends": "planned.yaml"}))
    assert "status" not in read_config(tmp_path / "grandchild.yaml")
