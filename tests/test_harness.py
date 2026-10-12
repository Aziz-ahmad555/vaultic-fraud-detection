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
    log = tmp_path / "log.md"
    kwargs = dict(runs_dir=tmp_path / "runs", experiment_log=log, decisions_log=decisions)
    run(_config(tmp_path), data=_data(), final=True, **kwargs)
    with pytest.raises(RuntimeError, match="already has a --final run"):
        run(_config(tmp_path), data=_data(), final=True, **kwargs)
    # development runs are never blocked
    run(_config(tmp_path), data=_data(), final=False, **kwargs)
    run(_config(tmp_path), data=_data(), final=True, rerun_reason="bug #1 fixed", **kwargs)
    text = decisions.read_text(encoding="utf-8")
    assert "FINAL-RERUN" in text and "bug #1 fixed" in text
    # review N6 (D81): the experiment log marks the re-run itself, not as a plain FINAL
    lines = [x for x in log.read_text(encoding="utf-8").splitlines() if x.startswith("| EXP-TEST")]
    assert ["FINAL-RERUN" in x for x in lines] == [False, False, True]
    assert "**FINAL**" in lines[0]


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


def test_drop_features_removes_a_feature_and_rejects_unknown_names(tmp_path):
    cfg = yaml.safe_load(_config(tmp_path, features="raw_base").read_text())
    cfg["drop_features"] = ["uid_n_past"]
    path = tmp_path / "EXP-DROP.yaml"
    path.write_text(yaml.safe_dump(cfg))
    full = json.loads(
        (
            _run(tmp_path, cfg=_config(tmp_path, features="raw_base"), runs="full") / "metrics.json"
        ).read_text()
    )
    dropped = json.loads((_run(tmp_path, cfg=path, runs="drop") / "metrics.json").read_text())
    assert dropped["n_features"] == full["n_features"] - 1
    assert dropped["dropped_features"] == ["uid_n_past"] and "dropped_features" not in full
    cfg["drop_features"] = ["no_such_column"]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="no_such_column"):
        _run(tmp_path, cfg=path, runs="bad")


def test_quantile_clipper_uses_training_quantiles_only():
    """D61, by hand: quantiles 0.25 / 0.75 of the training column [0, 1, 2, 3, 4] are 1 and 3."""
    from vaultic.views.tabular import QuantileClipper, make_model

    train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    clip = QuantileClipper(0.25, 0.75).fit(train)
    assert clip.transform(np.array([[-100.0], [2.5], [1e9]])).ravel().tolist() == [1.0, 2.5, 3.0]
    model = make_model(
        "logistic_regression", {"C": 1.0, "max_iter": 200, "clip_quantiles": [0.01, 0.99]}, seed=0
    )
    assert [type(s).__name__ for s in model] == [
        "SimpleImputer",
        "QuantileClipper",
        "StandardScaler",
        "LogisticRegression",
    ]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    extreme = np.array([[1e6, 0.0]])  # far outside training: the logit stays finite
    assert model.decision_function(extreme)[0] == pytest.approx(
        model.decision_function(np.array([[X[:, 0].max(), 0.0]]))[0], abs=0.5
    )
    plain = make_model("logistic_regression", {"C": 1.0}, seed=0)
    assert "QuantileClipper" not in [
        type(s).__name__ for s in plain
    ]  # unchanged without the option


def test_run_records_and_flags_convergence_warnings(tmp_path):
    """No -W ignore: an lbfgs fit stopped at max_iter is flagged in metrics.json, warnings.log
    and the experiment log line (the B1 failure of D61 would have been visible)."""
    log = tmp_path / "log.md"
    cfg = _config(tmp_path, "logistic_regression", {"max_iter": 2}, "raw_lr")
    out = _run(tmp_path, cfg=cfg, log=log)
    w = json.loads((out / "metrics.json").read_text())["warnings"]
    assert w["flagged"].get("ConvergenceWarning", 0) >= 2  # one per seed
    assert "ConvergenceWarning" in (out / "warnings.log").read_text(encoding="utf-8")
    assert "**WARNINGS: ConvergenceWarning x" in log.read_text(encoding="utf-8")
    clean = _run(tmp_path, cfg=_config(tmp_path), runs="runs_clean")
    assert json.loads((clean / "metrics.json").read_text())["warnings"]["flagged"] == {}


def test_smoke_run_is_small_separate_and_unlogged(tmp_path):
    log = tmp_path / "log.md"
    out = run(_config(tmp_path), runs_dir=tmp_path / "runs", experiment_log=log, data=_data(),
              decisions_log=tmp_path / "d.md", smoke=True)  # fmt: skip
    assert out.parent.name == "EXP-TEST-smoke" and not log.exists()
    m = json.loads((out / "metrics.json").read_text())
    assert m["seeds"] == [0] and m["mode"] == "development" and "test" not in m
    preds = pd.read_parquet(out / "predictions.parquet")
    assert (preds["TransactionID"] % 10 == 0).all()
    with pytest.raises(ValueError, match="smoke"):
        run(_config(tmp_path), runs_dir=tmp_path / "runs", data=_data(), final=True,
            decisions_log=tmp_path / "d.md", smoke=True)  # fmt: skip


def test_b5_extra_feature_sets_attach_precomputed_columns(monkeypatch):
    """b5_behavioral = b5 + the behavioral file's columns, row-aligned (D66)."""
    from vaultic.features import sets

    monkeypatch.setattr("vaultic.features.vreduce.load_kept", lambda: [])
    df, base = _data(60)
    extra = pd.DataFrame({"TransactionID": base["TransactionID"], "amt_z": np.arange(60.0)})
    both = sets.attach_features(base, extra, "behavioral")
    X = sets.design_matrix(df, both, "b5_behavioral")
    assert "amt_z" in X and "uid_n_past" in X and "TransactionID" not in X
    assert X.shape[1] == sets.design_matrix(df, base, "b5").shape[1] + 1
    with pytest.raises(ValueError, match="aligned"):
        sets.attach_features(base, extra.iloc[::-1], "behavioral")
    with pytest.raises(ValueError, match="repeat"):
        sets.attach_features(base, base, "behavioral")


def test_train_drop_fraud_removes_only_that_products_training_frauds(tmp_path):
    """Emerging fraud (D69): the held-out product's training frauds are removed; its legit
    training rows and all validation rows stay."""
    df, base = _data()
    cfg = _config(tmp_path)
    plain = json.loads((_run(tmp_path, cfg=cfg, runs="a") / "metrics.json").read_text())
    data = yaml.safe_load(cfg.read_text())
    data["train_drop_fraud"] = {"ProductCD": "C"}
    cfg.write_text(yaml.safe_dump(data))
    held = json.loads((_run(tmp_path, cfg=cfg, runs="b") / "metrics.json").read_text())
    part = (df["day"] >= 1) & (df["day"] <= 120)
    n_removed = int(((df["ProductCD"] == "C") & (df["isFraud"] == 1) & part).sum())
    assert n_removed > 0
    assert plain["rows"]["train"] - held["rows"]["train"] == n_removed
    assert held["rows"]["validation"] == plain["rows"]["validation"]
    assert held["train_drop_fraud"] == {"ProductCD": "C"}


def test_view_only_feature_set_uses_only_that_file():
    """D75: behavioral_only = the behavioral file's columns and nothing else."""
    from vaultic.features import sets

    df, base = _data(40)
    beh = pd.DataFrame({"TransactionID": base["TransactionID"], "amt_z": np.arange(40.0),
                        "vel_n_1h": np.ones(40)})  # fmt: skip
    X = sets.design_matrix(df, beh, "behavioral_only")
    assert list(X.columns) == ["amt_z", "vel_n_1h"]
    with pytest.raises(ValueError, match="aligned"):
        sets.design_matrix(df, beh.iloc[::-1], "behavioral_only")


def test_view_sets_split_label_features_between_tabular_and_behavioral(monkeypatch):
    """D89: the tabular view loses the label-derived features; the behavioral view gets them."""
    from vaultic.features import sets
    from vaultic.views.definitions import LABEL_DERIVED, behavioral_columns, tabular_columns

    monkeypatch.setattr("vaultic.features.vreduce.load_kept", lambda: [])
    df, base = _data(50)
    for c in LABEL_DERIVED:
        base[c] = np.arange(50.0)
    tab = sets.design_matrix(df, base, "tabular_view")
    b5 = sets.design_matrix(df, base, "b5")
    assert not set(LABEL_DERIVED) & set(tab.columns)
    assert list(tab.columns) == tabular_columns(b5.columns)
    assert set(b5.columns) - set(tab.columns) == set(LABEL_DERIVED)
    beh_file = pd.DataFrame({"TransactionID": base["TransactionID"], "amt_z": np.ones(50)})
    beh = sets.attach_features(beh_file, base[["TransactionID", *LABEL_DERIVED]], "labels")
    X = sets.design_matrix(df, beh, "behavioral_view")
    assert list(X.columns) == behavioral_columns(beh_file.columns) == ["amt_z", *LABEL_DERIVED]
    with pytest.raises(ValueError, match="label-derived"):
        sets.design_matrix(df, beh_file, "behavioral_view")
    assert "all_views" in sets.EXTRA_FEATURES and "sequence" in sets.EXTRA_FEATURES["all_views"]


def test_harness_refuses_frozen_final_configs(tmp_path):
    """D102: EXP-200-final runs only through fusion/final_run.py."""
    from vaultic.eval.run import load_config

    path = tmp_path / "EXP-X.yaml"
    path.write_text("id: EXP-X\nstatus: frozen-final\nquestion: q\n")
    with pytest.raises(ValueError, match="frozen final"):
        load_config(path)
