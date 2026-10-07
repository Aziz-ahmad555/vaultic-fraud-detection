"""Kaggle portability: data paths, device selection, run transfer. No GPU needed."""

import importlib
import json
import zipfile

import pytest

from vaultic.eval.transfer import export_runs, import_runs
from vaultic.views.tabular import effective_device, make_model, resolve_device


def test_data_dirs_follow_environment(monkeypatch, tmp_path):
    import vaultic.paths as paths

    monkeypatch.setenv("VAULTIC_DATA_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("VAULTIC_RAW_DIR", str(tmp_path / "input"))
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.DATA_DIR == tmp_path / "work"
        assert reloaded.RAW_DIR == tmp_path / "input"
        assert reloaded.MERGED_PATH == tmp_path / "work" / "interim" / "merged.parquet"
    finally:
        monkeypatch.delenv("VAULTIC_DATA_DIR")
        monkeypatch.delenv("VAULTIC_RAW_DIR")
        importlib.reload(paths)


def test_raw_dir_defaults_under_data_dir(monkeypatch, tmp_path):
    import vaultic.paths as paths

    monkeypatch.setenv("VAULTIC_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("VAULTIC_RAW_DIR", raising=False)
    try:
        assert importlib.reload(paths).RAW_DIR == tmp_path / "raw"
    finally:
        monkeypatch.delenv("VAULTIC_DATA_DIR")
        importlib.reload(paths)


def test_device_precedence(monkeypatch):
    monkeypatch.delenv("VAULTIC_DEVICE", raising=False)
    assert resolve_device() == "cpu"
    assert resolve_device(config="cuda") == "cuda"
    monkeypatch.setenv("VAULTIC_DEVICE", "cpu")
    assert resolve_device(config="cuda") == "cpu"  # environment beats config
    assert resolve_device(cli="cuda", config="cpu") == "cuda"  # flag beats everything
    with pytest.raises(ValueError):
        resolve_device(cli="tpu")


def test_effective_device_and_model_settings():
    assert effective_device("xgboost", "cuda") == "cuda"
    assert effective_device("random_forest", "cuda") == "cpu"
    assert make_model("xgboost", {}, 0, device="cuda").get_params()["device"] == "cuda"
    assert make_model("lightgbm", {}, 0, device="cuda").get_params()["device_type"] == "gpu"
    assert "device_type" not in make_model("lightgbm", {}, 0).get_params() or (
        make_model("lightgbm", {}, 0).get_params().get("device_type") in (None, "cpu")
    )


def _fake_run(root, exp, name, mode="development"):
    run = root / exp / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(f"id: {exp}\nmodel: {{name: xgboost}}\nfeatures: raw\n")
    (run / "metrics.json").write_text(
        json.dumps({"mode": mode, "uid_variant": "uid", "validation": {"pr_auc": {"mean": 0.5}}})
    )
    (run / "run_info.json").write_text(json.dumps({"hardware": {"gpu": "Tesla T4"}}))
    return run


def test_export_import_roundtrip_never_overwrites(tmp_path):
    kaggle, laptop = tmp_path / "kaggle", tmp_path / "laptop"
    _fake_run(kaggle, "EXP-009", "20261008-010000-000001", mode="final")
    _fake_run(kaggle, "EXP-011", "20261008-020000-000001")
    existing = _fake_run(laptop, "EXP-011", "20261008-020000-000001")
    (existing / "metrics.json").write_text('{"mode": "development", "local": true}')

    archive = tmp_path / "runs.zip"
    assert export_runs(archive, kaggle) == 2
    assert export_runs(tmp_path / "one.zip", kaggle, experiments=["EXP-009"]) == 1

    imported, skipped = import_runs(archive, laptop, log_to_mlflow=False)
    assert [p.parent.name for p in imported] == ["EXP-009"]
    assert skipped == ["EXP-011/20261008-020000-000001"]
    assert json.loads((existing / "metrics.json").read_text())["local"] is True

    from vaultic.eval.run import previous_final_runs

    assert len(previous_final_runs("EXP-009", laptop)) == 1  # imported finals count


def test_import_rejects_unexpected_paths(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="unexpected"):
        import_runs(bad, tmp_path / "runs", log_to_mlflow=False)


def test_harness_records_device(tmp_path, monkeypatch):
    """A run saves the device it really trained on in its config copy and its metrics."""
    import yaml
    from test_harness import _config, _data

    from vaultic.eval.run import run

    monkeypatch.delenv("VAULTIC_DEVICE", raising=False)
    out = run(_config(tmp_path), runs_dir=tmp_path / "r", experiment_log=None, data=_data(2000))
    assert yaml.safe_load((out / "config.yaml").read_text())["device"] == "cpu"
    assert json.loads((out / "metrics.json").read_text())["device"] == "cpu"

    # a GPU request for a CPU-only model is recorded as cpu, not as cuda
    rf = _config(tmp_path, "random_forest", {"n_estimators": 5}, "raw")
    out = run(rf, runs_dir=tmp_path / "r2", experiment_log=None, data=_data(2000), device="cuda")
    assert yaml.safe_load((out / "config.yaml").read_text())["device"] == "cpu"


def test_verify_compares_raw_files_with_dvc_pointers(tmp_path):
    from vaultic.data.verify import md5, verify

    raw, pointers = tmp_path / "raw", tmp_path / "pointers"
    raw.mkdir()
    pointers.mkdir()
    (raw / "a.csv").write_text("x,y\n1,2\n")
    (raw / "b.csv").write_text("changed\n")
    (pointers / "a.csv.dvc").write_text(f"outs:\n- md5: {md5(raw / 'a.csv')}\n  path: a.csv\n")
    (pointers / "b.csv.dvc").write_text("outs:\n- md5: 0123\n  path: b.csv\n")
    (pointers / "c.csv.dvc").write_text("outs:\n- md5: 0123\n  path: c.csv\n")
    assert verify(raw, pointers) == {"a.csv": "ok", "b.csv": "MISMATCH", "c.csv": "missing"}
