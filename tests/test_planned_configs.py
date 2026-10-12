"""Planned experiment configs E3-E15 and E25 (Phase 13): well formed, complete, never runnable
by accident."""

from pathlib import Path

import pytest
import yaml

from vaultic.eval.run import load_config
from vaultic.paths import CONFIG_DIR

PLANNED = sorted(Path(CONFIG_DIR).glob("EXP-1[0-9][0-9].yaml"))


def _raw(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_every_roadmap_experiment_has_a_config():
    roadmap = {_raw(p)["roadmap"] for p in PLANNED}
    assert roadmap == {f"E{i}" for i in range(3, 16)} | {"E25"}


@pytest.mark.parametrize("path", PLANNED, ids=lambda p: p.stem)
def test_planned_config_is_well_formed(path):
    cfg = _raw(path)
    assert cfg["id"] == path.stem
    assert int(path.stem[-2:]) == int(cfg["roadmap"][1:])  # EXP-1NN <-> ENN
    assert cfg["status"] == "planned" and cfg["needs"] and cfg["question"]
    if "extends" not in cfg:
        assert cfg["seeds"] == [0, 1, 2, 3, 4]  # rule 8
        assert cfg["bootstrap"]["n"] == 1000
    assert "test" not in str(cfg.get("tuning", "")).lower()  # rule 6: tuning on validation only


@pytest.mark.parametrize("path", PLANNED, ids=lambda p: p.stem)
def test_planned_config_is_refused_by_the_harness(path):
    if "extends" in _raw(path) and not (path.parent / _raw(path)["extends"]).exists():
        pytest.skip("parent config is produced by the Phase 2 chain")
    with pytest.raises(ValueError, match="planned experiment"):
        load_config(path)


def test_e25_drops_only_immature_training_labels():
    cfg = _raw(Path(CONFIG_DIR) / "EXP-125.yaml")
    assert cfg["extends"] == "EXP-009.yaml" and cfg["train_label_maturity_days"] == 30
    assert "model" not in cfg and "features" not in cfg  # same model and features as B5


APPENDIX = sorted(Path(CONFIG_DIR).glob("EXP-1[0-9][0-9]-F3.yaml"))


def test_ablation_ladder_uses_f4_with_f3_appendix():
    """D41: E4, E5, E8 stack with F4 (LightGBM, missing views as NaN); F3 only in appendix."""
    assert [p.stem for p in APPENDIX] == ["EXP-104-F3", "EXP-105-F3", "EXP-108-F3"]
    for path in APPENDIX:
        primary = Path(CONFIG_DIR) / f"{path.stem[:-3]}.yaml"
        assert _raw(primary)["fusion"] == {"method": "F4", "missing_views": "nan"}
        with pytest.raises(ValueError, match="planned experiment"):
            load_config(path)
        from vaultic.eval.run import read_config

        cfg = read_config(path)
        assert cfg["appendix"] is True and cfg["id"] == path.stem
        assert cfg["fusion"] == {"method": "F3", "missing_views": "constant"}
        assert cfg["views"] == _raw(primary)["views"]  # identical apart from the fusion


def test_e14_training_window_variants():
    """D45: train_window_days is an E14 experiment variable: all history, 45, 60, 90 days."""
    from vaultic.eval.run import read_config

    base = _raw(Path(CONFIG_DIR) / "EXP-114.yaml")
    assert base["retraining"]["train_window_days"] is None
    assert "margin" not in str(base["retraining"]["champion_challenger"])
    windows = {}
    for path in sorted(Path(CONFIG_DIR).glob("EXP-114-w*.yaml")):
        cfg = read_config(path)
        assert cfg["status"] == "planned" and cfg["id"] == path.stem
        assert cfg["retraining"]["policies"] == base["retraining"]["policies"]  # merged
        windows[path.stem] = cfg["retraining"]["train_window_days"]
        with pytest.raises(ValueError, match="planned experiment"):
            load_config(path)
    assert windows == {"EXP-114-w45": 45, "EXP-114-w60": 60, "EXP-114-w90": 90}


def test_no_planned_config_early_stops_on_validation():
    """D45 / D54: GRU early stopping uses the latest 20% of each fold's training rows."""
    from vaultic.eval.run import read_config

    for path in sorted(Path(CONFIG_DIR).glob("EXP-1*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert (
            "early_stopping: validation" not in text and "validation_pr_auc" not in text
        ), path.name
    temporal = read_config(Path(CONFIG_DIR) / "EXP-104.yaml")["views"]["temporal"]
    assert temporal["early_stopping"] == {
        "metric": "pr_auc",
        "rows": "latest_20pct_of_fold_training_rows",
    }


def test_e10_to_e12_declare_the_same_fold_calibration_scope():
    """Review N2 (D77): the fusion / calibration experiments use the gate plan with calibration
    fitted and applied on one fold of view models."""
    for n in ("110", "111", "112"):
        cfg = yaml.safe_load((CONFIG_DIR / f"EXP-{n}.yaml").read_text("utf-8"))
        assert cfg["plan"] == {"name": "gate", "calibration_scope": "same_fold"}, n
