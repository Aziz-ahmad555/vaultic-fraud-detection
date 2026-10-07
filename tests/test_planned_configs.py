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
