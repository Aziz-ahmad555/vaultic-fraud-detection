import json

import numpy as np
import pandas as pd
import pytest

from vaultic.reports.shap_summary import importance, save_plot, shap_values
from vaultic.reports.table1 import build_table, latest_run, to_markdown


def _entry(mean, ci=True):
    e = {"mean": mean, "std": 0.01, "per_seed": [mean]}
    if ci:
        e.update({"ci_low": mean - 0.02, "ci_high": mean + 0.02})
    return e


def _write_run(root, exp, name, mode, pr):
    run = root / exp / name
    run.mkdir(parents=True)
    period = {k: _entry(pr if k == "pr_auc" else 0.5) for k in [
        "pr_auc", "roc_auc", "recall_at_1pct_fpr", "recall_at_5pct_fpr",
        "precision_at_500", "brier", "ece",
    ]}  # fmt: skip
    result = {"mode": mode, "seeds": [0, 1], "validation": period}
    if mode == "final":
        result["test"] = dict(
            period, f1_at_val_threshold=_entry(0.4), cost_at_val_threshold=_entry(9.0)
        )
    (run / "metrics.json").write_text(json.dumps(result))
    timing = [{"train_seconds": 10.0, "inference_ms_per_1000": 2.0}] * 2
    (run / "run_info.json").write_text(json.dumps({"per_seed_timing": timing}))


SPEC = {
    "rows": [
        {"baseline": "B1", "experiment": "EXP-A", "model": "m", "features": "f", "tuning": "none"},
        {"baseline": "B2", "experiment": "EXP-B", "model": "m", "features": "f", "tuning": "none"},
    ]
}


def test_latest_run_respects_mode(tmp_path):
    _write_run(tmp_path, "EXP-A", "20260101-000000-000001", "final", 0.6)
    _write_run(tmp_path, "EXP-A", "20260102-000000-000001", "development", 0.7)
    assert latest_run("EXP-A", "final", tmp_path).name == "20260101-000000-000001"
    assert latest_run("EXP-A", "development", tmp_path).name == "20260102-000000-000001"
    assert latest_run("EXP-Z", "final", tmp_path) is None


def test_final_table_uses_test_metrics_and_marks_missing_runs(tmp_path):
    _write_run(tmp_path, "EXP-A", "20260101-000000-000001", "final", 0.6)
    table = build_table(SPEC, "final", tmp_path)
    a, b = table.iloc[0], table.iloc[1]
    assert a["PR-AUC"] == "0.6000 ± 0.0100 [0.5800, 0.6200]"
    assert a["F1 (val threshold)"] == "0.4000 ± 0.0100"
    assert a["Training s/seed"] == "10" and a["Seeds"] == 2
    assert b["PR-AUC"] == "not run"
    md = to_markdown(table, "final")
    assert "test period (FINAL runs)" in md and "| B1 |" in md and "| B2 |" in md


def test_development_table_has_no_test_only_columns(tmp_path):
    _write_run(tmp_path, "EXP-A", "20260101-000000-000001", "development", 0.7)
    table = build_table(SPEC, "development", tmp_path)
    assert "Cost ($)" not in table.columns
    assert table.iloc[0]["PR-AUC"].startswith("0.7000")


def test_shap_importance_and_plot(tmp_path):
    pytest.importorskip("shap")
    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 4)), columns=["a", "b", "c", "d"])
    y = (X["b"] > 0.5).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=3).fit(X, y)
    values = shap_values(model, X)
    top = importance(values, X.columns, top=2)
    assert top["feature"].iloc[0] == "b"  # the only informative feature
    save_plot(values, X, tmp_path / "shap.png", "test")
    assert (tmp_path / "shap.png").stat().st_size > 1000


def test_phase2_gate_requires_non_overlapping_cis():
    from vaultic.reports.phase2_gate import gate

    b3 = {"ci_low": 0.50, "ci_high": 0.55}
    assert gate(b3, {"ci_low": 0.56, "ci_high": 0.60})
    assert not gate(b3, {"ci_low": 0.54, "ci_high": 0.70})  # overlapping: not passed


def test_phase2_gate_needs_final_runs(tmp_path):
    from vaultic.reports.phase2_gate import check

    spec = tmp_path / "t.yaml"
    spec.write_text(
        "rows:\n  - {baseline: B3, experiment: EXP-A, model: m, features: f, tuning: x}\n"
        "  - {baseline: B5, experiment: EXP-B, model: m, features: f, tuning: x}\n"
    )
    _write_run(tmp_path, "EXP-A", "20260101-000000-000001", "development", 0.5)
    with pytest.raises(FileNotFoundError, match="--final"):
        check(runs_dir=tmp_path, spec_path=spec)


def test_fyp1_metrics_known_values():
    from vaultic.reports.e2_fyp1 import fyp1_metrics

    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.4])
    m = fyp1_metrics(y, p)
    # at 0.5: predictions 0,0,1,1,0 -> tp 1, fp 1, fn 1
    assert m["Precision@0.5"] == 0.5 and m["Recall@0.5"] == 0.5 and m["Accuracy@0.5"] == 0.6
    assert 0 < m["PR-AUC"] <= 1 and m["ROC-AUC"] == pytest.approx(
        5 / 6
    )  # 0.7 beats 3 negatives, 0.4 beats 2


def test_freeze_detects_later_changes(tmp_path, monkeypatch):
    import vaultic.reports.phase2_summary as ps

    monkeypatch.setattr(ps, "REPO_ROOT", tmp_path)
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("x: 1\n")
    b.write_text("y: 2\n")
    record = tmp_path / "frozen.md"
    ps.write_freeze([a, b], out=record)
    assert ps.check_freeze(record) == []
    b.write_text("y: 3\n")
    assert ps.check_freeze(record) == ["b.yaml"]
