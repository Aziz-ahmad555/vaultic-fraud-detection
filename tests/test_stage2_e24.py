"""E24 compute-cost helpers on synthetic inputs (D121 section 4)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from vaultic.stage2.e24 import (
    carbon,
    hardware,
    summarise,
    time_model,
    tuning_entry,
    tuning_minutes,
    write_report,
)


def test_time_model_medians_fresh_fits():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(500, 3)), rng.integers(0, 2, 500)
    made = []
    out = time_model(lambda: made.append(LogisticRegression()) or made[-1],
                     lambda m: m.fit(X, y), lambda m: m.predict_proba(X), 500)  # fmt: skip
    assert len(made) == 3 and len(out["fit_seconds_all"]) == 3  # a fresh model per repetition
    assert out["fit_seconds"] == sorted(out["fit_seconds_all"])[1]
    assert out["inference_ms_per_1000"] > 0 and out["predict_rows"] == 500


def test_tuning_minutes_and_entries(tmp_path):
    (tmp_path / "tuning_B5.md").write_text(
        "# x\n\n| weighting | trials | pruned | best val PR-AUC | trees | total minutes |\n"
        "|---|---|---|---|---|---|\n| none | 25 | 2 | 0.68 | 1350 | 67 |\n"
        "| scale_pos_weight | 25 | 8 | 0.66 | 1100 | 46 |\n\nWinner\n", "utf-8")  # fmt: skip
    assert tuning_minutes("B5", tmp_path) == 113
    assert tuning_minutes("missing", tmp_path) is None
    assert tuning_entry("B5", 10.0, tmp_path)["tuning_hours"] == pytest.approx(113 / 60)
    est = tuning_entry("MVAF", 36.0, tmp_path)
    assert est["tuning_hours"] == pytest.approx(0.5) and est["tuning_source"].startswith(
        "estimated"
    )
    assert tuning_entry("F4 rung 2", 72.0, tmp_path)["tuning_hours"] == pytest.approx(1.0)
    assert tuning_entry("B2", 5.0, tmp_path) == {"tuning_hours": 0.0, "tuning_source": "not tuned"}


def test_carbon_is_a_labelled_estimate():
    c = carbon(3600, tdp_watts=45, grams_per_kwh=400)  # one hour at 45 W
    assert c["energy_kwh"] == pytest.approx(0.045)
    assert c["co2_kg"] == pytest.approx(0.018)
    assert "estimate" in c["note"]


def test_summary_and_report(tmp_path):
    rows = [{"model": "B5", "kind": "baseline", "fit_seconds": 1800.0, "inference_ms_per_1000": 2.0,
             "tuning_hours": 1.5, "tuning_source": "log"},
            {"model": "MVAF", "kind": "fusion", "fit_seconds": 1800.0, "inference_ms_per_1000": 0.1,
             "tuning_hours": 0.5, "tuning_source": "estimated"}]  # fmt: skip
    settings = {"cpu_tdp_watts": 45, "grid_g_co2_per_kwh": 400, "tdp_source": "vendor page",
                "grid_source": "Ember", "grid_accessed": "2026-10-12"}  # fmt: skip
    s = summarise(rows, settings)
    assert s["compute_hours"] == pytest.approx(3.0)
    result = {"hardware": hardware(), "settings": settings, "models": rows, "summary": s}
    text = write_report(result, tmp_path).read_text("utf-8")
    assert "estimate, not a measurement" in text and "3.00 h" in text
    assert (tmp_path / "e24_compute.json").exists()
