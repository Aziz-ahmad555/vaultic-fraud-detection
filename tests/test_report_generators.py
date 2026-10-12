"""Report generators (D150): pinned runs, hand-written sections never overwritten."""

import inspect

import pytest

from vaultic.paths import RUNS_DIR


def test_emerging_reads_only_pinned_runs():
    from vaultic.reports import emerging

    for experiment, run in emerging.RUNS.items():
        assert (RUNS_DIR / experiment / run).is_dir() or not RUNS_DIR.exists(), experiment
    with pytest.raises(KeyError, match="no pinned run"):
        emerging._run_pr_auc("EXP-999", [])


@pytest.mark.parametrize("module", [
    "vaultic.fusion.f3_diagnosis", "vaultic.fusion.f3_sensitivity",
    "vaultic.reports.e5_graph_drift", "vaultic.trust.hold_sensitivity",
    "vaultic.fusion.gate_maturity_sensitivity",
])  # fmt: skip
def test_generators_write_only_through_keep_hand_section(module):
    import importlib

    mod = importlib.import_module(module)
    src = inspect.getsource(mod.main)
    assert "keep_hand_section(OUT_MD, lambda: write_md(result))" in src
    assert src.count("write_md(") == 1  # no bare call that would overwrite the reading


def test_keep_hand_section_with_a_generator_writer(tmp_path, monkeypatch):
    from vaultic.reports import e5_graph_drift as m
    from vaultic.reports.render_saved import keep_hand_section

    out = tmp_path / "e5.md"
    monkeypatch.setattr(m, "OUT_MD", out)
    out.write_text("# old\n\n## 3. Reading (added by hand, not generated)\n\nMy reading.\n")
    result = {"generated": "2026-10-11", "seeds": [0], "reproduction_seed0": {"max_abs_diff": 0.0, "corr": 1.0},
              "group_sizes": {"all validation rows": [10, 1]}, "drift": [],
              "models": [{"model": "B5", "all validation rows": 0.5}]}  # fmt: skip
    keep_hand_section(out, lambda: m.write_md(result))
    text = out.read_text()
    assert "My reading." in text and "_To be written._" not in text and "| B5 | 0.5000 |" in text
