"""Post-hoc test-data guard (D144)."""

import json

import pytest

from vaultic.eval.posthoc_guard import mark_test_data, open_test_data, require_final_on_record


def _run(root, name, **meta):
    d = root / "EXP-200-final" / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(meta))
    return d


def test_guard_needs_a_completed_final_run_and_its_final_row(tmp_path):
    runs, log = tmp_path / "runs", tmp_path / "experiment_log.md"
    log.write_text("| id | date |\n")
    kw = {"runs_dir": runs, "log": log}
    with pytest.raises(RuntimeError, match="no completed final run"):
        require_final_on_record(**kw)
    _run(runs, "20261011-000000-000001", mode="final", status="started")
    with pytest.raises(RuntimeError, match="no completed final run"):
        require_final_on_record(**kw)  # started but not completed
    done = _run(runs, "20261011-000000-000002", mode="final", status="completed")
    with open(log, "a") as f:
        f.write(f"| EXP-200-final | d | FINAL-STARTED: x | run `{done.name}` |\n")
    with pytest.raises(RuntimeError, match="no completed FINAL row"):
        require_final_on_record(**kw)  # the start line alone is not enough
    with open(log, "a") as f:
        f.write(f"| EXP-200-final | d | **FINAL-RERUN** (D115): results | run `{done.name}` |\n")
    assert require_final_on_record(**kw) == done


def test_cached_test_data_must_be_marked_and_match_the_run(tmp_path):
    runs, log = tmp_path / "runs", tmp_path / "experiment_log.md"
    done = _run(runs, "20261011-000000-000002", mode="final", status="completed")
    log.write_text(f"| EXP-200-final | d | **FINAL** | run `{done.name}` |\n")
    kw = {"runs_dir": runs, "log": log}
    cache = tmp_path / "inputs.npz"
    cache.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="not marked as test data"):
        open_test_data(cache, **kw)
    side = mark_test_data(cache, done, "test rows")
    assert json.loads(side.read_text())["contains_test_period"] is True
    assert open_test_data(cache, **kw) == cache
    side.write_text(json.dumps({"final_run": "another-run"}))
    with pytest.raises(RuntimeError, match="was built from final run"):
        open_test_data(cache, **kw)
