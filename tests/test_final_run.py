"""Startup checks of the final run (D108): settings vs D101 / D102, git state at freeze."""

import copy
import json
import subprocess

import pytest
import yaml

from vaultic.eval.run import _guard_final_rerun
from vaultic.fusion.final_run import CONFIG, check_git, check_settings, record_start


def test_config_settings_equal_the_preregistration():
    cfg = yaml.safe_load(CONFIG.read_text("utf-8"))
    check_settings(cfg)  # the frozen config passes
    for path, value in (
        (("e12", "adaptive", "gamma"), 0.1),
        (("bootstrap", "n"), 500),
        (("e10", "methods"), ["MVAF", "F3"]),
        (("e12", "lambda_grid"), [0, 1]),
    ):
        bad = copy.deepcopy(cfg)
        node = bad
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        with pytest.raises(ValueError, match="differ from the pre-registration"):
            check_settings(bad)


def _git(repo, *a):
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)


def test_check_git_needs_clean_tree_and_the_frozen_commit(tmp_path):
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "research").mkdir()
    (repo / "code.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "code")
    frozen_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                   text=True).stdout.strip()  # fmt: skip
    record = repo / "research" / "frozen_final.md"
    record.write_text(f"Frozen at git commit `{frozen_commit}`, before any run.\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "freeze")
    assert check_git(record, repo)  # freeze record committed after the frozen commit: allowed
    (repo / "research" / "experiment_log.md").write_text("FINAL-STARTED\n")
    with pytest.raises(RuntimeError, match="not clean"):
        check_git(record, repo)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "log")
    assert check_git(record, repo)  # the logs may differ
    (repo / "code.py").write_text("x = 2\n")
    _git(repo, "commit", "-qam", "change code")
    with pytest.raises(RuntimeError, match="changed since the frozen commit"):
        check_git(record, repo)


def test_record_start_commits_before_test_data_and_blocks_a_rerun(tmp_path):
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    log, decisions = repo / "experiment_log.md", repo / "decisions.md"
    log.write_text("| id | date |\n")
    decisions.write_text("| D1 |\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    runs = repo / "runs"
    run_dir = runs / "EXP-200-final" / "20261010-000000-000000"
    record_start(run_dir, "a" * 40, log, extra_logs=[decisions], repo=repo, push=False)
    m = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert m["mode"] == "final" and m["status"] == "started"
    assert "FINAL-STARTED" in log.read_text("utf-8")
    status = subprocess.run(["git", "status", "--porcelain", "--", "experiment_log.md"],
                            cwd=repo, capture_output=True, text=True).stdout  # fmt: skip
    assert status == ""  # the start line is committed
    with pytest.raises(RuntimeError, match="already has a --final run"):
        _guard_final_rerun("EXP-200-final", runs, None, decisions)  # "started" counts
