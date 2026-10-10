"""Startup checks of the final run (D108): settings vs D101 / D102, git state at freeze."""

import copy
import subprocess

import pytest
import yaml

from vaultic.fusion.final_run import CONFIG, check_git, check_settings


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
