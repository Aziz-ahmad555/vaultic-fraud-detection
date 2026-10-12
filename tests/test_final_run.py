"""Startup checks of the final run (D108): settings vs D101 / D102, git state at freeze."""

import copy
import json
import subprocess
import sys

import pytest
import yaml

from vaultic.eval.run import _guard_final_rerun
from vaultic.fusion.final_run import (
    CONFIG,
    check_data,
    check_git,
    check_libraries,
    check_origin,
    check_package_location,
    check_settings,
    count_final_rows,
    data_section,
    exclusive_lock,
    freeze_row,
    library_section,
    library_versions,
    log_guard,
    record_start,
    repo_relative,
)


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
    log, decisions = repo / "research" / "experiment_log.md", repo / "research" / "decisions.md"
    log.write_text("| id |\n| EXP-1 | old row |\n")
    decisions.write_text("| D1 |\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "code")
    frozen_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                   text=True).stdout.strip()  # fmt: skip
    record = repo / "research" / "frozen_final.md"
    record.write_text(f"Frozen at git commit `{frozen_commit}`, before any run.\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "freeze without its decisions row")
    with pytest.raises(RuntimeError, match="no row naming the frozen commit"):
        check_git(record, repo)  # D113
    _append(decisions, freeze_row(frozen_commit).rstrip())
    _git(repo, "commit", "-qam", "log the freeze")
    assert check_git(record, repo)  # freeze record + its row, after the frozen commit: allowed
    _append(log, "| EXP-200-final | FINAL-STARTED |")
    with pytest.raises(RuntimeError, match="not clean"):
        check_git(record, repo)
    _git(repo, "commit", "-qam", "log")
    assert check_git(record, repo)  # the logs may gain lines
    log.write_text(log.read_text().replace("| EXP-1 | old row |\n", ""))
    _git(repo, "commit", "-qam", "delete a log line")
    with pytest.raises(RuntimeError, match="may only gain lines"):
        check_git(record, repo)  # D113
    _git(repo, "revert", "--no-edit", "HEAD")
    assert check_git(record, repo)
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


def _repo_with_logs(path):
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    (path / "research").mkdir()
    log, decisions = path / "research" / "experiment_log.md", path / "research" / "decisions.md"
    log.write_text("| id | date |\n")
    decisions.write_text("| D1 |\n")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "init")
    _git(path, "remote", "add", "origin", str(path / "no-such-remote.git"))  # push will fail
    return log, decisions


def test_record_start_raises_when_the_push_fails(tmp_path):
    log, decisions = _repo_with_logs(tmp_path)
    run_dir = tmp_path / "runs" / "EXP-200-final" / "20261010-000000-000000"
    with pytest.raises(
        RuntimeError,
        match="push.*failed; the final run stops before any test data is scored or evaluated",
    ) as err:
        record_start(run_dir, "a" * 40, log, extra_logs=[decisions], repo=tmp_path, push=True)
    local = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True,
                           text=True).stdout.strip()  # fmt: skip
    assert f"unpushed FINAL-STARTED commit: {local}" in str(err.value)  # D110
    # the attempt stays on record locally, so the guard still blocks an unlogged re-run
    assert json.loads((run_dir / "metrics.json").read_text("utf-8"))["status"] == "started"
    with pytest.raises(RuntimeError, match="already has a --final run"):
        _guard_final_rerun("EXP-200-final", tmp_path / "runs", None, decisions)


def test_final_mode_aborts_before_scoring_test_data_when_the_push_fails(tmp_path, monkeypatch):
    """The whole final-mode start: a failed push must stop main() before build_table (the
    load, fit and scoring of the data) is called."""
    import vaultic.eval.run as harness
    import vaultic.fusion.dev_compare as dev_compare
    import vaultic.fusion.final_run as fr
    import vaultic.reports.phase2_summary as summary

    log, decisions = _repo_with_logs(tmp_path)
    frozen = tmp_path / "research" / "frozen_final.md"
    frozen.write_text("record\n")
    called = []
    monkeypatch.setattr(fr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fr, "RESEARCH_DIR", tmp_path / "research")
    monkeypatch.setattr(fr, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(fr, "FROZEN", frozen)
    monkeypatch.setattr(fr, "check_git", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(fr, "check_libraries", lambda *a, **k: {})
    monkeypatch.setattr(fr, "check_package_location", lambda *a, **k: None)
    monkeypatch.setattr(fr, "check_origin", lambda *a, **k: "main")
    monkeypatch.setattr(fr, "log_guard", lambda *a, **k: {})
    monkeypatch.setattr(fr, "data_files", lambda *a, **k: [])
    monkeypatch.setattr(fr, "check_data", lambda *a, **k: {"merged.parquet": "f" * 64})
    monkeypatch.setattr(summary, "check_freeze", lambda *a, **k: [])
    monkeypatch.setattr(harness, "_guard_final_rerun", lambda *a, **k: None)
    monkeypatch.setattr(dev_compare, "build_table", lambda *a, **k: called.append(1))
    monkeypatch.setattr(sys, "argv", ["final_run", "--mode", "final"])
    with pytest.raises(RuntimeError, match="push.*failed"):
        fr.main()
    assert called == []  # nothing was scored
    assert not (tmp_path / "runs" / "EXP-200-final" / ".final.lock").exists()  # released
    assert "FINAL-STARTED" in log.read_text("utf-8")


def _append(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _clone_pair(tmp_path):
    """A bare 'origin' with one commit on main, and two clones of it."""
    origin, a, b = tmp_path / "origin.git", tmp_path / "a", tmp_path / "b"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    for clone in (a, b):
        subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True,
                       capture_output=True)  # fmt: skip
        _git(clone, "config", "user.email", "t@t")
        _git(clone, "config", "user.name", "t")
        _git(clone, "checkout", "-q", "-B", "main")
    (a / "research").mkdir()
    _append(a / "research" / "experiment_log.md", "| id | date |")
    _append(a / "research" / "decisions.md", "| D1 |")
    _git(a, "add", ".")
    _git(a, "commit", "-qm", "init")
    _git(a, "push", "-q", "origin", "main")
    _git(b, "pull", "-q", "origin", "main")
    return a, b


def test_check_origin_refuses_behind_or_diverged(tmp_path):
    a, b = _clone_pair(tmp_path)
    assert check_origin(a) == "main"
    _append(b / "x.txt", "x")
    _git(b, "add", ".")
    _git(b, "commit", "-qm", "elsewhere")
    _git(b, "push", "-q", "origin", "main")
    with pytest.raises(RuntimeError, match="is behind origin/main"):
        check_origin(a)
    _append(a / "y.txt", "y")
    _git(a, "add", ".")
    _git(a, "commit", "-qm", "here")
    with pytest.raises(RuntimeError, match="diverged from origin/main"):
        check_origin(a)
    _git(a, "pull", "-q", "--no-rebase", "origin", "main")
    assert check_origin(a) == "main"  # ahead only: allowed (the start record will push)


def test_log_guard_counts_rows_at_head_and_at_origin(tmp_path):
    a, b = _clone_pair(tmp_path)
    text = "| EXP-200-final | d | FINAL-STARTED: x |\n| EXP-9 | FINAL-STARTED |"
    assert count_final_rows(text, "EXP-200-final") == 1
    assert not any(log_guard("EXP-200-final", None, "main", a).values())
    # a final run started on another machine: only origin has the row
    _append(
        b / "research" / "experiment_log.md",
        "| EXP-200-final | 2026-10-11 | FINAL-STARTED: E10-E12 final run started |",
    )
    _git(b, "commit", "-qam", "FINAL-STARTED elsewhere")
    _git(b, "push", "-q", "origin", "main")
    _git(a, "fetch", "-q", "origin", "main")
    with pytest.raises(RuntimeError, match="already has a final run on record in the logs"):
        log_guard("EXP-200-final", None, "main", a)
    found = log_guard("EXP-200-final", "bug in X", "main", a)  # a logged re-run may proceed
    assert found["origin/main:research/experiment_log.md"] == 1
    assert found["HEAD:research/experiment_log.md"] == 0
    # a FINAL-RERUN row in decisions.md at HEAD counts too
    _append(
        a / "research" / "decisions.md",
        "| FINAL-RERUN | 2026-10-11 | EXP-200-final | re-run | — | r | x |",
    )
    _git(a, "commit", "-qam", "rerun row")
    assert log_guard("EXP-200-final", "r", "main", a)["HEAD:research/decisions.md"] == 1


def test_exclusive_lock_blocks_a_second_holder_and_is_released(tmp_path):
    lock = tmp_path / "runs" / "EXP-200-final" / ".final.lock"
    with exclusive_lock(lock):
        assert lock.exists()
        with pytest.raises(RuntimeError, match="another final run is starting"):
            with exclusive_lock(lock):
                pass
        assert lock.exists()  # the failed second attempt does not remove the holder's lock
    assert not lock.exists()
    with pytest.raises(ValueError):
        with exclusive_lock(lock):
            raise ValueError("crash inside")
    assert not lock.exists()  # released on error too
    lock.write_text("stale")
    with pytest.raises(RuntimeError, match="delete it by hand"):
        with exclusive_lock(lock):
            pass


def test_data_freeze_detects_changed_added_files_and_other_data_paths(tmp_path):
    files = []
    for name in ("merged.parquet", "uids.parquet"):
        files.append(tmp_path / name)
        files[-1].write_bytes(name.encode() * 100)
    env = {"VAULTIC_DATA_DIR": str(tmp_path), "VAULTIC_RAW_DIR": "unset"}
    record = "# record\n" + data_section(files, env)
    hashes = check_data(record, files, env)
    assert set(hashes) == {f.as_posix() for f in files}
    with pytest.raises(RuntimeError, match="data paths differ"):
        check_data(record, files, {**env, "VAULTIC_DATA_DIR": "elsewhere"})
    extra = tmp_path / "graph_uid.parquet"
    extra.write_bytes(b"new")
    with pytest.raises(RuntimeError, match="added"):
        check_data(record, [*files, extra], env)
    files[0].write_bytes(b"changed" * 100)
    with pytest.raises(RuntimeError, match="changed since the freeze"):
        check_data(record, files, env)
    with pytest.raises(RuntimeError, match="no data section"):
        check_data("# record without data\n", files, env)


def test_record_start_writes_data_hashes_into_metrics_and_commit(tmp_path):
    log, decisions = _repo_with_logs(tmp_path)
    run_dir = tmp_path / "runs" / "EXP-200-final" / "20261010-000000-000001"
    hashes = {"E:/data/interim/merged.parquet": "ab" * 32}
    record_start(run_dir, "a" * 40, log, extra_logs=[decisions], repo=tmp_path, push=False,
                 data_hashes=hashes)  # fmt: skip
    assert json.loads((run_dir / "metrics.json").read_text("utf-8"))["data_hashes"] == hashes
    body = subprocess.run(["git", "log", "-1", "--format=%B"], cwd=tmp_path, capture_output=True,
                          text=True).stdout  # fmt: skip
    assert "ab" * 32 + "  E:/data/interim/merged.parquet" in body


def test_package_must_come_from_this_checkout(tmp_path):
    assert check_package_location().name == "__init__.py"  # tests import from this checkout
    with pytest.raises(RuntimeError, match="not from"):
        check_package_location(tmp_path)


def test_library_versions_must_match_the_freeze():
    now = library_versions()
    assert "python" in now and "xgboost" in now
    record = "# record\n" + library_section(now)
    assert check_libraries(record, now) == now
    with pytest.raises(RuntimeError, match="differ from the freeze"):
        check_libraries(record, {**now, "xgboost": "0.0.1"})
    with pytest.raises(RuntimeError, match="no library versions"):
        check_libraries("# record\n", now)


def test_record_start_with_runs_behind_a_junction():
    """D115: experiments/runs in a worktree is a junction to the main checkout's runs folder;
    the start record must not fail because the resolved run folder lies outside the repo."""
    import os
    import tempfile
    from pathlib import Path

    # under the system temp folder (E:\\dev-cache\\tmp in our runs): on C:\\...\\Temp nothing
    # can be created through a junction at all, so the test probes first and skips there
    with tempfile.TemporaryDirectory() as base:
        base = Path(base)
        repo, elsewhere = base / "repo", base / "main-checkout-runs"
        repo.mkdir()
        elsewhere.mkdir()
        if os.name == "nt":
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(repo / "runs"), str(elsewhere)],
                               capture_output=True, text=True)  # fmt: skip
        else:
            r = subprocess.run(["ln", "-s", str(elsewhere), str(repo / "runs")])
        try:
            if r.returncode != 0:
                raise OSError("no link")
            (repo / "runs" / "probe").mkdir()
        except OSError:
            pytest.skip("cannot create a writable link in this temp folder")
        log, decisions = _repo_with_logs(repo)
        run_dir = repo / "runs" / "EXP-200-final" / "20261010-000000-000002"
        assert repo_relative(run_dir, repo) == "runs/EXP-200-final/20261010-000000-000002"
        record_start(run_dir, "a" * 40, log, extra_logs=[decisions], repo=repo, push=False)
        assert "runs/EXP-200-final/20261010-000000-000002" in log.read_text("utf-8")
        assert (elsewhere / "EXP-200-final" / "20261010-000000-000002" / "metrics.json").exists()
        if os.name == "nt":
            os.rmdir(repo / "runs")  # remove the junction itself before the tree is deleted


def test_freeze_row_is_one_table_row_ending_in_a_newline():
    row = freeze_row("c" * 40)
    assert row.endswith("|\n") and row.count("\n") == 1 and "{NL}" not in row
    assert row.count(" | ") + 1 == 7 and ("`" + "c" * 40 + "`") in row
