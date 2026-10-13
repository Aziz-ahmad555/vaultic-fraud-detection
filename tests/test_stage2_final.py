"""Stage-2 freeze and guards (D164): settings, git check with stage-2 outputs, start record."""

import copy
import json
import subprocess

import pytest

from vaultic.fusion.final_run import check_git, record_start
from vaultic.stage2 import final as s2


def _git(repo, *a):
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)


@pytest.mark.parametrize("exp", s2.EXPERIMENTS)
def test_configs_equal_the_preregistration(exp):
    cfg = s2.load_config(exp)
    s2.check_settings(exp, cfg)
    for path in s2.EXPECTED[exp]:
        bad = copy.deepcopy(cfg)
        node = bad
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = "changed"
        with pytest.raises(ValueError, match="differ from the pre-registration"):
            s2.check_settings(exp, bad)


def test_frozen_files_cover_every_config_and_parameter_file():
    names = {p.name for p in s2.frozen_files()}
    assert {f"{e}.yaml" for e in s2.EXPERIMENTS} <= names
    assert {"stage2_ladder_tuned.json", "stage2_cf_setting.json", "final_results.json",
            "fusion_d95.json", "splits.yaml"} <= names  # fmt: skip


def test_check_git_allows_only_logs_record_and_stage2_outputs(tmp_path):
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "research" / "tables").mkdir(parents=True)
    (repo / "code.py").write_text("x = 1\n")
    decisions = repo / "research" / "decisions.md"
    decisions.write_text("| D1 |\n")
    (repo / "research" / "experiment_log.md").write_text("| id |\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "code")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                            text=True).stdout.strip()  # fmt: skip
    record = repo / "research" / "frozen_stage2.md"
    record.write_text(f"Frozen at git commit `{commit}`.\n")
    decisions.write_text(f"| D1 |\n| FREEZE | x | frozen at git commit `{commit}` |\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "freeze")

    def check():
        return check_git(record, repo, log_files=s2.LOG_FILES, freeze_record=s2.FREEZE_RECORD,
                         outputs=s2.OUTPUTS)  # fmt: skip

    assert check()
    out = repo / "research" / "tables" / "stage2"
    out.mkdir()
    (out / "e23_groups.csv").write_text("a\n")  # results of an earlier stage-2 final run
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "E23 results")
    assert check()
    (out / "ladder_steps.csv").write_text("b\n")  # a later run adds its own file: allowed
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "ladder results")
    assert check()
    (out / "e23_groups.csv").write_text("changed\n")  # an earlier run's file changed (D177)
    _git(repo, "commit", "-qam", "edit E23 results")
    with pytest.raises(RuntimeError, match="only be added"):
        check()
    _git(repo, "revert", "--no-edit", "HEAD")  # reverting is a further change, still refused
    with pytest.raises(RuntimeError, match="only be added"):
        check()
    _git(repo, "reset", "-q", "--hard", "HEAD~2")
    assert check()
    (out / "e23_groups.csv").unlink()  # removing one is refused too
    _git(repo, "commit", "-qam", "remove E23 results")
    with pytest.raises(RuntimeError, match="only be added"):
        check()
    _git(repo, "reset", "-q", "--hard", "HEAD~1")
    (repo / "research" / "tables" / "other.csv").write_text("a\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "another table")
    with pytest.raises(RuntimeError, match="changed since the frozen commit"):
        check()
    # the EXP-200-final record is not allowed to change under the stage-2 rules either
    with pytest.raises(RuntimeError):
        check_git(record, repo)


def test_start_record_names_the_stage2_experiment(tmp_path):
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    log, decisions = repo / "experiment_log.md", repo / "decisions.md"
    log.write_text("| id |\n")
    decisions.write_text("| D1 |\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    run_dir = repo / "runs" / "EXP-223-final" / "20261012-000000-000000"
    record_start(run_dir, "b" * 40, log, extra_logs=[decisions], repo=repo, push=False,
                 what=s2.WHAT["EXP-223-final"], prereg="D121",
                 freeze_record=s2.FREEZE_RECORD)  # fmt: skip
    text = log.read_text("utf-8")
    assert "FINAL-STARTED: E23 subgroup gaps final run started" in text and "(D121)" in text
    assert json.loads((run_dir / "metrics.json").read_text("utf-8"))["status"] == "started"


def test_final_procedure_is_not_called_when_the_start_fails(monkeypatch):
    from vaultic.stage2 import run as s2run

    called = []
    monkeypatch.setattr(
        s2run, "FINAL", {e: (lambda *a, e=e: called.append(e)) for e in s2.EXPERIMENTS}
    )

    def failing_start(*a, **k):
        raise RuntimeError("push failed: stops before any test data is read")

    monkeypatch.setattr(s2, "start", failing_start)
    monkeypatch.setattr("sys.argv", ["final", "--exp", "EXP-223-final"])
    with pytest.raises(RuntimeError, match="push failed"):
        s2.main()
    assert called == []


def test_stage2_library_list_adds_dice_ml_and_leaves_exp200_unchanged():
    from vaultic.fusion import final_run

    assert "dice-ml" not in final_run.LIBRARIES  # EXP-200-final's record is unchanged
    versions = s2.library_versions()
    assert "dice-ml" in versions and set(final_run.library_versions()) < set(versions)


def test_final_runs_go_in_order(tmp_path):
    def run(exp, status):
        d = tmp_path / exp / f"2026-{status}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"mode": "final", "status": status}), "utf-8")

    s2.require_order("EXP-223-final", tmp_path)  # E23 is first
    with pytest.raises(RuntimeError, match="must wait"):
        s2.require_order("EXP-210-final", tmp_path)
    run("EXP-223-final", "started")  # started but not completed does not count
    with pytest.raises(RuntimeError, match="must wait"):
        s2.require_order("EXP-210-final", tmp_path)
    run("EXP-223-final", "completed")
    s2.require_order("EXP-210-final", tmp_path)
    with pytest.raises(RuntimeError, match="EXP-210-final"):
        s2.require_order("EXP-215-final", tmp_path)
    run("EXP-210-final", "completed")
    s2.require_order("EXP-215-final", tmp_path)
