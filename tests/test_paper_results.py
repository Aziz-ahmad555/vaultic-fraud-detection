import subprocess
import sys

from vaultic.paths import REPO_ROOT
from vaultic.reports import paper_results
from vaultic.reports.paper_results import STEPS, Step, run_all


def test_registry_wires_table1_and_e2():
    modules = [s.args[1] for s in STEPS]
    assert modules == ["vaultic.reports.table1", "vaultic.reports.e2_fyp1"]
    assert ("--mode", "final") == STEPS[0].args[2:]


def test_every_step_runs_and_failures_are_reported():
    steps = (
        Step("passes", ("-c", "print('ok')"), "-"),
        Step("fails", ("-c", "import sys; print('boom', file=sys.stderr); sys.exit(2)"), "-"),
        Step("runs after a failure", ("-c", "pass"), "-"),
    )
    results = run_all(steps)
    assert [r["ok"] for r in results] == [True, False, True]
    assert "boom" in results[1]["error"]


def test_list_option(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["paper_results", "--list"])
    paper_results.main()
    out = capsys.readouterr().out
    assert "Table 1" in out and "e2_fyp1_comparison" in out


def test_exit_code_is_nonzero_when_a_step_fails(tmp_path):
    script = (
        "import vaultic.reports.paper_results as pr; "
        "pr.STEPS = (pr.Step('fails', ('-c', 'raise SystemExit(3)'), '-'),); pr.main()"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1 and "0 of 1 steps succeeded" in proc.stdout
