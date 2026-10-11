import subprocess
import sys

from vaultic.paths import REPO_ROOT
from vaultic.reports import paper_results
from vaultic.reports.paper_results import STEPS, Step, run_all


def test_registry_wires_every_paper_table():
    modules = [s.args[1] for s in STEPS]
    assert modules[:2] == ["vaultic.reports.table1", "vaultic.reports.e2_fyp1"]
    assert ("--mode", "final") == STEPS[0].args[2:]
    gains = {s.args[s.args.index("--name") + 1] for s in STEPS if "--name" in s.args}
    assert gains >= {"b5_drop_labels", "e3_tuned", "e5_tuned", "e8_tuned", "e25_L7", "e25_L30",
                     "e25_L60", "holdout_R_b5", "holdout_R_anomaly"}  # fmt: skip
    rendered = {s.args[-1] for s in STEPS if s.args[1] == "vaultic.reports.render_saved"}
    assert rendered == {"d95", "final", "f3-diagnosis", "f3-sensitivity", "e5-drift"}
    assert any(s.args[1:] == ("vaultic.reports.emerging", "report") for s in STEPS)
    for s in STEPS:  # every run a step reads exists
        for a in s.args:
            if a.startswith("experiments/runs/"):
                assert (REPO_ROOT / a).is_dir(), a


def test_render_keeps_the_hand_written_section(tmp_path):
    from vaultic.reports.render_saved import keep_hand_section

    path = tmp_path / "report.md"
    hand = "## 3. Reading (added by hand, not generated)"
    path.write_text(f"# R\n\nold table\n\n{hand}\n\nMy notes.\n")

    def write():
        path.write_text(f"# R\n\nnew table\n\n{hand}\n\n_To be written._\n")

    keep_hand_section(path, write)
    text = path.read_text()
    assert "new table" in text and "My notes." in text and "_To be written._" not in text


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
