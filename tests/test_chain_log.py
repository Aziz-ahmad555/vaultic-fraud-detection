"""The chain's Step helper writes one UTF-8 log, with output, errors and the exit code."""

import shutil
import subprocess
import sys

import pytest

from vaultic.paths import REPO_ROOT

POWERSHELL = shutil.which("powershell")


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell not available")
def test_step_writes_utf8_with_exit_code(tmp_path):
    log = tmp_path / "chain.log"
    snippet = (
        "import sys; print('trial 1: val PR-AUC 0.6549 ± 0.0012'); "
        "print('a warning', file=sys.stderr); sys.exit(3)"
    )
    script = (
        f". '{REPO_ROOT / 'tools' / 'chain_lib.ps1'}'; "
        f"$code = Step 'demo' @('-c', \"{snippet}\") '{log}' '{sys.executable}'; "
        "exit $code"
    )
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 3
    raw = log.read_bytes()
    assert b"\x00" not in raw  # no UTF-16 anywhere
    text = raw.decode("utf-8-sig")
    assert "trial 1: val PR-AUC 0.6549 ± 0.0012" in text
    assert "a warning" in text
    assert "START demo" in text and "END demo exit=3" in text


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell not available")
def test_step_keeps_warnings_and_flags_them(tmp_path):
    """The chain no longer passes -W ignore; flagged warnings are counted on the END line."""
    assert "& $py -W ignore" not in (REPO_ROOT / "tools" / "chain_lib.ps1").read_text(
        encoding="utf-8"
    )
    log = tmp_path / "chain.log"
    snippet = (
        "import warnings; from sklearn.exceptions import ConvergenceWarning; "
        "warnings.warn('lbfgs failed to converge', ConvergenceWarning)"
    )
    script = (
        f". '{REPO_ROOT / 'tools' / 'chain_lib.ps1'}'; "
        f"$code = Step 'demo' @('-c', \"{snippet}\") '{log}' '{sys.executable}'; "
        "exit $code"
    )
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0
    text = log.read_bytes().decode("utf-8-sig")
    assert "lbfgs failed to converge" in text
    assert "END demo exit=0 FLAGGED-WARNINGS=" in text
