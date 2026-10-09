# Windows equivalent of `make paper-results`: regenerate every paper table and figure from
# existing harness runs. Usage (repo root):  powershell -File tools/paper_results.ps1 [--list]
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"
$venv = Join-Path $root ".venv\Scripts\python.exe"
$py = if (Test-Path $venv) { $venv } else { "python" }   # e.g. a git worktree has no .venv
& $py -m vaultic.reports.paper_results @args
exit $LASTEXITCODE
