# Runs the remaining Phase 2 steps in order, unattended. Each step is logged with its exit
# code to experiments/tuning/phase2_chain.log; a failed step is logged and the chain goes on.
# Usage (repo root):  powershell -File tools/run_phase2.ps1
# Test-period runs (--final) are made ONLY for B3 and B5, for the Phase 2 exit gate.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$py = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "experiments\tuning\phase2_chain.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

function Step([string]$name, [string[]]$arguments) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') START $name" | Out-File $log -Append -Encoding utf8
    & $py -W ignore @arguments *>> $log
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') END $name exit=$LASTEXITCODE" | Out-File $log -Append -Encoding utf8
}

# 0. If a B5 tuning process is already running, wait for it (the study is resumable either way).
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -match 'vaultic\.eval\.tune --name B5' }) {
    Start-Sleep -Seconds 60
}

# 1-2. Tuning with equal budgets (resumes; writes EXP-009 / EXP-011 and research/tuning_*.md)
Step "tune B5" @("-m", "vaultic.eval.tune", "--name", "B5", "--features", "b5", "--trials-per-arm", "25", "--config-id", "EXP-009")
Step "tune B3" @("-m", "vaultic.eval.tune", "--name", "B3", "--features", "raw", "--trials-per-arm", "25", "--config-id", "EXP-011")

# 3. Development runs (validation only)
foreach ($exp in "EXP-009", "EXP-011", "EXP-001", "EXP-004", "EXP-002") {
    Step "dev $exp" @("-m", "vaultic.eval.run", "experiments/configs/$exp.yaml")
}

# 4. SHAP summary of tuned B5 and Table 1 on validation
Step "shap B5" @("-m", "vaultic.reports.shap_summary", "experiments/configs/EXP-009.yaml", "--name", "b5")
Step "table1 development" @("-m", "vaultic.reports.table1", "--mode", "development")

# 5. Phase 2 exit gate: final (test-period) runs of B3 and B5 only
Step "final EXP-011" @("-m", "vaultic.eval.run", "experiments/configs/EXP-011.yaml", "--final")
Step "final EXP-009" @("-m", "vaultic.eval.run", "experiments/configs/EXP-009.yaml", "--final")
Step "phase2 gate" @("-m", "vaultic.reports.phase2_gate")

"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') CHAIN FINISHED" | Out-File $log -Append -Encoding utf8
