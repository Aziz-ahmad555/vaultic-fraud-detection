# Runs the remaining Phase 2 steps in order, unattended. Each step is logged with its exit
# code to experiments/tuning/phase2_chain.log; a failed step is logged and the chain goes on.
# Usage (repo root):  powershell -File tools/run_phase2.ps1
#
# Order: all tuning first (B5, B3, B4 Optuna with equal budgets; B1 grid over C), then the
# configs are frozen (hashes in research/frozen_configs.md), then exactly ONE --final run per
# baseline B1-B6 (the harness refuses a second one without a logged reason). Final runs also
# report validation metrics, so no separate development runs are needed.

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

# 1. Tuning, validation only
Step "tune B5" @("-m", "vaultic.eval.tune", "--name", "B5", "--features", "b5", "--model", "xgboost", "--trials-per-arm", "25", "--config-id", "EXP-009")
Step "tune B3" @("-m", "vaultic.eval.tune", "--name", "B3", "--features", "raw", "--model", "xgboost", "--trials-per-arm", "25", "--config-id", "EXP-011")
Step "tune B4" @("-m", "vaultic.eval.tune", "--name", "B4", "--features", "raw", "--model", "lightgbm", "--trials-per-arm", "25", "--config-id", "EXP-013")
Step "grid B1" @("-m", "vaultic.eval.tune", "--name", "B1", "--features", "raw_lr", "--model", "logistic_regression", "--grid-c", "0.001", "0.01", "0.1", "1", "10", "--config-id", "EXP-012")

# 2. Freeze every Table 1 config before any test-period run
Step "freeze configs" @("-m", "vaultic.reports.phase2_summary", "freeze")

# 3. SHAP of tuned B5 (train + validation only)
Step "shap B5" @("-m", "vaultic.reports.shap_summary", "experiments/configs/EXP-009.yaml", "--name", "b5")

# 4. One --final run per baseline
foreach ($exp in "EXP-012", "EXP-002", "EXP-011", "EXP-013", "EXP-009", "EXP-010") {
    Step "final $exp" @("-m", "vaultic.eval.run", "experiments/configs/$exp.yaml", "--final")
}

# 5. Tables, gate and summary
Step "table1 final" @("-m", "vaultic.reports.table1", "--mode", "final")
Step "phase2 gate" @("-m", "vaultic.reports.phase2_gate")
Step "e2 table" @("-m", "vaultic.reports.e2_fyp1")
Step "summary" @("-m", "vaultic.reports.phase2_summary", "summary")

"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') CHAIN FINISHED" | Out-File $log -Append -Encoding utf8
