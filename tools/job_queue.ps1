# Sequential job queue (decision D73): at most one heavy job runs at a time.
#
# Each job is a file <queue>\NNN-name.ps1 holding PowerShell commands. The runner takes the
# lowest-named file, runs it in a fresh PowerShell from the repo root, appends its output to
# <queue>\log\NNN-name.log, records the exit code and moves the file to <queue>\done. Jobs can
# be added while it runs. It stops when <queue>\STOP exists or after `IdleMinutes` with an
# empty queue. Usage (repo root):
#   powershell -File tools/job_queue.ps1 -Queue E:\dev-cache\queue

param(
    [string]$Queue = "E:\dev-cache\queue",
    [int]$IdleMinutes = 60
)

$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $Queue "log"
$done = Join-Path $Queue "done"
New-Item -ItemType Directory -Force $Queue, $log, $done | Out-Null
$idleSince = Get-Date

while (-not (Test-Path (Join-Path $Queue "STOP"))) {
    $job = Get-ChildItem $Queue -Filter *.ps1 -File | Sort-Object Name | Select-Object -First 1
    if ($null -eq $job) {
        if (((Get-Date) - $idleSince).TotalMinutes -ge $IdleMinutes) { break }
        Start-Sleep -Seconds 20
        continue
    }
    $out = Join-Path $log ($job.BaseName + ".log")
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') START $($job.Name)" | Out-File $out -Append -Encoding utf8
    Write-Output "$(Get-Date -Format 'HH:mm:ss') START $($job.Name)"
    $proc = Start-Process powershell -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", $job.FullName) -WorkingDirectory $root -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$out.stdout" -RedirectStandardError "$out.stderr"
    Get-Content "$out.stdout", "$out.stderr" -ErrorAction SilentlyContinue | Out-File $out -Append -Encoding utf8
    Remove-Item "$out.stdout", "$out.stderr" -ErrorAction SilentlyContinue
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') END $($job.Name) exit=$($proc.ExitCode)" | Out-File $out -Append -Encoding utf8
    Write-Output "$(Get-Date -Format 'HH:mm:ss') END $($job.Name) exit=$($proc.ExitCode)"
    Move-Item $job.FullName (Join-Path $done $job.Name) -Force
    $idleSince = Get-Date
}
Write-Output "queue stopped"
