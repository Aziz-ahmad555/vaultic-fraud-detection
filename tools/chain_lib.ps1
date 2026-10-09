# Shared helpers for the unattended experiment chains (dot-source this file).
#
# Step runs one Python module and appends its output to a log as UTF-8. Windows PowerShell
# 5.1 writes redirected native output (`*>>`) as UTF-16, which made earlier chain logs
# unreadable ("t r i a l"); piping through Out-File -Encoding utf8 keeps one encoding.
#
# Python runs WITHOUT `-W ignore` (it hid B1's ConvergenceWarning, D61): the harness and the
# tuner capture every warning themselves (warnings.log, metrics.json, tuning report) and print
# them, so they also land in the chain log. After each step, flagged warnings are counted and
# written to the END line.

$env:PYTHONIOENCODING = "utf-8"            # Python writes UTF-8 ...
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8   # ... and PowerShell reads it as such

function Step([string]$name, [string[]]$arguments, [string]$log, [string]$py) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') START $name" | Out-File $log -Append -Encoding utf8
    $flagged = 0
    & $py @arguments 2>&1 | ForEach-Object {
        $line = "$_"
        if ($line -match "ConvergenceWarning|FitFailedWarning|UndefinedMetricWarning|RuntimeWarning") { $flagged++ }
        $line
    } | Out-File $log -Append -Encoding utf8
    $code = $LASTEXITCODE
    $note = if ($flagged) { " FLAGGED-WARNINGS=$flagged" } else { "" }
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') END $name exit=$code$note" | Out-File $log -Append -Encoding utf8
    return $code
}
