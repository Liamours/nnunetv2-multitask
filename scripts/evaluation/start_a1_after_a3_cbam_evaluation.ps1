$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$logDir = Join-Path $repoRoot "evaluation_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Start-Process powershell `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "evaluate_a1_after_a3_cbam.ps1")) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "a1_after_a3_cbam_evaluation_20260713.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "a1_after_a3_cbam_evaluation_20260713.stderr.log") `
    -PassThru
