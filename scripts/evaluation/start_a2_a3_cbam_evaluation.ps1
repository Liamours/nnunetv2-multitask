$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $PSScriptRoot "evaluate_a2_a3_cbam_sequential.ps1"
$logDir = Join-Path $repoRoot "evaluation_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Start-Process powershell `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runner) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "a2_a3_cbam_evaluation_20260713.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "a2_a3_cbam_evaluation_20260713.stderr.log") `
    -PassThru
