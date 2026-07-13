$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\lulay\Desktop\nnunetv2-multitask\repo\nnunetv2-multitask"
$logDir = Join-Path $repoRoot "evaluation_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Start-Process powershell `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $repoRoot "run_a1_evaluation_after_a3_cbam_20260713.ps1")) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "a1_after_a3_cbam_evaluation_20260713.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "a1_after_a3_cbam_evaluation_20260713.stderr.log") `
    -PassThru
