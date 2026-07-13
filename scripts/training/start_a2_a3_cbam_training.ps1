$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$runner = Join-Path $PSScriptRoot 'train_a2_a3_cbam_sequential.ps1'
$stdoutLog = Join-Path $repoRoot 'training_logs\\a2_a3_cbam_full_train_sequential_20260713.stdout.log'
$stderrLog = Join-Path $repoRoot 'training_logs\\a2_a3_cbam_full_train_sequential_20260713.stderr.log'

Start-Process powershell `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runner) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru
