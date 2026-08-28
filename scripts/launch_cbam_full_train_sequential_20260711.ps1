$ErrorActionPreference = 'Stop'

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$repoRoot = Join-Path $Workspace "repo\nnunetv2_multitask"
$runner = Join-Path $repoRoot 'run_cbam_full_train_sequential_20260711.ps1'
$stdoutLog = Join-Path $repoRoot 'training_logs\\cbam_full_train_sequential_20260711.stdout.log'
$stderrLog = Join-Path $repoRoot 'training_logs\\cbam_full_train_sequential_20260711.stderr.log'

Start-Process powershell `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runner) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru
