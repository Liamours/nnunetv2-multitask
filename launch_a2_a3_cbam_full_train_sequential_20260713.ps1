$ErrorActionPreference = 'Stop'

$repoRoot = 'C:\Users\lulay\Desktop\nnunetv2-multitask\repo\nnunetv2-multitask'
$runner = Join-Path $repoRoot 'run_a2_a3_cbam_full_train_sequential_20260713.ps1'
$stdoutLog = Join-Path $repoRoot 'training_logs\\a2_a3_cbam_full_train_sequential_20260713.stdout.log'
$stderrLog = Join-Path $repoRoot 'training_logs\\a2_a3_cbam_full_train_sequential_20260713.stderr.log'

Start-Process powershell `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runner) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru
