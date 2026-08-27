$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$repoRoot = Join-Path $Workspace "repo\nnunetv2_multitask"

while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "evaluate_a2_a3_cbam_sequential" }) {
    Start-Sleep -Seconds 30
}

$a3Summary = Join-Path $Workspace "analyses\evaluations\a3_dual_decoder_cbam_100epoch_best_latest\summary.csv"
if (-not (Test-Path -LiteralPath $a3Summary)) {
    throw "A3 + CBAM evaluation did not produce its summary; A1 evaluation was not started."
}

Set-Location -LiteralPath $repoRoot
& (Join-Path $PSScriptRoot "evaluate_a1_best_latest.ps1") `
    -PlanName "nnUNetPlansA1Lesion2GB" `
    -EvaluationName "a1_lesion_only_100epoch_best_latest" `
    -RunPrefix "A1_lesion_only"
if ($LASTEXITCODE -ne 0) {
    throw "A1 lesion-only evaluation failed."
}

& (Join-Path $PSScriptRoot "evaluate_a1_best_latest.ps1") `
    -PlanName "nnUNetPlansA1ControlledBatch4CBAM" `
    -EvaluationName "a1_lesion_only_cbam_100epoch_best_latest" `
    -RunPrefix "A1_lesion_only_cbam"
if ($LASTEXITCODE -ne 0) {
    throw "A1 lesion-only + CBAM evaluation failed."
}

& (Join-Path $PSScriptRoot "build_six_experiment_tables.ps1")
