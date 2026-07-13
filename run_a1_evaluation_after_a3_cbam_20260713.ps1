$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\lulay\Desktop\nnunetv2-multitask\repo\nnunetv2-multitask"
$dataRoot = "C:\Users\lulay\Desktop\nnunetv2-multitask\data"

while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "run_a2_a3_cbam_evaluation_20260713" }) {
    Start-Sleep -Seconds 30
}

$a3Summary = Join-Path $dataRoot "evaluation\a3_dual_decoder_cbam_100epoch_best_latest\summary.csv"
if (-not (Test-Path -LiteralPath $a3Summary)) {
    throw "A3 + CBAM evaluation did not produce its summary; A1 evaluation was not started."
}

Set-Location -LiteralPath $repoRoot
& (Join-Path $repoRoot "evaluate_a1_100epoch_best_latest_val_test.ps1") `
    -PlanName "nnUNetPlansA1Lesion2GB" `
    -EvaluationName "a1_lesion_only_100epoch_best_latest" `
    -RunPrefix "A1_lesion_only"
if ($LASTEXITCODE -ne 0) {
    throw "A1 lesion-only evaluation failed."
}

& (Join-Path $repoRoot "evaluate_a1_100epoch_best_latest_val_test.ps1") `
    -PlanName "nnUNetPlansA1ControlledBatch4CBAM" `
    -EvaluationName "a1_lesion_only_cbam_100epoch_best_latest" `
    -RunPrefix "A1_lesion_only_cbam"
if ($LASTEXITCODE -ne 0) {
    throw "A1 lesion-only + CBAM evaluation failed."
}

& (Join-Path $repoRoot "build_six_experiment_evaluation_tables.ps1")
