param(
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\lulay\Desktop\nnunetv2-multitask\repo\nnunetv2-multitask"
Set-Location -LiteralPath $repoRoot

$runs = @(
    @{
        Script = "evaluate_a2_100epoch_best_latest_val_test.ps1"
        Plan = "nnUNetPlansA2ControlledBatch4CBAM"
        EvaluationName = "a2_dual_head_cbam_100epoch_best_latest"
        RunPrefix = "A2_dual_head_cbam"
    },
    @{
        Script = "evaluate_a3_100epoch_best_latest_val_test.ps1"
        Plan = "nnUNetPlansA3ControlledBatch4CBAM"
        EvaluationName = "a3_dual_decoder_cbam_100epoch_best_latest"
        RunPrefix = "A3_dual_decoder_cbam"
    }
)

foreach ($run in $runs) {
    & (Join-Path $repoRoot $run.Script) `
        -Device $Device `
        -PlanName $run.Plan `
        -EvaluationName $run.EvaluationName `
        -RunPrefix $run.RunPrefix
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed: $($run.RunPrefix)"
    }
}
