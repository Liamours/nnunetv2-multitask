param(
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$repoRoot = Join-Path $Workspace "repo\nnunetv2_multitask"
Set-Location -LiteralPath $repoRoot

$runs = @(
    @{
        Script = "evaluate_a2_best_latest.ps1"
        Plan = "nnUNetPlansA2ControlledBatch4CBAM"
        EvaluationName = "a2_dual_head_cbam_100epoch_best_latest"
        RunPrefix = "A2_dual_head_cbam"
    },
    @{
        Script = "evaluate_a3_best_latest.ps1"
        Plan = "nnUNetPlansA3ControlledBatch4CBAM"
        EvaluationName = "a3_dual_decoder_cbam_100epoch_best_latest"
        RunPrefix = "A3_dual_decoder_cbam"
    }
)

foreach ($run in $runs) {
    & (Join-Path $PSScriptRoot $run.Script) `
        -Device $Device `
        -PlanName $run.Plan `
        -EvaluationName $run.EvaluationName `
        -RunPrefix $run.RunPrefix
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed: $($run.RunPrefix)"
    }
}
