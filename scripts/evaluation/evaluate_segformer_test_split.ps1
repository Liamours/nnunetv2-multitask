param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [Parameter(Mandatory=$true)][string]$ModelRegistryName,
    [string]$Checkpoint = "best.pt",
    [string]$Device = "cuda",
    [switch]$Multitask
)

$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$segformerRepoRoot = Join-Path $Workspace "repo\segformer_multitask"
$nnunetRepoRoot = Join-Path $Workspace "repo\nnunetv2_multitask"
# -Multitask picks the bone+lesion dataset/export and relays both tasks; without it (single-task
# checkpoints have no bone head) this stays the original lesion-only path.
$rawDatasetName = if ($Multitask) { "Dataset260_BS80KLesionBoneMT" } else { "Dataset261_BS80KLesionOnly" }
$rawDataset = Join-Path $NNUNetRaw $rawDatasetName
$rootDirName = if ($Multitask) { "bs80k_multitask" } else { "bs80k_lesion" }
$splitCsv = Join-Path $rawDataset "split_seed42.csv"
$predDir = Join-Path $InferencesRoot "segformer\$RunName\test"
$relayDir = Join-Path $Workspace "temp\segformer_relay\$ModelRegistryName"
$metricsDir = Join-Path $MetricsRoot $ModelRegistryName
$metricFile = Join-Path $metricsDir "test_metrics.json"

Set-Location -LiteralPath $segformerRepoRoot
uv run python services/inference-segmentation/batch_infer.py `
    --checkpoint "$WeightsRoot\$RunName\checkpoints\$Checkpoint" `
    --root-dir "$SegformerRoot\$rootDirName" `
    --split test `
    --output-dir $predDir `
    --device $Device `
    --height 1024 `
    --width 512

if (Test-Path -LiteralPath $relayDir) { Remove-Item -LiteralPath $relayDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $relayDir | Out-Null

Set-Location -LiteralPath $nnunetRepoRoot
$relayTasks = if ($Multitask) { @("lesion", "bone") } else { @("lesion") }
foreach ($task in $relayTasks) {
    uv run python scripts/evaluation/relay_segformer_predictions.py `
        --source $predDir `
        --dest $relayDir `
        --split-csv $splitCsv `
        --split test `
        --task $task
}

$env:nnUNet_raw = $NNUNetRaw
$env:nnUNet_preprocessed = $NNUNetPreprocessed
$env:nnUNet_results = $NNUNetResults

New-Item -ItemType Directory -Force -Path $metricsDir | Out-Null
uv run nnUNetv2_evaluate_multitask `
    --raw_dataset $rawDataset `
    --predictions $relayDir `
    --split test `
    --output $metricFile

Remove-Item -LiteralPath $relayDir -Recurse -Force

Write-Host ""
Write-Host "Predictions: $predDir"
Write-Host "Metrics: $metricFile"
Get-Content -Raw -LiteralPath $metricFile | ConvertFrom-Json | ConvertTo-Json -Depth 4
