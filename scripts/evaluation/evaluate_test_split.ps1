param(
    [Parameter(Mandatory=$true)][string]$DatasetId,
    [Parameter(Mandatory=$true)][string]$DatasetName,
    [Parameter(Mandatory=$true)][string]$PlanName,
    [Parameter(Mandatory=$true)][string]$TrainerName,
    [Parameter(Mandatory=$true)][string]$ModelRegistryName,
    [string]$Checkpoint = "checkpoint_best.pth",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$repoRoot = Join-Path $Workspace "repo\nnunetv2_multitask"
$rawDataset = Join-Path $NNUNetRaw $DatasetName
$images = Join-Path $rawDataset "imagesTr"
$tempInput = Join-Path $Workspace "temp\test_split_inputs\$ModelRegistryName"
$predDir = Join-Path $InferencesRoot "nnunet\$ModelRegistryName\test"
$metricsDir = Join-Path $MetricsRoot $ModelRegistryName
$metricFile = Join-Path $metricsDir "test_metrics.json"

Set-Location -LiteralPath $repoRoot
$env:nnUNet_raw = $NNUNetRaw
$env:nnUNet_preprocessed = $NNUNetPreprocessed
$env:nnUNet_results = $NNUNetResults

if (Test-Path -LiteralPath $tempInput) { Remove-Item -LiteralPath $tempInput -Recurse -Force }
New-Item -ItemType Directory -Force -Path $tempInput | Out-Null
$splitRows = Import-Csv (Join-Path $rawDataset "split_seed42.csv") | Where-Object { $_.split -eq "test" }
foreach ($case in $splitRows) {
    foreach ($channel in @("0000", "0001")) {
        $src = Join-Path $images "$($case.case_id)_$channel.png"
        $dst = Join-Path $tempInput "$($case.case_id)_$channel.png"
        New-Item -ItemType HardLink -Path $dst -Target $src | Out-Null
    }
}
Write-Host "Test split: $($splitRows.Count) cases hardlinked to $tempInput"

if (Test-Path -LiteralPath $predDir) { Remove-Item -LiteralPath $predDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $predDir | Out-Null

uv run nnUNetv2_predict `
    -i $tempInput `
    -o $predDir `
    -d $DatasetId `
    -c 2d `
    -tr $TrainerName `
    -p $PlanName `
    -f 0 `
    -chk $Checkpoint `
    -device $Device `
    --disable_tta

New-Item -ItemType Directory -Force -Path $metricsDir | Out-Null
uv run nnUNetv2_evaluate_multitask `
    --raw_dataset $rawDataset `
    --predictions $predDir `
    --split test `
    --output $metricFile

Remove-Item -LiteralPath $tempInput -Recurse -Force

Write-Host ""
Write-Host "Predictions: $predDir"
Write-Host "Metrics: $metricFile"
Get-Content -Raw -LiteralPath $metricFile | ConvertFrom-Json | ConvertTo-Json -Depth 4
