param(
    [switch]$SkipEntrypointCheck
)

$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$Repo = Join-Path $Workspace "repo\nnunetv2_multitask"
$DatasetId = "261"
$DatasetName = "Dataset261_BS80KLesionOnly"
$PlansName = "nnUNetPlansA1Lesion2GB"
$TrainerName = "nnUNetTrainer_50epochs"

Set-Location -LiteralPath $Repo

$env:nnUNet_raw = $NNUNetRaw
$env:nnUNet_preprocessed = $NNUNetPreprocessed
$env:nnUNet_results = $NNUNetResults
$env:nnUNet_n_proc_DA = "1"
$env:nnUNet_def_n_proc = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

$RawDataset = Join-Path $env:nnUNet_raw $DatasetName
$PreprocessedDataset = Join-Path $env:nnUNet_preprocessed $DatasetName
$DatasetJsonPath = Join-Path $RawDataset "dataset.json"
$PlanPath = Join-Path $PreprocessedDataset "$PlansName.json"
$PlanDataDir = Join-Path $PreprocessedDataset "${PlansName}_2d"
$SplitsPath = Join-Path $PreprocessedDataset "splits_final.json"

function Assert-PathExists {
    param([string]$Path, [string]$Meaning)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing $Meaning`: $Path"
    }
}

Assert-PathExists $RawDataset "raw Dataset261 folder"
Assert-PathExists $DatasetJsonPath "Dataset261 dataset.json"
Assert-PathExists (Join-Path $RawDataset "imagesTr") "Dataset261 imagesTr"
Assert-PathExists (Join-Path $RawDataset "labelsTr") "Dataset261 labelsTr"
Assert-PathExists $PreprocessedDataset "preprocessed Dataset261 folder"
Assert-PathExists $PlanPath "A1 plans file"
Assert-PathExists $PlanDataDir "A1 preprocessed 2d data folder"
Assert-PathExists $SplitsPath "A1 splits_final.json"

$datasetJson = Get-Content -LiteralPath $DatasetJsonPath -Raw | ConvertFrom-Json
if ($datasetJson.channel_names.PSObject.Properties.Name.Count -ne 2 -or $datasetJson.channel_names."0" -ne "anterior" -or $datasetJson.channel_names."1" -ne "posterior") {
    throw "Dataset261 must be paired-patient input with channel_names {'0':'anterior','1':'posterior'}."
}
if ($datasetJson.multitask.case_unit -ne "paired_anterior_posterior") {
    throw "Dataset261 must use paired_anterior_posterior case semantics."
}
if ($datasetJson.multitask.tasks.PSObject.Properties.Name.Count -ne 1 -or -not $datasetJson.multitask.tasks.lesion) {
    throw "Dataset261 must define exactly one task: lesion."
}
if ($datasetJson.multitask.tasks.lesion.labels.background -ne 0 -or $datasetJson.multitask.tasks.lesion.labels.benign -ne 1 -or $datasetJson.multitask.tasks.lesion.labels.malignant -ne 2) {
    throw "Dataset261 lesion labels must be background=0, benign=1, malignant=2."
}

$imageFiles = Get-ChildItem -LiteralPath (Join-Path $RawDataset "imagesTr") -Filter "*.png" -File
$labelFiles = Get-ChildItem -LiteralPath (Join-Path $RawDataset "labelsTr\lesion") -Filter "*.png" -File
if ($imageFiles.Count -ne ([int]$datasetJson.numTraining * 2)) {
    throw "imagesTr count $($imageFiles.Count) does not equal paired case count x 2: $($datasetJson.numTraining * 2)."
}
if ($labelFiles.Count -ne ([int]$datasetJson.numTraining * 2)) {
    throw "lesion label count $($labelFiles.Count) does not equal paired case count x 2: $($datasetJson.numTraining * 2)."
}

$badImages = $imageFiles | Where-Object { $_.BaseName -notmatch '^bs80k_\d{4}_000[01]$' } | Select-Object -First 5
if ($badImages) {
    throw "A1 image names must be bs80k_####_0000.png and bs80k_####_0001.png. Example bad file: $($badImages[0].Name)"
}

$missingLabels = foreach ($caseId in ($imageFiles | ForEach-Object { $_.BaseName -replace '_000[01]$','' } | Sort-Object -Unique)) {
    foreach ($viewIdx in 0, 1) {
        $labelPath = Join-Path (Join-Path $RawDataset "labelsTr\lesion") ("{0}_{1:D4}.png" -f $caseId, $viewIdx)
        if (-not (Test-Path -LiteralPath $labelPath)) { $labelPath }
    }
}
if ($missingLabels.Count -gt 0) {
    throw "Missing labels for $($missingLabels.Count) A1 images. First missing: $($missingLabels[0])"
}

if (-not $SkipEntrypointCheck) {
    & uv run nnUNetv2_train -h *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "uv run nnUNetv2_train is not available in this repository environment."
    }
}

Write-Host "A1 setup is ready."
Write-Host "Dataset: $RawDataset"
Write-Host "Preprocessed: $PreprocessedDataset"
Write-Host "Cases/images/lesion labels: $($datasetJson.numTraining) / $($imageFiles.Count) / $($labelFiles.Count)"
Write-Host "Plans: $PlanPath"
Write-Host ""
Write-Host "Train later with:"
Write-Host "cd $Repo"
Write-Host "`$env:nnUNet_raw='$env:nnUNet_raw'"
Write-Host "`$env:nnUNet_preprocessed='$env:nnUNet_preprocessed'"
Write-Host "`$env:nnUNet_results='$env:nnUNet_results'"
Write-Host "uv run nnUNetv2_train $DatasetId 2d 0 -tr nnUNetTrainerMultiTask_50epochs -p $PlansName -device cuda"
