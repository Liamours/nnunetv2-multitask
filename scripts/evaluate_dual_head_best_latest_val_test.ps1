param(
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$Repo = Join-Path $Workspace "repo\nnunetv2_multitask"
$RawDataset = Join-Path $NNUNetRaw "Dataset260_BS80KLesionBoneMT"
$Images = Join-Path $RawDataset "imagesTr"
$EvalRoot = Join-Path $Workspace "analyses\evaluations\dual_head"

Set-Location -LiteralPath $Repo
$env:nnUNet_raw = $NNUNetRaw
$env:nnUNet_preprocessed = $NNUNetPreprocessed
$env:nnUNet_results = $NNUNetResults
$env:nnUNet_n_proc_DA = "1"
$env:nnUNet_def_n_proc = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

$runs = @(
    @{ Name = "best"; Checkpoint = "checkpoint_best.pth" },
    @{ Name = "latest"; Checkpoint = "checkpoint_final.pth" }
)
$splits = @("val", "test")

foreach ($run in $runs) {
    $PredDir = Join-Path $EvalRoot "$($run.Name)\predictions"
    New-Item -ItemType Directory -Force -Path $PredDir | Out-Null

    uv run nnUNetv2_predict `
        -i $Images `
        -o $PredDir `
        -d 260 `
        -c 2d `
        -tr nnUNetTrainerMultiTask_100epochs `
        -p nnUNetPlansMultiTask2GB `
        -f 0 `
        -chk $run.Checkpoint `
        -device $Device `
        -npp 1 `
        -nps 1 `
        --disable_progress_bar

    foreach ($split in $splits) {
        $OutFile = Join-Path $EvalRoot "$($run.Name)\metrics_$split.json"
        uv run nnUNetv2_evaluate_multitask `
            --raw_dataset $RawDataset `
            --predictions $PredDir `
            --split $split `
            --output $OutFile
    }
}
