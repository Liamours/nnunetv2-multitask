param(
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$Repo = Join-Path $Workspace "repo\nnunetv2_multitask"
$RawMT = Join-Path $NNUNetRaw "Dataset260_BS80KLesionBoneMT"
$ImageAll = Join-Path $RawMT "imagesTr"
$EvalRoot = Join-Path $Workspace "analyses\evaluations\available_now"
$InputRoot = Join-Path $EvalRoot "inputs"
$SummaryFile = Join-Path $EvalRoot "available_now_summary.csv"

Set-Location -LiteralPath $Repo
$env:nnUNet_raw = $NNUNetRaw
$env:nnUNet_preprocessed = $NNUNetPreprocessed
$env:nnUNet_results = $NNUNetResults
$env:nnUNet_n_proc_DA = "1"
$env:nnUNet_def_n_proc = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

New-Item -ItemType Directory -Force -Path $EvalRoot | Out-Null

$splitRows = Import-Csv (Join-Path $RawMT "split_seed42.csv")
foreach ($split in @("val", "test")) {
    $splitInput = Join-Path $InputRoot $split
    New-Item -ItemType Directory -Force -Path $splitInput | Out-Null
    $cases = $splitRows | Where-Object { $_.split -eq $split }
    foreach ($case in $cases) {
        foreach ($channel in @("0000", "0001")) {
            $src = Join-Path $ImageAll "$($case.case_id)_$channel.png"
            $dst = Join-Path $splitInput "$($case.case_id)_$channel.png"
            if (-not (Test-Path -LiteralPath $dst)) {
                New-Item -ItemType HardLink -Path $dst -Target $src | Out-Null
            }
        }
    }
}

$runs = @(
    @{
        Id = "A2_dual_head_best"
        Dataset = "260"
        Trainer = "nnUNetTrainerMultiTask_100epochs"
        Plans = "nnUNetPlansMultiTask2GB"
        Checkpoint = "checkpoint_best.pth"
    },
    @{
        Id = "A2_dual_head_latest"
        Dataset = "260"
        Trainer = "nnUNetTrainerMultiTask_100epochs"
        Plans = "nnUNetPlansMultiTask2GB"
        Checkpoint = "checkpoint_latest.pth"
    },
    @{
        Id = "A3_dual_decoder_best"
        Dataset = "260"
        Trainer = "nnUNetTrainerMultiTask_100epochs"
        Plans = "nnUNetPlansMultiTaskDualDecoder"
        Checkpoint = "checkpoint_best.pth"
    }
)

$summary = @()
foreach ($run in $runs) {
    foreach ($split in @("val", "test")) {
        $inputDir = Join-Path $InputRoot $split
        $predDir = Join-Path $EvalRoot "$($run.Id)\$split\predictions"
        $metricFile = Join-Path $EvalRoot "$($run.Id)\$split\metrics.json"
        New-Item -ItemType Directory -Force -Path $predDir | Out-Null

        uv run nnUNetv2_predict `
            -i $inputDir `
            -o $predDir `
            -d $run.Dataset `
            -c 2d `
            -tr $run.Trainer `
            -p $run.Plans `
            -f 0 `
            -chk $run.Checkpoint `
            -device $Device `
            -npp 1 `
            -nps 1 `
            --disable_tta `
            --continue_prediction `
            --disable_progress_bar

        uv run nnUNetv2_evaluate_multitask `
            --raw_dataset $RawMT `
            --predictions $predDir `
            --split $split `
            --output $metricFile

        $metrics = Get-Content -Raw -LiteralPath $metricFile | ConvertFrom-Json
        $lesion = $metrics.tasks.lesion
        $bone = $metrics.tasks.bone
        $summary += [PSCustomObject]@{
            model = $run.Id
            split = $split
            checkpoint = $run.Checkpoint
            lesion_dice = $lesion.pixel_mean.dice
            lesion_sensitivity = $lesion.pixel_mean.sensitivity
            lesion_specificity = $lesion.pixel_mean.specificity
            lesionwise_f1 = $lesion.lesionwise_class_matched.lesionwise_f1
            lesion_froc_llf = $lesion.lesionwise_class_matched.froc.lesion_localization_fraction
            lesion_froc_fp_per_case = $lesion.lesionwise_class_matched.froc.false_positives_per_case
            bone_dice = $bone.pixel_mean.dice
            bone_sensitivity = $bone.pixel_mean.sensitivity
            bone_specificity = $bone.pixel_mean.specificity
            metrics_file = $metricFile
        }
    }
}

$summary | Export-Csv -NoTypeInformation -Path $SummaryFile
$summary | Format-Table -AutoSize
Write-Host "Saved summary: $SummaryFile"
