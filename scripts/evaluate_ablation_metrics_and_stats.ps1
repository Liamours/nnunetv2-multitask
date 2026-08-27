param(
    [string]$Device = "cuda",
    [string]$Checkpoint = "checkpoint_best.pth"
)

$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$Repo = Join-Path $Workspace "repo\nnunetv2_multitask"
$EvalRoot = Join-Path $Workspace "analyses\evaluations\ablation"
$RawA1 = Join-Path $NNUNetRaw "Dataset261_BS80KLesionOnly"
$RawMT = Join-Path $NNUNetRaw "Dataset260_BS80KLesionBoneMT"

Set-Location -LiteralPath $Repo
$env:nnUNet_raw = $NNUNetRaw
$env:nnUNet_preprocessed = $NNUNetPreprocessed
$env:nnUNet_results = $NNUNetResults
$env:nnUNet_n_proc_DA = "1"
$env:nnUNet_def_n_proc = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

$runs = @(
    @{
        Name = "A1_lesion_only"
        Dataset = "261"
        Raw = $RawA1
        Images = Join-Path $RawA1 "imagesTr"
        Trainer = "nnUNetTrainer_50epochs"
        Plans = "nnUNetPlansA1Lesion2GB"
        Evaluator = "nnUNetv2_evaluate_lesion_only"
    },
    @{
        Name = "A2_dual_head"
        Dataset = "260"
        Raw = $RawMT
        Images = Join-Path $RawMT "imagesTr"
        Trainer = "nnUNetTrainerMultiTask_50epochs"
        Plans = "nnUNetPlansMultiTask2GB"
        Evaluator = "nnUNetv2_evaluate_multitask"
    },
    @{
        Name = "A3_dual_decoder"
        Dataset = "260"
        Raw = $RawMT
        Images = Join-Path $RawMT "imagesTr"
        Trainer = "nnUNetTrainerMultiTask_50epochs"
        Plans = "nnUNetPlansMultiTaskDualDecoder"
        Evaluator = "nnUNetv2_evaluate_multitask"
    }
)

foreach ($run in $runs) {
    $PredDir = Join-Path $EvalRoot "$($run.Name)\$Checkpoint\predictions"
    New-Item -ItemType Directory -Force -Path $PredDir | Out-Null

    uv run nnUNetv2_predict `
        -i $run.Images `
        -o $PredDir `
        -d $run.Dataset `
        -c 2d `
        -tr $run.Trainer `
        -p $run.Plans `
        -f 0 `
        -chk $Checkpoint `
        -device $Device `
        -npp 1 `
        -nps 1 `
        --disable_progress_bar

    foreach ($split in @("val", "test")) {
        $OutFile = Join-Path $EvalRoot "$($run.Name)\$Checkpoint\metrics_$split.json"
        uv run $run.Evaluator `
            --raw_dataset $run.Raw `
            --predictions $PredDir `
            --split $split `
            --output $OutFile
    }
}

foreach ($split in @("val", "test")) {
    $StatsDir = Join-Path $EvalRoot "statistics\$Checkpoint\$split"
    New-Item -ItemType Directory -Force -Path $StatsDir | Out-Null

    foreach ($label in @("benign", "malignant")) {
        uv run nnUNetv2_statistical_tests_multitask `
            --test cochran_q `
            --model A1="$(Join-Path $EvalRoot "A1_lesion_only\$Checkpoint\metrics_$split.json")" `
            --model A2="$(Join-Path $EvalRoot "A2_dual_head\$Checkpoint\metrics_$split.json")" `
            --model A3="$(Join-Path $EvalRoot "A3_dual_decoder\$Checkpoint\metrics_$split.json")" `
            --task lesion `
            --label_name $label `
            --binary_field binary_all_gt_detected `
            --output "$(Join-Path $StatsDir "cochran_q_lesion_${label}.json")"

        uv run nnUNetv2_statistical_tests_multitask `
            --test wilcoxon `
            --model A1="$(Join-Path $EvalRoot "A1_lesion_only\$Checkpoint\metrics_$split.json")" `
            --model A2="$(Join-Path $EvalRoot "A2_dual_head\$Checkpoint\metrics_$split.json")" `
            --model_a A1 `
            --model_b A2 `
            --task lesion `
            --label_name $label `
            --metric dice `
            --output "$(Join-Path $StatsDir "wilcoxon_A1_A2_lesion_${label}_dice.json")"

        uv run nnUNetv2_statistical_tests_multitask `
            --test wilcoxon `
            --model A2="$(Join-Path $EvalRoot "A2_dual_head\$Checkpoint\metrics_$split.json")" `
            --model A3="$(Join-Path $EvalRoot "A3_dual_decoder\$Checkpoint\metrics_$split.json")" `
            --model_a A2 `
            --model_b A3 `
            --task lesion `
            --label_name $label `
            --metric dice `
            --output "$(Join-Path $StatsDir "wilcoxon_A2_A3_lesion_${label}_dice.json")"
    }

    foreach ($label in 1..12) {
        $labelName = "label_$label"
        uv run nnUNetv2_statistical_tests_multitask `
            --test wilcoxon `
            --model A2="$(Join-Path $EvalRoot "A2_dual_head\$Checkpoint\metrics_$split.json")" `
            --model A3="$(Join-Path $EvalRoot "A3_dual_decoder\$Checkpoint\metrics_$split.json")" `
            --model_a A2 `
            --model_b A3 `
            --task bone `
            --label_name $labelName `
            --metric dice `
            --output "$(Join-Path $StatsDir "wilcoxon_A2_A3_bone_${labelName}_dice.json")"
    }
}
