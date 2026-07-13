param(
    [string]$Device = "cuda",
    [string]$PlanName = "nnUNetPlansA3ControlledBatch4",
    [string]$EvaluationName = "a3_dual_decoder_100epoch_best_latest",
    [string]$RunPrefix = "A3_dual_decoder"
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\lulay\Desktop\nnunetv2-multitask\repo\nnunetv2-multitask"
$DataRoot = "C:\Users\lulay\Desktop\nnunetv2-multitask\data"
$RawDataset = Join-Path $DataRoot "nnUNet_raw\Dataset260_BS80KLesionBoneMT"
$Images = Join-Path $RawDataset "imagesTr"
$EvalRoot = Join-Path $DataRoot ("evaluation\" + $EvaluationName)
$InputRoot = Join-Path $EvalRoot "inputs"
$SummaryFile = Join-Path $EvalRoot "summary.csv"

Set-Location -LiteralPath $Repo
$env:nnUNet_raw = Join-Path $DataRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $DataRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $DataRoot "nnUNet_results"
$env:nnUNet_n_proc_DA = "1"
$env:nnUNet_def_n_proc = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

New-Item -ItemType Directory -Force -Path $EvalRoot | Out-Null

$splitRows = Import-Csv (Join-Path $RawDataset "split_seed42.csv")
foreach ($split in @("val", "test")) {
    $splitInput = Join-Path $InputRoot $split
    if (Test-Path -LiteralPath $splitInput) {
        Remove-Item -LiteralPath $splitInput -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $splitInput | Out-Null
    $cases = $splitRows | Where-Object { $_.split -eq $split }
    foreach ($case in $cases) {
        foreach ($channel in @("0000", "0001")) {
            $src = Join-Path $Images "$($case.case_id)_$channel.png"
            $dst = Join-Path $splitInput "$($case.case_id)_$channel.png"
            New-Item -ItemType HardLink -Path $dst -Target $src | Out-Null
        }
    }
}

$runs = @(
    @{ Id = "$RunPrefix`_best"; Checkpoint = "checkpoint_best.pth" },
    @{ Id = "$RunPrefix`_latest"; Checkpoint = "checkpoint_final.pth" }
)

$summary = @()
foreach ($run in $runs) {
    foreach ($split in @("val", "test")) {
        $inputDir = Join-Path $InputRoot $split
        $predDir = Join-Path $EvalRoot "$($run.Id)\$split\predictions"
        $metricFile = Join-Path $EvalRoot "$($run.Id)\$split\metrics.json"
        if (Test-Path -LiteralPath $predDir) {
            Remove-Item -LiteralPath $predDir -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path $predDir | Out-Null

        uv run nnUNetv2_predict `
            -i $inputDir `
            -o $predDir `
            -d 260 `
            -c 2d `
            -tr nnUNetTrainerMultiTask_100epochs `
            -p $PlanName `
            -f 0 `
            -chk $run.Checkpoint `
            -device $Device `
            -npp 1 `
            -nps 1 `
            --disable_tta `
            --disable_progress_bar

        uv run nnUNetv2_evaluate_multitask `
            --raw_dataset $RawDataset `
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
            lesion_benign_dice = $lesion.by_label.benign.dice
            lesion_malignant_dice = $lesion.by_label.malignant.dice
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
