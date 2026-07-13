param(
    [string]$Device = "cuda",
    [string]$PlanName = "nnUNetPlansA1Lesion2GB",
    [string]$EvaluationName = "a1_lesion_only_100epoch_best_latest",
    [string]$RunPrefix = "A1_lesion_only"
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\lulay\Desktop\nnunetv2-multitask\repo\nnunetv2-multitask"
$dataRoot = "C:\Users\lulay\Desktop\nnunetv2-multitask\data"
$rawDataset = Join-Path $dataRoot "nnUNet_raw\Dataset261_BS80KLesionOnly"
$images = Join-Path $rawDataset "imagesTr"
$evalRoot = Join-Path $dataRoot ("evaluation\" + $EvaluationName)
$inputRoot = Join-Path $evalRoot "inputs"
$summaryFile = Join-Path $evalRoot "summary.csv"

Set-Location -LiteralPath $repoRoot
$env:nnUNet_raw = Join-Path $dataRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $dataRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $dataRoot "nnUNet_results"
$env:nnUNet_n_proc_DA = "1"
$env:nnUNet_def_n_proc = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

New-Item -ItemType Directory -Force -Path $evalRoot | Out-Null
$splitRows = Import-Csv (Join-Path $rawDataset "split_seed42.csv")
foreach ($split in @("val", "test")) {
    $splitInput = Join-Path $inputRoot $split
    if (Test-Path -LiteralPath $splitInput) {
        Remove-Item -LiteralPath $splitInput -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $splitInput | Out-Null
    foreach ($case in ($splitRows | Where-Object { $_.split -eq $split })) {
        foreach ($channel in @("0000", "0001")) {
            $src = Join-Path $images "$($case.case_id)_$channel.png"
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
        $inputDir = Join-Path $inputRoot $split
        $predDir = Join-Path $evalRoot "$($run.Id)\$split\predictions"
        $metricFile = Join-Path $evalRoot "$($run.Id)\$split\metrics.json"
        if (Test-Path -LiteralPath $predDir) {
            Remove-Item -LiteralPath $predDir -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path $predDir | Out-Null

        uv run nnUNetv2_predict `
            -i $inputDir `
            -o $predDir `
            -d 261 `
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
            --raw_dataset $rawDataset `
            --predictions $predDir `
            --split $split `
            --output $metricFile

        $metrics = Get-Content -Raw -LiteralPath $metricFile | ConvertFrom-Json
        $lesion = $metrics.tasks.lesion
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
            metrics_file = $metricFile
        }
    }
}

$summary | Export-Csv -NoTypeInformation -Path $summaryFile
$summary | Format-Table -AutoSize
