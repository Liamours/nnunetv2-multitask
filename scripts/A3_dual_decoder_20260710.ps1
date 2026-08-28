$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$Repo = Join-Path $Workspace "repo\nnunetv2_multitask"
$LogRoot = Join-Path $Repo "training_logs"
$RunName = "A3_dual_decoder_20260710"
$LogFile = Join-Path $LogRoot "$RunName.log"
$StatusFile = Join-Path $LogRoot "$RunName.csv"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Set-Location -LiteralPath $Repo

$env:nnUNet_raw = $NNUNetRaw
$env:nnUNet_preprocessed = $NNUNetPreprocessed
$env:nnUNet_results = $NNUNetResults
$env:nnUNet_n_proc_DA = "1"
$env:nnUNet_def_n_proc = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

$row = [PSCustomObject]@{
    run = $RunName
    started_at = (Get-Date).ToString("o")
    command = "uv run nnUNetv2_train 260 2d 0 -tr nnUNetTrainerMultiTask_100epochs -p nnUNetPlansA3ControlledBatch4 -device cuda --c"
    exit_code = ""
    ended_at = ""
}
$row | Export-Csv -NoTypeInformation -Path $StatusFile

try {
    & uv run nnUNetv2_train 260 2d 0 `
        -tr nnUNetTrainerMultiTask_100epochs `
        -p nnUNetPlansA3ControlledBatch4 `
        -device cuda `
        --c *>&1 | Tee-Object -FilePath $LogFile
    $exitCode = $LASTEXITCODE
}
catch {
    $_ | Out-String | Tee-Object -FilePath $LogFile -Append
    $exitCode = 1
}

$row.exit_code = $exitCode
$row.ended_at = (Get-Date).ToString("o")
$row | Export-Csv -NoTypeInformation -Path $StatusFile
exit $exitCode
