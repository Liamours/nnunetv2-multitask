$ErrorActionPreference = "Continue"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$Repo = Join-Path $Workspace "repo\nnunetv2_multitask"
$LogRoot = Join-Path $Repo "training_logs"
$RunId = Get-Date -Format "yyyyMMdd-HHmmss"
$StatusCsv = Join-Path $LogRoot "overnight_continue_A2_then_A3_$RunId.csv"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Set-Location $Repo

$env:nnUNet_raw = $NNUNetRaw
$env:nnUNet_preprocessed = $NNUNetPreprocessed
$env:nnUNet_results = $NNUNetResults
$env:nnUNet_compile = "0"

"timestamp,experiment,attempt,status,exit_code,log_file" | Set-Content -LiteralPath $StatusCsv -Encoding UTF8

function Write-Status {
    param(
        [string]$Experiment,
        [int]$Attempt,
        [string]$Status,
        [int]$ExitCode,
        [string]$LogFile
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp,$Experiment,$Attempt,$Status,$ExitCode,$LogFile" | Add-Content -LiteralPath $StatusCsv -Encoding UTF8
}

function Run-Training {
    param(
        [string]$Experiment,
        [int]$Attempt,
        [string]$PlansName
    )
    $logFile = Join-Path $LogRoot ("{0}_attempt{1}_{2}.log" -f $Experiment, $Attempt, $RunId)
    Write-Status -Experiment $Experiment -Attempt $Attempt -Status "started" -ExitCode 0 -LogFile $logFile
    $cmd = "uv run nnUNetv2_train 260 2d 0 -tr nnUNetTrainerMultiTask_100epochs -p $PlansName -device cuda --c"
    "Command: $cmd" | Set-Content -LiteralPath $logFile -Encoding UTF8
    "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content -LiteralPath $logFile -Encoding UTF8
    Invoke-Expression "$cmd *>> `"$logFile`""
    $exitCode = $LASTEXITCODE
    "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content -LiteralPath $logFile -Encoding UTF8
    "ExitCode: $exitCode" | Add-Content -LiteralPath $logFile -Encoding UTF8
    if ($exitCode -eq 0) {
        Write-Status -Experiment $Experiment -Attempt $Attempt -Status "completed" -ExitCode $exitCode -LogFile $logFile
    } else {
        Write-Status -Experiment $Experiment -Attempt $Attempt -Status "failed" -ExitCode $exitCode -LogFile $logFile
    }
    return @{ ExitCode = $exitCode; LogFile = $logFile }
}

$a2Plans = "nnUNetPlansMultiTask2GB"
$a3Plans = "nnUNetPlansMultiTaskDualDecoder"

$a2First = Run-Training -Experiment "A2_dual_head_100epochs" -Attempt 1 -PlansName $a2Plans
if ($a2First.ExitCode -ne 0) {
    Run-Training -Experiment "A2_dual_head_100epochs" -Attempt 2 -PlansName $a2Plans | Out-Null
}

"Status CSV: $StatusCsv"
