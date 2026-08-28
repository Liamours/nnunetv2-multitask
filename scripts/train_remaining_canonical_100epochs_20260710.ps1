param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$Repo = Join-Path $Workspace "repo\nnunetv2_multitask"
$LogDir = Join-Path $Repo "training_logs"
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$SummaryCsv = Join-Path $LogDir "remaining_canonical_100epochs_$RunStamp.csv"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location -LiteralPath $Repo

$env:nnUNet_raw = $NNUNetRaw
$env:nnUNet_preprocessed = $NNUNetPreprocessed
$env:nnUNet_results = $NNUNetResults
$env:nnUNet_n_proc_DA = "1"
$env:nnUNet_def_n_proc = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:TORCHINDUCTOR_COMPILE_THREADS = "1"

if (-not $Force) {
    $runningTraining = Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "nnUNetv2_train" -and $_.ProcessId -ne $PID }
    if ($runningTraining) {
        Write-Host "Existing nnUNet training process detected. Refusing to start a second runner."
        $runningTraining | Select-Object ProcessId, Name, CommandLine | Format-Table -Wrap
        exit 2
    }
}

& uv run nnUNetv2_train -h *> $null
if ($LASTEXITCODE -ne 0) {
    throw "uv run nnUNetv2_train is not callable from $Repo"
}

$runs = @(
    @{
        Name = "A1_cbam"
        DatasetId = "261"
        PlansName = "nnUNetPlansA1ControlledBatch4CBAM"
        OutputFolder = Join-Path $env:nnUNet_results "Dataset261_BS80KLesionOnly\nnUNetTrainerMultiTask_100epochs__nnUNetPlansA1ControlledBatch4CBAM__2d\fold_0"
        RequiredPaths = @(
            (Join-Path $env:nnUNet_preprocessed "Dataset261_BS80KLesionOnly\nnUNetPlansA1ControlledBatch4CBAM.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset261_BS80KLesionOnly\nnUNetPlansA1Lesion2GB_2d")
        )
    },
    @{
        Name = "A2_cbam"
        DatasetId = "260"
        PlansName = "nnUNetPlansA2ControlledBatch4CBAM"
        OutputFolder = Join-Path $env:nnUNet_results "Dataset260_BS80KLesionBoneMT\nnUNetTrainerMultiTask_100epochs__nnUNetPlansA2ControlledBatch4CBAM__2d\fold_0"
        RequiredPaths = @(
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansA2ControlledBatch4CBAM.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansMultiTask2GB_2d")
        )
    },
    @{
        Name = "A3_cbam"
        DatasetId = "260"
        PlansName = "nnUNetPlansA3ControlledBatch4CBAM"
        OutputFolder = Join-Path $env:nnUNet_results "Dataset260_BS80KLesionBoneMT\nnUNetTrainerMultiTask_100epochs__nnUNetPlansA3ControlledBatch4CBAM__2d\fold_0"
        RequiredPaths = @(
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansA3ControlledBatch4CBAM.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansMultiTask2GB_2d")
        )
    },
    @{
        Name = "A3_non_cbam"
        DatasetId = "260"
        PlansName = "nnUNetPlansA3ControlledBatch4"
        OutputFolder = Join-Path $env:nnUNet_results "Dataset260_BS80KLesionBoneMT\nnUNetTrainerMultiTask_100epochs__nnUNetPlansA3ControlledBatch4__2d\fold_0"
        RequiredPaths = @(
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansA3ControlledBatch4.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansMultiTask2GB_2d")
        )
    }
)

function Get-CheckpointEpoch {
    param([string]$CheckpointPath)
    if (-not (Test-Path -LiteralPath $CheckpointPath)) {
        return $null
    }
    $script = "import pickle, torch; ck=torch.load(r'$CheckpointPath', map_location='cpu'); print(ck.get('current_epoch', ''))"
    $value = (& .venv\Scripts\python.exe -W ignore -c $script 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($value)) {
        return $null
    }
    return [int]$value
}

function Get-CheckpointState {
    param([string]$OutputFolder)
    $final = Join-Path $OutputFolder "checkpoint_final.pth"
    $latest = Join-Path $OutputFolder "checkpoint_latest.pth"
    [pscustomobject]@{
        FinalExists = Test-Path -LiteralPath $final
        LatestExists = Test-Path -LiteralPath $latest
        FinalEpoch = Get-CheckpointEpoch $final
        LatestEpoch = Get-CheckpointEpoch $latest
    }
}

$summary = @()

foreach ($run in $runs) {
    $missingPath = $null
    foreach ($path in $run.RequiredPaths) {
        if (-not (Test-Path -LiteralPath $path)) {
            $missingPath = $path
            break
        }
    }
    if ($missingPath) {
        $summary += [pscustomobject]@{
            name = $run.Name
            status = "missing_required_path"
            exit_code = -1
            output_folder = $run.OutputFolder
            stdout_log = ""
            stderr_log = $missingPath
            finished_at = (Get-Date).ToString("o")
        }
        $summary | Export-Csv -NoTypeInformation -Path $SummaryCsv
        continue
    }

    $state = Get-CheckpointState $run.OutputFolder
    if ($state.FinalExists -and $state.FinalEpoch -ge 100) {
        $summary += [pscustomobject]@{
            name = $run.Name
            status = "skipped_completed"
            exit_code = 0
            output_folder = $run.OutputFolder
            stdout_log = ""
            stderr_log = ""
            finished_at = (Get-Date).ToString("o")
        }
        $summary | Export-Csv -NoTypeInformation -Path $SummaryCsv
        continue
    }

    $outLog = Join-Path $LogDir "$($run.Name)_$RunStamp.out.log"
    $errLog = Join-Path $LogDir "$($run.Name)_$RunStamp.err.log"
    $args = @(
        "run",
        "nnUNetv2_train",
        $run.DatasetId,
        "2d",
        "0",
        "-tr",
        "nnUNetTrainerMultiTask_100epochs",
        "-p",
        $run.PlansName,
        "-device",
        "cuda"
    )
    if ($state.LatestExists) {
        $args += "--c"
    }

    $proc = Start-Process -FilePath "uv" `
        -ArgumentList $args `
        -WorkingDirectory $Repo `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -WindowStyle Hidden `
        -Wait `
        -PassThru

    $status = if ($proc.ExitCode -eq 0) { "completed_or_stopped_cleanly" } else { "failed_continue_next" }
    $summary += [pscustomobject]@{
        name = $run.Name
        status = $status
        exit_code = $proc.ExitCode
        output_folder = $run.OutputFolder
        stdout_log = $outLog
        stderr_log = $errLog
        finished_at = (Get-Date).ToString("o")
    }
    $summary | Export-Csv -NoTypeInformation -Path $SummaryCsv
}

Write-Host "Summary written to $SummaryCsv"
