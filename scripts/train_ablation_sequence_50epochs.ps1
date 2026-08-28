param(
    [switch]$Force,
    [int]$MaxAttempts = 2,
    [switch]$RestartPartial,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$Repo = Join-Path $Workspace "repo\nnunetv2_multitask"
$LogDir = Join-Path $Repo "training_logs"
$RunSummary = Join-Path $LogDir ("ablation_sequence_50epochs_{0}.csv" -f (Get-Date -Format "yyyyMMdd-HHmmss"))

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location -LiteralPath $Repo

$env:nnUNet_raw = $NNUNetRaw
$env:nnUNet_preprocessed = $NNUNetPreprocessed
$env:nnUNet_results = $NNUNetResults

# Low worker counts keep this laptop stable and avoid RAM pressure.
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
        Write-Host "Existing nnUNet training process detected. Stop it first, or rerun with -Force."
        $runningTraining | Select-Object ProcessId, Name, CommandLine | Format-Table -Wrap
        exit 2
    }
}

$runs = @(
    @{
        Name = "A1_lesion_only_single_task"
        Dataset = "Dataset261_BS80KLesionOnly"
        OutputFolder = Join-Path $env:nnUNet_results "Dataset261_BS80KLesionOnly\nnUNetTrainerMultiTask_50epochs__nnUNetPlansA1Lesion2GB__2d\fold_0"
        RequiredPaths = @(
            (Join-Path $env:nnUNet_raw "Dataset261_BS80KLesionOnly\dataset.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset261_BS80KLesionOnly\dataset.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset261_BS80KLesionOnly\nnUNetPlansA1Lesion2GB.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset261_BS80KLesionOnly\nnUNetPlansA1Lesion2GB_2d")
        )
        Args = @("nnUNetv2_train", "261", "2d", "0", "-tr", "nnUNetTrainerMultiTask_50epochs", "-p", "nnUNetPlansA1Lesion2GB", "-device", "cuda")
    },
    @{
        Name = "A2_multitask_dual_head"
        Dataset = "Dataset260_BS80KLesionBoneMT"
        OutputFolder = Join-Path $env:nnUNet_results "Dataset260_BS80KLesionBoneMT\nnUNetTrainerMultiTask_50epochs__nnUNetPlansMultiTask2GB__2d\fold_0"
        RequiredPaths = @(
            (Join-Path $env:nnUNet_raw "Dataset260_BS80KLesionBoneMT\dataset.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\dataset.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansMultiTask2GB.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansMultiTask2GB_2d")
        )
        Args = @("nnUNetv2_train", "260", "2d", "0", "-tr", "nnUNetTrainerMultiTask_50epochs", "-p", "nnUNetPlansMultiTask2GB", "-device", "cuda")
    },
    @{
        Name = "A3_multitask_dual_decoder"
        Dataset = "Dataset260_BS80KLesionBoneMT"
        OutputFolder = Join-Path $env:nnUNet_results "Dataset260_BS80KLesionBoneMT\nnUNetTrainerMultiTask_50epochs__nnUNetPlansMultiTaskDualDecoder__2d\fold_0"
        RequiredPaths = @(
            (Join-Path $env:nnUNet_raw "Dataset260_BS80KLesionBoneMT\dataset.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\dataset.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansMultiTaskDualDecoder.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansMultiTaskDualDecoder_2d")
        )
        Args = @("nnUNetv2_train", "260", "2d", "0", "-tr", "nnUNetTrainerMultiTask_50epochs", "-p", "nnUNetPlansMultiTaskDualDecoder", "-device", "cuda")
    }
)

$summaryRows = @()

function Get-CheckpointStatus {
    param([string]$OutputFolder)
    [pscustomobject]@{
        Final = Test-Path -LiteralPath (Join-Path $OutputFolder "checkpoint_final.pth")
        Latest = Test-Path -LiteralPath (Join-Path $OutputFolder "checkpoint_latest.pth")
        Best = Test-Path -LiteralPath (Join-Path $OutputFolder "checkpoint_best.pth")
    }
}

function Assert-RunPreflight {
    param($Run)
    foreach ($path in $Run.RequiredPaths) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing required path for $($Run.Name): $path"
        }
    }

    & uv run nnUNetv2_train -h *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "uv run nnUNetv2_train is not callable from $Repo"
    }
}

function Get-LogTail {
    param([string]$Path, [int]$Lines = 80)
    if (-not (Test-Path -LiteralPath $Path)) {
        return ""
    }
    return ((Get-Content -LiteralPath $Path -Tail $Lines -ErrorAction SilentlyContinue) -join [Environment]::NewLine)
}

foreach ($run in $runs) {
    $status = Get-CheckpointStatus $run.OutputFolder
    if ($status.Final) {
        Write-Host "SKIP completed: $($run.Name)"
        $summaryRows += [pscustomobject]@{
            Name = $run.Name
            Status = "skipped_completed"
            Attempts = 0
            ExitCode = 0
            Stdout = ""
            Stderr = ""
            OutputFolder = $run.OutputFolder
            FinishedAt = Get-Date
        }
        continue
    }

    try {
        Assert-RunPreflight $run
    } catch {
        Write-Host "PREFLIGHT FAILED: $($run.Name)"
        Write-Host $_.Exception.Message
        $summaryRows += [pscustomobject]@{
            Name = $run.Name
            Status = "preflight_failed"
            Attempts = 0
            ExitCode = -1
            Stdout = ""
            Stderr = $_.Exception.Message
            OutputFolder = $run.OutputFolder
            FinishedAt = Get-Date
        }
        continue
    }

    if ($PreflightOnly) {
        Write-Host "PREFLIGHT OK: $($run.Name)"
        $summaryRows += [pscustomobject]@{
            Name = $run.Name
            Status = "preflight_ok"
            Attempts = 0
            ExitCode = 0
            Stdout = ""
            Stderr = ""
            OutputFolder = $run.OutputFolder
            FinishedAt = Get-Date
        }
        continue
    }

    $attempt = 0
    $completed = $false
    $lastExitCode = $null
    $lastOutLog = ""
    $lastErrLog = ""

    while (-not $completed -and $attempt -lt $MaxAttempts) {
        $attempt += 1
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $outLog = Join-Path $LogDir "$($run.Name)_50epochs_attempt${attempt}_$stamp.out.log"
        $errLog = Join-Path $LogDir "$($run.Name)_50epochs_attempt${attempt}_$stamp.err.log"
        $lastOutLog = $outLog
        $lastErrLog = $errLog

        $argsForAttempt = @($run.Args)
        $checkpointStatus = Get-CheckpointStatus $run.OutputFolder
        if ($checkpointStatus.Latest -and -not $RestartPartial) {
            $argsForAttempt += "--c"
        }

        Write-Host "Starting $($run.Name), attempt $attempt/$MaxAttempts"
        Write-Host "stdout: $outLog"
        Write-Host "stderr: $errLog"
        Write-Host ("command: uv run {0}" -f ($argsForAttempt -join " "))

        $process = Start-Process -FilePath "uv" `
            -ArgumentList (@("run") + $argsForAttempt) `
            -WorkingDirectory $Repo `
            -RedirectStandardOutput $outLog `
            -RedirectStandardError $errLog `
            -NoNewWindow `
            -Wait `
            -PassThru
        $lastExitCode = $process.ExitCode

        $checkpointStatus = Get-CheckpointStatus $run.OutputFolder
        if ($lastExitCode -eq 0 -and $checkpointStatus.Final) {
            Write-Host "Finished $($run.Name)"
            $completed = $true
            break
        }

        Write-Host "FAILED: $($run.Name), attempt $attempt/$MaxAttempts, exit=$lastExitCode"
        Write-Host "stderr tail:"
        Write-Host (Get-LogTail $errLog 80)
        Write-Host "stdout tail:"
        Write-Host (Get-LogTail $outLog 80)
        Write-Host "Continuing retry/next run policy."
    }

    $summaryRows += [pscustomobject]@{
        Name = $run.Name
        Status = if ($completed) { "completed" } else { "failed_skipped" }
        Attempts = $attempt
        ExitCode = $lastExitCode
        Stdout = $lastOutLog
        Stderr = $lastErrLog
        OutputFolder = $run.OutputFolder
        FinishedAt = Get-Date
    }

    $summaryRows | Export-Csv -LiteralPath $RunSummary -NoTypeInformation

    if (-not $completed) {
        Write-Host "SKIPPING to next experiment after failed attempts: $($run.Name)"
    }
}

$summaryRows | Export-Csv -LiteralPath $RunSummary -NoTypeInformation
Write-Host "Sequential ablation runner finished. Summary: $RunSummary"
