param(
    [switch]$Force,
    [int]$MaxAttempts = 2,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$Repo = Join-Path $Workspace "repo\nnunetv2_multitask"
$LogDir = Join-Path $Repo "training_logs"
$RunSummary = Join-Path $LogDir ("ablation_sequence_100epochs_{0}.csv" -f (Get-Date -Format "yyyyMMdd-HHmmss"))

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
        Write-Host "Existing nnUNet training process detected. Stop it first, or rerun with -Force."
        $runningTraining | Select-Object ProcessId, Name, CommandLine | Format-Table -Wrap
        exit 2
    }
}

$runs = @(
    @{
        Name = "A1_lesion_only_single_task"
        OutputFolder = Join-Path $env:nnUNet_results "Dataset261_BS80KLesionOnly\nnUNetTrainerMultiTask_100epochs__nnUNetPlansA1Lesion2GB__2d\fold_0"
        SplitCsv = Join-Path $env:nnUNet_raw "Dataset261_BS80KLesionOnly\split_seed42.csv"
        SplitJson = Join-Path $env:nnUNet_preprocessed "Dataset261_BS80KLesionOnly\splits_final.json"
        RequiredPaths = @(
            (Join-Path $env:nnUNet_raw "Dataset261_BS80KLesionOnly\dataset.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset261_BS80KLesionOnly\dataset.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset261_BS80KLesionOnly\nnUNetPlansA1Lesion2GB.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset261_BS80KLesionOnly\nnUNetPlansA1Lesion2GB_2d")
        )
        Args = @("nnUNetv2_train", "261", "2d", "0", "-tr", "nnUNetTrainerMultiTask_100epochs", "-p", "nnUNetPlansA1Lesion2GB", "-device", "cuda")
    },
    @{
        Name = "A2_multitask_dual_head"
        OutputFolder = Join-Path $env:nnUNet_results "Dataset260_BS80KLesionBoneMT\nnUNetTrainerMultiTask_100epochs__nnUNetPlansMultiTask2GB__2d\fold_0"
        SplitCsv = Join-Path $env:nnUNet_raw "Dataset260_BS80KLesionBoneMT\split_seed42.csv"
        SplitJson = Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\splits_final.json"
        RequiredPaths = @(
            (Join-Path $env:nnUNet_raw "Dataset260_BS80KLesionBoneMT\dataset.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\dataset.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansMultiTask2GB.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansMultiTask2GB_2d")
        )
        Args = @("nnUNetv2_train", "260", "2d", "0", "-tr", "nnUNetTrainerMultiTask_100epochs", "-p", "nnUNetPlansMultiTask2GB", "-device", "cuda")
    },
    @{
        Name = "A3_multitask_dual_decoder"
        OutputFolder = Join-Path $env:nnUNet_results "Dataset260_BS80KLesionBoneMT\nnUNetTrainerMultiTask_100epochs__nnUNetPlansMultiTaskDualDecoder__2d\fold_0"
        SplitCsv = Join-Path $env:nnUNet_raw "Dataset260_BS80KLesionBoneMT\split_seed42.csv"
        SplitJson = Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\splits_final.json"
        RequiredPaths = @(
            (Join-Path $env:nnUNet_raw "Dataset260_BS80KLesionBoneMT\dataset.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\dataset.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansMultiTaskDualDecoder.json"),
            (Join-Path $env:nnUNet_preprocessed "Dataset260_BS80KLesionBoneMT\nnUNetPlansMultiTaskDualDecoder_2d")
        )
        Args = @("nnUNetv2_train", "260", "2d", "0", "-tr", "nnUNetTrainerMultiTask_100epochs", "-p", "nnUNetPlansMultiTaskDualDecoder", "-device", "cuda")
    }
)

function Get-CheckpointEpoch {
    param([string]$CheckpointPath)
    if (-not (Test-Path -LiteralPath $CheckpointPath)) {
        return $null
    }
    $script = "import torch; ck=torch.load(r'$CheckpointPath', map_location='cpu'); print(ck.get('current_epoch', ''))"
    $value = (& uv run python -W ignore -c $script 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($value)) {
        return $null
    }
    return [int]$value
}

function Get-CheckpointStatus {
    param([string]$OutputFolder)
    $final = Join-Path $OutputFolder "checkpoint_final.pth"
    $latest = Join-Path $OutputFolder "checkpoint_latest.pth"
    $best = Join-Path $OutputFolder "checkpoint_best.pth"
    [pscustomobject]@{
        Final = Test-Path -LiteralPath $final
        Latest = Test-Path -LiteralPath $latest
        Best = Test-Path -LiteralPath $best
        FinalEpoch = Get-CheckpointEpoch $final
        LatestEpoch = Get-CheckpointEpoch $latest
        BestEpoch = Get-CheckpointEpoch $best
    }
}

function Promote-MostAdvancedCheckpointForContinue {
    param([string]$OutputFolder)
    $latest = Join-Path $OutputFolder "checkpoint_latest.pth"
    $best = Join-Path $OutputFolder "checkpoint_best.pth"
    $status = Get-CheckpointStatus $OutputFolder
    if ($status.Best -and ((-not $status.Latest) -or ($status.BestEpoch -gt $status.LatestEpoch))) {
        if (Test-Path -LiteralPath $latest) {
            $backup = Join-Path $OutputFolder ("checkpoint_latest.before_promote_{0}.pth" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
            Move-Item -LiteralPath $latest -Destination $backup
            Write-Host "Backed up stale latest checkpoint: $backup"
        }
        Copy-Item -LiteralPath $best -Destination $latest
        Write-Host "Promoted checkpoint_best epoch $($status.BestEpoch) to checkpoint_latest for continuation."
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

function Repair-FixedSplit {
    param($Run)
    if (-not (Test-Path -LiteralPath $Run.SplitCsv)) {
        throw "Missing split CSV for $($Run.Name): $($Run.SplitCsv)"
    }

    $rows = Import-Csv -LiteralPath $Run.SplitCsv
    $train = @($rows | Where-Object split -eq "train" | ForEach-Object case_id | Sort-Object)
    $val = @($rows | Where-Object split -eq "val" | ForEach-Object case_id | Sort-Object)
    $test = @($rows | Where-Object split -eq "test" | ForEach-Object case_id | Sort-Object)
    if ($train.Count -ne 2340 -or $val.Count -ne 292 -or $test.Count -ne 293) {
        throw "Unexpected split counts for $($Run.Name): train=$($train.Count), val=$($val.Count), test=$($test.Count)"
    }

    $split = @([ordered]@{ train = $train; val = $val })
    $json = ConvertTo-Json -InputObject $split -Depth 5
    $tmp = "$($Run.SplitJson).tmp"
    [System.IO.File]::WriteAllText($tmp, $json, [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $tmp -Destination $Run.SplitJson -Force

    $validationScript = @"
import json
p = r'''$($Run.SplitJson)'''
with open(p, encoding='utf-8') as f:
    parsed = json.load(f)
assert isinstance(parsed, list) and len(parsed) == 1
assert len(parsed[0]['train']) == 2340
assert len(parsed[0]['val']) == 292
print('ok')
"@
    $validation = (($validationScript | uv run python -) | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0 -or "$validation".Trim() -ne "ok") {
        throw "Split JSON validation failed for $($Run.Name): $($Run.SplitJson)"
    }
}

function Get-LogTail {
    param([string]$Path, [int]$Lines = 80)
    if (-not (Test-Path -LiteralPath $Path)) {
        return ""
    }
    return ((Get-Content -LiteralPath $Path -Tail $Lines -ErrorAction SilentlyContinue) -join [Environment]::NewLine)
}

$summaryRows = @()

foreach ($run in $runs) {
    $status = Get-CheckpointStatus $run.OutputFolder
    if ($status.Final -and $status.FinalEpoch -ge 100) {
        Write-Host "SKIP completed: $($run.Name), final epoch $($status.FinalEpoch)"
        $summaryRows += [pscustomobject]@{Name=$run.Name; Status="skipped_completed"; Attempts=0; ExitCode=0; Stdout=""; Stderr=""; OutputFolder=$run.OutputFolder; FinishedAt=Get-Date}
        continue
    }

    try {
        Assert-RunPreflight $run
        Repair-FixedSplit $run
    } catch {
        Write-Host "PREFLIGHT FAILED: $($run.Name)"
        Write-Host $_.Exception.Message
        $summaryRows += [pscustomobject]@{Name=$run.Name; Status="preflight_failed"; Attempts=0; ExitCode=-1; Stdout=""; Stderr=$_.Exception.Message; OutputFolder=$run.OutputFolder; FinishedAt=Get-Date}
        $summaryRows | Export-Csv -LiteralPath $RunSummary -NoTypeInformation
        continue
    }

    if ($PreflightOnly) {
        Write-Host "PREFLIGHT OK: $($run.Name). Final=$($status.FinalEpoch), Latest=$($status.LatestEpoch), Best=$($status.BestEpoch)"
        $summaryRows += [pscustomobject]@{Name=$run.Name; Status="preflight_ok"; Attempts=0; ExitCode=0; Stdout=""; Stderr=""; OutputFolder=$run.OutputFolder; FinishedAt=Get-Date}
        continue
    }

    Promote-MostAdvancedCheckpointForContinue $run.OutputFolder
    $status = Get-CheckpointStatus $run.OutputFolder

    $attempt = 0
    $completed = $false
    $lastExitCode = $null
    $lastOutLog = ""
    $lastErrLog = ""

    while (-not $completed -and $attempt -lt $MaxAttempts) {
        $attempt += 1
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $outLog = Join-Path $LogDir "$($run.Name)_100epochs_attempt${attempt}_$stamp.out.log"
        $errLog = Join-Path $LogDir "$($run.Name)_100epochs_attempt${attempt}_$stamp.err.log"
        $lastOutLog = $outLog
        $lastErrLog = $errLog

        $argsForAttempt = @($run.Args)
        $checkpointStatus = Get-CheckpointStatus $run.OutputFolder
        if ($checkpointStatus.Latest) {
            $argsForAttempt += "--c"
        }

        Write-Host "Starting $($run.Name), attempt $attempt/$MaxAttempts"
        Write-Host ("command: uv run {0}" -f ($argsForAttempt -join " "))
        Write-Host "stdout: $outLog"
        Write-Host "stderr: $errLog"

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
        if ($lastExitCode -eq 0 -and $checkpointStatus.Final -and $checkpointStatus.FinalEpoch -ge 100) {
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
}

$summaryRows | Export-Csv -LiteralPath $RunSummary -NoTypeInformation
Write-Host "Sequential 100-epoch ablation runner finished. Summary: $RunSummary"
