$ErrorActionPreference = 'Stop'

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$repoRoot = Join-Path $Workspace "repo\nnunetv2_multitask"
$logDir = Join-Path $repoRoot 'training_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:nnUNet_raw = $NNUNetRaw
$env:nnUNet_preprocessed = $NNUNetPreprocessed
$env:nnUNet_results = $NNUNetResults
# Twelve augmentation workers exhaust host memory while reading Blosc2 arrays on this 16 GiB machine.
# This limits loader concurrency only; it does not alter the saved plans or augmentation transforms.
$env:nnUNet_n_proc_DA = '2'

$runs = @(
    @{
        Name = 'A2_cbam_full_20260713'
        Dataset = '260'
        Plan = 'nnUNetPlansA2ControlledBatch4CBAM'
        ResultDir = Join-Path $NNUNetResults 'Dataset260_BS80KLesionBoneMT\nnUNetTrainerMultiTask_100epochs__nnUNetPlansA2ControlledBatch4CBAM__2d\fold_0'
    },
    @{
        Name = 'A3_cbam_full_20260713'
        Dataset = '260'
        Plan = 'nnUNetPlansA3ControlledBatch4CBAM'
        ResultDir = Join-Path $NNUNetResults 'Dataset260_BS80KLesionBoneMT\nnUNetTrainerMultiTask_100epochs__nnUNetPlansA3ControlledBatch4CBAM__2d\fold_0'
    }
)

function Get-CheckpointEpoch([string]$path) {
    $script = @'
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
print(ckpt.get("current_epoch", -1))
'@
    $value = $script | uv run python - $path
    return ($value | Out-String).Trim()
}

Set-Location $repoRoot

foreach ($run in $runs) {
    $csvPath = Join-Path $logDir ($run.Name + '.csv')
    "run_name,started_at,ended_at,exit_code,status,command" | Set-Content -Encoding UTF8 $csvPath

    $finalCheckpoint = Join-Path $run.ResultDir 'checkpoint_final.pth'
    $latestCheckpoint = Join-Path $run.ResultDir 'checkpoint_latest.pth'
    $baseCommand = "uv run nnUNetv2_train $($run.Dataset) 2d 0 -tr nnUNetTrainerMultiTask_100epochs -p $($run.Plan) -device cuda"
    $command = $baseCommand
    $status = 'ran_fresh'

    if (Test-Path $finalCheckpoint) {
        $epoch = Get-CheckpointEpoch $finalCheckpoint
        if ($epoch -eq '100') {
            $started = (Get-Date).ToString('o')
            $ended = (Get-Date).ToString('o')
            $escapedCommand = '"' + ($baseCommand + ' --c').Replace('"', '""') + '"'
            "$($run.Name),$started,$ended,0,skipped_completed,$escapedCommand" | Add-Content -Encoding UTF8 $csvPath
            continue
        }
    }

    if (Test-Path $latestCheckpoint) {
        $command = $baseCommand + ' --c'
        $status = 'continued'
    }

    $started = (Get-Date).ToString('o')
    Invoke-Expression $command
    $exitCode = $LASTEXITCODE
    $ended = (Get-Date).ToString('o')
    $escapedCommand = '"' + $command.Replace('"', '""') + '"'
    "$($run.Name),$started,$ended,$exitCode,$status,$escapedCommand" | Add-Content -Encoding UTF8 $csvPath
    if ($exitCode -ne 0) {
        throw "CBAM full run failed: $($run.Name)"
    }
}
