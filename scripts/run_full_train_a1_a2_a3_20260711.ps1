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

$runs = @(
    @{
        Name = 'A1_full_20260711'
        Dataset = '261'
        Plan = 'nnUNetPlansA1Lesion2GB'
        ResultDir = Join-Path $NNUNetResults 'Dataset261_BS80KLesionOnly\nnUNetTrainerMultiTask_100epochs__nnUNetPlansA1Lesion2GB__2d\fold_0'
    },
    @{
        Name = 'A2_full_20260711'
        Dataset = '260'
        Plan = 'nnUNetPlansMultiTask2GB'
        ResultDir = Join-Path $NNUNetResults 'Dataset260_BS80KLesionBoneMT\nnUNetTrainerMultiTask_100epochs__nnUNetPlansMultiTask2GB__2d\fold_0'
    },
    @{
        Name = 'A3_full_20260711'
        Dataset = '260'
        Plan = 'nnUNetPlansA3ControlledBatch4'
        ResultDir = Join-Path $NNUNetResults 'Dataset260_BS80KLesionBoneMT\nnUNetTrainerMultiTask_100epochs__nnUNetPlansA3ControlledBatch4__2d\fold_0'
    }
)

Set-Location $repoRoot

foreach ($run in $runs) {
    $csvPath = Join-Path $logDir ($run.Name + '.csv')
    "run_name,started_at,ended_at,exit_code,status,command" | Set-Content -Encoding UTF8 $csvPath
    $finalCheckpoint = Join-Path $run.ResultDir 'checkpoint_final.pth'
    $command = "uv run nnUNetv2_train $($run.Dataset) 2d 0 -tr nnUNetTrainerMultiTask_100epochs -p $($run.Plan) -device cuda --c"
    $started = (Get-Date).ToString('o')

    if (Test-Path $finalCheckpoint) {
        $epoch = @'
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
print(ckpt.get("current_epoch", -1))
'@ | uv run python - $finalCheckpoint
        if (($epoch | Out-String).Trim() -eq '100') {
            $ended = (Get-Date).ToString('o')
            $escapedCommand = '"' + $command.Replace('"', '""') + '"'
            "$($run.Name),$started,$ended,0,skipped_completed,$escapedCommand" | Add-Content -Encoding UTF8 $csvPath
            continue
        }
    }

    Invoke-Expression $command
    $exitCode = $LASTEXITCODE
    $ended = (Get-Date).ToString('o')
    $escapedCommand = '"' + $command.Replace('"', '""') + '"'
    "$($run.Name),$started,$ended,$exitCode,ran,$escapedCommand" | Add-Content -Encoding UTF8 $csvPath
    if ($exitCode -ne 0) {
        throw "Full run failed: $($run.Name)"
    }
}
