$ErrorActionPreference = 'Stop'

$repoRoot = 'C:\Users\lulay\Desktop\nnunetv2-multitask\repo\nnunetv2-multitask'
$dataRoot = 'C:\Users\lulay\Desktop\nnunetv2-multitask\data'
$logDir = Join-Path $repoRoot 'training_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:nnUNet_raw = Join-Path $dataRoot 'nnUNet_raw'
$env:nnUNet_preprocessed = Join-Path $dataRoot 'nnUNet_preprocessed'
$env:nnUNet_results = Join-Path $dataRoot 'nnUNet_results'

$runs = @(
    @{
        Name = 'A1_smoke_20260711'
        Dataset = '261'
        Plan = 'nnUNetPlansA1Lesion2GB'
    },
    @{
        Name = 'A2_smoke_20260711'
        Dataset = '260'
        Plan = 'nnUNetPlansMultiTask2GB'
    },
    @{
        Name = 'A3_smoke_20260711'
        Dataset = '260'
        Plan = 'nnUNetPlansA3ControlledBatch4'
    }
)

Set-Location $repoRoot

foreach ($run in $runs) {
    $csvPath = Join-Path $logDir ($run.Name + '.csv')
    "run_name,started_at,ended_at,exit_code,command" | Set-Content -Encoding UTF8 $csvPath
    $command = "uv run nnUNetv2_train $($run.Dataset) 2d 0 -tr nnUNetTrainerMultiTask_2epochs -p $($run.Plan) -device cuda"
    $started = (Get-Date).ToString('o')
    Invoke-Expression $command
    $exitCode = $LASTEXITCODE
    $ended = (Get-Date).ToString('o')
    $escapedCommand = '"' + $command.Replace('"', '""') + '"'
    "$($run.Name),$started,$ended,$exitCode,$escapedCommand" | Add-Content -Encoding UTF8 $csvPath
    if ($exitCode -ne 0) {
        throw "Smoke run failed: $($run.Name)"
    }
}
