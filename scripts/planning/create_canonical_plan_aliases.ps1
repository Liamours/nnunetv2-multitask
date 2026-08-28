$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$dataRoot = $NNUNetPreprocessed
$aliases = @(
    @{ Dataset = "Dataset261_BS80KLesionOnly"; Source = "nnUNetPlansA1Lesion2GB"; Alias = "nnUNetPlans_A1_SingleTask_B4" },
    @{ Dataset = "Dataset261_BS80KLesionOnly"; Source = "nnUNetPlansA1ControlledBatch4CBAM"; Alias = "nnUNetPlans_A1_SingleTask_CBAM_B4" },
    @{ Dataset = "Dataset260_BS80KLesionBoneMT"; Source = "nnUNetPlansMultiTask2GB"; Alias = "nnUNetPlans_A2_DualHead_B4" },
    @{ Dataset = "Dataset260_BS80KLesionBoneMT"; Source = "nnUNetPlansA2ControlledBatch4CBAM"; Alias = "nnUNetPlans_A2_DualHead_CBAM_B4" },
    @{ Dataset = "Dataset260_BS80KLesionBoneMT"; Source = "nnUNetPlansA3ControlledBatch4"; Alias = "nnUNetPlans_A3_DualDecoder_B4" },
    @{ Dataset = "Dataset260_BS80KLesionBoneMT"; Source = "nnUNetPlansA3ControlledBatch4CBAM"; Alias = "nnUNetPlans_A3_DualDecoder_CBAM_B4" }
)

foreach ($entry in $aliases) {
    $folder = Join-Path $dataRoot $entry.Dataset
    $source = Join-Path $folder "$($entry.Source).json"
    $destination = Join-Path $folder "$($entry.Alias).json"
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Missing source plan: $source"
    }
    $plan = Get-Content -LiteralPath $source -Raw | ConvertFrom-Json
    $plan.plans_name = $entry.Alias
    $plan | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $destination -Encoding UTF8
    Write-Output "$($entry.Alias) -> $($entry.Source)"
}
