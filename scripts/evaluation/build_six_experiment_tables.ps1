$ErrorActionPreference = "Stop"

$__d = $PSScriptRoot
while (-not (Test-Path (Join-Path $__d 'dataset_paths.ps1'))) { $__d = Split-Path $__d -Parent }
. (Join-Path $__d 'dataset_paths.ps1')

$evaluationRoot = Join-Path $Workspace "analyses\evaluations"
$runs = @(
    @{ Model = "A1 lesion-only"; Summary = "a1_lesion_only_100epoch_best_latest"; Id = "A1_lesion_only_best" },
    @{ Model = "A1 lesion-only + CBAM"; Summary = "a1_lesion_only_cbam_100epoch_best_latest"; Id = "A1_lesion_only_cbam_best" },
    @{ Model = "A2 dual-head"; Summary = "a2_dual_head_100epoch_best_latest"; Id = "A2_dual_head_best" },
    @{ Model = "A2 dual-head + CBAM"; Summary = "a2_dual_head_cbam_100epoch_best_latest"; Id = "A2_dual_head_cbam_best" },
    @{ Model = "A3 dual-decoder"; Summary = "a3_dual_decoder_100epoch_best_latest"; Id = "A3_dual_decoder_best" },
    @{ Model = "A3 dual-decoder + CBAM"; Summary = "a3_dual_decoder_cbam_100epoch_best_latest"; Id = "A3_dual_decoder_cbam_best" }
)

$allRows = foreach ($run in $runs) {
    $summary = Import-Csv (Join-Path $evaluationRoot "$($run.Summary)\summary.csv")
    foreach ($split in @("val", "test")) {
        $row = $summary | Where-Object { $_.model -eq $run.Id -and $_.split -eq $split }
        if (@($row).Count -ne 1) {
            throw "Expected one best-checkpoint row for $($run.Model) on $split."
        }
        [PSCustomObject]@{
            model = $run.Model
            split = $split
            lesion_dice = [double]$row.lesion_dice
            lesion_benign_dice = [double]$row.lesion_benign_dice
            lesion_malignant_dice = [double]$row.lesion_malignant_dice
            lesion_sensitivity = [double]$row.lesion_sensitivity
            lesion_specificity = [double]$row.lesion_specificity
            lesionwise_f1 = [double]$row.lesionwise_f1
            lesion_froc_llf = [double]$row.lesion_froc_llf
            lesion_froc_fp_per_case = [double]$row.lesion_froc_fp_per_case
            bone_dice = if ($row.PSObject.Properties.Name -contains "bone_dice") { [double]$row.bone_dice } else { $null }
            bone_sensitivity = if ($row.PSObject.Properties.Name -contains "bone_sensitivity") { [double]$row.bone_sensitivity } else { $null }
            bone_specificity = if ($row.PSObject.Properties.Name -contains "bone_specificity") { [double]$row.bone_specificity } else { $null }
        }
    }
}

$headers = @("Model", "Lesion Dice", "Benign Dice", "Malignant Dice", "Lesion Sens.", "Lesion Spec.", "Lesion-wise F1", "FROC LLF", "FROC FP/case", "Bone Dice", "Bone Sens.", "Bone Spec.")
function Format-Value($value) { if ($null -eq $value) { return "N/A" }; return ([double]$value).ToString("0.0000", [cultureinfo]::InvariantCulture) }
function New-MarkdownTable($rows) {
    $lines = @("| " + ($headers -join " | ") + " |", "|" + (($headers | ForEach-Object { "---" }) -join "|") + "|")
    foreach ($row in $rows) {
        $values = @(
            $row.model,
            (Format-Value $row.lesion_dice),
            (Format-Value $row.lesion_benign_dice),
            (Format-Value $row.lesion_malignant_dice),
            (Format-Value $row.lesion_sensitivity),
            (Format-Value $row.lesion_specificity),
            (Format-Value $row.lesionwise_f1),
            (Format-Value $row.lesion_froc_llf),
            (Format-Value $row.lesion_froc_fp_per_case),
            (Format-Value $row.bone_dice),
            (Format-Value $row.bone_sensitivity),
            (Format-Value $row.bone_specificity)
        )
        $lines += "| " + ($values -join " | ") + " |"
    }
    return $lines -join [Environment]::NewLine
}

$valRows = @($allRows | Where-Object { $_.split -eq "val" })
$testRows = @($allRows | Where-Object { $_.split -eq "test" })
$valRows | Export-Csv -NoTypeInformation -Path (Join-Path $evaluationRoot "six_experiment_best_val.csv")
$testRows | Export-Csv -NoTypeInformation -Path (Join-Path $evaluationRoot "six_experiment_best_test.csv")
$markdown = "# Six-experiment best-checkpoint evaluation`n`n## Validation`n`n$(New-MarkdownTable $valRows)`n`n## Test`n`n$(New-MarkdownTable $testRows)`n"
$markdown | Set-Content -Encoding UTF8 -Path (Join-Path $evaluationRoot "six_experiment_best_val_test.md")
