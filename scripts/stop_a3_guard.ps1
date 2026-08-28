$LogPath = Join-Path $PSScriptRoot "training_logs\stop_a3_guard.log"
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null

"$(Get-Date -Format o) guard started" | Add-Content -Path $LogPath

while ($true) {
    $a2 = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'nnUNetv2_train' -and
        $_.CommandLine -match 'nnUNetTrainerMultiTask_100epochs' -and
        $_.CommandLine -match 'nnUNetPlansMultiTask2GB'
    }

    $a3 = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'nnUNetv2_train|uv.exe' -and
        ($_.CommandLine -match 'DualDecoder' -or $_.CommandLine -match 'nnUNetPlansMultiTaskDualDecoder')
    }

    foreach ($proc in $a3) {
        "$(Get-Date -Format o) stopping A3 process pid=$($proc.ProcessId) command=$($proc.CommandLine)" | Add-Content -Path $LogPath
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }

    if (-not $a2) {
        "$(Get-Date -Format o) A2 no longer running; guard exiting" | Add-Content -Path $LogPath
        break
    }

    Start-Sleep -Seconds 10
}
