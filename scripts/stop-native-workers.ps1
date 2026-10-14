# Stop the native workers started by start-native-workers.ps1.
$root = Split-Path -Parent $PSScriptRoot
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'cardsim\.dist\.worker' -and $_.CommandLine -notmatch 'docker' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped pid $($_.ProcessId)" }
