# Start a GPU worker and a CPU worker natively on Windows (no Docker VM: about 50% more
# CPU throughput here than inside Docker Desktop). They run hidden in the background and
# log to logs\. Stop them with scripts\stop-native-workers.ps1.
#
#   .\scripts\start-native-workers.ps1 [-Url http://localhost:8700] [-CpuThreads 12] [-AutoUpdate]
# -AutoUpdate: on a study submitted with different code, fetch that code from GitHub and
# restart on it (cardsim/dist/update.py).
param(
    [string]$Url = "http://localhost:8700",
    [int]$CpuThreads = 12,
    [switch]$AutoUpdate
)
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logs | Out-Null

# the coordinator token lives in docker\.env (git-ignored)
$env:COORDINATOR_TOKEN = (Get-Content (Join-Path $root "docker\.env") |
    Where-Object { $_ -match '^COORDINATOR_TOKEN=' } | Select-Object -First 1) -replace '^COORDINATOR_TOKEN=', ''
$env:COORDINATOR_URL = $Url
$env:PYTHONUNBUFFERED = "1"

$started = @()
foreach ($w in @(@{name = "$env:COMPUTERNAME-gpu"; device = "auto"; threads = $null},
                 @{name = "$env:COMPUTERNAME-cpu"; device = "cpu"; threads = $CpuThreads})) {
    if ($w.threads) { $env:NUMBA_NUM_THREADS = "$($w.threads)" } else { Remove-Item Env:NUMBA_NUM_THREADS -ErrorAction SilentlyContinue }
    $p = Start-Process -FilePath $python -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -ArgumentList (@("-m", "cardsim.dist.worker", "--device", $w.device, "--name", $w.name) + $(if ($AutoUpdate) { @("--auto-update") } else { @() })) `
        -RedirectStandardOutput (Join-Path $logs "$($w.name).log") -RedirectStandardError (Join-Path $logs "$($w.name).err.log")
    $started += "$($w.name) (pid $($p.Id))"
}
Set-Content (Join-Path $logs "native-workers.pids") ($started -join "`n")
"started: " + ($started -join ", ")
