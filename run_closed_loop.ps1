param([switch]$Status)
$ErrorActionPreference='Stop'
if($Status) {
    foreach($window in @(5,10,30)) {
        foreach($condition in @('standard','push')) {
            $path=Join-Path $PSScriptRoot "results/closed_loop/test/$condition-h$window/summary.json"
            if(Test-Path -LiteralPath $path) {
                $s=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
                [ordered]@{window=$window;condition=$condition;successes=$s.successes;episodes=$s.episodes;latency_p95_ms=$s.query_latency_ms_p95} | ConvertTo-Json -Compress
            } else {
                $partial=Join-Path (Split-Path $path -Parent) 'episodes.json'
                if(Test-Path -LiteralPath $partial) {
                    $rows=@(Get-Content -LiteralPath $partial -Raw | ConvertFrom-Json)
                    [ordered]@{window=$window;condition=$condition;status='running-or-interrupted';completed=$rows.Count;planned=100} | ConvertTo-Json -Compress
                }
            }
        }
    }
    exit
}
$entry=Join-Path $PSScriptRoot 'scripts/python.sh'
$linuxPath='/mnt/'+$entry.Substring(0,1).ToLower()+$entry.Substring(2).Replace('\','/')
$environment=Join-Path $PSScriptRoot 'results/closed_loop/environment.json'
$original=Join-Path $PSScriptRoot 'results/closed_loop/original_environment.json'
if((Test-Path -LiteralPath $environment) -and -not (Test-Path -LiteralPath $original)) {
    Copy-Item -LiteralPath $environment -Destination $original
}
& wsl.exe -d Ubuntu-24.04 -- bash $linuxPath scripts/closed_loop_experiment.py
if($LASTEXITCODE -ne 0) {throw 'Closed-loop experiment failed; inspect results/closed_loop.'}
if(-not (Test-Path -LiteralPath $original)) { Copy-Item -LiteralPath $environment -Destination $original }
& wsl.exe -d Ubuntu-24.04 -- bash $linuxPath scripts/closed_loop_report.py
if($LASTEXITCODE -ne 0) {throw 'Closed-loop report audit failed.'}
& (Join-Path $PSScriptRoot 'scripts/audit_closed_loop.ps1')
