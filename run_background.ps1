param([switch]$Status)
$ErrorActionPreference='Stop'
if($Status) {
    foreach($name in @('background_control','background_aug')) {
        $row=[ordered]@{variant=$name}
        $training=Join-Path $PSScriptRoot ("checkpoints/"+$name+"/training_summary.json")
        if(Test-Path -LiteralPath $training) {$row.steps=(Get-Content -LiteralPath $training -Raw | ConvertFrom-Json).steps}
        foreach($condition in @('standard','background_test')) {
            $summary=Join-Path $PSScriptRoot ("results/"+$name+"/background_test-"+$condition+"/summary.json")
            if(Test-Path -LiteralPath $summary) {$s=Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json;$row[$condition]="$($s.successes)/$($s.episodes)"}
        }
        $row | ConvertTo-Json -Compress
    }
    exit
}
$entry=Join-Path $PSScriptRoot 'scripts/python.sh'
$linuxPath='/mnt/'+$entry.Substring(0,1).ToLower()+$entry.Substring(2).Replace('\','/')
& wsl.exe -d Ubuntu-24.04 -- bash $linuxPath scripts/build_background_masks.py
if($LASTEXITCODE -ne 0) {throw 'Background mask replay failed; inspect logs.'}
& wsl.exe -d Ubuntu-24.04 -- bash $linuxPath scripts/background_experiment.py
if($LASTEXITCODE -ne 0) {throw 'Background experiment failed; inspect logs.'}
