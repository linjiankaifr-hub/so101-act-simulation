param([switch]$Status)
$ErrorActionPreference='Stop'
if($Status) {
    foreach($name in @('brightness_control','brightness_aug')) {
        $row=[ordered]@{variant=$name}
        $training=Join-Path $PSScriptRoot ("checkpoints/"+$name+"/training_summary.json")
        if(Test-Path -LiteralPath $training) {$row.steps=(Get-Content -LiteralPath $training -Raw | ConvertFrom-Json).steps}
        foreach($condition in @('standard','lighting')) {
            $summary=Join-Path $PSScriptRoot ("results/"+$name+"/brightness_test-"+$condition+"/summary.json")
            if(Test-Path -LiteralPath $summary) {$s=Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json;$row[$condition]="$($s.successes)/$($s.episodes)"}
        }
        $row | ConvertTo-Json -Compress
    }
    exit
}
$entry=Join-Path $PSScriptRoot 'scripts/python.sh'
$linuxPath='/mnt/'+$entry.Substring(0,1).ToLower()+$entry.Substring(2).Replace('\','/')
& wsl.exe -d Ubuntu-24.04 -- bash $linuxPath scripts/brightness_experiment.py
if($LASTEXITCODE -ne 0) {throw 'Brightness experiment failed; inspect logs.'}
