$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
function Read-Json([string]$path) {Get-Content -LiteralPath $path -Raw | ConvertFrom-Json}
function Check($condition,[string]$message) {if(-not $condition){throw $message}}
$output=Join-Path $projectRoot 'results/brightness'
$verified=Read-Json (Join-Path $output 'verification.json')
$protocol=Read-Json (Join-Path $output 'protocol.json')
Check ($verified.passed -and $verified.final_episodes -eq 400 -and $verified.development_episodes -eq 80) 'Experiment incomplete'
foreach($kind in @('data_sha256','manifest_sha256')) {
    foreach($property in $protocol.$kind.PSObject.Properties) {
        $file=if($kind -eq 'data_sha256') {'episodes.h5'} else {'manifest.json'}
        $actual=(Get-FileHash -LiteralPath (Join-Path $projectRoot ("data/"+$property.Name+"/"+$file)) -Algorithm SHA256).Hash.ToLower()
        Check ($actual -eq $property.Value) 'Dataset or ordered split manifest changed'
    }
}
foreach($property in $protocol.code_sha256.PSObject.Properties) {
    $actual=(Get-FileHash -LiteralPath (Join-Path $projectRoot ("scripts/"+$property.Name)) -Algorithm SHA256).Hash.ToLower()
    Check ($actual -eq $property.Value) 'Training/evaluation source changed'
}
$scores=@{}
foreach($name in @('brightness_control','brightness_aug')) {
    $run=Read-Json (Join-Path $projectRoot ("checkpoints/"+$name+"/run.json"))
    $training=Read-Json (Join-Path $projectRoot ("checkpoints/"+$name+"/training_summary.json"))
    Check ($training.steps -eq 6000 -and $training.parameter_change_l2 -gt 0) 'Actual training evidence missing'
    Check ($run.initial_parameter_sha256 -eq $verified.same_initial_parameter_sha256) 'Initialization differs'
    Check ($run.validation_augmentation -eq $null) 'Validation augmentation enabled'
    if($name -eq 'brightness_control') {Check ($null -eq $run.run.brightness_range) 'Control modified'}
    else {Check ($run.run.brightness_range.Count -eq 2 -and $run.run.brightness_range[0] -eq .4 -and $run.run.brightness_range[1] -eq 1.4) 'Brightness recipe differs'}
    $hashes=@{}
    foreach($file in @('model.safetensors','config.json','normalization.json')) {
        $hashes[$file]=(Get-FileHash -LiteralPath (Join-Path $projectRoot ("checkpoints/"+$name+"/deploy/"+$file)) -Algorithm SHA256).Hash.ToLower()
    }
    $scores[$name]=@{}
    foreach($split in @('brightness_development','brightness_test')) {
        $wanted=if($split -eq 'brightness_test') {@(50000..50099)} else {@(21000..21019)}
        foreach($condition in @('standard','lighting')) {
            $directory=Join-Path $projectRoot ("results/"+$name+"/"+$split+"-"+$condition)
            $summary=Read-Json (Join-Path $directory 'summary.json')
            $rows=@(Read-Json (Join-Path $directory 'episodes.json'))
            Check ($rows.Count -eq $wanted.Count -and ($rows.seed -join ',') -eq ($wanted -join ',')) 'Count or seed mismatch'
            Check ($summary.successes -eq @($rows | Where-Object success).Count) 'Score mismatch'
            Check (-not $summary.expert_used -and $summary.execution_steps -eq 30) 'Policy protocol mismatch'
            Check (@($rows | Where-Object {$_.success -and -not $_.verified_grasp_lift}).Count -eq 0) 'False grasp success'
            foreach($property in $summary.evaluation_identity.files.PSObject.Properties) {Check ($hashes[$property.Name] -eq $property.Value) 'Stale model score'}
            if($split -eq 'brightness_test') {$scores[$name][$condition]=$summary.successes}
        }
    }
}
foreach($old in @('base','targeted','random')) {
    $summary=Read-Json (Join-Path $projectRoot ("results/"+$old+"/test-standard/summary.json"))
    foreach($property in $summary.evaluation_identity.files.PSObject.Properties) {
        $actual=(Get-FileHash -LiteralPath (Join-Path $projectRoot ("checkpoints/"+$old+"/deploy/"+$property.Name)) -Algorithm SHA256).Hash.ToLower()
        Check ($actual -eq $property.Value) 'Original model changed'
    }
}
Check ((Get-Content -LiteralPath (Join-Path $projectRoot 'logs/brightness-final-tests.log') -Raw).Contains('37 passed')) 'Final tests missing'
$result=[ordered]@{passed=$true;checked_at=(Get-Date).ToString('o');tests=37;formal_episodes=400;development_episodes=80;scores=$scores;old_model_fingerprints_preserved=$true}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'windows_audit.json') -Encoding utf8
$result | ConvertTo-Json -Depth 8
