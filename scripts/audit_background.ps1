$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
function Read-Json([string]$path) {Get-Content -LiteralPath $path -Raw | ConvertFrom-Json}
function Check($condition,[string]$message) {if(-not $condition){throw $message}}
function Hash([string]$path) {(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()}
$output=Join-Path $projectRoot 'results/background'
$verified=Read-Json (Join-Path $output 'verification.json')
$protocol=Read-Json (Join-Path $output 'protocol.json')
Check ($verified.passed -and $verified.final_episodes -eq 400 -and $verified.new_development_episodes -eq 80) 'Incomplete experiment'
foreach($kind in @('data_sha256','manifest_sha256')) {
    foreach($property in $protocol.$kind.PSObject.Properties) {
        $file=if($kind -eq 'data_sha256') {'episodes.h5'} else {'manifest.json'}
        Check ((Hash (Join-Path $projectRoot ("data/"+$property.Name+"/"+$file))) -eq $property.Value) 'Original data changed'
    }
}
foreach($property in $protocol.code_sha256.PSObject.Properties) {
    Check ((Hash (Join-Path $projectRoot ("scripts/"+$property.Name))) -eq $property.Value) 'Experiment code changed'
}
foreach($property in $protocol.mask_sha256.PSObject.Properties) {
    Check ((Hash (Join-Path $projectRoot ("data/background_masks/"+$property.Name+".h5"))) -eq $property.Value) 'Mask artifact changed'
}
foreach($old in $protocol.previous_models.PSObject.Properties) {
    foreach($file in $old.Value.PSObject.Properties) {
        Check ((Hash (Join-Path $projectRoot ("checkpoints/"+$old.Name+"/deploy/"+$file.Name))) -eq $file.Value) 'Previous model changed'
    }
}
foreach($oldName in @('brightness_control','brightness_aug')) {
    $oldDirectory=Join-Path $projectRoot ("results/"+$oldName+"/background_development-background")
    $oldRows=@(Read-Json (Join-Path $oldDirectory 'episodes.json'))
    $oldSummary=Read-Json (Join-Path $oldDirectory 'summary.json')
    Check ($oldRows.Count -eq 20 -and ($oldRows.seed -join ',') -eq ((22000..22019) -join ',')) 'Old diagnostics seeds differ'
    Check ($oldSummary.successes -eq @($oldRows | Where-Object success).Count) 'Old diagnostic score differs'
}
$scores=@{}
foreach($name in @('background_control','background_aug')) {
    $run=Read-Json (Join-Path $projectRoot ("checkpoints/"+$name+"/run.json"))
    $training=Read-Json (Join-Path $projectRoot ("checkpoints/"+$name+"/training_summary.json"))
    Check ($training.steps -eq 6000 -and $training.parameter_change_l2 -gt 0) 'Actual updates missing'
    Check ($run.run.steps -eq 6000 -and $run.run.batch_size -eq 8 -and $run.run.seed -eq 42) 'Budget mismatch'
    Check ($run.initial_parameter_sha256 -eq $verified.same_initial_parameter_sha256) 'Initialization differs'
    Check ($run.validation_augmentation -eq $null -and $run.run.brightness_range -eq $null) 'Unexpected image augmentation'
    Check ($run.run.table_augmentation -eq ($name -eq 'background_aug')) 'Incorrect table augmentation'
    $hashes=@{}
    foreach($file in @('model.safetensors','config.json','normalization.json')) {
        $hashes[$file]=Hash (Join-Path $projectRoot ("checkpoints/"+$name+"/deploy/"+$file))
    }
    $curves=@(Get-Content -LiteralPath (Join-Path $projectRoot ("logs/train-"+$name+".jsonl")) | ForEach-Object {$_ | ConvertFrom-Json})
    Check ($curves.Count -eq 60 -and ($curves.step -join ',') -eq ((1..60 | ForEach-Object {$_*100}) -join ',')) 'Training curve incomplete'
    $scores[$name]=@{}
    foreach($split in @('background_development','background_test')) {
        $wanted=if($split -eq 'background_test') {@(60000..60099)} else {@(22000..22019)}
        $conditions=if($split -eq 'background_test') {@('standard','background_test')} else {@('standard','background')}
        foreach($condition in $conditions) {
            $directory=Join-Path $projectRoot ("results/"+$name+"/"+$split+"-"+$condition)
            $summary=Read-Json (Join-Path $directory 'summary.json')
            $rows=@(Read-Json (Join-Path $directory 'episodes.json'))
            Check ($rows.Count -eq $wanted.Count -and ($rows.seed -join ',') -eq ($wanted -join ',')) 'Seed mismatch'
            Check ($summary.episodes -eq $rows.Count -and $summary.successes -eq @($rows | Where-Object success).Count) 'Incorrect score'
            Check (-not $summary.expert_used -and $summary.execution_steps -eq 30) 'Policy protocol mismatch'
            Check (($summary.policy_inputs | Sort-Object) -join ',' -eq 'observation.images.front,observation.images.wrist,observation.state') 'Policy inputs differ'
            Check (@($rows | Where-Object {$_.success -and -not $_.verified_grasp_lift}).Count -eq 0) 'Invalid success'
            foreach($file in $summary.evaluation_identity.files.PSObject.Properties) {Check ($hashes[$file.Name] -eq $file.Value) 'Stale score'}
            if($split -eq 'background_test') {$scores[$name][$condition]=$summary.successes}
        }
    }
}
foreach($property in $verified.artifacts_sha256.PSObject.Properties) {
    Check ((Hash (Join-Path $output $property.Name)) -eq $property.Value) 'Report attachment changed'
}
$vault=Join-Path (Split-Path -Parent (Split-Path -Parent $projectRoot)) 'obsdian知识库'
foreach($pair in @(@('comparison.png','SO101-ACT-背景泛化对照.png'),@('augmentation_example.png','SO101-ACT-桌面增强实例.png'),@('demo.mp4','SO101-ACT-新背景自主.mp4'),@('demo.gif','SO101-ACT-新背景演示.gif'))) {
    Check ((Hash (Join-Path $output $pair[0])) -eq (Hash (Join-Path $vault ("附件/"+$pair[1])))) 'Obsidian attachment mismatch'
}
$audit=[ordered]@{passed=$true;scores=$scores;final_episodes=400;development_episodes=80;previous_models_preserved=5;checked_at=(Get-Date).ToString('o')}
$audit | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'windows_audit.json') -Encoding UTF8
$audit | ConvertTo-Json -Compress -Depth 8
