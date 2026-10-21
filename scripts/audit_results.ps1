$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
function Read-Json([string]$path) {Get-Content -LiteralPath $path -Raw | ConvertFrom-Json}
function Assert-True($condition,[string]$message) {if(-not $condition){throw $message}}
$verified=Read-Json (Join-Path $projectRoot 'results/verification.json')
Assert-True ($verified.passed -and $verified.final_episodes -eq 300) 'Python experiment audit not complete'
$finalSeeds=@(30000..30099)
$scoreRows=@()
foreach($name in @('base','targeted','random')) {
    $model=Join-Path $projectRoot ("checkpoints/"+$name+"/deploy")
    $training=Read-Json (Join-Path $projectRoot ("checkpoints/"+$name+"/training_summary.json"))
    Assert-True ($training.steps -eq 6000 -and $training.parameter_change_l2 -gt 0) 'Actual training evidence missing'
    $testFolder=Join-Path $projectRoot ("results/"+$name+"/test-standard")
    $summary=Read-Json (Join-Path $testFolder 'summary.json')
    $rows=@(Read-Json (Join-Path $testFolder 'episodes.json'))
    Assert-True ($rows.Count -eq 100 -and $summary.episodes -eq 100) 'Final test count mismatch'
    Assert-True (($rows.seed -join ',') -eq ($finalSeeds -join ',')) 'Final seeds mismatch'
    Assert-True (-not $summary.expert_used) 'Expert used during evaluation'
    Assert-True ($summary.execution_steps -eq 30 -and $summary.task_version -eq 'verified-pinch-lift-release-v2') 'Protocol mismatch'
    $successful=@($rows | Where-Object success)
    Assert-True ($successful.Count -eq $summary.successes) 'Score mismatch'
    Assert-True (@($successful | Where-Object {-not $_.verified_grasp_lift}).Count -eq 0) 'Success without verified grasp'
    foreach($property in $summary.evaluation_identity.files.PSObject.Properties) {
        $actual=(Get-FileHash -LiteralPath (Join-Path $model $property.Name) -Algorithm SHA256).Hash.ToLower()
        Assert-True ($actual -eq $property.Value) ("Model fingerprint mismatch: "+$name+"/"+$property.Name)
    }
    foreach($variation in @('position','lighting','background')) {
        $perturb=Read-Json (Join-Path $projectRoot ("results/"+$name+"/perturbation-"+$variation+"/summary.json"))
        Assert-True ($perturb.episodes -eq 10 -and -not $perturb.expert_used) 'Perturbation count mismatch'
        Assert-True (($perturb.evaluation_identity.seeds -join ',') -eq (@(40000..40009) -join ',')) 'Perturbation seed mismatch'
    }
    $scoreRows+=[ordered]@{variant=$name;successes=$summary.successes;episodes=100;training_steps=$training.steps}
}
$unitLog=Get-Content -LiteralPath (Join-Path $projectRoot 'logs/tests-review-green.log') -Raw
$visualLog=Get-Content -LiteralPath (Join-Path $projectRoot 'logs/test-perturbations.log') -Raw
Assert-True ($unitLog.Contains('20 passed') -and $visualLog.Contains('1 passed')) 'Test evidence missing'
foreach($file in @('results/demo.mp4','results/demo.gif','results/comparison.png','results/实验报告.md')) {
    Assert-True ((Get-Item -LiteralPath (Join-Path $projectRoot $file)).Length -gt 0) ("Artifact missing: "+$file)
}
$result=[ordered]@{passed=$true;checked_at=(Get-Date).ToString('o');final_episodes=300;perturbation_episodes=90;python_full_suite_passed=20;additional_visual_test_passed=1;variants=$scoreRows}
$result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $projectRoot 'results/windows_final_audit.json') -Encoding utf8
$result | ConvertTo-Json -Depth 10
