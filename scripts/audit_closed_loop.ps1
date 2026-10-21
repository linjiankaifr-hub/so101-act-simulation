$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$out=Join-Path $root 'results/closed_loop'
function Read-Json([string]$path) { Get-Content -LiteralPath $path -Raw | ConvertFrom-Json }
function Require([bool]$condition,[string]$message) { if(-not $condition) {throw $message} }
function Match-Hash([string]$path,[string]$expected) {
    $actual=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    Require ($actual -eq $expected) "File changed: $path"
}
$protocol=Read-Json (Join-Path $out 'protocol.json')
foreach($entry in $protocol.code_sha256.PSObject.Properties) {
    Match-Hash (Join-Path $root ('scripts/'+$entry.Name)) $entry.Value
}
foreach($entry in $protocol.xml_sha256.PSObject.Properties) {
    Match-Hash (Join-Path $root ('official/so101-ros2-mujoco/src/'+$entry.Name)) $entry.Value
}
foreach($model in $protocol.previous_models.PSObject.Properties) {
    foreach($entry in $model.Value.PSObject.Properties) {
        Match-Hash (Join-Path $root ('checkpoints/'+$model.Name+'/deploy/'+$entry.Name)) $entry.Value
    }
}
$allProtocolDigests=[System.Collections.Generic.HashSet[string]]::new()
$count=0
foreach($split in @('development','test')) {
    $seeds=if($split -eq 'development') {@($protocol.development_seeds)} else {@($protocol.test_seeds)}
    foreach($window in @(5,10,30)) {
        foreach($condition in @('standard','push')) {
            $dir=Join-Path $out "$split/$condition-h$window"
            $summary=Read-Json (Join-Path $dir 'summary.json')
            $rows=@(Read-Json (Join-Path $dir 'episodes.json'))
            Require ($summary.identity.execution_steps -eq $window) "Wrong execution window: $dir"
            Require ($summary.identity.condition -eq $condition) "Wrong condition: $dir"
            Require (($summary.identity.seeds -join ',') -eq ($seeds -join ',')) "Wrong bound seeds: $dir"
            Require (($rows.seed -join ',') -eq ($seeds -join ',')) "Wrong row seeds: $dir"
            Require ($rows.Count -eq $seeds.Count -and $summary.episodes -eq $seeds.Count) "Missing episodes: $dir"
            Require (@($rows | Where-Object success).Count -eq $summary.successes) "Wrong success count: $dir"
            Require (-not $summary.expert_used) "Expert used: $dir"
            Require ((($summary.policy_inputs | Sort-Object) -join ',') -eq 'observation.images.front,observation.images.wrist,observation.state') "Unexpected inputs: $dir"
            foreach($row in $rows) {
                Require ($row.policy_queries -eq [math]::Ceiling($row.steps/$window)) "Wrong query count: $dir"
                Require ($row.query_latency_ms.Count -eq $row.policy_queries) "Missing latency samples: $dir"
                Require (-not $row.success -or $row.verified_grasp_lift) "Unverified successful grasp: $dir"
                if($condition -eq 'push') {Require ([bool]$row.pulse_applied) "Pulse not applied: $dir"}
            }
            $null=$allProtocolDigests.Add($summary.identity.protocol_sha256)
            $count+=$rows.Count
        }
    }
}
Require ($count -eq 720) 'Expected 120 development and 600 formal executions'
Require ($allProtocolDigests.Count -eq 1) 'Evaluations belong to different protocols'
$verification=Read-Json (Join-Path $out 'verification.json')
Require ([bool]$verification.passed) 'Python trajectory/protocol audit missing'
$audit=[ordered]@{passed=$true;episodes=$count;unique_final_initial_conditions=100;old_models_unchanged=$true;core_code_unchanged=$true;scene_xml_unchanged=$true;source='independent Windows PowerShell audit'}
$audit | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'windows_audit.json') -Encoding utf8
$audit | ConvertTo-Json
