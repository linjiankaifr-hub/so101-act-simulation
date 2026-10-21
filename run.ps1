param(
    [ValidateSet('status','test','demo','evaluate','collect','train','full','ros')][string]$Task='status',
    [ValidateSet('base','targeted','random')][string]$Variant='base',
    [ValidateRange(1,100)][int]$Episodes=100
)
$ErrorActionPreference='Stop'
if($Task -eq 'demo') {
    $video=Join-Path $PSScriptRoot ("results\"+$Variant+"\test-standard-dense\success.mp4")
    if(-not (Test-Path -LiteralPath $video)) { $video=Join-Path $PSScriptRoot 'results\expert_demo.mp4' }
    if(-not (Test-Path -LiteralPath $video)) { throw 'No video exists yet.' }
    Write-Host $video
    Invoke-Item -LiteralPath $video
    exit
}
$scriptPath=Join-Path $PSScriptRoot 'scripts\stage.sh'
$linuxPath='/mnt/'+$scriptPath.Substring(0,1).ToLower()+$scriptPath.Substring(2).Replace('\','/')
& wsl.exe -d Ubuntu-24.04 -- bash $linuxPath $Task $Variant $Episodes
if($LASTEXITCODE -ne 0) { throw 'The stage failed; inspect the terminal or logs.' }
