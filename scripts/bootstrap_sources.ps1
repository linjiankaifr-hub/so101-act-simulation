$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$sourceRoot=Join-Path $projectRoot 'official'
New-Item -ItemType Directory -Force -Path $sourceRoot | Out-Null
$sources=Get-Content -LiteralPath (Join-Path $projectRoot 'sources.json') -Raw | ConvertFrom-Json
foreach($entry in $sources.PSObject.Properties) {
    $checkout=Join-Path $sourceRoot $entry.Name
    if(Test-Path -LiteralPath $checkout) {
        $actual=(& git -C $checkout rev-parse HEAD).Trim()
        if($LASTEXITCODE -ne 0 -or $actual -ne $entry.Value.commit) { throw "Existing source differs: $checkout. Preserve it and inspect manually." }
        Write-Host "Verified $($entry.Name) $actual"
        continue
    }
    & git clone $entry.Value.url $checkout
    if($LASTEXITCODE -ne 0) { throw "Clone failed; retained $checkout for inspection." }
    & git -C $checkout checkout --detach $entry.Value.commit
    if($LASTEXITCODE -ne 0) { throw 'Cannot check out the pinned revision.' }
}
