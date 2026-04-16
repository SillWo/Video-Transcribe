param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"

$sourcePath = (Resolve-Path $SourceDir).Path
$uiPath = Join-Path $sourcePath "ui"
$outputParent = Split-Path $OutputDir -Parent

if (!(Test-Path $uiPath)) {
    throw "UI directory not found at $uiPath"
}

if ($outputParent) {
    New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
}

Push-Location $uiPath
try {
    npm ci
    npm run build
} finally {
    Pop-Location
}

if (Test-Path $OutputDir) {
    Remove-Item -Recurse -Force $OutputDir
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Copy-Item -Recurse -Force (Join-Path $uiPath "dist\*") $OutputDir
