param(
    [Parameter(Mandatory = $true)]
    [string]$PyInstallerDistDir,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeBinariesDir,
    [Parameter(Mandatory = $true)]
    [string]$StagingDir,
    [Parameter(Mandatory = $true)]
    [string]$SourceDir
)

$ErrorActionPreference = "Stop"

$resolvedDist = (Resolve-Path $PyInstallerDistDir).Path
$resolvedRuntime = (Resolve-Path $RuntimeBinariesDir).Path
$resolvedSource = (Resolve-Path $SourceDir).Path

if (Test-Path $StagingDir) {
    Remove-Item -Recurse -Force $StagingDir
}

$appDir = Join-Path $StagingDir "app"
$binDir = Join-Path $appDir "bin"
$docsDir = Join-Path $appDir "docs"

New-Item -ItemType Directory -Force -Path $appDir, $binDir, $docsDir | Out-Null

Copy-Item -Recurse -Force (Join-Path $resolvedDist "*") $appDir
Copy-Item -Force (Join-Path $resolvedRuntime "bin\*") $binDir
Copy-Item -Force (Join-Path $resolvedSource "README.md") $docsDir
