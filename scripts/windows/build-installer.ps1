param(
    [Parameter(Mandatory = $true)]
    [string]$PackagingDir,
    [Parameter(Mandatory = $true)]
    [string]$StagingDir,
    [Parameter(Mandatory = $true)]
    [string]$AppVersion,
    [Parameter(Mandatory = $true)]
    [string]$IsccPath
)

$ErrorActionPreference = "Stop"

$resolvedPackaging = (Resolve-Path $PackagingDir).Path
$resolvedStaging = (Resolve-Path $StagingDir).Path
$scriptPath = Join-Path $resolvedPackaging "VideoTranscribe.iss"

& $IsccPath "/DSourceDir=$resolvedStaging" "/DAppVersion=$AppVersion" $scriptPath
