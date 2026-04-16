param(
    [Parameter(Mandatory = $true)]
    [string]$ReleaseTag,
    [string]$SourceRef,
    [string]$SourceSha,
    [string]$PackagingRef = "windows_dist",
    [string]$SourceDir,
    [string]$PackagingDir,
    [string]$RepositoryRoot,
    [string]$WorkRoot,
    [string]$PythonExe = "python",
    [string]$IsccPath
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
} else {
    $scriptRoot = $PSScriptRoot
}

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
}

if ([string]::IsNullOrWhiteSpace($WorkRoot)) {
    $WorkRoot = Join-Path $RepositoryRoot "build\windows"
}

function Expand-GitRef {
    param(
        [string]$RepoRoot,
        [string]$RefName,
        [string]$DestinationDir,
        [string]$ArchiveName
    )

    if (Test-Path $DestinationDir) {
        Remove-Item -Recurse -Force $DestinationDir
    }

    New-Item -ItemType Directory -Force -Path $DestinationDir | Out-Null
    $archivePath = Join-Path $env:TEMP $ArchiveName
    if (Test-Path $archivePath) {
        Remove-Item -Force $archivePath
    }

    cmd /c "git -C ""$RepoRoot"" rev-parse --verify ""$RefName"" 1>nul 2>nul"
    if ($LASTEXITCODE -ne 0) {
        throw "Git ref '$RefName' was not found in $RepoRoot"
    }

    cmd /c "git -C ""$RepoRoot"" archive --format=zip --output=""$archivePath"" ""$RefName"" 2>nul"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create archive for git ref '$RefName'"
    }

    if (!(Test-Path $archivePath)) {
        throw "Archive for git ref '$RefName' was not created"
    }

    Expand-Archive -Path $archivePath -DestinationPath $DestinationDir
}

if ([string]::IsNullOrWhiteSpace($SourceRef) -and [string]::IsNullOrWhiteSpace($SourceSha)) {
    if ([string]::IsNullOrWhiteSpace($SourceDir)) {
        throw "Either -SourceRef or -SourceSha must be provided when -SourceDir is not used"
    }
}

$resolvedRepoRoot = (Resolve-Path $RepositoryRoot).Path
$resolvedWorkRoot = (New-Item -ItemType Directory -Force -Path $WorkRoot).FullName
$effectiveSourceDir = Join-Path $resolvedWorkRoot "source"
$effectivePackagingDir = Join-Path $resolvedWorkRoot "packaging"
$frontendDir = Join-Path $resolvedWorkRoot "frontend"
$runtimeDir = Join-Path $resolvedWorkRoot "runtime-binaries"
$stagingDir = Join-Path $resolvedWorkRoot "staging"

if (![string]::IsNullOrWhiteSpace($SourceDir)) {
    $effectiveSourceDir = (Resolve-Path $SourceDir).Path
} else {
    $sourceRefValue = if (![string]::IsNullOrWhiteSpace($SourceRef)) { $SourceRef } else { $SourceSha }
    Expand-GitRef -RepoRoot $resolvedRepoRoot -RefName $sourceRefValue -DestinationDir $effectiveSourceDir -ArchiveName "vt-source.work.zip"
}

if (![string]::IsNullOrWhiteSpace($PackagingDir)) {
    $effectivePackagingDir = (Resolve-Path $PackagingDir).Path
} else {
    Expand-GitRef -RepoRoot $resolvedRepoRoot -RefName $PackagingRef -DestinationDir $effectivePackagingDir -ArchiveName "vt-packaging.work.zip"
}

$requiredPackagingFiles = @(
    (Join-Path $effectivePackagingDir "scripts\windows\build-frontend.ps1"),
    (Join-Path $effectivePackagingDir "scripts\windows\build-backend.ps1"),
    (Join-Path $effectivePackagingDir "scripts\windows\fetch-runtime-binaries.ps1"),
    (Join-Path $effectivePackagingDir "scripts\windows\prepare-staging.ps1"),
    (Join-Path $effectivePackagingDir "scripts\windows\build-installer.ps1"),
    (Join-Path $effectivePackagingDir "packaging\windows\VideoTranscribe.spec"),
    (Join-Path $effectivePackagingDir "packaging\windows\VideoTranscribe.iss")
)

$missingPackagingFiles = $requiredPackagingFiles | Where-Object { !(Test-Path $_) }
if ($missingPackagingFiles.Count -gt 0) {
    $missingList = ($missingPackagingFiles | ForEach-Object { " - $_" }) -join [Environment]::NewLine
    throw @"
Packaging layer is incomplete at '$effectivePackagingDir'.
Missing files:
$missingList

If you are testing local uncommitted changes from the current workspace, rerun with:
  -PackagingDir .

If you expect a reproducible ref build, commit the packaging files to '$PackagingRef' first.
"@
}

$version = $ReleaseTag.Trim()
if ($version.StartsWith("v")) {
    $version = $version.Substring(1)
}

& (Join-Path $effectivePackagingDir "scripts\windows\build-frontend.ps1") -SourceDir $effectiveSourceDir -OutputDir $frontendDir
& (Join-Path $effectivePackagingDir "scripts\windows\fetch-runtime-binaries.ps1") -PackagingDir (Join-Path $effectivePackagingDir "packaging\windows") -OutputDir $runtimeDir
& (Join-Path $effectivePackagingDir "scripts\windows\build-backend.ps1") `
    -SourceDir $effectiveSourceDir `
    -PackagingDir (Join-Path $effectivePackagingDir "packaging\windows") `
    -FrontendBuildDir $frontendDir `
    -BuildRoot $resolvedWorkRoot `
    -AppVersion $version `
    -RuntimeBinariesDir $runtimeDir `
    -PythonExe $PythonExe
& (Join-Path $effectivePackagingDir "scripts\windows\prepare-staging.ps1") `
    -PyInstallerDistDir (Join-Path $resolvedWorkRoot "pyinstaller\VideoTranscribe") `
    -RuntimeBinariesDir $runtimeDir `
    -StagingDir $stagingDir `
    -SourceDir $effectiveSourceDir

if ([string]::IsNullOrWhiteSpace($IsccPath)) {
    throw "-IsccPath is required for installer build"
}

& (Join-Path $effectivePackagingDir "scripts\windows\build-installer.ps1") `
    -PackagingDir (Join-Path $effectivePackagingDir "packaging\windows") `
    -StagingDir $stagingDir `
    -AppVersion $version `
    -IsccPath $IsccPath

$installerDir = Join-Path $resolvedWorkRoot "installer"
$installer = Get-ChildItem -Path $installerDir -Filter "VideoTranscribe-Setup-$version.exe" | Select-Object -First 1
if (!$installer) {
    throw "Installer was not produced in $installerDir"
}

$checksumPath = "$($installer.FullName).sha256"
((Get-FileHash -Algorithm SHA256 -Path $installer.FullName).Hash.ToLowerInvariant() + "  " + $installer.Name) | Set-Content -Encoding ASCII $checksumPath

Write-Output $installer.FullName
