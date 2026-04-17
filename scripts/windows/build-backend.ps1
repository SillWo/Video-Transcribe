param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,
    [Parameter(Mandatory = $true)]
    [string]$PackagingDir,
    [Parameter(Mandatory = $true)]
    [string]$FrontendBuildDir,
    [Parameter(Mandatory = $true)]
    [string]$BuildRoot,
    [Parameter(Mandatory = $true)]
    [string]$AppVersion,
[string]$RuntimeBinariesDir,
[string]$PythonExe = "python",
[string]$PyInstallerVersion = "6.16.0"
)

$ErrorActionPreference = "Stop"

$gpuRuntimePackages = @(
    @{ Name = "nvidia-cublas-cu12"; Version = "12.9.2.10" },
    @{ Name = "nvidia-cuda-runtime-cu12"; Version = "12.9.79" },
    @{ Name = "nvidia-cudnn-cu12"; Version = "9.21.0.82" },
    @{ Name = "nvidia-nvjitlink-cu12"; Version = "12.9.86" }
)

$resolvedSource = (Resolve-Path $SourceDir).Path
$resolvedPackaging = (Resolve-Path $PackagingDir).Path
$resolvedFrontend = (Resolve-Path $FrontendBuildDir).Path
$resolvedBuildRoot = (New-Item -ItemType Directory -Force -Path $BuildRoot).FullName
$venvDir = Join-Path $resolvedBuildRoot "venv"

if (Test-Path $venvDir) {
    Remove-Item -Recurse -Force $venvDir
}

& $PythonExe -m venv $venvDir
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required for backend packaging builds"
}

& git lfs version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "git-lfs is required to install kontur-ai/sbert_punc_case_ru during the packaging build"
}

& git lfs install --skip-smudge | Out-Null

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $resolvedSource "requirements.txt")
& $venvPython -m pip install "pyinstaller==$PyInstallerVersion"
& $venvPython -m pip install ($gpuRuntimePackages | ForEach-Object { "$($_.Name)==$($_.Version)" })

$env:VT_SOURCE_DIR = $resolvedSource
$env:VT_PACKAGING_DIR = $resolvedPackaging
$env:VT_FRONTEND_BUILD_DIR = $resolvedFrontend
$env:VT_BUILD_ROOT = $resolvedBuildRoot
$env:VT_APP_VERSION = $AppVersion

$distPath = Join-Path $resolvedBuildRoot "pyinstaller"
$workPath = Join-Path $resolvedBuildRoot "pyinstaller-work"
$appDistDir = Join-Path $distPath "VideoTranscribe"
$appBinDir = Join-Path $appDistDir "bin"

New-Item -ItemType Directory -Force -Path $distPath, $workPath | Out-Null

if (Test-Path $appDistDir) {
    try {
        Remove-Item -Recurse -Force $appDistDir
    }
    catch {
        throw "Unable to clean '$appDistDir'. Close any running VideoTranscribe.exe processes and retry the build."
    }
}

& $venvPython -m PyInstaller `
    --noconfirm `
    --distpath $distPath `
    --workpath $workPath `
    (Join-Path $resolvedPackaging "VideoTranscribe.spec")

if (![string]::IsNullOrWhiteSpace($RuntimeBinariesDir)) {
    $resolvedRuntime = (Resolve-Path $RuntimeBinariesDir).Path
    $runtimeBinDir = Join-Path $resolvedRuntime "bin"
    if (!(Test-Path $runtimeBinDir)) {
        throw "Runtime binaries directory '$runtimeBinDir' does not exist"
    }

    if (Test-Path $appBinDir) {
        Remove-Item -Recurse -Force $appBinDir
    }

    New-Item -ItemType Directory -Force -Path $appBinDir | Out-Null
    Copy-Item -Force (Join-Path $runtimeBinDir "*") $appBinDir
}

$gpuPackageRoot = Join-Path $venvDir "Lib\site-packages\nvidia"
if (Test-Path $gpuPackageRoot) {
    $gpuDlls = Get-ChildItem -Path $gpuPackageRoot -Recurse -Filter *.dll -File
    if ($gpuDlls.Count -gt 0) {
        $gpuTargets = @($appBinDir)
        if (![string]::IsNullOrWhiteSpace($RuntimeBinariesDir)) {
            $gpuTargets += (Join-Path $resolvedRuntime "bin")
        }

        foreach ($target in ($gpuTargets | Select-Object -Unique)) {
            New-Item -ItemType Directory -Force -Path $target | Out-Null
        }

        foreach ($dll in $gpuDlls) {
            foreach ($target in ($gpuTargets | Select-Object -Unique)) {
                Copy-Item -Force $dll.FullName (Join-Path $target $dll.Name)
            }
        }
    }
}
