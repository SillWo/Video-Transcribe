[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$CheckOnly,
    [switch]$SetupOnly,
    [switch]$InstallPostprocess,
    [switch]$SkipPostprocess,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$OriginalPSModulePath = $env:PSModulePath
$modulePaths = @([IO.Path]::Combine($PSHOME, 'Modules'))
if ($env:SystemRoot) { $modulePaths += [IO.Path]::Combine($env:SystemRoot, 'System32\WindowsPowerShell\v1.0\Modules') }
if ($env:ProgramFiles) { $modulePaths += [IO.Path]::Combine($env:ProgramFiles, 'WindowsPowerShell\Modules') }
if ($env:USERPROFILE) { $modulePaths += [IO.Path]::Combine($env:USERPROFILE, 'Documents\WindowsPowerShell\Modules') }
$modulePaths += ($OriginalPSModulePath -split ';')
$modulePaths = @($modulePaths | Where-Object { $_ -and [IO.Directory]::Exists($_) } | Select-Object -Unique)
$env:PSModulePath = $modulePaths -join ';'
$AppId = 'video-transcribe'
$BackendUrl = 'http://127.0.0.1:8000'
$Root = (Resolve-Path $PSScriptRoot).Path
$Runtime = Join-Path $Root '.runtime'
$Logs = Join-Path $Runtime 'logs'
$PidFile = Join-Path $Runtime 'video-transcribe.pid'
$WhisperUrl = 'https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-windows-vulkan-x64.zip'
$WhisperSha256 = 'e0d20a0f92e31b98adc0faf71172efc810b701e6391a9d858ca045bff26f77cd'
$WhisperHome = Join-Path $Runtime 'whisper-vulkan'
$NodeHome = Join-Path $Runtime 'node'
$script:IncludePostprocess = (-not $SkipPostprocess -or $InstallPostprocess)

function Assert-RuntimePath([string]$Path) {
    $runtimeFull = [IO.Path]::GetFullPath($Runtime).TrimEnd('\') + '\'
    $runtimeBase = $runtimeFull.TrimEnd('\')
    $targetFull = [IO.Path]::GetFullPath($Path)
    if (-not $targetFull.StartsWith($runtimeFull, [StringComparison]::OrdinalIgnoreCase)) {
        Fail "Refusing to alter a path outside .runtime: $targetFull"
    }
    $probe = $targetFull
    while (-not (Test-Path -LiteralPath $probe)) {
        $parent = Split-Path -Parent $probe
        if (-not $parent -or $parent -eq $probe) { break }
        $probe = $parent
    }
    if (Test-Path -LiteralPath $probe) {
        $resolved = (Resolve-Path -LiteralPath $probe).Path
        if ($resolved -ne $runtimeBase -and -not $resolved.StartsWith($runtimeFull, [StringComparison]::OrdinalIgnoreCase)) {
            Fail "Refusing to alter a path that resolves outside .runtime: $resolved"
        }
    }
}

function Fail([string]$Message) {
    throw $Message
}

function Refresh-ProcessPath {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $parts = @($env:Path -split ';') + @($userPath -split ';') + @($machinePath -split ';')
    $env:Path = (($parts | Where-Object { $_ } | Select-Object -Unique) -join ';')
}

function Set-ProjectCacheEnvironment {
    $modelHome = Join-Path $Runtime 'models'
    $huggingFaceHome = Join-Path $modelHome 'huggingface'
    $cacheHome = Join-Path $Runtime 'cache'
    $tempHome = Join-Path $Runtime 'tmp'
    $pipCache = Join-Path $cacheHome 'pip'
    $npmCache = Join-Path $cacheHome 'npm'
    foreach ($directory in @($huggingFaceHome, (Join-Path $huggingFaceHome 'hub'), (Join-Path $huggingFaceHome 'xet'), (Join-Path $huggingFaceHome 'assets'), (Join-Path $modelHome 'torch'), $cacheHome, $tempHome, $pipCache, $npmCache)) {
        if (-not $CheckOnly) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
    }
    $env:HF_HOME = $huggingFaceHome
    $env:HF_HUB_CACHE = Join-Path $huggingFaceHome 'hub'
    $env:HF_XET_CACHE = Join-Path $huggingFaceHome 'xet'
    $env:HF_ASSETS_CACHE = Join-Path $huggingFaceHome 'assets'
    $env:TORCH_HOME = Join-Path $modelHome 'torch'
    $env:XDG_CACHE_HOME = $cacheHome
    $env:PIP_CACHE_DIR = $pipCache
    $env:npm_config_cache = $npmCache
    $env:TMP = $tempHome
    $env:TEMP = $tempHome
    $env:TMPDIR = $tempHome
}

function Find-CommandPath([string]$Name) {
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { return $command.Source }
    return $null
}

function Invoke-WingetUserInstall([string]$PackageId, [string]$FriendlyName) {
    $winget = Find-CommandPath 'winget.exe'
    if (-not $winget) {
        Fail "$FriendlyName не найден, а winget недоступен. Установите $FriendlyName вручную с официального сайта и повторите запуск."
    }
    Write-Host "Устанавливаю $FriendlyName для текущего пользователя через winget..."
    & $winget install --id $PackageId --exact --scope user --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
    if ($LASTEXITCODE -ne 0) {
        Fail "winget не смог установить $FriendlyName (код $LASTEXITCODE). Установите его вручную без прав администратора и повторите запуск."
    }
    Refresh-ProcessPath
}

function Get-PythonLauncher {
    $launcher = Find-CommandPath 'py.exe'
    if ($launcher) { return $launcher }
    return $null
}

function Select-Python {
    $venvPython = Join-Path $Root '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) {
        & $venvPython -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)' 2>$null
        if ($LASTEXITCODE -ne 0) { Fail 'Локальное .venv создано на Python старше или младше поддерживаемого диапазона. Переместите .venv и повторите запуск с Python 3.12+.' }
        return $venvPython
    }

    $py = Get-PythonLauncher
    if ($py) {
        & $py -3.12 -c 'import sys; print(sys.executable)' 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { return ,@($py, '-3.12') }
        & $py -3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { return ,@($py, '-3') }
    }
    $python = Find-CommandPath 'python.exe'
    if ($python) {
        & $python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { return $python }
    }
    if (-not $CheckOnly) { Invoke-WingetUserInstall 'Python.Python.3.12' 'Python 3.12' }
    Refresh-ProcessPath
    $py = Get-PythonLauncher
    if ($py) {
        & $py -3.12 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { return ,@($py, '-3.12') }
    }
    $python = Find-CommandPath 'python.exe'
    if ($python) {
        & $python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { return $python }
    }
    $localPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    if (Test-Path $localPython) {
        & $localPython -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { return $localPython }
    }
    Fail 'Python 3.12+ не найден после установки. Перезапустите Windows Terminal или установите Python с python.org для текущего пользователя.'
}

function Invoke-Python([string[]]$Arguments) {
    if ($script:PythonSpec -is [array]) {
        $exe = $script:PythonSpec[0]
        $prefix = @($script:PythonSpec | Select-Object -Skip 1)
        & $exe @prefix @Arguments
    } else {
        & $script:PythonSpec @Arguments
    }
}

function Invoke-CudaRuntimePreparation([switch]$NoDownload) {
    $arguments = @('-m', 'services.cuda_runtime')
    if ($NoDownload) { $arguments += '--check-only' }
    Invoke-Python $arguments
    if ($LASTEXITCODE -ne 0) { Fail 'Подготовка локального CUDA runtime не завершена. Исправьте указанную выше причину и повторите запуск.' }
}

function Invoke-RequiredModelPreparation([switch]$NoDownload) {
    $arguments = @('-m', 'services.prepare_models')
    if ($NoDownload) { $arguments += '--check-only' }
    if ($SetupOnly) { $arguments += '--setup-only' }
    if (-not $script:IncludePostprocess) { $arguments += '--skip-postprocess' }
    Invoke-Python $arguments
    if ($LASTEXITCODE -ne 0) { Fail 'Подготовка обязательных моделей не завершена. Повторный запуск продолжит загрузку с уже сохранённых файлов.' }
}

function Invoke-DiarizationPreparation([switch]$NoDownload) {
    $arguments = @('-m', 'services.diarization_runtime')
    if ($NoDownload) { $arguments += '--check-only' }
    Write-Host 'Проверяю локальную диаризацию Nemotron-3...'
    Invoke-Python $arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'ПРЕДУПРЕЖДЕНИЕ: диаризация не полностью подготовлена. Приложение продолжит запуск; проверьте сообщения выше и повторите start.ps1 для загрузки недостающих файлов.' -ForegroundColor Yellow
    }
}

function Get-ExecutableVersion([string]$Path, [string]$Argument) {
    if (-not $Path) { return $null }
    try { return ((& $Path $Argument 2>$null | Select-Object -First 1).Trim()) } catch { return $null }
}

function Test-NodeVersion([string]$NodePath) {
    $versionText = Get-ExecutableVersion $NodePath '--version'
    if ($versionText -notmatch '^v(\d+)\.(\d+)\.(\d+)') { return $false }
    $major = [int]$Matches[1]; $minor = [int]$Matches[2]; $patch = [int]$Matches[3]
    return ($major -ge 22 -and ($major -gt 22 -or $minor -ge 12)) -or ($major -eq 20 -and $patch -ge 19)
}

function Test-NodeElectronVersion([string]$NodePath) {
    $versionText = Get-ExecutableVersion $NodePath '--version'
    if ($versionText -notmatch '^v(\d+)\.(\d+)\.(\d+)') { return $false }
    $major = [int]$Matches[1]; $minor = [int]$Matches[2]
    return ($major -ge 22 -and ($major -gt 22 -or $minor -ge 12))
}

function Install-LocalNode {
    $manifestUrl = 'https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt'
    $manifest = Join-Path $Runtime 'node-SHASUMS256.txt'
    $archive = Join-Path $Runtime 'node-win-x64.zip'
    $staging = Join-Path $Runtime 'node.extracting'
    New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
    Assert-RuntimePath $manifest
    Assert-RuntimePath $archive
    Assert-RuntimePath $staging
    Write-Host 'Загружаю официальный Node.js 22 LTS в локальную папку проекта...'
    Invoke-WebRequest -Uri $manifestUrl -OutFile $manifest
    $entry = Get-Content -LiteralPath $manifest | Where-Object { $_ -match '^([0-9a-fA-F]{64})\s+(node-(v22\.\d+\.\d+)-win-x64\.zip)$' } | Select-Object -First 1
    if (-not $entry -or $entry -notmatch '^([0-9a-fA-F]{64})\s+(node-(v22\.\d+\.\d+)-win-x64\.zip)$') { Fail 'Не удалось найти архив Windows x64 Node.js 22 в официальном SHA256-манифесте.' }
    $expectedHash = $Matches[1].ToLowerInvariant()
    $archiveName = $Matches[2]
    $archiveUrl = 'https://nodejs.org/dist/latest-v22.x/' + $archiveName
    Invoke-WebRequest -Uri $archiveUrl -OutFile $archive
    $actualHash = Get-FileHashText $archive
    if ($actualHash -ne $expectedHash) { Remove-Item -LiteralPath $archive -Force; Fail "Проверка SHA-256 Node.js не прошла. Ожидался $expectedHash, получен $actualHash." }
    if (Test-Path -LiteralPath $staging) { Assert-RuntimePath $staging; Remove-Item -LiteralPath $staging -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $staging -Force
    $nodeExe = Get-ChildItem -LiteralPath $staging -Filter node.exe -File -Recurse | Select-Object -First 1
    if (-not $nodeExe) { Fail 'В официальном архиве Node.js не найден node.exe.' }
    if (Test-Path -LiteralPath $NodeHome) { Assert-RuntimePath $NodeHome; Remove-Item -LiteralPath $NodeHome -Recurse -Force }
    Move-Item -LiteralPath $staging -Destination $NodeHome
    Remove-Item -LiteralPath $archive -Force
    Remove-Item -LiteralPath $manifest -Force
}

function Ensure-Node([switch]$ElectronRuntime) {
    $node = Find-CommandPath 'node.exe'
    $npm = Find-CommandPath 'npm.cmd'
    if ($node -and $npm -and (Test-NodeVersion $node) -and (-not $ElectronRuntime -or (Test-NodeElectronVersion $node))) { return @($node, $npm) }
    if (Test-Path -LiteralPath $NodeHome) {
        $nodeLocal = Get-ChildItem -LiteralPath $NodeHome -Filter node.exe -File -Recurse | Select-Object -First 1
        if ($nodeLocal) {
            $npmLocal = Join-Path $nodeLocal.DirectoryName 'npm.cmd'
            if ((Test-Path -LiteralPath $npmLocal) -and (Test-NodeVersion $nodeLocal.FullName) -and (-not $ElectronRuntime -or (Test-NodeElectronVersion $nodeLocal.FullName))) {
                $env:Path = $nodeLocal.DirectoryName + ';' + $env:Path
                return @($nodeLocal.FullName, $npmLocal)
            }
        }
    }
    if ($CheckOnly) {
        if ($ElectronRuntime) { Fail 'Для Electron 45 нужен Node.js 22.12+ с npm. Запустите start.ps1, чтобы установить локальный Node.js 22 LTS.' }
        Fail 'Нужен Node.js 20.19+ или 22.12+ с npm для сборки Vite. Запустите start.ps1 для локальной установки Node.js LTS.'
    }
    Install-LocalNode
    $nodeLocal = Get-ChildItem -LiteralPath $NodeHome -Filter node.exe -File -Recurse | Select-Object -First 1
    $npmLocal = Join-Path $nodeLocal.DirectoryName 'npm.cmd'
    if (-not $nodeLocal -or -not (Test-Path -LiteralPath $npmLocal) -or -not (Test-NodeVersion $nodeLocal.FullName) -or ($ElectronRuntime -and -not (Test-NodeElectronVersion $nodeLocal.FullName))) { Fail 'Локальный Node.js был загружен, но не прошёл проверку версии или npm. Для Electron нужен Node.js 22.12+.' }
    $env:Path = $nodeLocal.DirectoryName + ';' + $env:Path
    return @($nodeLocal.FullName, $npmLocal)
}

function Find-MediaTool([string]$Name) {
    $found = Find-CommandPath ($Name + '.exe')
    if ($found) { return $found }
    $local = Join-Path (Join-Path $Runtime 'ffmpeg\bin') ($Name + '.exe')
    if (Test-Path $local) { return $local }
    return $null
}

function Ensure-Ffmpeg {
    $ffmpeg = Find-MediaTool 'ffmpeg'
    $ffprobe = Find-MediaTool 'ffprobe'
    if ($ffmpeg -and $ffprobe) { return @($ffmpeg, $ffprobe) }
    if ($CheckOnly) { Fail 'Не найдены ffmpeg.exe и ffprobe.exe. Запустите start.ps1 для установки или поставьте FFmpeg с https://ffmpeg.org/download.html.' }
    Invoke-WingetUserInstall 'Gyan.FFmpeg.Shared' 'FFmpeg (ffmpeg и ffprobe)'
    Refresh-ProcessPath
    $ffmpeg = Find-MediaTool 'ffmpeg'
    $ffprobe = Find-MediaTool 'ffprobe'
    if (-not $ffmpeg -or -not $ffprobe) { Fail 'FFmpeg установлен, но ffmpeg.exe/ffprobe.exe не найдены в PATH. Добавьте папку bin FFmpeg в PATH текущего пользователя и повторите запуск.' }
    return @($ffmpeg, $ffprobe)
}

function Ensure-PostprocessVcs {
    $git = Find-CommandPath 'git.exe'
    if (-not $git) {
        if ($CheckOnly) { Fail 'Для русской постобработки нужны Git и Git LFS. Запустите start.ps1 -InstallPostprocess, чтобы установить их.' }
        Invoke-WingetUserInstall 'Git.Git' 'Git for Windows'
        $git = Find-CommandPath 'git.exe'
    }
    if (-not $git) { Fail 'Git не найден после установки. Установите Git for Windows и повторите: start.ps1 -InstallPostprocess.' }
    $oldPreference = $ErrorActionPreference
    try { $ErrorActionPreference = 'Continue'; & $git lfs version *> $null; $lfsAvailable = ($LASTEXITCODE -eq 0) }
    finally { $ErrorActionPreference = $oldPreference }
    if (-not $lfsAvailable) {
        if ($CheckOnly) { Fail 'Для русской постобработки нужен Git LFS. Запустите start.ps1 -InstallPostprocess, чтобы установить его.' }
        Invoke-WingetUserInstall 'GitHub.GitLFS' 'Git LFS'
        Refresh-ProcessPath
        $git = Find-CommandPath 'git.exe'
        if (-not $git) { Fail 'Git не виден в PATH после установки. Перезапустите start.ps1 -InstallPostprocess.' }
        $oldPreference = $ErrorActionPreference
        try { $ErrorActionPreference = 'Continue'; & $git lfs version *> $null; $lfsAvailable = ($LASTEXITCODE -eq 0) }
        finally { $ErrorActionPreference = $oldPreference }
        if (-not $lfsAvailable) { Fail 'Git LFS не найден после установки. Установите Git LFS и повторите: start.ps1 -InstallPostprocess.' }
    }
}

function Get-WhisperExecutable {
    if (Test-Path $WhisperHome) {
        $existing = Get-ChildItem -LiteralPath $WhisperHome -Filter 'whisper-cli.exe' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $existing) { $existing = Get-ChildItem -LiteralPath $WhisperHome -Filter 'main.exe' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 }
        if ($existing) {
            if (-not (Test-WhisperNative $existing.FullName)) { Fail "whisper.cpp не прошёл проверку запуска. Установите Microsoft Visual C++ 2015-2022 x64 с https://aka.ms/vs/17/release/vc_redist.x64.exe. Если runtime уже установлен, обновите драйвер видеокарты с поддержкой Vulkan от NVIDIA, AMD или производителя устройства." }
            return $existing.FullName
        }
    }
    if ($CheckOnly) { Fail 'Локальный whisper.cpp Vulkan runtime отсутствует. Запустите start.ps1 для загрузки.' }
    New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
    $archive = Join-Path $Runtime 'whisper-v1.8.4-windows-vulkan-x64.zip'
    $staging = Join-Path $Runtime 'whisper-vulkan.extracting'
    Assert-RuntimePath $archive
    Assert-RuntimePath $staging
    Assert-RuntimePath $WhisperHome
    Write-Host 'Загружаю закреплённый whisper.cpp Vulkan runtime...'
    Invoke-WebRequest -Uri $WhisperUrl -OutFile $archive
    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $WhisperSha256) { Remove-Item -LiteralPath $archive -Force; Fail "Проверка SHA-256 whisper.cpp не прошла. Ожидался $WhisperSha256, получен $actualHash. Загрузка удалена." }
    if (Test-Path -LiteralPath $staging) { Assert-RuntimePath $staging; Remove-Item -LiteralPath $staging -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $staging -Force
    $exe = Get-ChildItem -LiteralPath $staging -Filter 'whisper-cli.exe' -File -Recurse | Select-Object -First 1
    if (-not $exe) { $exe = Get-ChildItem -LiteralPath $staging -Filter 'main.exe' -File -Recurse | Select-Object -First 1 }
    if (-not $exe) { Fail 'В проверенном архиве whisper.cpp не найден whisper-cli.exe. Архив оставлен в .runtime для диагностики.' }
    if (Test-Path -LiteralPath $WhisperHome) { Assert-RuntimePath $WhisperHome; Remove-Item -LiteralPath $WhisperHome -Recurse -Force }
    Move-Item -LiteralPath $staging -Destination $WhisperHome
    Remove-Item -LiteralPath $archive -Force
    $relative = $exe.FullName.Substring($staging.Length).TrimStart('\')
    $resolvedExe = Join-Path $WhisperHome $relative
    if (-not (Test-WhisperNative $resolvedExe)) { Fail "whisper.cpp не прошёл проверку запуска. Установите Microsoft Visual C++ 2015-2022 x64 с https://aka.ms/vs/17/release/vc_redist.x64.exe. Если runtime уже установлен, обновите драйвер видеокарты с поддержкой Vulkan от NVIDIA, AMD или производителя устройства." }
    return $resolvedExe
}

function Test-WhisperNative([string]$Executable) {
    $previousPreference = $ErrorActionPreference
    try {
        # whisper.cpp writes device discovery to stderr even when --help exits successfully.
        $ErrorActionPreference = 'Continue'
        & $Executable --help *> $null
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Get-FileHashText([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-CorePackages([string]$PythonExe) {
    & $PythonExe -c "import importlib.util,sys; names=['fastapi','faster_whisper','ffmpeg','huggingface_hub','psutil','multipart','uvicorn','validators','yt_dlp']; missing=[name for name in names if importlib.util.find_spec(name) is None]; print('Missing core packages: '+', '.join(missing)) if missing else None; sys.exit(bool(missing))" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Test-PostprocessPackages([string]$PythonExe) {
    $checkPath = Join-Path $Runtime 'postprocess-check.py'
    $checkScript = @'
import importlib.metadata as metadata
import json
import sys

import razdel
import sbert_punc_case_ru
import torch
import transformers
from packaging.version import Version

distribution = metadata.distribution("sbert_punc_case_ru")
direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
commit = direct_url.get("vcs_info", {}).get("commit_id")
razdel_version = Version(metadata.version("razdel"))
matches = (
    commit == "f778dc6c63bb0ec235a220488862509810e54583"
    and razdel_version >= Version("0.5.0")
    and razdel_version < Version("1")
)
sys.exit(0 if matches else 1)
'@
    Set-Content -LiteralPath $checkPath -Value $checkScript -Encoding ascii
    try {
        & $PythonExe $checkPath *> $null
        return ($LASTEXITCODE -eq 0)
    } finally {
        Remove-Item -LiteralPath $checkPath -Force -ErrorAction SilentlyContinue
    }
}

function Get-UiFingerprint {
    $inputs = @('package.json', 'package-lock.json')
    foreach ($folder in @('src', 'public')) {
        $base = Join-Path $Root ('ui\' + $folder)
        if (Test-Path $base) {
            $inputs += Get-ChildItem -LiteralPath $base -File -Recurse | ForEach-Object { $_.FullName.Substring((Join-Path $Root 'ui').Length + 1) }
        }
    }
    $inputs += @('vite.config.ts', 'tsconfig.json', 'tsconfig.app.json', 'tsconfig.node.json', 'index.html', '.env', '.env.production', '.env.local', '.env.production.local')
    $lines = foreach ($relative in ($inputs | Sort-Object -Unique)) {
        $path = Join-Path $Root ('ui\' + $relative)
        if (Test-Path $path -PathType Leaf) { $relative + ':' + (Get-FileHashText $path) }
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($lines -join "`n"))
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Ensure-Environment {
    New-Item -ItemType Directory -Force -Path $Logs | Out-Null
    Set-ProjectCacheEnvironment
    $script:PythonSpec = Select-Python
    if (-not $CheckOnly -and -not (Test-Path (Join-Path $Root '.venv\Scripts\python.exe'))) {
        Write-Host 'Создаю локальное окружение Python...'
        Invoke-Python @('-m', 'venv', (Join-Path $Root '.venv'))
        if ($LASTEXITCODE -ne 0) { Fail 'Не удалось создать .venv. Убедитесь, что установлен полноценный Python, а не только псевдоним Microsoft Store.' }
        $script:PythonSpec = Join-Path $Root '.venv\Scripts\python.exe'
    }
    if (-not $NoBrowser -or $SetupOnly) { $nodeTools = Ensure-Node -ElectronRuntime }
    else { $nodeTools = Ensure-Node }
    $mediaTools = Ensure-Ffmpeg
    $script:WhisperExe = Get-WhisperExecutable
    $runtimeBinDirs = @((Split-Path $script:WhisperExe -Parent))
    $runtimeBinDirs += Get-ChildItem -LiteralPath $WhisperHome -Directory -Recurse -ErrorAction SilentlyContinue | Where-Object {
        Get-ChildItem -LiteralPath $_.FullName -Filter '*.dll' -File -ErrorAction SilentlyContinue | Select-Object -First 1
    } | ForEach-Object { $_.FullName }
    $runtimeBinDirs = $runtimeBinDirs | Select-Object -Unique
    $env:Path = (($runtimeBinDirs + (Split-Path $mediaTools[0] -Parent) + $env:Path) -join ';')
    $env:VT_WHISPER_CPP_EXE = $script:WhisperExe

    if ($CheckOnly) {
        if (-not (Test-Path (Join-Path $Root '.venv\Scripts\python.exe'))) { Fail 'Локальное окружение .venv ещё не создано. Запустите start.ps1 для настройки.' }
        $reqMarker = Join-Path $Runtime 'core-requirements.sha256'
        if (-not (Test-Path $reqMarker) -or (Get-Content -Raw $reqMarker).Trim() -ne (Get-FileHashText (Join-Path $Root 'requirements.txt'))) { Fail 'Python-зависимости не установлены для текущего requirements.txt. Запустите start.ps1.' }
        if (-not (Test-CorePackages (Join-Path $Root '.venv\Scripts\python.exe'))) { Fail 'В .venv отсутствуют основные Python-модули. Запустите start.ps1, чтобы восстановить зависимости.' }
        if ($script:IncludePostprocess) {
            if (Test-PostprocessPackages (Join-Path $Root '.venv\Scripts\python.exe')) { Write-Host 'Русская постобработка доступна.' }
            else { Write-Host 'Русская постобработка не установлена; запустите start.ps1 или start.ps1 -InstallPostprocess для установки.' }
        } else { Write-Host 'Проверка русской постобработки пропущена по -SkipPostprocess.' }
        if (-not (Test-Path (Join-Path $Root 'ui\dist\index.html'))) { Fail 'Production UI ещё не собран. Запустите start.ps1.' }
        if (-not $NoBrowser) {
            $electronCli = Join-Path $Root 'desktop\node_modules\electron\cli.js'
            $electronExe = Join-Path $Root 'desktop\node_modules\electron\dist\electron.exe'
            $desktopLockHash = Get-FileHashText (Join-Path $Root 'desktop\package-lock.json')
            $desktopMarker = Join-Path $Runtime 'desktop-lock.sha256'
            if (-not (Test-Path $electronExe) -or -not (Test-Path $desktopMarker) -or (Get-Content -Raw $desktopMarker).Trim() -ne $desktopLockHash) { Fail 'Electron desktop зависимости ещё не установлены для текущего lock-файла. Запустите start.ps1.' }
            $savedElectronCache = $env:ELECTRON_CACHE
            $env:ELECTRON_CACHE = Join-Path $Runtime 'cache\electron'
            Push-Location (Join-Path $Root 'desktop')
            try {
                $electronVersionOutput = @(& $nodeTools[0] $electronCli '--version' 2>&1)
                $electronExitCode = $LASTEXITCODE
                $electronVersion = ($electronVersionOutput | Where-Object { $_ -match '^v\d+\.\d+\.\d+$' } | Select-Object -First 1)
                if ($electronExitCode -ne 0 -or $electronVersion -ne 'v45.2.3') { Fail 'Electron desktop зависимости повреждены или имеют неверную версию. Запустите start.ps1 для восстановления.' }
            }
            finally { Pop-Location; $env:ELECTRON_CACHE = $savedElectronCache }
        }
        Invoke-CudaRuntimePreparation -NoDownload
        Invoke-DiarizationPreparation -NoDownload
        if ($SetupOnly) {
            Invoke-RequiredModelPreparation -NoDownload
        } else {
            Write-Host 'Проверка моделей пропущена: выбранные веса будут подготовлены при запуске задачи.'
        }
        Write-Host 'Проверки окружения пройдены.'
        return
    }

    $pythonExe = Join-Path $Root '.venv\Scripts\python.exe'
    $reqPath = Join-Path $Root 'requirements.txt'
    $reqHash = Get-FileHashText $reqPath
    $markerPath = Join-Path $Runtime 'core-requirements.sha256'
    $installedHash = if (Test-Path $markerPath) { (Get-Content -Raw $markerPath).Trim() } else { '' }
    $pipCheck = $false
    if ($installedHash -eq $reqHash) {
        & $pythonExe -m pip check *> $null
        $pipCheck = ($LASTEXITCODE -eq 0)
    }
    $importsReady = Test-CorePackages $pythonExe
    if ($installedHash -ne $reqHash -or -not $pipCheck -or -not $importsReady) {
        Write-Host 'Устанавливаю/проверяю основные Python-зависимости...'
        & $pythonExe -m pip install --disable-pip-version-check -r $reqPath
        if ($LASTEXITCODE -ne 0) { Fail 'Установка Python-зависимостей не завершилась. Проверьте интернет и доступность PyPI; подробности выше.' }
        & $pythonExe -m pip check
        if ($LASTEXITCODE -ne 0) { Fail 'Python обнаружил конфликт или отсутствующую зависимость. Исправьте вывод pip check выше.' }
        Set-Content -LiteralPath $markerPath -Value $reqHash -NoNewline -Encoding ascii
    }
    if ($script:IncludePostprocess) {
        Ensure-PostprocessVcs
        $postMarker = Join-Path $Runtime 'postprocess-requirements.sha256'
        $postHash = Get-FileHashText (Join-Path $Root 'requirements-postprocess.txt')
        $postprocessReady = Test-PostprocessPackages $pythonExe
        $postprocessEnvironmentReady = $false
        if ($postprocessReady) {
            & $pythonExe -m pip check *> $null
            $postprocessEnvironmentReady = ($LASTEXITCODE -eq 0)
        }
        $postMarkerMatches = (Test-Path $postMarker) -and ((Get-Content -Raw $postMarker).Trim() -eq $postHash)
        if ($postprocessReady -and $postprocessEnvironmentReady -and -not $postMarkerMatches) {
            Set-Content -LiteralPath $postMarker -Value $postHash -NoNewline -Encoding ascii
        } elseif (-not $postprocessReady -or -not $postprocessEnvironmentReady -or -not $postMarkerMatches) {
            Write-Host 'Устанавливаю необязательную русскую пунктуацию (Git LFS требуется заранее)...'
            $previousGitLfsSkipSmudge = $env:GIT_LFS_SKIP_SMUDGE
            $env:GIT_LFS_SKIP_SMUDGE = '1'
            try {
                & $pythonExe -m pip install --disable-pip-version-check -r (Join-Path $Root 'requirements-postprocess.txt')
                if ($LASTEXITCODE -ne 0) { Fail 'Не удалось установить постобработку. Установите Git и Git LFS, затем повторите: start.ps1 -InstallPostprocess.' }
                if (-not (Test-PostprocessPackages $pythonExe)) { Fail 'Пакеты постобработки завершили установку, но Python не находит razdel и sbert_punc_case_ru.' }
                Set-Content -LiteralPath $postMarker -Value $postHash -NoNewline -Encoding ascii
            } finally {
                if ($null -eq $previousGitLfsSkipSmudge) { Remove-Item Env:\GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue }
                else { $env:GIT_LFS_SKIP_SMUDGE = $previousGitLfsSkipSmudge }
            }
        }
    }

    $ui = Join-Path $Root 'ui'
    $nodeModules = Join-Path $ui 'node_modules'
    $npmMarker = Join-Path $Runtime 'ui-lock.sha256'
    $lockHash = Get-FileHashText (Join-Path $ui 'package-lock.json')
    if (-not (Test-Path $nodeModules) -or -not (Test-Path $npmMarker) -or (Get-Content -Raw $npmMarker).Trim() -ne $lockHash) {
        Write-Host 'Устанавливаю frontend-зависимости через npm ci...'
        Push-Location $ui
        try { & $nodeTools[1] ci --no-audit; if ($LASTEXITCODE -ne 0) { Fail 'npm ci завершился с ошибкой. Проверьте доступность npm registry и повторите запуск.' } }
        finally { Pop-Location }
        Set-Content -LiteralPath $npmMarker -Value $lockHash -NoNewline -Encoding ascii
    }
    $desktopDir = Join-Path $Root 'desktop'
    $desktopLockHash = Get-FileHashText (Join-Path $desktopDir 'package-lock.json')
    $desktopMarker = Join-Path $Runtime 'desktop-lock.sha256'
    $installedDesktopHash = if (Test-Path $desktopMarker) { (Get-Content -Raw $desktopMarker).Trim() } else { '' }
    if (-not (Test-Path (Join-Path $desktopDir 'node_modules\electron\dist\electron.exe')) -or $installedDesktopHash -ne $desktopLockHash) {
        Write-Host 'Устанавливаю закреплённые Electron-зависимости в папку проекта...'
        $savedElectronCache = $env:ELECTRON_CACHE
        $env:ELECTRON_CACHE = Join-Path $Runtime 'cache\electron'
        Push-Location $desktopDir
        try {
            & $nodeTools[1] ci --no-audit
            if ($LASTEXITCODE -ne 0) { Fail 'Установка Electron через npm ci завершилась с ошибкой. Проверьте доступность npm registry и повторите запуск.' }
            Set-Content -LiteralPath $desktopMarker -Value $desktopLockHash -NoNewline -Encoding ascii
        }
        finally { Pop-Location; $env:ELECTRON_CACHE = $savedElectronCache }
    }
    $uiFingerprint = Get-UiFingerprint
    $buildMarker = Join-Path $Runtime 'ui-build.sha256'
    $distIndex = Join-Path $ui 'dist\index.html'
    if (-not (Test-Path $distIndex) -or -not (Test-Path $buildMarker) -or (Get-Content -Raw $buildMarker).Trim() -ne $uiFingerprint) {
        Write-Host 'Собираю production-интерфейс...'
        Push-Location $ui
        try { & $nodeTools[1] run build; if ($LASTEXITCODE -ne 0) { Fail 'Сборка UI завершилась с ошибкой. Проверьте сообщения npm run build выше.' } }
        finally { Pop-Location }
        if (-not (Test-Path $distIndex)) { Fail 'npm завершился успешно, но ui/dist/index.html не создан.' }
        Set-Content -LiteralPath $buildMarker -Value $uiFingerprint -NoNewline -Encoding ascii
    }
    Invoke-CudaRuntimePreparation
    Invoke-DiarizationPreparation
    if ($SetupOnly) { Invoke-RequiredModelPreparation }
}

function Get-HealthyApp {
    try {
        $response = Invoke-RestMethod -Uri ($BackendUrl + '/api/health') -TimeoutSec 2
        if ($response.status -eq 'ok' -and $response.appId -eq $AppId) { return $response }
    } catch { }
    return $null
}

function Get-DesktopHostInfo {
    try {
        $response = Invoke-RestMethod -Uri ($BackendUrl + '/api/desktop/host') -TimeoutSec 2
        if ($response.appId -eq $AppId -and $response.projectRoot -eq $Root) { return $response }
    } catch { }
    return $null
}

function Get-ListenerProcessId {
    $connection = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($connection) { return [int]$connection.OwningProcess }
    return 0
}

function Get-ProcessInfo([int]$ProcessId) {
    return Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
}

function Test-BackendCommand($ProcessInfo) {
    if (-not $ProcessInfo -or -not $ProcessInfo.CommandLine) { return $false }
    $expectedPython = Join-Path $Root '.venv\Scripts\python.exe'
    $command = $ProcessInfo.CommandLine
    return ($command.IndexOf($expectedPython, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $command.Contains('web_api:app') -and
        $command.Contains('--host 127.0.0.1') -and
        $command.Contains('--port 8000'))
}

function Test-ProcessDescendsFrom([int]$ProcessId, [int]$AncestorId) {
    $currentId = $ProcessId
    for ($depth = 0; $depth -lt 8; $depth++) {
        if ($currentId -eq $AncestorId) { return $true }
        $current = Get-ProcessInfo $currentId
        if (-not $current -or [int]$current.ParentProcessId -le 0) { return $false }
        $currentId = [int]$current.ParentProcessId
    }
    return $false
}

function Get-ProcessDescendants([int]$RootProcessId) {
    $allProcesses = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $queue = New-Object System.Collections.Queue
    $queue.Enqueue([PSCustomObject]@{ ProcessId = $RootProcessId; Depth = 0 })
    $descendants = New-Object System.Collections.ArrayList
    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        foreach ($child in ($allProcesses | Where-Object { [int]$_.ParentProcessId -eq [int]$current.ProcessId })) {
            $entry = [PSCustomObject]@{ ProcessId = [int]$child.ProcessId; Depth = [int]$current.Depth + 1 }
            [void]$descendants.Add($entry)
            $queue.Enqueue($entry)
        }
    }
    return @($descendants | Sort-Object Depth -Descending)
}

function Stop-VerifiedProcessTree([int]$RootProcessId) {
    $rootProcess = Get-ProcessInfo $RootProcessId
    $expectedPython = Join-Path $Root '.venv\Scripts\python.exe'
    if (-not $rootProcess -or $rootProcess.ExecutablePath -ine $expectedPython -or -not (Test-BackendCommand $rootProcess)) { return $false }
    $descendants = Get-ProcessDescendants $RootProcessId
    foreach ($child in $descendants) {
        Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $RootProcessId -Force -ErrorAction SilentlyContinue
    return $true
}

function Write-LauncherPid([int]$LauncherProcessId, [int]$BackendProcessId) {
    $record = [PSCustomObject]@{ launcherPid = $LauncherProcessId; backendPid = $BackendProcessId }
    Set-Content -LiteralPath $PidFile -Value ($record | ConvertTo-Json -Compress) -NoNewline -Encoding ascii
}

function Stop-LaunchedBackend {
    if (-not (Test-Path $PidFile)) { Write-Host 'PID-файл не найден; останавливать нечего.'; return }
    $pidContent = (Get-Content -Raw $PidFile).Trim()
    $launcherPid = 0
    $backendPid = 0
    try {
        $record = $pidContent | ConvertFrom-Json
        if ($record -and $record.PSObject.Properties['launcherPid']) {
            $launcherPid = [int]$record.launcherPid
            $backendPid = [int]$record.backendPid
        } elseif (-not [int]::TryParse($pidContent, [ref]$launcherPid)) {
            throw 'Invalid PID file'
        } else {
            $backendPid = $launcherPid
        }
    } catch {
        if (-not [int]::TryParse($pidContent, [ref]$launcherPid)) {
            Remove-Item -LiteralPath $PidFile -Force
            Write-Host 'Удалён некорректный PID-файл.'
            return
        }
        $backendPid = $launcherPid
    }
    if (-not (Stop-VerifiedProcessTree $launcherPid)) {
        Remove-Item -LiteralPath $PidFile -Force
        Write-Host 'PID устарел или процесс не принадлежит этому запуску; чужой процесс не остановлен.'
        return
    }
    Remove-Item -LiteralPath $PidFile -Force
    Write-Host 'Video Transcribe остановлен.'
}

function Start-Backend {
    $pythonExe = Join-Path $Root '.venv\Scripts\python.exe'
    if ($NoBrowser) {
        if (Get-DesktopHostInfo) { Write-Host 'Backend уже запущен.'; return }
        $stdout = Join-Path $Logs 'backend.out.log'
        $stderr = Join-Path $Logs 'backend.err.log'
        $process = Start-Process -FilePath $pythonExe -ArgumentList @('-m', 'services.app_host') -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $deadline = (Get-Date).AddSeconds(45)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Milliseconds 500
            if (Get-DesktopHostInfo) { Write-Host 'Backend готов: http://127.0.0.1:8000'; return }
            $process.Refresh()
            if ($process.HasExited) {
                $tail = if (Test-Path $stderr) { (Get-Content -LiteralPath $stderr -Tail 12) -join "`n" } else { '' }
                Fail "Backend не запустился. Проверьте порт 8000 и принадлежность существующего сервера Video Transcribe.`n$tail"
            }
        }
        Fail "Backend не подтвердил готовность за 45 секунд. Проверьте $stderr."
    }
    $node = Ensure-Node -ElectronRuntime
    $env:ELECTRON_CACHE = Join-Path $Runtime 'cache\electron'
    $env:npm_config_cache = Join-Path $Runtime 'cache\npm'
    $electron = Join-Path $Root 'desktop\node_modules\electron\cli.js'
    & $node[0] $electron (Join-Path $Root 'desktop')
    if ($LASTEXITCODE -ne 0) { Fail 'Desktop-приложение завершилось с ошибкой.' }
}

try {
    if ($Stop) {
        $pythonExe = Join-Path $Root '.venv\Scripts\python.exe'
        if (Test-Path $pythonExe) {
            & $pythonExe -m services.app_host --stop
            if ($LASTEXITCODE -eq 0) { Write-Host 'Video Transcribe остановлен.'; return }
        }
        Stop-LaunchedBackend
        return
    }
    Ensure-Environment
    if ($CheckOnly -or $SetupOnly) { return }
    Start-Backend
} catch {
    Write-Host "ОШИБКА: $($_.Exception.Message) (строка $($_.InvocationInfo.ScriptLineNumber))" -ForegroundColor Red
    exit 1
} finally {
    $env:PSModulePath = $OriginalPSModulePath
}
