[CmdletBinding()]
param(
    [Parameter()]
    [string]$LinkPath
)

$ErrorActionPreference = 'Stop'

function Stop-WithMessage([string]$Message) {
    [Console]::Error.WriteLine("Ошибка: $Message")
    exit 1
}

try {
    $projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $electronPath = Join-Path $projectRoot 'desktop\node_modules\electron\dist\electron.exe'
    $desktopAppPath = Join-Path $projectRoot 'desktop'
    $uiIndexPath = Join-Path $projectRoot 'ui\dist\index.html'
    $uiAssetsPath = Join-Path $projectRoot 'ui\dist\assets'

    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        Stop-WithMessage "Не найден Python приложения: $pythonPath. Сначала запустите start.ps1 для установки." 
    }
    if (-not (Test-Path -LiteralPath $electronPath -PathType Leaf)) {
        Stop-WithMessage "Не найден Electron: $electronPath. Сначала запустите start.ps1 для установки." 
    }
    if (-not (Test-Path -LiteralPath $uiIndexPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $uiAssetsPath -PathType Container)) {
        Stop-WithMessage "Не собран интерфейс приложения: $uiIndexPath или папка assets отсутствует. Запустите start.ps1 и дождитесь сборки." 
    }

    if ([string]::IsNullOrWhiteSpace($LinkPath)) {
        $desktopDirectory = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
        if ([string]::IsNullOrWhiteSpace($desktopDirectory) -or
            -not (Test-Path -LiteralPath $desktopDirectory -PathType Container)) {
            Stop-WithMessage 'Не удалось определить папку рабочего стола текущего пользователя.'
        }
        $LinkPath = Join-Path $desktopDirectory 'Video Transcribe — встречи.lnk'
    } else {
        $LinkPath = [System.IO.Path]::GetFullPath($LinkPath)
        if ([System.IO.Path]::GetExtension($LinkPath) -ine '.lnk') {
            Stop-WithMessage 'Параметр -LinkPath должен указывать на файл с расширением .lnk.'
        }
    }

    $linkDirectory = [System.IO.Path]::GetDirectoryName($LinkPath)
    if (-not (Test-Path -LiteralPath $linkDirectory -PathType Container)) {
        Stop-WithMessage "Папка для ярлыка не существует: $linkDirectory"
    }

    # Electron receives the desktop app directory as one quoted argument. The
    # project root is the working directory used by the desktop host bootstrap.
    $arguments = '"' + $desktopAppPath + '"'
    if (Test-Path -LiteralPath $LinkPath -PathType Leaf) {
        $existing = (New-Object -ComObject WScript.Shell).CreateShortcut($LinkPath)
        $sameTarget = [string]::Equals(
            [System.IO.Path]::GetFullPath($existing.TargetPath),
            [System.IO.Path]::GetFullPath($electronPath),
            [System.StringComparison]::OrdinalIgnoreCase)
        if (-not $sameTarget -or $existing.Arguments -cne $arguments) {
            Stop-WithMessage "Файл уже существует и не является ярлыком Video Transcribe; он оставлен без изменений: $LinkPath"
        }
    }

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($LinkPath)
    $shortcut.TargetPath = $electronPath
    $shortcut.Arguments = $arguments
    $shortcut.WorkingDirectory = $projectRoot
    # electron.exe contains a valid Windows icon resource; the UI's tray.png
    # cannot be used directly by a .lnk without converting it to .ico.
    $shortcut.IconLocation = "$electronPath,0"
    $shortcut.Description = 'Video Transcribe — архив и обработка встреч'
    $shortcut.WindowStyle = 1
    $shortcut.Save()

    Write-Output "Ярлык создан: $LinkPath"
} catch {
    Stop-WithMessage "Не удалось создать ярлык: $($_.Exception.Message)"
}
