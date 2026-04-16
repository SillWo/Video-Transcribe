param(
    [string]$InstallerUrl = "https://github.com/jrsoftware/issrc/releases/download/is-6_7_1/innosetup-6.7.1.exe",
    [string]$ExpectedSha256 = "4D11E8050B6185E0D49BD9E8CC661A7A59F44959A621D31D11033124C4E8A7B0",
    [string]$DownloadDir = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if ([string]::IsNullOrWhiteSpace($DownloadDir)) {
    $DownloadDir = Join-Path ([System.IO.Path]::GetTempPath()) "inno-setup"
}

New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
$installerPath = Join-Path $DownloadDir "innosetup-6.7.1.exe"

Invoke-WebRequest -Uri $InstallerUrl -OutFile $installerPath
$actualHash = (Get-FileHash -Algorithm SHA256 -Path $installerPath).Hash
if ($actualHash -ne $ExpectedSha256) {
    throw "SHA256 mismatch for Inno Setup installer"
}

Start-Process -FilePath $installerPath -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-" -Wait

$candidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)

$isccPath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (!$isccPath) {
    throw "ISCC.exe was not found after installing Inno Setup"
}

Write-Output $isccPath
