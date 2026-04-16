param(
    [Parameter(Mandatory = $true)]
    [string]$PackagingDir,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Get-Sha256 {
    param([string]$Path)
    return ((Get-FileHash -Algorithm SHA256 -Path $Path).Hash).ToLowerInvariant()
}

$resolvedPackaging = (Resolve-Path $PackagingDir).Path
$manifestPath = Join-Path $resolvedPackaging "runtime-binaries.json"
$outputPath = (New-Item -ItemType Directory -Force -Path $OutputDir).FullName
$downloadPath = Join-Path $outputPath "downloads"
$binPath = Join-Path $outputPath "bin"

New-Item -ItemType Directory -Force -Path $downloadPath, $binPath | Out-Null

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

$ytPath = Join-Path $downloadPath $manifest.ytDlp.filename
Invoke-WebRequest -Uri $manifest.ytDlp.url -OutFile $ytPath
if ((Get-Sha256 $ytPath) -ne $manifest.ytDlp.sha256.ToLowerInvariant()) {
    throw "SHA256 mismatch for yt-dlp.exe"
}
Copy-Item -Force $ytPath (Join-Path $binPath "yt-dlp.exe")

$denoArchive = Join-Path $downloadPath $manifest.deno.archiveName
Invoke-WebRequest -Uri $manifest.deno.archiveUrl -OutFile $denoArchive
if ((Get-Sha256 $denoArchive) -ne $manifest.deno.archiveSha256.ToLowerInvariant()) {
    throw "SHA256 mismatch for deno archive"
}

$denoExtractRoot = Join-Path $downloadPath "deno-extracted"
if (Test-Path $denoExtractRoot) {
    Remove-Item -Recurse -Force $denoExtractRoot
}
Expand-Archive -Path $denoArchive -DestinationPath $denoExtractRoot

$denoSource = Join-Path $denoExtractRoot $manifest.deno.denoRelativePath
if (!(Test-Path $denoSource)) {
    throw "Extracted deno payload does not contain deno.exe"
}

Copy-Item -Force $denoSource (Join-Path $binPath "deno.exe")

$ffmpegArchive = Join-Path $downloadPath $manifest.ffmpeg.archiveName
Invoke-WebRequest -Uri $manifest.ffmpeg.archiveUrl -OutFile $ffmpegArchive
if ((Get-Sha256 $ffmpegArchive) -ne $manifest.ffmpeg.archiveSha256.ToLowerInvariant()) {
    throw "SHA256 mismatch for ffmpeg archive"
}

$extractRoot = Join-Path $downloadPath "ffmpeg-extracted"
if (Test-Path $extractRoot) {
    Remove-Item -Recurse -Force $extractRoot
}
Expand-Archive -Path $ffmpegArchive -DestinationPath $extractRoot

$ffmpegSource = Join-Path $extractRoot $manifest.ffmpeg.ffmpegRelativePath
$ffprobeSource = Join-Path $extractRoot $manifest.ffmpeg.ffprobeRelativePath

if (!(Test-Path $ffmpegSource) -or !(Test-Path $ffprobeSource)) {
    throw "Extracted ffmpeg payload does not contain ffmpeg.exe and ffprobe.exe"
}

Copy-Item -Force $ffmpegSource (Join-Path $binPath "ffmpeg.exe")
Copy-Item -Force $ffprobeSource (Join-Path $binPath "ffprobe.exe")
