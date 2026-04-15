from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

try:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError, ExtractorError, UnsupportedError
except ImportError:  # pragma: no cover - handled at runtime
    YoutubeDL = None
    DownloadError = Exception
    ExtractorError = Exception
    UnsupportedError = Exception

CHECK_TIMEOUT_SECONDS = 15
PLAYLIST_PREVIEW_LIMIT = 3
FORMAT_SAMPLE_LIMIT = 5
DETAIL_LIMIT = 6
AUDIO_FILE_EXTENSIONS = {
    "aac",
    "aiff",
    "alac",
    "flac",
    "m4a",
    "mid",
    "midi",
    "mp2",
    "mp3",
    "oga",
    "ogg",
    "opus",
    "wav",
    "weba",
    "wma",
}
MEDIA_CONTAINER_EXTENSIONS = {
    "3gp",
    "avi",
    "flv",
    "m2ts",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "mts",
    "ts",
    "webm",
}


@dataclass
class SourceCheckResult:
    ok: bool
    canProcess: bool
    message: str
    details: list[str]
    sourceInfo: dict[str, Any] | None
    formatsAvailable: bool
    audioExtractable: bool
    extractor: str | None
    title: str | None
    id: str | None
    diagnosticCode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class YtDlpLogCollector:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def _append(self, message: str) -> None:
        cleaned = clean_text(message)
        if not cleaned or cleaned.startswith("[debug]"):
            return
        if cleaned not in self.messages:
            self.messages.append(cleaned)

    def debug(self, message: str) -> None:
        self._append(message)

    def warning(self, message: str) -> None:
        self._append(message)

    def error(self, message: str) -> None:
        self._append(message)


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    while text.startswith("ERROR: "):
        text = text[len("ERROR: ") :]
    return " ".join(text.split())


def append_detail(details: list[str], value: Any) -> None:
    cleaned = clean_text(value)
    if not cleaned or cleaned in details:
        return
    details.append(cleaned)


def is_supported_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def ydl_options(logger: YtDlpLogCollector) -> dict[str, Any]:
    return {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "logger": logger,
        "socket_timeout": CHECK_TIMEOUT_SECONDS,
        "source_address": "0.0.0.0",
        "geo_bypass": True,
        "extract_flat": False,
        "playlistend": PLAYLIST_PREVIEW_LIMIT,
        "ignoreerrors": False,
        "noprogress": True,
        "cachedir": False,
    }


def get_formats(info: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not info:
        return []

    formats = [fmt for fmt in (info.get("formats") or []) if isinstance(fmt, dict)]
    if formats:
        return formats

    if info.get("url"):
        return [info]

    return []


def inspect_audio_support(format_info: dict[str, Any]) -> tuple[bool, bool]:
    acodec = str(format_info.get("acodec") or "").lower()
    audio_ext = str(format_info.get("audio_ext") or "").lower()
    resolution = str(format_info.get("resolution") or "").lower()
    ext = str(format_info.get("ext") or "").lower()
    protocol = str(format_info.get("protocol") or "").lower()

    if acodec not in {"", "none"}:
        return True, False

    if audio_ext not in {"", "none"}:
        return True, False

    if resolution == "audio only":
        return True, False

    if ext in AUDIO_FILE_EXTENSIONS:
        return True, False

    # Generic direct-file URLs often omit codec metadata. In these cases,
    # treat common media containers as audio-extractable candidates.
    if format_info.get("url") and protocol in {"http", "https"} and ext in MEDIA_CONTAINER_EXTENSIONS:
        return True, True

    return False, False


def has_audio_stream(format_info: dict[str, Any]) -> bool:
    supported, _ = inspect_audio_support(format_info)
    return supported


def summarize_format(format_info: dict[str, Any]) -> str:
    parts: list[str] = []
    audio_supported, audio_inferred = inspect_audio_support(format_info)
    format_id = format_info.get("format_id")
    if format_id:
        parts.append(str(format_id))

    ext = format_info.get("ext")
    if ext:
        parts.append(str(ext))

    resolution = format_info.get("resolution")
    if resolution and resolution != "audio only":
        parts.append(str(resolution))
    elif format_info.get("height"):
        parts.append(f"{format_info['height']}p")

    abr = format_info.get("abr")
    if abr:
        try:
            parts.append(f"{int(float(abr))} kbps")
        except (TypeError, ValueError):
            parts.append(f"{abr} kbps")

    if audio_supported:
        acodec = format_info.get("acodec")
        if acodec not in {None, "", "none"}:
            parts.append(f"audio:{acodec}")
        elif audio_inferred:
            parts.append("audio:likely")
        else:
            parts.append("audio:available")
    elif format_info.get("vcodec") not in {None, "", "none"}:
        parts.append(f"video:{format_info.get('vcodec')}")

    if format_info.get("protocol"):
        parts.append(str(format_info["protocol"]))

    return " | ".join(parts) if parts else "format available"


def collect_playlist_entries(info: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for entry in info.get("entries") or []:
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def build_source_info(url: str, info: dict[str, Any], details: list[str]) -> SourceCheckResult:
    root_entries = collect_playlist_entries(info)
    representative = next(
        (
            entry
            for entry in root_entries
            if get_formats(entry) and any(has_audio_stream(fmt) for fmt in get_formats(entry))
        ),
        None,
    )
    if representative is None:
        representative = next((entry for entry in root_entries if get_formats(entry)), None)
    if representative is None:
        representative = next((entry for entry in root_entries if entry), None)
    if representative is None:
        representative = info

    formats = get_formats(representative)
    formats_available = bool(formats)
    audio_flags = [inspect_audio_support(fmt) for fmt in formats]
    audio_extractable = any(flag[0] for flag in audio_flags) or inspect_audio_support(representative)[0]
    inferred_audio = any(flag[1] for flag in audio_flags) or inspect_audio_support(representative)[1]
    can_process = True

    is_playlist = bool(root_entries) or info.get("_type") == "playlist"
    entry_count = info.get("playlist_count") or (len(root_entries) if root_entries else None)
    checked_entries = len(root_entries) if root_entries else None
    extractor = (
        representative.get("extractor_key")
        or representative.get("extractor")
        or info.get("extractor_key")
        or info.get("extractor")
    )
    title = representative.get("title") or info.get("title")
    source_id = representative.get("id") or info.get("id")
    site = representative.get("webpage_url_domain") or info.get("webpage_url_domain") or extractor
    format_sample = [summarize_format(fmt) for fmt in formats[:FORMAT_SAMPLE_LIMIT]]
    audio_formats_count = sum(1 for supported, _ in audio_flags if supported)

    source_info = {
        "title": title,
        "id": source_id,
        "extractor": representative.get("extractor") or info.get("extractor"),
        "extractorKey": representative.get("extractor_key") or info.get("extractor_key"),
        "site": site,
        "webpageUrl": representative.get("webpage_url") or info.get("webpage_url") or url,
        "kind": info.get("_type") or "video",
        "isPlaylist": is_playlist,
        "playlistTitle": info.get("title") if is_playlist else None,
        "entryCount": entry_count,
        "checkedEntries": checked_entries,
        "formatsCount": len(formats),
        "audioFormatsCount": audio_formats_count,
        "audioExtractable": audio_extractable,
        "availability": representative.get("availability") or info.get("availability"),
        "formatSample": format_sample,
    }

    if is_playlist:
        checked_text = checked_entries if checked_entries is not None else 0
        append_detail(details, f"Playlist source detected. Checked up to {checked_text} entries.")

    if formats_available:
        append_detail(details, f"Found {len(formats)} format(s); {audio_formats_count} with audio streams.")

    if inferred_audio:
        inferred_ext = representative.get("ext") or "media"
        append_detail(
            details,
            f"Audio extractability was inferred from the direct .{inferred_ext} media URL because codec metadata was unavailable.",
        )

    if source_info["availability"]:
        append_detail(details, f"Availability: {source_info['availability']}")

    diagnostic_code = "ok"
    if not formats_available:
        diagnostic_code = "no_formats"
        message = "yt-dlp recognized the source, but no downloadable formats were returned."
    elif not audio_extractable:
        diagnostic_code = "no_audio_formats"
        message = "yt-dlp found formats, but none of them expose an audio stream for transcription."
    elif inferred_audio:
        message = "yt-dlp extracted metadata and the source looks suitable for audio extraction."
    else:
        message = "yt-dlp extracted metadata and found audio-capable formats."

    return SourceCheckResult(
        ok=can_process and formats_available and audio_extractable,
        canProcess=can_process,
        message=message,
        details=details[:DETAIL_LIMIT],
        sourceInfo=source_info,
        formatsAvailable=formats_available,
        audioExtractable=audio_extractable,
        extractor=extractor,
        title=title,
        id=source_id,
        diagnosticCode=diagnostic_code,
    )


def classify_error(url: str, exc: Exception, logger: YtDlpLogCollector) -> SourceCheckResult:
    details: list[str] = []
    append_detail(details, exc)
    for item in logger.messages:
        append_detail(details, item)

    lower = " ".join([clean_text(exc), *[clean_text(item) for item in logger.messages]]).lower()
    diagnostic_code = "check_failed"
    message = "yt-dlp could not validate this URL."

    if not is_supported_url(url):
        diagnostic_code = "invalid_url"
        message = "Enter a valid HTTP or HTTPS URL."
    elif "no supported javascript runtime could be found" in lower:
        diagnostic_code = "js_runtime_missing"
        message = "yt-dlp needs a JavaScript runtime in this environment to inspect this source reliably."
    elif isinstance(exc, TimeoutError) or "timed out" in lower or "timeout" in lower:
        diagnostic_code = "timeout"
        message = "The source check timed out before yt-dlp could finish."
    elif isinstance(exc, UnsupportedError) or "unsupported url" in lower or "no suitable extractor" in lower:
        diagnostic_code = "unsupported_url"
        message = "This URL is not supported by yt-dlp."
    elif "private" in lower or "members-only" in lower or "login" in lower or "sign in" in lower:
        diagnostic_code = "access_restricted"
        message = "The source is private, restricted, or requires authentication."
    elif "requested format is not available" in lower or "no formats" in lower:
        diagnostic_code = "no_formats"
        message = "yt-dlp recognized the source, but no formats are currently available."
    elif "video unavailable" in lower or "content unavailable" in lower or "not available" in lower:
        diagnostic_code = "unavailable"
        message = "The source is unavailable or cannot be accessed."
    elif (
        "http error" in lower
        or "urlopen error" in lower
        or "connection" in lower
        or "network" in lower
        or "temporarily unavailable" in lower
    ):
        diagnostic_code = "network_error"
        message = "Network error while contacting the source."

    return SourceCheckResult(
        ok=False,
        canProcess=False,
        message=message,
        details=details[:DETAIL_LIMIT],
        sourceInfo=None,
        formatsAvailable=False,
        audioExtractable=False,
        extractor=None,
        title=None,
        id=None,
        diagnosticCode=diagnostic_code,
    )


def check_source(url: str) -> dict[str, Any]:
    normalized_url = (url or "").strip()
    if not normalized_url:
        return SourceCheckResult(
            ok=False,
            canProcess=False,
            message="Enter a URL to check.",
            details=[],
            sourceInfo=None,
            formatsAvailable=False,
            audioExtractable=False,
            extractor=None,
            title=None,
            id=None,
            diagnosticCode="empty_url",
        ).to_dict()

    if not is_supported_url(normalized_url):
        return SourceCheckResult(
            ok=False,
            canProcess=False,
            message="Enter a valid HTTP or HTTPS URL.",
            details=[],
            sourceInfo=None,
            formatsAvailable=False,
            audioExtractable=False,
            extractor=None,
            title=None,
            id=None,
            diagnosticCode="invalid_url",
        ).to_dict()

    if YoutubeDL is None:
        return SourceCheckResult(
            ok=False,
            canProcess=False,
            message="yt-dlp is not installed in this Python environment.",
            details=["Install the Python package yt-dlp or make it available to the backend runtime."],
            sourceInfo=None,
            formatsAvailable=False,
            audioExtractable=False,
            extractor=None,
            title=None,
            id=None,
            diagnosticCode="yt_dlp_missing",
        ).to_dict()

    logger = YtDlpLogCollector()

    try:
        with YoutubeDL(ydl_options(logger)) as ydl:
            info = ydl.extract_info(normalized_url, download=False)

        if not isinstance(info, dict):
            return SourceCheckResult(
                ok=False,
                canProcess=False,
                message="yt-dlp did not return structured metadata for this URL.",
                details=[],
                sourceInfo=None,
                formatsAvailable=False,
                audioExtractable=False,
                extractor=None,
                title=None,
                id=None,
                diagnosticCode="no_metadata",
            ).to_dict()

        details: list[str] = []
        for item in logger.messages:
            append_detail(details, item)

        return build_source_info(normalized_url, info, details).to_dict()
    except (DownloadError, ExtractorError, UnsupportedError, TimeoutError) as exc:
        return classify_error(normalized_url, exc, logger).to_dict()
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return classify_error(normalized_url, exc, logger).to_dict()
