"""Lightweight, cancellable media classification for manually selected files."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from services.task_errors import TaskFailure


def _ffprobe_path() -> str | None:
    return shutil.which("ffprobe") or shutil.which("ffprobe.exe")


def _ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def _run_media_probe(command: list[str], *, timeout: float, controller=None) -> tuple[int, bytes, bytes]:
    """Run a decoder/probe command with cancellation and bounded execution time."""
    if controller is not None:
        controller.check_cancelled()
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as exc:
        raise TaskFailure(f"Не удалось запустить средство чтения медиа: {exc}",
                          "media_decode_unavailable", stage="source") from exc
    registered = False
    try:
        if controller is not None:
            controller.register(process)
            registered = True
        deadline = time.monotonic() + max(.1, timeout)
        while True:
            if controller is not None:
                controller.check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TaskFailure("Декодирование медиа не завершилось за отведённое время.",
                                  "media_decode_failed", stage="source")
            try:
                stdout, stderr = process.communicate(timeout=min(.25, remaining))
                return process.returncode or 0, stdout or b"", stderr or b""
            except subprocess.TimeoutExpired:
                continue
    finally:
        if process.poll() is None:
            try:
                process.kill()
                process.communicate(timeout=1)
            except (OSError, subprocess.SubprocessError):
                pass
        if registered:
            controller.unregister(process)


def _decoded_streams(path: Path, streams: list[dict[str, Any]], *, kind: str,
                     controller=None) -> list[str]:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        raise TaskFailure("Не удалось проверить медиа: ffmpeg недоступен.",
                          "media_decode_unavailable", stage="source")
    usable: list[str] = []
    for stream in streams:
        index = int(stream.get("index", -1))
        if index < 0:
            raise TaskFailure("ffprobe вернул некорректный индекс медиапотока.",
                              "media_probe_failed", stage="source")
        if kind == "audio":
            command = [ffmpeg, "-hide_banner", "-nostats", "-v", "info", "-i", str(path),
                       "-map", f"0:{index}", "-vn", "-af", "astats=metadata=0:reset=0",
                       "-f", "null", "-"]
            code, _stdout, stderr = _run_media_probe(command, timeout=3600, controller=controller)
            log = stderr.decode("utf-8", "replace")
            peak = re.findall(r"Peak level dB:\s*(-?inf|[-+]?\d+(?:\.\d+)?)", log, re.IGNORECASE)
            if not peak:
                lower = log.casefold()
                if any(marker in lower for marker in ("unknown decoder", "decoder not found", "not currently supported")):
                    raise TaskFailure("Для одного из медиапотоков недоступен необходимый декодер.",
                                      "media_decode_unavailable", stage="source")
                if any(marker in lower for marker in ("invalid data found", "error while decoding", "corrupt", "invalid packet")):
                    usable.append("failed")
                    continue
                if code != 0:
                    raise TaskFailure("Не удалось декодировать звуковой поток.",
                                      "media_decode_failed", stage="source")
                raise TaskFailure("Не удалось проверить декодированные звуковые отсчёты.",
                                  "media_decode_failed", stage="source")
            if code != 0:
                usable.append("failed")
            elif any(value.casefold() != "-inf" for value in peak):
                usable.append("audible")
            else:
                usable.append("silent")
        else:
            command = [ffmpeg, "-v", "error", "-i", str(path), "-map", f"0:{index}",
                       "-frames:v", "1", "-an", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
            code, stdout, _stderr = _run_media_probe(command, timeout=120, controller=controller)
            if stdout:
                try:
                    from PIL import Image
                    import io
                    with Image.open(io.BytesIO(stdout)) as frame:
                        frame.verify()
                    usable.append("decoded")
                except Exception:
                    usable.append("failed")
            else:
                detail = _stderr.decode("utf-8", "replace").casefold()
                if any(marker in detail for marker in ("decoder not found", "unknown decoder", "not currently supported")):
                    raise TaskFailure("Для одного из медиапотоков недоступен необходимый декодер.",
                                      "media_decode_unavailable", stage="source")
                usable.append("failed")
    return usable


def probe_local_media(source: str | Path, *, controller=None, timeout: float = 30.0) -> dict[str, Any]:
    """Probe actual streams, excluding attached-picture cover art from video."""
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise TaskFailure("Исходный файл недоступен. Проверьте путь и повторите попытку.",
                          "source_unavailable", stage="source")
    if path.stat().st_size == 0:
        raise TaskFailure("Файл пустой: в нём нет ни одного байта.", "media_empty", stage="source")
    ffprobe = _ffprobe_path()
    if not ffprobe:
        raise TaskFailure("Не удалось определить тип локального файла: ffprobe недоступен.",
                          "media_probe_unavailable", stage="source")
    command = [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    if controller is not None:
        controller.check_cancelled()
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as exc:
        raise TaskFailure(f"Не удалось запустить ffprobe: {exc}", "media_probe_unavailable", stage="source") from exc
    registered = False
    try:
        if controller is not None:
            controller.register(process)
            registered = True
        deadline = time.monotonic() + max(0.1, float(timeout))
        while True:
            if controller is not None:
                controller.check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TaskFailure("ffprobe не завершил проверку файла за отведённое время.",
                                  "media_probe_failed", stage="source")
            try:
                stdout, stderr = process.communicate(timeout=min(0.25, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode != 0:
            detail = (stderr or b"").decode("utf-8", "replace").strip()
            if "invalid data found" in detail.casefold() or "end of file" in detail.casefold():
                raise TaskFailure(
                    "Не удалось распознать содержимое файла как аудио или видео. "
                    "Проверьте, что файл завершённый и имеет поддерживаемый медиаформат.",
                    "media_format_unsupported", stage="source")
            raise TaskFailure(
                "Не удалось прочитать структуру локального файла через ffprobe. "
                + (detail[-500:] if detail else "Проверьте файл и повторите попытку."),
                "media_probe_failed", stage="source")
        try:
            payload = json.loads((stdout or b"").decode("utf-8", "replace"))
            streams = payload.get("streams")
            if not isinstance(streams, list):
                raise ValueError("ffprobe returned no stream list")
            if not streams:
                raise TaskFailure(
                    "В файле не найдены аудио- или видеопотоки. Выберите файл с аудио или видео.",
                    "media_format_unsupported", stage="source")
            format_info = payload.get("format") if isinstance(payload.get("format"), dict) else {}
            format_names = str(format_info.get("format_name") or "").casefold().split(",")
            if any(name in {"image2", "image2pipe", "jpeg_pipe", "png_pipe", "webp_pipe"}
                   for name in format_names):
                raise TaskFailure(
                    "Этот файл содержит только изображение, без аудио или видео. "
                    "Выберите файл с аудио или видео.",
                    "media_format_unsupported", stage="source")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise TaskFailure("ffprobe вернул некорректные сведения о локальном файле.",
                              "media_probe_failed", stage="source") from exc
        videos = []
        audios = []
        for index, row in enumerate(streams):
            if not isinstance(row, dict):
                continue
            if row.get("codec_type") == "audio":
                audios.append({**row, "index": row.get("index", index)})
            elif row.get("codec_type") == "video":
                disposition = row.get("disposition") if isinstance(row.get("disposition"), dict) else {}
                attached = disposition.get("attached_pic")
                if attached not in (1, "1", True):
                    videos.append({**row, "index": row.get("index", index)})
        video_decoded = _decoded_streams(path, videos, kind="video", controller=controller) if videos else []
        audio_decoded = _decoded_streams(path, audios, kind="audio", controller=controller) if audios else []
        has_video = any(result == "decoded" for result in video_decoded)
        has_audio = any(result == "audible" for result in audio_decoded)
        if not has_video and not has_audio:
            if audios and not videos and audio_decoded and all(result == "silent" for result in audio_decoded):
                raise TaskFailure(
                    "Звуковая дорожка декодирована, но содержит только цифровую тишину. "
                    "Выберите файл с записанным звуком.",
                    "audio_silent", stage="source")
            if not audios and not videos:
                raise TaskFailure(
                    "В файле нет пригодного аудио- или видеопотока. Выберите медиафайл поддерживаемого формата.",
                    "media_format_unsupported", stage="source")
            raise TaskFailure("Не удалось декодировать аудио- или видеопотоки. Проверьте файл и повторите попытку.",
                              "media_decode_failed", stage="source")
        return {"video": videos, "audio": audios, "isVideo": has_video,
                "hasAudio": has_audio, "streams": streams,
                "decodedVideoStreams": [stream for stream, result in zip(videos, video_decoded) if result == "decoded"],
                "decodedAudioStreams": [stream for stream, result in zip(audios, audio_decoded) if result == "audible"]}
    finally:
        if process.poll() is None:
            try:
                process.kill()
                process.communicate(timeout=1)
            except (OSError, subprocess.SubprocessError):
                pass
        if registered:
            controller.unregister(process)
