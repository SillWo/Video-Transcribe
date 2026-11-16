"""Lightweight archive image readers without OpenCV or NumPy imports."""
from __future__ import annotations

import io
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image


def _executable(name: str) -> str | None:
    return shutil.which(name) or shutil.which(name + ".exe")


def _inside(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Image path must stay inside the meeting directory")
    return candidate


def frame_image(meeting_dir: str | Path, event: dict[str, Any],
                crop: list[int] | tuple[int, int, int, int] | None = None,
                max_width: int = 1920) -> bytes:
    root = Path(meeting_dir).resolve()
    relative = Path(str(event.get("frame", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("frame path must stay inside the meeting directory")
    path = _inside(root, root / relative)
    with Image.open(path) as source:
        image = source.convert("RGB")
    if crop is not None:
        if len(crop) != 4:
            raise ValueError("crop must be [x, y, width, height]")
        x, y, width, height = [int(value) for value in crop]
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image.width or y + height > image.height:
            raise ValueError("crop is outside the frame")
        image = image.crop((x, y, x + width, y + height))
    if max_width > 0 and image.width > max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def probe_media(source: str | Path) -> dict[str, Any]:
    ffprobe = _executable("ffprobe")
    if not ffprobe:
        return {"duration": None, "video": [], "audio": [], "error": "ffprobe unavailable"}
    try:
        result = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)],
                                capture_output=True, check=False, timeout=30, shell=False)
        if result.returncode:
            return {"duration": None, "video": [], "audio": [], "error": result.stderr.decode("utf-8", "replace").strip()}
        payload = json.loads(result.stdout.decode("utf-8", "replace"))
        streams = payload.get("streams", [])

        def number(value: Any) -> float | None:
            try:
                result_value = float(value)
                return result_value if math.isfinite(result_value) else None
            except (TypeError, ValueError):
                return None

        def brief(stream: dict[str, Any], index: int) -> dict[str, Any]:
            return {"index": index, "codec": stream.get("codec_name"), "start_time": number(stream.get("start_time")),
                    "duration": number(stream.get("duration")), "width": stream.get("width"), "height": stream.get("height"),
                    "sample_rate": stream.get("sample_rate"), "channels": stream.get("channels"),
                    "language": stream.get("tags", {}).get("language")}

        duration = number(payload.get("format", {}).get("duration"))
        if duration is None:
            duration = max((value for value in (number(stream.get("duration")) for stream in streams) if value is not None), default=None)
        return {"duration": duration,
                "video": [brief(stream, i) for i, stream in enumerate(streams) if stream.get("codec_type") == "video"],
                "audio": [brief(stream, i) for i, stream in enumerate(streams) if stream.get("codec_type") == "audio"]}
    except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError) as exc:
        return {"duration": None, "video": [], "audio": [], "error": str(exc)}


def extract_episode_frames(source: str | Path, start: float, end: float, *,
                           step: float = 1.0, limit: int = 8,
                           max_width: int | None = 1920) -> list[dict[str, Any]]:
    """Decode a bounded sequence via FFmpeg pipes, without OpenCV/NumPy."""
    ffmpeg = _executable("ffmpeg")
    if not ffmpeg:
        return []
    start = max(0.0, float(start))
    end = max(0.0, float(end))
    step = max(.1, float(step))
    limit = max(0, min(8, int(limit)))
    if end <= start or limit == 0:
        return []
    results: list[dict[str, Any]] = []
    second = start
    while second < end and len(results) < limit:
        command = [ffmpeg, "-v", "error", "-ss", f"{second:.3f}", "-i", str(source), "-map", "0:v:0", "-frames:v", "1",
                   "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
        try:
            decoded = subprocess.run(command, capture_output=True, check=False, timeout=30, shell=False)
        except (OSError, subprocess.TimeoutExpired):
            break
        if decoded.returncode or not decoded.stdout:
            break
        try:
            with Image.open(io.BytesIO(decoded.stdout)) as image_file:
                image = image_file.convert("RGB")
            source_width, source_height = image.size
            if max_width is not None and max_width > 0 and image.width > max_width:
                height = max(1, round(image.height * max_width / image.width))
                image = image.resize((max_width, height), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85)
            results.append({"time": round(second, 3), "image": output.getvalue(),
                            "width": int(image.width), "height": int(image.height),
                            "source_width": int(source_width), "source_height": int(source_height)})
        except (OSError, ValueError):
            break
        second += step
    return results
