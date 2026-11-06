"""Adapter for the Vulkan-enabled whisper.cpp command line runtime."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

from services.model_paths import HF_HUB_CACHE
from services.download_progress import create_download_progress_tqdm
from services.runtime_devices import VULKAN_MODELS, find_whisper_cpp_executable
from services.task_errors import TaskFailure, is_invalid_source_error, is_out_of_memory_error

MODEL_REPO = os.environ.get("VT_WHISPER_CPP_MODEL_REPO", "ggerganov/whisper.cpp")
MODEL_CACHE = HF_HUB_CACHE
_DLL_DIRECTORY_HANDLES = []


@dataclass
class WhisperCppSegment:
    start: float
    end: float
    text: str


def model_filename(model_name: str) -> str:
    if model_name not in VULKAN_MODELS:
        if model_name.startswith("distil-"):
            raise ValueError(f"Model '{model_name}' is not supported by the Vulkan backend. Select a standard Whisper model.")
        raise ValueError(f"Unsupported Vulkan model '{model_name}'. Supported models: {', '.join(VULKAN_MODELS)}")
    normalized = "large-v3" if model_name == "large" else model_name
    return f"ggml-{normalized}.bin"


def resolve_model_path(model_name: str, *, progress=None, byte_progress=None) -> Path:
    try:
        filename = model_filename(model_name)
    except ValueError as exc:
        raise TaskFailure(str(exc), "model_unavailable", stage="preparation") from exc
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except ImportError as exc:
        raise TaskFailure("Install huggingface_hub to download Vulkan Whisper models.",
                          "dependency_missing", stage="preparation") from exc
    cached_path = try_to_load_from_cache(repo_id=MODEL_REPO, filename=filename, cache_dir=str(MODEL_CACHE))
    if isinstance(cached_path, str) and Path(cached_path).is_file():
        if byte_progress:
            cached_size = Path(cached_path).stat().st_size
            byte_progress(cached_size, cached_size)
        if progress:
            progress(f"Using cached Vulkan model {model_name}.")
        return Path(cached_path)
    if progress:
        progress(f"Downloading Vulkan model {model_name} from {MODEL_REPO} (starting at 10%).")
    progress_tqdm = create_download_progress_tqdm(
        initial_percent=10,
        callback=(lambda _percent, message: progress(message)) if progress else None,
        byte_callback=byte_progress,
        label=f"Downloading Vulkan model {model_name}",
    )
    try:
        # HF handles locking, resume, and atomic placement in its shared cache.
        cached = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            cache_dir=str(MODEL_CACHE),
            tqdm_class=progress_tqdm,
        )
    except Exception as exc:
        raise TaskFailure(f"Could not download Vulkan model '{model_name}': {exc}",
                          "model_download_failed", stage="preparation") from exc
    path = Path(cached)
    if not path.is_file():
        raise TaskFailure(f"Hugging Face did not return the Vulkan model file '{filename}'.",
                          "model_unavailable", stage="preparation")
    if progress:
        progress(f"Vulkan model {model_name} is ready.")
    return path


def _timestamp_ms(value) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    value = str(value).strip()
    if value.endswith("ms"):
        value = value[:-2]
    if ":" in value:
        hours, minutes, seconds = value.replace(",", ".").split(":")
        return int(round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000))
    return int(float(value))


def parse_whisper_cpp_json(payload: dict) -> tuple[list[WhisperCppSegment], SimpleNamespace]:
    rows = payload.get("transcription")
    if not isinstance(rows, list):
        raise RuntimeError("whisper.cpp returned malformed JSON: missing transcription array.")
    segments = []
    for index, row in enumerate(rows):
        try:
            # whisper.cpp's JSON writer uses `offsets` in milliseconds and
            # `timestamps` in HH:MM:SS,mmm; accept either shape.
            stamps = row.get("offsets") or row["timestamps"]
            start = _timestamp_ms(stamps["from"]) / 1000
            end = _timestamp_ms(stamps["to"]) / 1000
            text = str(row["text"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"whisper.cpp returned malformed segment {index}.") from exc
        segments.append(WhisperCppSegment(start, end, text))
    language = payload.get("result", {}).get("language") or "unknown"
    return segments, SimpleNamespace(language=language, language_probability=None)


def _find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")
    if not exe:
        raise TaskFailure("FFmpeg is required to prepare audio for Vulkan transcription. Install it and restart the app.",
                          "dependency_missing", stage="source")
    return exe


def _add_runtime_dll_path(executable: str) -> None:
    if os.name != "nt":
        return
    folder = Path(executable).parent
    if hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(folder)))
    os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")


class WhisperCppModel:
    """whisper.cpp adapter; `transcription_progress` accepts progress dictionaries during native inference."""
    def __init__(self, model_name: str, *, threads: int = 1, progress=None, gpu_index: int | None = None,
                 transcription_progress=None):
        self.executable = find_whisper_cpp_executable()
        if not self.executable:
            raise TaskFailure("Vulkan Whisper runtime is not installed. Run the application's GPU setup and restart it.",
                              "device_unavailable", stage="recognition")
        self.model_path = resolve_model_path(model_name, progress=progress)
        self.threads = max(1, int(threads))
        self.model_name = model_name
        self.gpu_index = gpu_index
        self.transcription_progress = transcription_progress
        _add_runtime_dll_path(self.executable)

    def transcribe(self, media_path: str, beam_size: int = 5, language: str | None = None):
        ffmpeg = _find_ffmpeg()
        with tempfile.TemporaryDirectory(prefix="vt-whisper-") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            output_base = Path(temp_dir) / "transcript"
            converted = subprocess.run(
                [ffmpeg, "-y", "-i", str(media_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if converted.returncode:
                detail = (converted.stderr or converted.stdout or "unknown FFmpeg error").strip()
                raise TaskFailure(f"FFmpeg could not prepare audio: {detail[-1200:]}",
                                  "source_invalid", stage="source")
            command = [self.executable, "-m", str(self.model_path), "-f", str(wav_path), "-l", language or "auto",
                       "-t", str(self.threads), "-bs", str(max(1, int(beam_size))), "-oj", "-of", str(output_base)]
            selected_index = getattr(self, "gpu_index", None)
            if selected_index is None and os.environ.get("VT_VULKAN_DEVICE_INDEX"):
                try:
                    selected_index = int(os.environ["VT_VULKAN_DEVICE_INDEX"])
                except ValueError as exc:
                    raise RuntimeError("VT_VULKAN_DEVICE_INDEX must be an integer physical GPU index.") from exc
            if selected_index is not None:
                if selected_index < 0:
                    raise RuntimeError("Vulkan device index cannot be negative.")
                command += ["-dev", str(selected_index)]
            callback = getattr(self, "transcription_progress", None)
            if callback is None:
                completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
                return_code = completed.returncode
                runtime_log = f"{completed.stdout or ''}\n{completed.stderr or ''}"
                saw_vulkan = "using vulkan" in runtime_log.lower()
            else:
                command.append("-pp")
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace", bufsize=1)
                runtime_log = ""
                saw_vulkan = False
                last_percent = None
                try:
                    assert process.stdout is not None
                    for line in process.stdout:
                        if "using vulkan" in line.lower():
                            saw_vulkan = True
                        runtime_log = (runtime_log + line)[-8000:]
                        match = re.search(r"progress\s*=\s*(\d+(?:\.\d+)?)\s*%", line, re.IGNORECASE)
                        if match:
                            value = min(99.0, float(match.group(1)))
                            if last_percent is None or value > last_percent:
                                last_percent = value
                                callback({"percent": value, "processedSeconds": None, "totalSeconds": None})
                    return_code = process.wait()
                except Exception:
                    process.kill()
                    process.wait()
                    raise
                finally:
                    if process.stdout is not None:
                        process.stdout.close()
            if return_code:
                detail = (runtime_log or "unknown whisper.cpp error").strip()
                if is_out_of_memory_error(detail):
                    code = "out_of_memory"
                    stage = "recognition"
                elif is_invalid_source_error(detail):
                    code = "source_invalid"
                    stage = "source"
                else:
                    code = "device_error"
                    stage = "recognition"
                raise TaskFailure(f"Vulkan transcription failed (whisper.cpp exit {return_code}): {detail[-1600:]}",
                                  code, stage=stage)
            if not saw_vulkan:
                raise TaskFailure(
                    "whisper.cpp did not confirm Vulkan GPU execution. Check that the installed runtime includes Vulkan support and restart the app.",
                    "device_error", stage="recognition",
                )
            json_path = Path(str(output_base) + ".json")
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise TaskFailure("whisper.cpp completed without creating its JSON transcription output.",
                                  "transcription_failed", stage="recognition") from exc
            except json.JSONDecodeError as exc:
                raise TaskFailure(f"whisper.cpp created malformed JSON output: {exc}",
                                  "transcription_failed", stage="recognition") from exc
            try:
                result = parse_whisper_cpp_json(payload)
            except TaskFailure:
                raise
            except Exception as exc:
                raise TaskFailure(str(exc), "transcription_failed", stage="recognition") from exc
            if callback is not None:
                callback({"percent": 100.0, "processedSeconds": None, "totalSeconds": None})
            return result
