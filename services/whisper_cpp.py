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
from services.model_paths import MODEL_TEMP_DIR
from services.download_progress import create_download_progress_tqdm
from services.runtime_devices import VULKAN_MODELS, find_whisper_cpp_executable
from services.task_errors import TaskFailure, is_invalid_source_error, is_out_of_memory_error

MODEL_REPO = os.environ.get("VT_WHISPER_CPP_MODEL_REPO", "ggerganov/whisper.cpp")
MODEL_CACHE = HF_HUB_CACHE
TOKENIZER_REPO = "Systran/faster-whisper-{}"
_DLL_DIRECTORY_HANDLES = []


def cached_tokenizer_path(model_name: str) -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
        value = try_to_load_from_cache(repo_id=TOKENIZER_REPO.format(model_name),
                                       filename="tokenizer.json", cache_dir=str(HF_HUB_CACHE))
    except Exception:
        return None
    return Path(value) if isinstance(value, str) and Path(value).is_file() else None


def resolve_tokenizer_path(model_name: str, *, progress=None) -> Path:
    """Load only Whisper's tokenizer asset, without preparing CPU model weights."""
    if model_name not in {"small", "large-v3"}:
        raise TaskFailure(f"Dictionary hints are not supported for Whisper model '{model_name}'.",
                          "dictionary_tokenizer_unavailable", stage="preparation")
    repo = TOKENIZER_REPO.format(model_name)
    try:
        from huggingface_hub import hf_hub_download
        cached = cached_tokenizer_path(model_name)
        if cached:
            return cached
        if progress:
            progress(f"Preparing the Whisper {model_name} tokenizer for dictionary hints.")
        path = Path(hf_hub_download(repo_id=repo, filename="tokenizer.json", cache_dir=str(HF_HUB_CACHE)))
        if not path.is_file():
            raise FileNotFoundError("Hugging Face did not return tokenizer.json")
        return path
    except Exception as exc:
        raise TaskFailure(
            f"Could not prepare the Whisper {model_name} tokenizer for dictionary hints: {exc}",
            "dictionary_tokenizer_unavailable", stage="preparation") from exc


def bound_whisper_vocabulary(terms, model_name: str, limit: int = 224) -> tuple[str, dict]:
    """Build a bounded prompt with the exact cached Hugging Face Whisper tokenizer; fail closed."""
    raw_terms = [str(term).strip() for term in terms if isinstance(term, str) and term.strip()]
    diagnostics = {"usedTerms": [], "omittedTerms": list(raw_terms), "usedCount": 0,
                   "omittedCount": len(raw_terms), "truncated": bool(raw_terms), "tokenCount": 0,
                   "tokenLimit": limit, "tokenizer": None,
                   "reason": "tokenizer_missing" if raw_terms else None}
    try:
        from huggingface_hub import try_to_load_from_cache
        from tokenizers import Tokenizer
        repo = f"Systran/faster-whisper-{model_name}"
        tokenizer_path = try_to_load_from_cache(repo_id=repo, filename="tokenizer.json", cache_dir=str(HF_HUB_CACHE))
        if not isinstance(tokenizer_path, str) or not Path(tokenizer_path).is_file():
            return "", diagnostics
        tokenizer = Tokenizer.from_file(tokenizer_path)
        diagnostics["tokenizer"] = str(Path(tokenizer_path).resolve())
        diagnostics["reason"] = None
    except Exception as exc:
        diagnostics["reason"] = f"tokenizer_unavailable:{type(exc).__name__}"
        return "", diagnostics
    prefix = "Vocabulary: "
    selected: list[str] = []
    for term in raw_terms:
        candidate = [*selected, term]
        prompt = prefix + ", ".join(candidate)
        try:
            count = len(tokenizer.encode(prompt).ids)
        except Exception:
            break
        if count > limit:
            continue
        selected = candidate
        diagnostics["tokenCount"] = count
    omitted = [term for term in raw_terms if term not in selected]
    diagnostics.update({"usedTerms": selected, "omittedTerms": omitted,
                        "usedCount": len(selected), "omittedCount": len(omitted),
                        "truncated": bool(omitted)})
    return (prefix + ", ".join(selected) if selected else ""), diagnostics


@dataclass
class WhisperCppSegment:
    start: float
    end: float
    text: str
    words: list[dict] | None = None


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


def _decode_surrogateescaped_utf8(value: str) -> str:
    """Decode token byte fragments while rejecting malformed exposed UTF-8."""
    return value.encode("utf-8", errors="surrogateescape").decode("utf-8")


def _token_words(tokens, segment_start, segment_end, expected_text):
    """Reassemble real -ojf token text and millisecond offsets into lexical words."""
    groups = []
    current = None
    special = re.compile(r"^\[.*\]$")
    for token in tokens:
        value = str(token.get("text", ""))
        if not value or special.match(value.strip()):
            continue
        try:
            offsets = token.get("offsets") or {}
            start, end = float(offsets["from"]) / 1000, float(offsets["to"]) / 1000
        except (KeyError, TypeError, ValueError):
            start = end = 0.0
        # Whisper text tokens use a leading space to mark word boundaries;
        # punctuation tokens have no leading space and stay with the word.
        begins_word = bool(value[:1].isspace())
        if begins_word and current is not None:
            groups.append(current)
            current = None
        if current is None:
            current = {"text": value, "start": start, "end": end}
        else:
            current["text"] += value
            if end > start:
                if current["end"] <= current["start"]:
                    current["start"] = start
                current["end"] = max(current["end"], end)
    if current is not None:
        groups.append(current)
    words = groups
    token_texts = [str(token.get("text", "")) for token in tokens
                   if token.get("text") and not special.match(str(token.get("text", "")).strip())]
    try:
        reconstructed = _decode_surrogateescaped_utf8("".join(token_texts)).strip()
        for group in groups:
            group["text"] = _decode_surrogateescaped_utf8(group["text"])
    except (UnicodeDecodeError, UnicodeEncodeError):
        return None
    if reconstructed != expected_text.strip():
        return None
    return words


def parse_whisper_cpp_json(payload: dict, *, word_timestamps: bool = False) -> tuple[list[WhisperCppSegment], SimpleNamespace]:
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
            if not isinstance(row["text"], str):
                raise TypeError("segment text must be a string")
            text = _decode_surrogateescaped_utf8(row["text"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"whisper.cpp returned malformed segment {index}.") from exc
        words = None
        if word_timestamps and isinstance(row.get("tokens"), list):
            words = _token_words(row["tokens"], start, end, text)
        segments.append(WhisperCppSegment(start, end, text, words))
    result_metadata = payload.get("result") or {}
    if not isinstance(result_metadata, dict):
        raise RuntimeError("whisper.cpp returned malformed language metadata.")
    language = result_metadata.get("language") or "unknown"
    if not isinstance(language, str):
        raise RuntimeError("whisper.cpp returned malformed language metadata.")
    try:
        language = _decode_surrogateescaped_utf8(language)
    except (UnicodeDecodeError, UnicodeEncodeError) as exc:
        raise RuntimeError("whisper.cpp returned invalid UTF-8 language metadata.") from exc
    return segments, SimpleNamespace(language=language, language_probability=None)


class _RepeatedTextDetector:
    """Recognize sustained, advancing repetitions without retaining segment history."""

    _segment_line = re.compile(
        r"^\s*\[?(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
        r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\]?\s*(.*?)\s*$"
    )

    def __init__(self):
        self.key = None
        self.count = 0
        self.first_start = None
        self.last_end = None
        self.advancing_segments = 0

    @staticmethod
    def _normalize(text: str) -> tuple[str, int]:
        words = re.findall(r"\w+", str(text).casefold())
        return " ".join(words), len(words)

    def add(self, start: float, end: float, text: str) -> bool:
        key, word_count = self._normalize(text)
        if word_count < 4:
            self._reset()
            return False
        if key != self.key:
            self._reset()
            self.key = key
            self.count = 1
            self.first_start = start
            self.last_end = end if end > start else None
            self.advancing_segments = 1 if end > start else 0
            return False

        self.count += 1
        if end > start and (self.last_end is None or end > self.last_end):
            self.advancing_segments += 1
            self.last_end = end
        return bool(
            self.count >= 6
            and self.advancing_segments >= 3
            and self.first_start is not None
            and self.last_end is not None
            and self.last_end - self.first_start >= 8.0
        )

    def add_cli_line(self, line: str) -> bool:
        match = self._segment_line.match(line)
        if not match:
            return False
        try:
            start = _timestamp_ms(match.group(1)) / 1000
            end = _timestamp_ms(match.group(2)) / 1000
        except (TypeError, ValueError):
            return False
        return self.add(start, end, match.group(3))

    def _reset(self):
        self.key = None
        self.count = 0
        self.first_start = None
        self.last_end = None
        self.advancing_segments = 0


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


def _kill_and_reap(process) -> int:
    try:
        process.kill()
    except OSError:
        # It may have exited between the final output line and kill().
        pass
    return process.wait()


def _run_whisper_attempt(command: list[str], callback, *, repetition_guard: bool):
    """Run one native attempt and report whether streaming output proved a loop."""
    if callback is None:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
        log = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        return completed.returncode, log, "using vulkan" in log.lower(), False

    command.append("-pp")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
    runtime_log = ""
    saw_vulkan = False
    last_percent = None
    detector = _RepeatedTextDetector() if repetition_guard else None
    pending_loop = False
    stream_loop = False
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if "using vulkan" in line.lower():
                saw_vulkan = True
            runtime_log = (runtime_log + line)[-8000:]
            if detector is not None and detector.add_cli_line(line):
                pending_loop = True
            if pending_loop and saw_vulkan:
                stream_loop = True
                return_code = _kill_and_reap(process)
                return return_code, runtime_log, saw_vulkan, stream_loop
            match = re.search(r"progress\s*=\s*(\d+(?:\.\d+)?)\s*%", line, re.IGNORECASE)
            if match:
                value = min(99.0, float(match.group(1)))
                if last_percent is None or value > last_percent:
                    last_percent = value
                    callback({"percent": value, "processedSeconds": None, "totalSeconds": None})
        return_code = process.wait()
    except Exception:
        _kill_and_reap(process)
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
    return return_code, runtime_log, saw_vulkan, stream_loop


def _load_whisper_result(output_base: Path, *, word_timestamps: bool):
    json_path = Path(str(output_base) + ".json")
    try:
        payload = json.loads(json_path.read_bytes().decode("utf-8", errors="surrogateescape"))
    except FileNotFoundError as exc:
        raise TaskFailure("whisper.cpp completed without creating its JSON transcription output.",
                          "transcription_failed", stage="recognition") from exc
    except json.JSONDecodeError as exc:
        raise TaskFailure(f"whisper.cpp created malformed JSON output: {exc}",
                          "transcription_failed", stage="recognition") from exc
    try:
        return parse_whisper_cpp_json(payload, word_timestamps=word_timestamps)
    except TaskFailure:
        raise
    except Exception as exc:
        raise TaskFailure(str(exc), "transcription_failed", stage="recognition") from exc


def _raise_whisper_failure(return_code: int, runtime_log: str) -> None:
    detail = (runtime_log or "unknown whisper.cpp error").strip()
    if is_out_of_memory_error(detail):
        code, stage = "out_of_memory", "recognition"
    elif is_invalid_source_error(detail):
        code, stage = "source_invalid", "source"
    else:
        code, stage = "device_error", "recognition"
    raise TaskFailure(f"Vulkan transcription failed (whisper.cpp exit {return_code}): {detail[-1600:]}",
                      code, stage=stage)


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

    def transcribe(self, media_path: str, beam_size: int = 5, language: str | None = None,
                   word_timestamps: bool = False, vocabulary=()):
        ffmpeg = _find_ffmpeg()
        with tempfile.TemporaryDirectory(prefix="vt-whisper-", dir=MODEL_TEMP_DIR) as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            converted = subprocess.run(
                [ffmpeg, "-y", "-i", str(media_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if converted.returncode:
                detail = (converted.stderr or converted.stdout or "unknown FFmpeg error").strip()
                raise TaskFailure(f"FFmpeg could not prepare audio: {detail[-1200:]}",
                                  "source_invalid", stage="source")
            is_large_v3 = getattr(self, "model_name", "") in {"large-v3", "large"}
            command = [self.executable, "-m", str(self.model_path), "-f", str(wav_path), "-l", language or "auto",
                       "-t", str(self.threads), "-bs", str(max(1, int(beam_size))),
                       "-ojf" if word_timestamps else "-oj"]
            if is_large_v3:
                # The pinned v1.8.4 runtime has the unmasked padded-K/V attention defect fixed upstream in #4127.
                command.append("--no-flash-attn")
            vocabulary = tuple(vocabulary or ())
            prompt, self.dictionary_diagnostics = bound_whisper_vocabulary(
                vocabulary, getattr(self, "model_name", "small"))
            if word_timestamps:
                profile = str(getattr(self, "model_name", "small")).lower().replace("-", ".")
                dtw = "large.v3.turbo" if "turbo" in profile else ("large.v3" if "large" in profile else "small")
                command += ["-dtw", dtw]
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
            attempts = 2 if is_large_v3 else 1
            original_terms = [str(term).strip() for term in vocabulary
                              if isinstance(term, str) and term.strip()]

            for attempt in range(attempts):
                output_base = Path(temp_dir) / f"transcript-attempt-{attempt + 1}"
                attempt_command = [*command, "-of", str(output_base)]
                if prompt and not attempt:
                    attempt_command += ["--prompt", prompt, "--carry-initial-prompt"]
                if attempt:
                    attempt_command += ["--max-context", "0"]

                return_code, runtime_log, saw_vulkan, stream_loop = _run_whisper_attempt(
                    attempt_command, callback, repetition_guard=is_large_v3)
                if return_code and not stream_loop:
                    _raise_whisper_failure(return_code, runtime_log)
                if not saw_vulkan:
                    raise TaskFailure(
                        "whisper.cpp did not confirm Vulkan GPU execution. Check that the installed runtime includes Vulkan support and restart the app.",
                        "device_error", stage="recognition",
                    )
                self.last_vulkan_confirmed = True

                result = None
                json_loop = False
                if not stream_loop:
                    result = _load_whisper_result(output_base, word_timestamps=word_timestamps)
                    if is_large_v3:
                        detector = _RepeatedTextDetector()
                        json_loop = any(detector.add(segment.start, segment.end, segment.text)
                                        for segment in result[0])

                if not (stream_loop or json_loop):
                    if attempt:
                        diagnostics = dict(getattr(self, "dictionary_diagnostics", {}) or {})
                        diagnostics.update({
                            "usedTerms": [], "omittedTerms": original_terms,
                            "usedCount": 0, "omittedCount": len(original_terms),
                            "truncated": bool(original_terms), "tokenCount": 0,
                        })
                        if prompt:
                            diagnostics["reason"] = "repetition_retry_no_context"
                        self.dictionary_diagnostics = diagnostics
                    if callback is not None:
                        callback({"percent": 100.0, "processedSeconds": None, "totalSeconds": None})
                    return result

                if attempt + 1 >= attempts:
                    raise TaskFailure("Whisper repeatedly produced the same sentence during transcription.",
                                      "repetitive_transcription", stage="recognition")
                if callback is not None:
                    callback({"percent": 0.0, "processedSeconds": None, "totalSeconds": None,
                              "message": "Whisper зациклил текст. Повторяем распознавание без предыдущего контекста.",
                              "retry": True})
