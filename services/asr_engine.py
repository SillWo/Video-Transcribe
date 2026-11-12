"""Parent-process adapter for isolated ASR workers."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from services.asr_resources import runtime_python
from services.runtime_devices import asr_gpu_candidate
from services.model_paths import MODEL_TEMP_DIR
from services.task_errors import TaskFailure, is_invalid_source_error, is_out_of_memory_error

APP_ROOT = Path(__file__).resolve().parents[1]
WORKER_MODULE = "services.asr_worker"
PARAKEET_LANGUAGES = frozenset({
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
})


@dataclass
class WorkerSegment:
    start: float
    end: float
    text: str
    words: list[dict]


@dataclass
class WorkerStats:
    language: str | None
    language_probability: float | None
    duration: float


def _is_pcm_16k_mono(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as audio:
            return (audio.getnchannels() == 1 and audio.getsampwidth() == 2
                    and audio.getframerate() == 16000 and audio.getcomptype() == "NONE"
                    and audio.getnframes() > 0)
    except (OSError, wave.Error, EOFError):
        return False


def _audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getframerate() <= 0 or audio.getnframes() <= 0:
                raise ValueError("empty audio")
            return audio.getnframes() / audio.getframerate()
    except (OSError, wave.Error, EOFError, ValueError) as exc:
        raise TaskFailure(f"Не удалось прочитать подготовленный WAV: {exc}", "source_invalid", stage="source") from exc


def _prepare_audio(source: Path, workspace: Path) -> Path:
    if not source.is_file():
        raise TaskFailure(f"Аудиофайл не найден: {source}", "source_invalid", stage="source")
    if _is_pcm_16k_mono(source):
        return source
    from services.whisper_cpp import _find_ffmpeg

    ffmpeg = _find_ffmpeg()
    target = workspace / "input-pcm16k-mono.wav"
    command = [ffmpeg, "-nostdin", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", "-f", "wav", str(target)]
    try:
        completed = subprocess.run(command, cwd=APP_ROOT, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", check=False)
    except OSError as exc:
        raise TaskFailure(f"Не удалось запустить FFmpeg: {exc}", "source_invalid", stage="source") from exc
    if completed.returncode or not target.is_file() or not _is_pcm_16k_mono(target):
        detail = (completed.stderr or completed.stdout or "FFmpeg did not produce PCM16 mono 16000 Hz WAV").strip()
        raise TaskFailure(f"Не удалось подготовить аудио: {detail[-1200:]}", "source_invalid", stage="source")
    return target


def _map_worker_failure(message: str, default_stage: str, *, device: str) -> TaskFailure:
    text = message.strip() or "ASR worker завершился без результата."
    lowered = text.casefold()
    public_stage = "recognition" if default_stage == "alignment" else default_stage
    if is_invalid_source_error(text) or any(token in lowered for token in ("worker audio must", "wav contains no samples", "audio file does not exist")):
        return TaskFailure(text, "source_invalid", stage=public_stage)
    if is_out_of_memory_error(text):
        return TaskFailure(text, "out_of_memory", stage=public_stage)
    if any(token in lowered for token in ("unavailable", "not available", "device", "provider", "cuda", "vulkan", "directml")):
        return TaskFailure(text, "device_unavailable", stage=public_stage)
    if any(token in lowered for token in ("model file does not exist", "model is unavailable", "no such file", "model.safetensors")):
        return TaskFailure(text, "model_unavailable", stage=public_stage)
    if default_stage == "alignment" and any(token in lowered for token in ("unsupported for language", "supported codes")):
        return TaskFailure(text, "model_unavailable", stage="recognition")
    code = "transcription_failed" if default_stage == "recognition" else "model_unavailable"
    return TaskFailure(text, code, stage=public_stage)


class NewAsrModel:
    """Lightweight facade that runs one short-lived isolated child per phase."""

    def __init__(self, model_id: str, device: str, threads: int = 8, device_index: int = 0,
                 progress: Callable[[dict], None] | None = None):
        if model_id not in {"parakeet-tdt-0.6b-v3", "qwen3-asr-1.7b"}:
            raise ValueError(f"Unsupported new ASR model: {model_id}")
        if device not in {"cpu", "cuda", "vulkan", "directml"}:
            raise ValueError(f"Unsupported ASR device: {device}")
        self.model_id = model_id
        self.effective_device = device
        self.effective_backend = "ASR worker"
        self.precision: str | None = None
        self.diagnostics: dict = {}
        self.threads = max(1, int(threads))
        self.device_index = max(0, int(device_index))
        self.transcription_progress = progress

    def _runtime(self, mode: str) -> tuple[Path, str]:
        if mode == "align":
            if self.effective_device == "cuda":
                return runtime_python("aligner", "cuda"), "cuda"
            if self.effective_device == "directml":
                return runtime_python("aligner", "cpu"), "directml"
            if self.effective_device == "vulkan":
                return runtime_python("aligner", "cpu"), "vulkan"
            return runtime_python("aligner", "cpu"), "cpu"
        if mode == "language-id":
            return runtime_python("native", "cpu"), "cpu"
        if mode == "devices":
            backend = {"directml": "vulkan"}.get(self.effective_device, self.effective_device)
            return runtime_python("native", backend), backend
        kind = "native" if self.model_id == "qwen3-asr-1.7b" else "parakeet"
        return runtime_python(kind, self.effective_device), self.effective_device

    def _run_worker(self, mode: str, audio: Path, workspace: Path, *, language: str | None = None,
                    segments_path: Path | None = None, device: str | None = None,
                    device_index: int | None = None, default_stage: str = "recognition") -> dict:
        if mode == "align":
            aligner_runtime = "cuda" if device == "cuda" else "cpu"
            python, runtime_device = runtime_python("aligner", aligner_runtime), device or "cpu"
        elif mode == "language-id" and device is not None:
            # GPU preparation guarantees the native GPU environment, which also
            # supports CPU language-ID. Reuse it for auxiliary CPU fallback.
            native_backend = {"directml": "vulkan"}.get(self.effective_device, self.effective_device)
            runtime_backend = native_backend if device == "cpu" and native_backend != "cpu" else device
            python, runtime_device = runtime_python("native", runtime_backend), device
        else:
            python, runtime_device = self._runtime(mode)
        selected_device = device or runtime_device
        selected_index = 0 if selected_device == "cpu" else (self.device_index if device_index is None else device_index)
        output_path = workspace / f"{mode}-{selected_device}.json"
        worker_model = "qwen3-asr-1.7b" if mode == "devices" else self.model_id
        command = [str(python), "-m", WORKER_MODULE, "--mode", mode, "--model", worker_model,
                   "--device", selected_device, "--device-index", str(selected_index)]
        if mode != "devices":
            command += ["--audio", str(audio)]
        command += [
                   "--output", str(output_path), "--threads", str(self.threads)]
        if language:
            command += ["--language", language]
        if segments_path is not None:
            command += ["--segments-json", str(segments_path)]
        tail: deque[str] = deque(maxlen=40)
        process = None
        started = time.monotonic()
        try:
            process = subprocess.Popen(command, cwd=APP_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                tail.append(line[-1000:])
                if line.startswith("__VT_ASR__ "):
                    try:
                        event = json.loads(line[len("__VT_ASR__ "):])
                    except (TypeError, ValueError):
                        print(line, flush=True)
                        continue
                    self._forward_worker_event(event, default_stage)
                else:
                    print(line, flush=True)
            returncode = process.wait()
        except OSError as exc:
            raise TaskFailure(f"Не удалось запустить ASR worker: {exc}", "dependency_missing", stage=default_stage) from exc
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if not output_path.is_file():
            log = "\n".join(tail)
            raise _map_worker_failure(log or f"Worker exited with code {returncode} without JSON output.", default_stage,
                                      device=selected_device)
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _map_worker_failure(f"Worker result is invalid: {exc}\n" + "\n".join(tail), default_stage,
                                      device=selected_device) from exc
        if returncode != 0 or not isinstance(result, dict) or result.get("ok") is not True:
            error = result.get("error", {}) if isinstance(result, dict) else {}
            message = str(error.get("message") or "Worker returned no successful result.")
            if tail:
                message += "\n" + "\n".join(tail)
            raise _map_worker_failure(message, default_stage, device=selected_device)
        if mode != "devices":
            actual_device = result.get("effective_device")
            if mode == "align":
                reported_backend = str(result.get("effective_backend") or "").casefold()
                backend_matches = f"forced aligner {selected_device}" in reported_backend
            else:
                backend_matches = actual_device == selected_device
            if not backend_matches:
                raise TaskFailure(f"Worker backend mismatch: requested {selected_device}, reported {actual_device}.",
                                  "device_unavailable", stage="recognition" if default_stage == "alignment" else default_stage)
            expected_precision = "F16" if mode == "transcribe" and self.model_id == "qwen3-asr-1.7b" else (
                "FP32" if mode == "transcribe" else None)
            if expected_precision and result.get("precision") != expected_precision:
                raise TaskFailure(f"Unexpected model precision: expected {expected_precision}, got {result.get('precision')}.",
                                  "model_unavailable", stage="recognition" if default_stage == "alignment" else default_stage)
        result["_worker_elapsed_seconds"] = time.monotonic() - started
        return result

    def _language_id_target(self, audio: Path, workspace: Path) -> tuple[str, int, str | None]:
        """Resolve the ASR adapter to its native runtime without crossing index namespaces."""
        if self.effective_device == "cpu":
            return "cpu", 0, "ASR is configured for CPU."
        target_backend = "cuda" if self.effective_device == "cuda" else "vulkan"
        candidate = asr_gpu_candidate(self.model_id)
        if candidate.get("backend") != self.effective_device:
            return "cpu", 0, f"ASR GPU backend {self.effective_device} has no matching language-ID GPU backend."
        try:
            listed = self._run_worker("devices", audio, workspace, device=target_backend)
        except TaskFailure as exc:
            return "cpu", 0, f"Native {target_backend} device inventory failed: {exc}"
        import re
        expected_name = re.sub(r"[^a-z0-9]", "", str(candidate.get("name") or "").casefold())
        matches = [item for item in listed.get("devices", [])
                   if item.get("backend") == target_backend
                   and str(item.get("deviceType", "")).casefold() in {"gpu", "igpu"}
                   and expected_name and any(
                       re.sub(r"[^a-z0-9]", "", str(item.get(field) or "").casefold()) == expected_name
                       for field in ("description", "name")
                   )]
        if len(matches) != 1:
            reason = "No matching native GPU was found." if not matches else "Native GPU name match is ambiguous."
            return "cpu", 0, f"{reason} ASR adapter: {candidate.get('name') or 'unknown'}; runtime: {target_backend}."
        target = matches[0]
        index = target.get("index")
        if not isinstance(index, int) or index < 0:
            return "cpu", 0, f"Matching native GPU has no valid {target_backend} index."
        # Native indices are selected from backend-filtered devices.
        index = sum(1 for item in listed.get("devices", [])
                    if item.get("backend") == target_backend and isinstance(item.get("index"), int)
                    and item["index"] < target["index"])
        return target_backend, index, None

    def _forward_worker_event(self, event: dict, default_stage: str) -> None:
        message = str(event.get("message") or "")
        if event.get("type") == "log":
            print(message, flush=True)
        callback = self.transcription_progress
        if not callable(callback):
            return
        processed = event.get("processedSeconds")
        total = event.get("totalSeconds")
        try:
            processed = float(processed) if processed is not None else None
            total = float(total) if total is not None else None
        except (TypeError, ValueError):
            processed = total = None
        percent = min(99.0, processed * 100.0 / total) if processed is not None and total and total > 0 else None
        # Alignment is an internal subphase. Preserve the established public
        # recognition stage while exposing the true phase in diagnostics.
        phase = event.get("stage", default_stage)
        callback({"percent": percent, "processedSeconds": processed, "totalSeconds": total,
                  "message": message, "stage": "recognition" if phase == "alignment" else phase,
                  "phase": phase, "workerStage": phase})

    def _language_from_observations(self, observations: list[dict], segments: list[dict]) -> tuple[str | None, float | None]:
        scores: dict[str, float] = {}
        confidence: dict[str, list[tuple[float, float]]] = {}
        for item in observations:
            code = item.get("code") or item.get("language")
            if not code:
                continue
            try:
                start, end = float(item.get("start", 0)), float(item.get("end", 0))
            except (TypeError, ValueError):
                continue
            overlap = sum(max(0.0, min(end, float(segment["end"])) - max(start, float(segment["start"])))
                          for segment in segments if segment.get("text", "").strip())
            if overlap <= 0:
                continue
            scores[code] = scores.get(code, 0.0) + overlap
            probability = item.get("probability")
            if isinstance(probability, (int, float)) and 0 <= float(probability) <= 1:
                confidence.setdefault(code, []).append((float(probability), overlap))
        if not scores:
            return None, None
        selected = max(scores, key=lambda code: (scores[code],
                                                sum(p * weight for p, weight in confidence.get(code, ())), code))
        values = confidence.get(selected, ())
        total_weight = sum(weight for _, weight in values)
        probability = sum(value * weight for value, weight in values) / total_weight if total_weight else None
        return selected, probability

    def _transcript_cyrillic_constraint(self, rows: list[dict] | str) -> tuple[str | None, dict]:
        text = rows if isinstance(rows, str) else " ".join(str(row.get("text", "")) for row in rows)
        letters = [char.casefold() for char in text if char.isalpha()]
        cyrillic = [char for char in letters if "а" <= char <= "я" or char in "ёєіїґ"]
        info = {"applied": False, "reason": "insufficient_cyrillic_text", "cyrillic_letters": len(cyrillic),
                "letter_count": len(letters)}
        if len(cyrillic) < 20 or not letters or len(cyrillic) / len(letters) < 0.8:
            return None, info

        alphabets = {
            "bg": set("абвгдежзийклмнопрстуфхцчшщъьюя"),
            "ru": set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
            "uk": set("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"),
        }
        observed = set(cyrillic)
        compatible = [code for code, alphabet in alphabets.items() if observed <= alphabet]
        discriminatory = {
            "ru": sum(char in "ыэ" for char in cyrillic),
            "uk": sum(char in "іїєґ" for char in cyrillic),
        }
        selected = None
        if compatible == ["ru"] and discriminatory["ru"] >= 2:
            selected = "ru"
        elif compatible == ["uk"] and discriminatory["uk"] >= 2:
            selected = "uk"
        info.update({"compatible_alphabets": compatible,
                     "discriminatory_letter_counts": discriminatory,
                     "reason": "unique_russian_alphabet" if selected == "ru" else
                               "unique_ukrainian_alphabet" if selected == "uk" else
                               "no_unique_discriminatory_alphabet"})
        info["applied"] = selected is not None
        return selected, info

    def _supported_language_observations(self, observations: list[dict], rows: list[dict] | str | None = None) -> list[dict]:
        """Restrict Parakeet language-ID results to its documented language set."""
        transcript_language, alphabet_constraint = self._transcript_cyrillic_constraint(rows or "")
        supported: list[dict] = []
        for item in observations:
            top_code = item.get("code") or item.get("language")
            top_probability = item.get("probability")
            candidates = item.get("candidates") or []
            valid = []
            for candidate in candidates:
                code = candidate.get("code") or candidate.get("language")
                probability = candidate.get("p", candidate.get("probability"))
                if code not in PARAKEET_LANGUAGES or not isinstance(probability, (int, float)):
                    continue
                probability = float(probability)
                if math.isfinite(probability) and 0.0 <= probability <= 1.0:
                    valid.append((code, probability))
            if transcript_language:
                matching = [pair for pair in valid if pair[0] == transcript_language]
                if matching:
                    valid = matching
            if not candidates and top_code in PARAKEET_LANGUAGES and isinstance(top_probability, (int, float)):
                top_probability = float(top_probability)
                if math.isfinite(top_probability) and 0.0 <= top_probability <= 1.0:
                    valid.append((top_code, top_probability))
            selected = max(valid, key=lambda pair: (pair[1], pair[0])) if valid else (None, None)
            supported.append({**item, "code": selected[0], "probability": selected[1],
                              "unrestricted_top": {"code": top_code,
                                                   "probability": top_probability},
                              "alphabet_constraint": alphabet_constraint})
        return supported

    def _segments(self, rows: list[dict]) -> list[WorkerSegment]:
        result = []
        for row in rows:
            try:
                start, end, text = float(row["start"]), float(row["end"]), str(row.get("text", ""))
            except (KeyError, TypeError, ValueError) as exc:
                raise TaskFailure(f"ASR worker returned an invalid segment: {exc}", "transcription_failed", stage="recognition") from exc
            words = []
            for word in row.get("words", []) or []:
                words.append({"text": str(word.get("text", "")), "start": word.get("start"), "end": word.get("end")})
            result.append(WorkerSegment(start, end, text, words))
        return result

    def transcribe(self, path, beam_size=5, language=None, word_timestamps=False):
        source = Path(path).expanduser().resolve()
        MODEL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vt-asr-", dir=MODEL_TEMP_DIR) as directory:
            workspace = Path(directory)
            audio = _prepare_audio(source, workspace)
            asr = self._run_worker("transcribe", audio, workspace, language=language)
            rows = asr.get("segments")
            if not isinstance(rows, list):
                raise TaskFailure("ASR worker result contains no segments.", "transcription_failed", stage="recognition")
            language_code = language or asr.get("language")
            probability = asr.get("language_probability")
            diagnostics = {"asr": {"effective_device": asr.get("effective_device"),
                                    "effective_backend": asr.get("effective_backend"),
                                    "precision": asr.get("precision"),
                                    "timing_seconds": asr.get("timing_seconds") or {},
                                    "worker_elapsed_seconds": asr.get("_worker_elapsed_seconds"),
                                    "diagnostics": asr.get("diagnostics") or {}}}
            if self.model_id == "qwen3-asr-1.7b" and not language:
                observations = asr.get("language_observations") or []
                observed_language, observed_probability = self._language_from_observations(observations, rows)
                if observed_language:
                    language_code, probability = observed_language, observed_probability
                diagnostics["language_observations"] = observations
                diagnostics["asr"]["language_selection"] = "duration-weighted observations"
            if self.model_id == "parakeet-tdt-0.6b-v3" and not language:
                nonempty_recognition = any(str(row.get("text", "")).strip() for row in rows)
                target_device, target_index, fallback_reason = self._language_id_target(audio, workspace)
                if fallback_reason:
                    print(f"Language-ID GPU unavailable; using CPU: {fallback_reason}", flush=True)
                try:
                    lang_result = self._run_worker("language-id", audio, workspace, device=target_device,
                                                   device_index=target_index, default_stage="recognition")
                    observations = lang_result.get("language_observations") or []
                    scored_observations = self._supported_language_observations(observations, rows)
                    language_code, probability = self._language_from_observations(scored_observations, rows)
                    diagnostics["language_identification"] = {
                        "effective_device": lang_result.get("effective_device"),
                        "effective_backend": lang_result.get("effective_backend"),
                        "timing_seconds": lang_result.get("timing_seconds") or {},
                        "worker_elapsed_seconds": lang_result.get("_worker_elapsed_seconds"),
                        "fallback_reason": fallback_reason,
                        "observations": observations,
                        "supported_observations": scored_observations,
                        "diagnostics": lang_result.get("diagnostics") or {},
                    }
                    if lang_result.get("effective_device") != target_device:
                        raise TaskFailure(f"Language-ID worker backend mismatch: requested {target_device}.",
                                          "device_unavailable", stage="recognition")
                    if nonempty_recognition and not language_code and observations:
                        raise TaskFailure("Language-ID observations did not overlap recognized speech.",
                                          "transcription_failed", stage="recognition")
                    if language_code:
                        print(f"Language identification selected '{language_code}' from duration-weighted observations.", flush=True)
                    else:
                        if nonempty_recognition:
                            raise TaskFailure("Не удалось определить язык распознанной речи.",
                                              "transcription_failed", stage="recognition")
                        print("Language identification found no speech observations in silent audio.", flush=True)
                except TaskFailure as exc:
                    if target_device != "cpu":
                        fallback_reason = f"{target_device} language-ID failed: {exc}"
                        print(f"{fallback_reason}; retrying language-ID on CPU.", flush=True)
                        try:
                            lang_result = self._run_worker("language-id", audio, workspace, device="cpu",
                                                           device_index=0, default_stage="recognition")
                            observations = lang_result.get("language_observations") or []
                            scored_observations = self._supported_language_observations(observations, rows)
                            language_code, probability = self._language_from_observations(scored_observations, rows)
                            diagnostics["language_identification"] = {
                                "effective_device": lang_result.get("effective_device"),
                                "effective_backend": lang_result.get("effective_backend"),
                                "timing_seconds": lang_result.get("timing_seconds") or {},
                                "worker_elapsed_seconds": lang_result.get("_worker_elapsed_seconds"),
                                "fallback_reason": fallback_reason, "observations": observations,
                                "supported_observations": scored_observations,
                                "diagnostics": lang_result.get("diagnostics") or {},
                            }
                        except TaskFailure as cpu_exc:
                            if nonempty_recognition:
                                raise cpu_exc
                            diagnostics["language_identification"] = {"error": str(cpu_exc),
                                                                       "effective_device": "cpu",
                                                                       "fallback_reason": fallback_reason}
                    elif nonempty_recognition:
                        raise
                    else:
                        diagnostics["language_identification"] = {"error": str(exc), "effective_device": "cpu"}
                if nonempty_recognition and not language_code:
                    raise TaskFailure("Не удалось определить язык распознанной речи.",
                                      "transcription_failed", stage="recognition")
            if word_timestamps and rows and self.model_id == "qwen3-asr-1.7b":
                if not language_code:
                    raise TaskFailure("Нельзя выровнять слова: язык аудио не определён.", "model_unavailable", stage="recognition")
                raw_segments = workspace / "segments.json"
                raw_segments.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
                align_backend = "cpu"
                if self.effective_device == "cuda":
                    align_backend = "cuda"
                elif self.effective_device == "directml":
                    align_backend = "directml"
                elif self.effective_device == "vulkan":
                    align_backend = "vulkan"
                aligned = None
                if align_backend == "vulkan":
                    print("Forced aligner does not support Vulkan; using CPU.", flush=True)
                    diagnostics["alignment_fallback"] = "Forced aligner has no compatible Vulkan runtime."
                else:
                    try:
                        self._forward_worker_event(
                            {"type": "progress", "stage": "alignment", "message": "Выравнивание слов"},
                            "alignment",
                        )
                        aligned = self._run_worker("align", audio, workspace, language=language_code,
                                                   segments_path=raw_segments, device=align_backend,
                                                   default_stage="alignment")
                    except TaskFailure as exc:
                        if align_backend == "cpu":
                            raise
                        print(f"{align_backend} word alignment failed ({exc}); retrying on CPU.", flush=True)
                        diagnostics["alignment_fallback"] = str(exc)
                if aligned is None:
                    self._forward_worker_event(
                        {"type": "progress", "stage": "alignment", "message": "Выравнивание слов"},
                        "alignment",
                    )
                    aligned = self._run_worker("align", audio, workspace, language=language_code,
                                               segments_path=raw_segments, device="cpu", default_stage="alignment")
                rows = aligned.get("segments")
                if not isinstance(rows, list):
                    raise TaskFailure("Aligner result contains no segments.", "transcription_failed", stage="recognition")
                diagnostics["alignment"] = {"diagnostics": aligned.get("diagnostics") or {},
                                             "effective_device": aligned.get("effective_device"),
                                             "effective_backend": aligned.get("effective_backend"),
                                             "precision": aligned.get("precision"),
                                             "timing_seconds": aligned.get("timing_seconds") or {},
                                             "worker_elapsed_seconds": aligned.get("_worker_elapsed_seconds")}
            segments = self._segments(rows)
            self.effective_device = str(asr.get("effective_device") or self.effective_device)
            self.effective_backend = str(asr.get("effective_backend") or "ASR worker")
            self.precision = asr.get("precision")
            self.diagnostics = diagnostics
            stats = WorkerStats(language_code, probability, float(asr.get("duration") or _audio_duration(audio)))
            return segments, stats

    def unload_model(self) -> None:
        """Workers exit synchronously after each phase, releasing their model memory."""
        return None
