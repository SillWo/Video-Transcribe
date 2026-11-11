import argparse
import errno
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import timeit
from os.path import isfile, join
from pathlib import Path
from typing import TextIO

from services.model_paths import HF_HUB_CACHE
from services.model_paths import MODEL_TEMP_DIR
import psutil
import validators
from faster_whisper import WhisperModel as whisper
from faster_whisper.utils import available_models

from utils.download_best import Download
from utils.get_audio import AudioProcess
from utils.text_postprocess import build_polished_russian_text_from_segments, restore_russian_segment_text
from services.runtime_devices import VULKAN_MODELS, get_device_details
from services.task_errors import TaskFailure, failure_stage, is_invalid_source_error, is_out_of_memory_error
from services.whisper_cpp import WhisperCppModel

audio_file = "intermediate_audio_file.mp3"
JSON_EVENT_PREFIX = "__VT_JSON__ "


def sizes_supported() -> list[str]:
    return available_models()


def logical_cpu_count() -> int:
    return psutil.cpu_count(logical=True) or os.cpu_count() or 1


def srt_format_timestamp(seconds: float) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(segments: list[dict], file: TextIO) -> None:
    print("\nBegin transcription and creating subtitle file:")
    print("-------------------------------------------------------")

    for count, segment in enumerate(segments, start=1):
        print(
            f"{count}\n"
            f"{srt_format_timestamp(segment['start'])} --> {srt_format_timestamp(segment['end'])}\n"
            f"{segment['text'].replace('-->', '->').strip()}\n",
            file=file,
            flush=True,
        )


def findarg(args, key: str) -> bool:
    return key in args and getattr(args, key)


def emit_event(args, event_type: str, **payload) -> None:
    if not getattr(args, "web_json", False):
        return

    event = {"type": event_type, **payload}
    print(f"{JSON_EVENT_PREFIX}{json.dumps(event, ensure_ascii=False)}", flush=True)


def progress_callback(args, stage: str, message: str):
    """Build a throttled structured progress callback for the simple UI path."""
    last_time = 0.0
    last_percent = None

    def report(values):
        nonlocal last_time, last_percent
        values = values or {}
        percent = values.get("percent")
        now = timeit.default_timer()
        if percent is not None:
            try:
                percent = max(0.0, min(100.0, float(percent)))
            except (TypeError, ValueError):
                percent = None
        changed_enough = percent is not None and (last_percent is None or abs(percent - last_percent) >= 1.0)
        if last_time and now - last_time < 1.0 and not changed_enough:
            return
        last_time = now
        if percent is not None:
            last_percent = percent
        emit_event(args, "progress", stage=stage, percent=percent,
                   completedBytes=values.get("completedBytes"), totalBytes=values.get("totalBytes"),
                   processedSeconds=values.get("processedSeconds"), totalSeconds=values.get("totalSeconds"),
                   message=values.get("message", message))

    return report


def configure_web_stdio(args) -> None:
    if not getattr(args, "web_json", False):
        return

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def get_fullpath(output_dir, output_file) -> tuple[Path, str]:
    output_filepath = Path(output_file).resolve()

    if output_filepath.parent != Path(output_dir).resolve():
        output_dir = output_filepath.parent
        output_file = Path(output_file).name

    return Path(output_dir, output_file), output_dir


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    suffix = path.suffix
    stem = path.stem
    parent = path.parent
    index = 2

    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def serialize_segments(segments, *, word_timestamps=False) -> list[dict]:
    result = []
    for segment in segments:
        row = {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
        }
        if word_timestamps:
            words = getattr(segment, "words", None)
            serialized_words = []
            for word in words or []:
                if isinstance(word, dict):
                    word_text = word.get("text", word.get("word", ""))
                    raw_start, raw_end = word.get("start"), word.get("end")
                else:
                    word_text = getattr(word, "word", getattr(word, "text", ""))
                    raw_start, raw_end = getattr(word, "start", None), getattr(word, "end", None)
                try:
                    start = float(raw_start)
                except (TypeError, ValueError):
                    start = None
                try:
                    end = float(raw_end)
                except (TypeError, ValueError):
                    end = None
                serialized_words.append({"text": str(word_text), "start": start, "end": end})
            row["words"] = serialized_words
        result.append(row)
    return result


def build_plain_text(segments: list[dict], combined_text: str, use_timestamps: bool) -> str:
    if not use_timestamps:
        return combined_text.strip() + "\n"

    lines = [f"[{srt_format_timestamp(segment['start'])}] {segment['text']}" for segment in segments]
    return "\n".join(lines).strip() + "\n"


def build_output_payload(segments: list[dict], stats, combined_text: str, use_timestamps: bool,
                         diarization=None) -> dict:
    payload = {
        "language": stats.language,
        "language_probability": stats.language_probability,
        "text": combined_text,
    }

    if use_timestamps:
        payload["segments"] = segments
    else:
        payload["segments"] = [{"text": segment["text"]} for segment in segments]
    if diarization is not None:
        payload["diarization"] = diarization

    return payload


def _recognition_failure(exc: Exception, args, model) -> TaskFailure:
    if isinstance(exc, TaskFailure):
        return exc
    if is_out_of_memory_error(exc):
        return TaskFailure(str(exc), "out_of_memory", stage="recognition")
    if is_invalid_source_error(exc):
        return TaskFailure(str(exc), "source_invalid", stage="source")
    effective_device = getattr(model, "effective_device", getattr(args, "device", "cpu"))
    if effective_device in {"cuda", "vulkan"}:
        return TaskFailure(str(exc), "device_error", stage="recognition")
    return TaskFailure(str(exc), "transcription_failed", stage="recognition")


def _initialization_failure(exc: Exception, device: str) -> TaskFailure:
    if isinstance(exc, TaskFailure):
        return exc
    if is_out_of_memory_error(exc):
        code = "out_of_memory"
    elif device in {"cuda", "vulkan"}:
        code = "device_unavailable"
    else:
        code = "model_unavailable"
    return TaskFailure(str(exc), code, stage="recognition")


def write_output(args, segments: list[dict], combined_text: str, stats, filename: str,
                 diarization=None, diarization_text: str | None = None) -> tuple[Path, str]:
    output_suffix = args.output_format
    output_language = stats.language if args.language == "auto" else args.language
    output_path = Path(
        args.output_dir,
        Path(filename).stem + "." + time.strftime("%Y%m%d-%H%M%S") + f".{output_language}.{output_suffix}",
    )

    if args.output_format == "srt":
        if diarization is not None and diarization.get("status") == "completed":
            segments = [{"start": turn["start"], "end": turn["end"],
                         "text": f"{turn['label']}: {turn['text']}"}
                        for turn in diarization["turns"]]
        with open(output_path, "w", encoding="utf-8") as srt_file:
            write_srt(segments, file=srt_file)
        rendered_result = output_path.read_text(encoding="utf-8")

    elif args.output_format == "txt":
        rendered_result = (diarization_text if diarization_text is not None else
                           build_plain_text(segments, combined_text, use_timestamps=args.use_timestamps))
        if diarization_text is not None and not rendered_result.endswith("\n"):
            rendered_result += "\n"
        output_path.write_text(rendered_result, encoding="utf-8")

    else:
        payload = build_output_payload(segments, stats, combined_text, use_timestamps=args.use_timestamps,
                                       diarization=diarization)
        rendered_result = json.dumps(payload, ensure_ascii=False, indent=2)
        output_path.write_text(rendered_result, encoding="utf-8")

    return output_path, rendered_result


def transcribe(args, model, full_filepath: str, filename: str | None = None) -> dict:
    assert isinstance(full_filepath, str)

    if filename is None:
        filename = full_filepath

    emit_event(args, "stage", stage="recognition", status="in_progress", message=f"Transcribing {Path(filename).name}")

    start_time = timeit.default_timer()
    language = None if args.language == "auto" else args.language
    total_seconds = None
    diarization_enabled = bool(getattr(args, "diarization", False))
    asr_effective_device = getattr(model, "effective_device", getattr(args, "device", "cpu"))
    asr_effective_backend = getattr(model, "effective_backend", "faster-whisper")
    diarization_wav = None
    native_progress = None
    uses_native_whisper_cpp = isinstance(model, WhisperCppModel) or (
        isinstance(model, AutoFallbackModel) and isinstance(model.model, WhisperCppModel)
    )
    if getattr(args, "single_source", False):
        if isinstance(model, WhisperCppModel):
            native_progress = progress_callback(args, "recognition", "Распознавание речи.")
            model.transcription_progress = native_progress
        elif isinstance(model, AutoFallbackModel) and isinstance(model.model, WhisperCppModel):
            native_progress = progress_callback(args, "recognition", "Распознавание речи.")
            model.model.transcription_progress = native_progress
        elif not isinstance(model, AutoFallbackModel):
            native_progress = progress_callback(args, "recognition", "Распознавание речи.")

    def report_segment_progress(segment, completed, total_hint=None):
        nonlocal total_seconds
        if native_progress is None:
            return
        total = total_hint if total_hint is not None else getattr(stats, "duration", None)
        try:
            total = float(total)
            if total <= 0:
                total = None
        except (TypeError, ValueError):
            total = None
        total_seconds = total
        processed = max(0.0, float(getattr(segment, "end", 0.0)))
        percent = min(99.0, processed * 100.0 / total) if total else None
        native_progress({"percent": percent, "processedSeconds": processed, "totalSeconds": total})

    try:
        transcription_path = full_filepath
        if diarization_enabled:
            from services.whisper_cpp import _find_ffmpeg
            ffmpeg = _find_ffmpeg()
            wav_fd, wav_name = tempfile.mkstemp(prefix="vt-diarization-audio-", suffix=".wav", dir=MODEL_TEMP_DIR)
            os.close(wav_fd)
            diarization_wav = Path(wav_name)
            converted = subprocess.run([ffmpeg, "-y", "-i", str(full_filepath), "-vn", "-ac", "1", "-ar", "16000",
                                        "-c:a", "pcm_s16le", str(diarization_wav)], capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", cwd=str(Path(__file__).resolve().parent))
            if converted.returncode:
                detail = (converted.stderr or converted.stdout or "FFmpeg audio conversion failed").strip()
                raise TaskFailure(f"FFmpeg could not prepare diarization audio: {detail[-1200:]}",
                                  "source_invalid", stage="source")
            transcription_path = str(diarization_wav)
        transcribe_kwargs = {"beam_size": args.beam_size, "language": language}
        if diarization_enabled:
            transcribe_kwargs["word_timestamps"] = True
        segments_iter, stats = model.transcribe(transcription_path, **transcribe_kwargs)
        if native_progress is not None and not uses_native_whisper_cpp:
            # Consume the real lazy generator so progress follows completed segments.
            segment_list = []
            for index, segment in enumerate(segments_iter, start=1):
                segment_list.append(segment)
                report_segment_progress(segment, index)
            segments = serialize_segments(segment_list, word_timestamps=diarization_enabled)
        else:
            segments = serialize_segments(list(segments_iter), word_timestamps=diarization_enabled)
    except Exception as exc:
        if diarization_wav is not None:
            diarization_wav.unlink(missing_ok=True)
        failure = _recognition_failure(exc, args, model)
        if failure is exc:
            raise
        raise failure from exc
    asr_effective_device = getattr(model, "effective_device", getattr(args, "device", "cpu"))
    asr_effective_backend = getattr(model, "effective_backend", "faster-whisper")
    asr_segments = [{**segment, "words": list(segment.get("words", []))} for segment in segments]
    if native_progress is not None and not uses_native_whisper_cpp:
        native_progress({"percent": 100.0 if total_seconds else None,
                         "processedSeconds": total_seconds, "totalSeconds": total_seconds})
    emit_event(args, "stage", stage="recognition", status="completed", message="Распознавание завершено.")
    effective_language = stats.language if args.language == "auto" else args.language
    effective_restore_punctuation = bool(args.restore_punctuation and effective_language.lower() == "ru")

    probability = stats.language_probability
    probability_text = f"{probability:f}" if probability is not None else "unavailable"
    print(f"\nDetected language '{stats.language}' with probability {probability_text}")

    postprocess_notice = None
    diarization = {"status": "disabled", "requestedDevice": getattr(args, "diarization_device", "auto"),
                   "effectiveDevice": None, "turns": []}
    diarization_text = None
    if diarization_enabled:
        emit_event(args, "stage", stage="diarization", status="in_progress",
                   message="Определяем, кто говорит.")
        try:
            from services.diarization import assign_speakers, run_diarization
            # faster-whisper keeps CTranslate2 weights loaded after transcription. Release
            # them before Nemotron allocates its model; CLI recognition has exited already.
            active_model = model.model if isinstance(model, AutoFallbackModel) else model
            ctranslate_model = getattr(active_model, "model", None)
            unload = getattr(ctranslate_model, "unload_model", None)
            if not callable(unload):
                unload = getattr(active_model, "unload_model", None)
            if callable(unload):
                unload()
            diarized = run_diarization(diarization_wav, getattr(args, "diarization_device", "auto"))
            turns = assign_speakers(asr_segments, diarized["intervals"])
            diarization = {"status": "completed", "requestedDevice": getattr(args, "diarization_device", "auto"),
                           "effectiveDevice": diarized["effectiveDevice"], "turns": turns}
            emit_event(args, "stage", stage="diarization", status="completed", message="Диаризация завершена.")
        finally:
            if diarization_wav is not None:
                diarization_wav.unlink(missing_ok=True)

        emit_event(args, "stage", stage="postprocess",
                   status="in_progress" if effective_restore_punctuation else "skipped",
                   message="Восстанавливаем пунктуацию в русских репликах." if effective_restore_punctuation else
                   ("Пунктуация пропущена: обнаружен не русский язык." if args.restore_punctuation else
                    "Восстановление пунктуации выключено."))
        if effective_restore_punctuation:
            for turn in turns:
                turn["text"] = restore_russian_segment_text(turn["text"])
            emit_event(args, "stage", stage="postprocess", status="completed", message="Пунктуация восстановлена.")
        elif args.restore_punctuation:
            postprocess_notice = "Пунктуация не восстановлена: обнаружен не русский язык."
            print(postprocess_notice)
        from services.diarization import render_turns
        diarization_text = render_turns(turns)
        combined_text = diarization_text
    else:
        # Keep the established non-diarization recognition and postprocessing order intact.
        emit_event(args, "stage", stage="postprocess",
                   status="in_progress" if effective_restore_punctuation else "skipped",
                   message="Восстанавливаем пунктуацию в русском тексте." if effective_restore_punctuation else
                   ("Пунктуация пропущена: обнаружен не русский язык." if args.restore_punctuation else
                    "Восстановление пунктуации выключено."))
        if effective_restore_punctuation:
            print("Russian text post-processing enabled.")
            print("Applying punctuation restoration to Russian segments.")
            postprocess_progress = progress_callback(args, "postprocess", "Восстанавливаем пунктуацию.") if getattr(args, "single_source", False) else None
            total_segments = len(segments)
            for index, segment in enumerate(segments, start=1):
                segment["text"] = restore_russian_segment_text(segment["text"])
                if postprocess_progress:
                    postprocess_progress({"percent": min(99.0, index * 100.0 / max(1, total_segments)),
                                          "message": f"Обработано фрагментов {index} из {total_segments}."})
            combined_text = build_polished_russian_text_from_segments([segment["text"] for segment in segments])
            if postprocess_progress:
                postprocess_progress({"percent": 100.0, "message": "Пунктуация восстановлена."})
            print("Russian text post-processing completed.")
            emit_event(args, "stage", stage="postprocess", status="completed", message="Пунктуация восстановлена.")
        else:
            combined_text = "\n".join(segment["text"] for segment in segments).strip()
            if args.restore_punctuation:
                postprocess_notice = "Пунктуация не восстановлена: обнаружен не русский язык."
                print(postprocess_notice)

    try:
        output_path, rendered_result = write_output(args, segments, combined_text, stats, filename,
                                                    diarization=diarization if diarization_enabled else None,
                                                    diarization_text=diarization_text)
    except TaskFailure:
        raise
    except Exception as exc:
        storage_errors = {getattr(errno, "ENOSPC", -1), getattr(errno, "EDQUOT", -1)}
        code = "storage_unavailable" if isinstance(exc, OSError) and exc.errno in storage_errors else "output_failed"
        raise TaskFailure(str(exc), code, stage="result") from exc
    elapsed_seconds = timeit.default_timer() - start_time

    print(filename, " took ", "{:.1f}".format(elapsed_seconds), " seconds")
    print("Saved result to", output_path)
    print("-------------------------------------------------------")

    result = {
        "input": str(filename),
        "output_path": str(output_path),
        "output_format": args.output_format,
        "detected_language": stats.language,
        "detected_language_probability": stats.language_probability,
        "text": combined_text,
        "rendered_result": rendered_result,
        "segments": segments if args.use_timestamps else [{"text": segment["text"]} for segment in segments],
        "elapsed_seconds": round(elapsed_seconds, 2),
        "effective_device": asr_effective_device,
        "effective_backend": asr_effective_backend,
    }
    result["diarization"] = diarization
    if postprocess_notice:
        result["notice"] = postprocess_notice

    emit_event(
        args,
        "result",
        stage="result",
        status="completed",
        message=f"Completed {Path(filename).name}",
        result=result,
        diarization=diarization,
    )
    return result


def _faster_model(args, device: str):
    threads = max(1, min(int(args.nproc or logical_cpu_count()), logical_cpu_count()))
    precision = "int8" if device == "cpu" and args.precision == "auto" else args.precision
    if device == "cuda":
        from services.cuda_runtime import register_local_cuda_runtime
        register_local_cuda_runtime()
    if device == "cpu":
        return whisper(args.model_size, cpu_threads=threads, num_workers=1, device="cpu", compute_type=precision,
                       download_root=str(HF_HUB_CACHE))
    return whisper(args.model_size, device=device, compute_type=precision, download_root=str(HF_HUB_CACHE))


class AutoFallbackModel:
    """Fallback to CPU if GPU initialization succeeded but inference later failed."""
    def __init__(self, model, args, backend: str):
        self.model = model
        self.args = args
        self.effective_device = "vulkan" if backend == "whisper.cpp Vulkan" else "cuda"
        self.effective_backend = backend

    def transcribe(self, *args, **kwargs):
        try:
            segments, stats = self.model.transcribe(*args, **kwargs)
            # faster-whisper performs GPU work lazily while iterating segments;
            # consume inside the fallback boundary so deferred OOM/runtime errors are caught.
            return list(segments), stats
        except Exception as exc:
            print(f"Automatic GPU transcription failed ({exc}); retrying this input on CPU.", flush=True)
            self.model = _faster_model(self.args, "cpu")
            self.effective_device = "cpu"
            self.effective_backend = "faster-whisper CPU (int8)"
            return self.model.transcribe(*args, **kwargs)


def initialize(args):
    print("--------------------INITIALIZING-----------------------")

    if not getattr(args, "single_source", False) and isfile(audio_file):
        os.remove(audio_file)

    requested = args.device
    choices = [item for item in get_device_details() if item["available"]]
    available = {item["id"] for item in choices}
    backend = requested
    if requested == "auto":
        backend = "vulkan" if "vulkan" in available else "cuda" if "cuda" in available else "cpu"
        if backend == "vulkan" and args.model_size not in VULKAN_MODELS:
            backend = "cuda" if "cuda" in available else "cpu"
            print(f"{args.model_size} is not supported by Vulkan; automatic mode selected {backend}.", flush=True)
        print(f"Automatic device selected: {backend}.", flush=True)
    elif requested in {"vulkan", "cuda"} and requested not in available:
        detail = next((item.get("reason") for item in choices if item["id"] == requested), None)
        # Include the unavailable entry's reason, which isn't part of `choices`.
        if not detail:
            all_details = get_device_details()
            detail = next((item.get("reason") for item in all_details if item["id"] == requested), None)
        raise TaskFailure(f"{requested.upper()} is unavailable: {detail or 'no usable device was detected.'}",
                          "device_unavailable", stage="recognition")
    if backend == "vulkan":
        if args.model_size not in VULKAN_MODELS:
            raise TaskFailure(
                f"Model '{args.model_size}' cannot run on Vulkan. Supported models: {', '.join(VULKAN_MODELS)}",
                "model_unavailable", stage="recognition",
            )
        try:
            model = WhisperCppModel(args.model_size, threads=args.nproc or logical_cpu_count(),
                                    progress=lambda message: print(message, flush=True))
            if getattr(args, "single_source", False):
                model.transcription_progress = progress_callback(args, "recognition", "Распознавание речи.")
            model.effective_device = "vulkan"
            model.effective_backend = "whisper.cpp Vulkan"
            return AutoFallbackModel(model, args, "whisper.cpp Vulkan") if requested == "auto" else model
        except Exception as exc:
            if requested != "auto":
                raise _initialization_failure(exc, requested) from exc
            print(f"Vulkan initialization failed ({exc}); trying the next available backend.", flush=True)
            if "cuda" in available:
                backend = "cuda"
            else:
                backend = "cpu"
    if backend == "cuda":
        try:
            model = _faster_model(args, "cuda")
            model.effective_device = "cuda"
            model.effective_backend = "faster-whisper CUDA"
            return AutoFallbackModel(model, args, "faster-whisper CUDA") if requested == "auto" else model
        except Exception as exc:
            if requested != "auto":
                raise _initialization_failure(exc, requested) from exc
            print(f"CUDA initialization failed ({exc}); retrying on CPU.", flush=True)
            backend = "cpu"
    try:
        model = _faster_model(args, "cpu")
    except Exception as exc:
        raise _initialization_failure(exc, "cpu") from exc
    model.effective_device = "cpu"
    model.effective_backend = "faster-whisper CPU (int8)"
    return model


def close(args):
    if not findarg(args, "keep") and isfile(audio_file):
        print("Delete temp file")
        os.remove(audio_file)


def add_media_files(args, media_files, debug=False, verbose=False):
    if findarg(args, "filename"):
        if validators.url(args.filename):
            args.url = args.filename
            args.audio_only = True
            args.restrict_filenames = True
            args.overwrite = True
            args.verbose = False
            args.audio_format = "mp3"

            emit_event(args, "stage", stage="source", status="in_progress", message=f"Downloading source from {args.filename}")

            download_progress = progress_callback(args, "source", "Загружаем исходный файл.") if getattr(args, "single_source", False) else None
            try:
                download = Download(args, debug, progress_callback=download_progress)
            except Exception as exc:
                raise TaskFailure(str(exc), "source_invalid", stage="source") from exc
            try:
                audio_file_list, retcode = download.run()
            except Exception as exc:
                raise TaskFailure(str(exc), "source_unavailable", stage="source") from exc

            if retcode == 0:
                media_files += audio_file_list
                emit_event(args, "stage", stage="source", status="completed", message=f"Downloaded {len(audio_file_list)} source file(s)")
            else:
                print(f"Could not process the URL, code {retcode} and audio file {audio_file_list} and args.output_name {args.output_name}")
                close(args)
                raise TaskFailure(f"URL download failed with code {retcode}",
                                  "source_unavailable", stage="source")

        elif Path(args.filename).exists():
            args.filename = Path(args.filename).resolve()
            media_files.append(str(args.filename))
            emit_event(args, "stage", stage="source", status="completed", message=f"Loaded local source {Path(args.filename).name}")
        else:
            raise TaskFailure("The selected source file does not exist.", "source_invalid", stage="source")

    elif findarg(args, "input_dir"):
        if Path(args.input_dir).exists():
            for filename in os.listdir(args.input_dir):
                filename = Path(args.input_dir, filename)
                if filename.is_file():
                    media_files.append(str(filename))
        else:
            raise TaskFailure(f"--input_dir argument does not specify a valid dir: {args.input_dir}",
                              "source_invalid", stage="source")

        emit_event(args, "stage", stage="source", status="completed", message=f"Discovered {len(media_files)} local file(s)")

    if len(media_files) == 0:
        if args.filename:
            print("URL is not valid")
            close(args)
            raise TaskFailure("URL is not valid", "source_invalid", stage="source")
        else:
            print("There were no media to process")
            close(args)
            raise TaskFailure("There were no media to process", "source_invalid", stage="source")

    print(f"Media files found {len(media_files)}")


def preserve_audio_copy(args, source_media_file: str) -> str | None:
    if not isfile(audio_file):
        return None

    target = Path(args.output_dir, Path(source_media_file).stem + Path(audio_file).suffix)
    target = next_available_path(target)
    shutil.move(audio_file, target)
    print(f"Saved intermediate audio to {target}")
    return str(target)


def build_argument_parser():
    parser = argparse.ArgumentParser("Generates subtitiles of the video file as an input")
    parser.add_argument("-f", "--filename", help="Name of the media file stored in the filesystem or URL of a video/audio file that needs to subtitles. URL can also be a list of media")
    parser.add_argument("-i", "--input_dir", help="Input directory where video files are. --filename overrides this")
    parser.add_argument("-af", "--audio_filter", help="Audio or video filters to use before transcription (for ffmpeg), no spaces, just comma-separated")
    parser.add_argument("-o", "--output_name", help="Output filename in case of issues with title")
    parser.add_argument("-od", "--output_dir", help="Ouput directory", default=os.getcwd())
    parser.add_argument("-l", "--language", help="Language to be translated from", default="ru", type=str)
    parser.add_argument("-b", "--beam_size", help="Beam size parameter or best_of equivalent from Open-AI whisper", type=int, default=5)
    parser.add_argument("-p", "--precision", help="Precision to use to create the model", type=str, default="auto")
    parser.add_argument("-d", "--device", help="Device to use", choices=["auto", "cpu", "vulkan", "cuda"], default="auto")
    parser.add_argument("-s", "--model_size", help="Size of the model, default is small.", choices=sizes_supported(), nargs="?", default="small")
    parser.add_argument("--start", help="Start time in 00:00:00 format", type=AudioProcess.valid_time)
    parser.add_argument("--end", help="End time in 00:00:00 format", type=AudioProcess.valid_time)
    parser.add_argument("-n", "--nproc", help="Number of CPUs to use", default=logical_cpu_count(), type=int)
    parser.add_argument("-k", "--keep", help="Keep intermediate files", action="store_true")
    parser.add_argument("--verbose", help="Verbose print from dependent processes", action="store_true")
    parser.add_argument("--quiet", help="Debug print off", action="store_true")
    parser.add_argument("--playlist_start", help="Starting position from a list of media, to start downloading from")
    parser.add_argument("--playlist_end", help="Ending position from a list of media, to stop downloading at")
    parser.add_argument("--codec", help="Audio codec to use")
    parser.add_argument("--bitrate", help="Audio bitrate to use")
    parser.add_argument("--output_format", help="Output format to create", choices=["txt", "srt", "json"], default="srt")
    parser.add_argument("--no_timestamps", help="Disable timestamps in txt/json output. Ignored for srt.", action="store_true")
    parser.add_argument("--restore_punctuation", help="Restore punctuation for explicitly selected Russian transcription.", action="store_true")
    parser.add_argument("--diarization", help="Enable Nemotron speaker diarization.", action="store_true", default=False)
    parser.add_argument("--diarization-device", help="Nemotron device ID (auto, cpu, cuda:N, or vulkan:N).", type=str, default="auto")
    parser.add_argument("--single-source", help=argparse.SUPPRESS, action="store_true")
    parser.add_argument("--web_json", help=argparse.SUPPRESS, action="store_true")
    return parser


def main():
    global audio_file
    media_files = []

    parser = build_argument_parser()

    args = parser.parse_args()
    configure_web_stdio(args)
    args.use_timestamps = False if args.no_timestamps else True

    if args.output_format == "srt":
        args.use_timestamps = True

    if "model_size" in args and getattr(args, "model_size") is None:
        supported = sizes_supported()
        print("\nSupported size in faster-whisper")
        print("-------------------------------------------------------")
        for size in supported:
            print(f"*  {size}")
        close(args)
        return 0

    if findarg(args, "output_name"):
        args.output_name, args.output_dir = get_fullpath(args.output_dir, args.output_name)

    if args.quiet is False:
        print("-----------------------SETTINGS------------------------")
        arguments = vars(args)
        for arg in arguments:
            print(arg, "\t", getattr(args, arg))

    audio_file = str(Path(args.output_dir, audio_file))
    print(f"audio_file={audio_file}")

    try:
        try:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TaskFailure(str(exc), "output_failed", stage="result") from exc
        add_media_files(args, media_files, debug=not args.quiet, verbose=args.verbose)
        model = None if getattr(args, "single_source", False) or args.diarization else initialize(args)
        results = []

        for media_file in media_files:
            if args.quiet is False:
                print(f"Processing file {media_file} and using audio filter")

            if args.quiet is False:
                if args.output_name:
                    print(f"Set file {args.output_name}")

                if args.audio_filter:
                    print(f"Set filter {args.audio_filter}")

                if args.codec:
                    print(f"Set codec {args.codec}")

                if args.bitrate:
                    print(f"Set bitrate {args.bitrate}")

            transcription_input = media_file
            extracted_audio = False
            audio_processor = AudioProcess(args)

            if (
                not audio_processor.audio_only(media_file)
                or findarg(args, "audio_filter")
                or findarg(args, "start")
                or findarg(args, "end")
                or findarg(args, "codec")
                or findarg(args, "bitrate")
            ):
                print("Extracting audio")
                emit_event(args, "stage", stage="audio", status="in_progress", message=f"Extracting audio from {Path(media_file).name}")
                audio_progress = progress_callback(args, "audio", "Подготавливаем аудио.") if getattr(args, "single_source", False) else None
                try:
                    audio_processor.extract_audio(input_filepath=media_file, output_filepath=audio_file, overwrite=True,
                                                  progress_callback=audio_progress)
                except TaskFailure:
                    raise
                except Exception as exc:
                    raise TaskFailure(str(exc), "source_invalid", stage="source") from exc
                transcription_input = audio_file
                extracted_audio = True
                emit_event(args, "stage", stage="audio", status="completed", message=f"Audio ready for {Path(media_file).name}")
            elif getattr(args, "single_source", False):
                emit_event(args, "stage", stage="audio", status="skipped", message="Подготовка аудио не требуется.")

            if getattr(args, "single_source", False) or args.diarization:
                emit_event(args, "stage", stage="recognition", status="in_progress",
                           message="Загружаем модель в память.")
                model = initialize(args)

            result = transcribe(args, model, transcription_input, media_file)

            if extracted_audio and isfile(audio_file):
                if findarg(args, "keep"):
                    result["saved_audio"] = preserve_audio_copy(args, media_file)
                else:
                    os.remove(audio_file)

            results.append(result)

        emit_event(args, "complete", stage="result", status="completed", message="All media processed", results=results)
        close(args)
        print("Done.")
        return 0

    except Exception as exc:
        failure = exc if isinstance(exc, TaskFailure) else TaskFailure(str(exc), "internal_error", stage="result")
        emit_event(args, "error", stage=failure_stage(failure), status="failed",
                   code=failure.code, message=str(exc))
        close(args)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
