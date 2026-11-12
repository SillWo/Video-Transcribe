"""Short-lived isolated workers for the additional ASR runtimes."""

from __future__ import annotations

import argparse
import array
import ctypes
import dataclasses
import json
import importlib
import math
import os
import re
import struct
import sys
import tempfile
import threading
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from services.asr_resources import alignment_model_dir, model_paths
from services.parakeet_timestamps import recognize_with_native_durations, words_from_timestamped_result

APP_ROOT = Path(__file__).resolve().parents[1]
EVENT_PREFIX = "__VT_ASR__ "
SAMPLE_RATE = 16_000
MAX_CHUNK_SECONDS = 30
MIN_RETRY_SECONDS = 3
NATIVE_NAME = "qwen3-asr-1.7b"
PARAKEET_NAME = "parakeet-tdt-0.6b-v3"
LANGID_NAME = "lang-id-voxlingua107-ecapa-F32.gguf"
PARAKEET_ONNX_NAME = "nemo-parakeet-tdt-0.6b-v3"
PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
}
CPU_HEAVY_OPS = {
    "Conv", "ConvTranspose", "MatMul", "Gemm", "LSTM", "Attention",
    "MultiHeadAttention", "GRU", "RNN", "FusedMatMul", "FusedConv", "DmlFusedConv",
}


class WorkerError(RuntimeError):
    """A failure that should be returned to the parent process as JSON."""


@dataclasses.dataclass
class AudioChunk:
    start_frame: int
    end_frame: int
    pcm_i16: array.array
    pcm_f32: array.array

    @property
    def duration(self) -> float:
        return (self.end_frame - self.start_frame) / SAMPLE_RATE

    @property
    def start(self) -> float:
        return self.start_frame / SAMPLE_RATE

    @property
    def end(self) -> float:
        return self.end_frame / SAMPLE_RATE

    @property
    def is_silent(self) -> bool:
        return all(sample == 0 for sample in self.pcm_i16)


def configure_runtime(runtime: str, *, verify_gpu: bool = False) -> None:
    """Create process-local caches and set them before importing ML packages."""
    base = APP_ROOT / ".runtime" / "asr-worker" / runtime
    cache = base / "cache"
    scratch = APP_ROOT / ".runtime" / "asr-worker" / "tmp"
    for directory in (cache, scratch, cache / "huggingface", cache / "huggingface" / "hub",
                      cache / "torch", cache / "xdg"):
        directory.mkdir(parents=True, exist_ok=True)
    for key, value in {
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache / "huggingface" / "hub",
        "TORCH_HOME": cache / "torch",
        "XDG_CACHE_HOME": cache / "xdg",
        "TMP": scratch,
        "TEMP": scratch,
        "TMPDIR": scratch,
    }.items():
        os.environ[key] = str(value)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if runtime == "native":
        os.environ["GGML_SCHED_DEBUG"] = "2" if verify_gpu else "0"
    if tempfile.tempdir is not None:
        tempfile.tempdir = None


def emit(event_type: str, message: str, *, stage: str = "recognition",
         processed: float | None = None, total: float | None = None) -> None:
    payload: dict[str, Any] = {"type": event_type, "stage": stage, "message": message}
    if processed is not None:
        payload["processedSeconds"] = processed
    if total is not None:
        payload["totalSeconds"] = total
    print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
                if not field.name.startswith("_")}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def peak_memory_bytes() -> dict[str, int | None]:
    """Read this worker's peak memory without an extra runtime dependency."""
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        try:
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
            if ok:
                return {"peakWorkingSetBytes": int(counters.PeakWorkingSetSize),
                        "peakPrivateBytes": int(counters.PeakPagefileUsage)}
        except (AttributeError, OSError):
            pass
        return {"peakWorkingSetBytes": None, "peakPrivateBytes": None}
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                working = int(line.split()[1]) * 1024
                return {"peakWorkingSetBytes": working, "peakPrivateBytes": None}
    except (OSError, ValueError, IndexError):
        pass
    return {"peakWorkingSetBytes": None, "peakPrivateBytes": None}


def _choose_quiet_boundary(samples: array.array, *, search_frames: int = SAMPLE_RATE) -> int:
    """Pick a nearby quiet point without dropping or overlapping any samples."""
    count = len(samples)
    minimum = MIN_RETRY_SECONDS * SAMPLE_RATE
    if count - minimum < 1:
        return count
    window = max(1, SAMPLE_RATE // 100)  # 10 ms energy windows
    lo = max(minimum, count - search_frames)
    lo = min(lo, count - window)
    best_frame = count
    best_energy = None
    for start in range(lo, count - window + 1, window):
        energy = sum(abs(samples[index]) for index in range(start, start + window))
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_frame = start + window // 2
    return min(max(minimum, best_frame), count)


def iter_audio_chunks(path: Path, max_seconds: int = MAX_CHUNK_SECONDS) -> Iterable[AudioChunk]:
    """Yield exact, contiguous PCM chunks while keeping at most one 30s block."""
    max_frames = max_seconds * SAMPLE_RATE
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
            raise WorkerError("Worker audio must be uncompressed PCM16 mono 16000 Hz WAV")
        total_frames = wav.getnframes()
        if total_frames <= 0:
            raise WorkerError("Audio WAV contains no samples")
        start_frame = 0
        pending = b""
        while start_frame < total_frames:
            pending_frames = len(pending) // 2
            read_limit = max_frames - pending_frames
            block = pending + wav.readframes(read_limit)
            available = len(block) // 2
            if available <= 0:
                if start_frame < total_frames:
                    raise WorkerError(
                        f"WAV ended at frame {start_frame}, before its declared {total_frames} frames"
                    )
                break
            if len(block) % 2:
                raise WorkerError("WAV ended on a partial PCM16 sample")
            pcm_i16 = array.array("h")
            pcm_i16.frombytes(block)
            if sys.byteorder != "little":
                pcm_i16.byteswap()
            split = available
            if available == max_frames and start_frame + available < total_frames:
                split = _choose_quiet_boundary(pcm_i16)
            chunk_i16 = array.array("h", pcm_i16[:split])
            pcm_f32 = array.array("f", (sample / 32768.0 for sample in chunk_i16))
            end_frame = start_frame + split
            yield AudioChunk(start_frame, end_frame, chunk_i16, pcm_f32)
            pending = block[split * 2:]
            start_frame = end_frame


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
            raise WorkerError("Worker audio must be uncompressed PCM16 mono 16000 Hz WAV")
        if wav.getnframes() <= 0:
            raise WorkerError("Audio WAV contains no samples")
        return wav.getnframes() / SAMPLE_RATE


def _is_native_gpu(device: Any, backend: str) -> bool:
    if backend == "cpu":
        return getattr(device, "device_type", None) == "cpu"
    return (str(getattr(device, "kind", "")).casefold() == backend
            and getattr(device, "device_type", None) in ("gpu", "igpu"))


def _select_native_device(transcribe_cpp: Any, backend: str, index: int):
    candidates = [device for device in transcribe_cpp.backends() if _is_native_gpu(device, backend)]
    if index < 0 or index >= len(candidates):
        raise WorkerError(f"Requested {backend} device index {index} is unavailable; found {len(candidates)} matching devices")
    return candidates[index]


def _native_descriptor(device: Any) -> dict[str, Any]:
    return {field.name: _jsonable(getattr(device, field.name)) for field in dataclasses.fields(device)
            if not field.name.startswith("_")}


class QwenPlacementDiagnostics:
    """Count scheduler node records across arbitrarily fragmented native callbacks."""

    _header = re.compile(r"node\s+#\s*\d+\s+\(")

    _node = re.compile(
        r"node #\s*(?P<index>\d+)\s+\((?P<op>[^)]*)\):\s*"
        r"(?P<name>.*?)\s+\([^)]*\)\s+\[(?P<backend>[^\s\]]+)\s+"
        r"[^\]]*\]\s+use=\d+,c=(?P<compute>[01]):"
    )
    _heavy = re.compile(r"MUL_MAT|FLASH_ATTN|CONV|MATMUL|GEMM", re.IGNORECASE)
    _encoder = re.compile(r"encoder|audio[_./-]?enc|\benc[._-]", re.IGNORECASE)
    _decoder = re.compile(
        r"\b(?:lm|prefill|decode|decoder)\b|language[_./-]?model|"
        r"\b(?:dec|decoder)[._/-]|\bblk[._-]",
        re.IGNORECASE,
    )

    def __init__(self, selected_backend: str):
        self.selected_backend = selected_backend
        self.lock = threading.Lock()
        self.pending = ""
        self.gpu_by_stage: Counter[str] = Counter()
        self.gpu_heavy_by_stage: Counter[str] = Counter()
        self.cpu_assignments = 0
        self.gpu_assignments = 0
        self.other_gpu_assignments: Counter[str] = Counter()

    def __call__(self, _level: int, message: str) -> None:
        with self.lock:
            self.pending += str(message)
            headers = list(self._header.finditer(self.pending))
            while len(headers) >= 2:
                cut = headers[1].start()
                self._consume_record(self.pending[:cut])
                self.pending = self.pending[cut:]
                headers = list(self._header.finditer(self.pending))

    def _consume_record(self, record: str) -> None:
        match = self._node.search(record)
        if not match or match.group("compute") != "1":
            return
        backend_name = match.group("backend")
        folded = backend_name.casefold()
        if folded == "cpu":
            self.cpu_assignments += 1
            return
        is_requested_backend = (
            self.selected_backend == "vulkan" and folded.startswith(("vulk", "gpu"))
        ) or (
            self.selected_backend == "cuda" and folded.startswith(("cuda", "gpu"))
        )
        if not re.match(r"^(cuda|vulk|rocm|metal|gpu|sycl)", folded):
            return
        if not is_requested_backend:
            self.other_gpu_assignments[backend_name] += 1
            return
        self.gpu_assignments += 1
        # Node names are often generic (node_123); GGML_SCHED_DEBUG emits
        # tensor/source names after each header, which identify encoder/decoder.
        stage_text = f"{match.group('name')} {match.group('op')} {record}".lower()
        encoder = bool(self._encoder.search(stage_text))
        decoder = bool(self._decoder.search(stage_text))
        if encoder:
            self.gpu_by_stage["encoder"] += 1
            if self._heavy.search(match.group("op")):
                self.gpu_heavy_by_stage["encoder"] += 1
        if decoder:
            self.gpu_by_stage["decoder"] += 1
            if self._heavy.search(match.group("op")):
                self.gpu_heavy_by_stage["decoder"] += 1

    def finish(self) -> dict[str, Any]:
        with self.lock:
            if self.pending.strip():
                self._consume_record(self.pending)
                self.pending = ""
            verified = (
                self.selected_backend in {"vulkan", "cuda"}
                and self.gpu_heavy_by_stage["encoder"] > 0
                and self.gpu_heavy_by_stage["decoder"] > 0
                and self.gpu_assignments > self.cpu_assignments
            )
            return {
                "gpu_verified": verified,
                "verification_basis": (
                    "Successful model inference with GGML_SCHED_DEBUG=2 encoder and language-model "
                    "heavy-op assignments on the selected GPU, plus nonzero native encode/decode timings."
                    if verified else
                    "No complete encoder and language-model GPU assignment evidence was captured; "
                    "native scheduler assignments are reported for diagnostics."
                ),
                "gpu_assignments": self.gpu_assignments,
                "gpu_assignments_by_stage": dict(self.gpu_by_stage),
                "gpu_heavy_assignments_by_stage": dict(self.gpu_heavy_by_stage),
                "cpu_assignments": self.cpu_assignments,
                "other_gpu_assignments": dict(self.other_gpu_assignments),
            }


def _native_inventory(transcribe_cpp: Any) -> list[dict[str, Any]]:
    return [
        {"backend": device.kind, "index": index, "name": device.name, "deviceId": device.device_id,
         "deviceType": device.device_type, "description": device.description}
        for index, device in enumerate(transcribe_cpp.backends())
    ]


def _onnx_inventory(ort: Any) -> list[dict[str, Any]]:
    entries = []
    for backend, provider in PROVIDERS.items():
        if provider in ort.get_available_providers():
            entries.append({"backend": backend, "index": 0, "name": provider,
                            "deviceId": "0" if backend != "cpu" else None,
                            "deviceType": "cpu" if backend == "cpu" else "gpu"})
    return entries


def list_devices(model_id: str) -> dict[str, Any]:
    if model_id == NATIVE_NAME:
        configure_runtime("native")
        import transcribe_cpp
        devices = _native_inventory(transcribe_cpp)
    elif model_id == PARAKEET_NAME:
        configure_runtime("parakeet")
        import onnxruntime as ort
        devices = _onnx_inventory(ort)
    else:
        raise WorkerError(f"Unsupported ASR model id: {model_id}")
    return {"ok": True, "model": model_id, "devices": devices}


def _native_timings(result: Any) -> dict[str, float]:
    timings = getattr(result, "timings", None)
    if timings is None:
        return {}
    return _jsonable(timings)


def _result_segments(result: Any, chunk: AudioChunk) -> list[dict[str, Any]]:
    text = str(getattr(result, "text", "") or "").strip()
    source_segments = getattr(result, "segments", None)
    rows: list[dict[str, Any]] = []
    for segment in source_segments or ():
        raw_start = getattr(segment, "t0_ms", None)
        raw_end = getattr(segment, "t1_ms", None)
        try:
            local_start = max(0.0, float(raw_start) / 1000.0)
            local_end = min(chunk.duration, float(raw_end) / 1000.0)
        except (TypeError, ValueError):
            continue
        segment_text = str(getattr(segment, "text", "") or "").strip()
        if segment_text and math.isfinite(local_start) and math.isfinite(local_end) and local_end > local_start:
            rows.append({"start": chunk.start + local_start, "end": chunk.start + local_end, "text": segment_text})
    if rows and text and " ".join(" ".join(row["text"].split()) for row in rows) != " ".join(text.split()):
        rows = []
    if not rows and text:
        rows.append({"start": chunk.start, "end": chunk.end, "text": text})
    return rows


def _split_retry_chunk(chunk: AudioChunk) -> tuple[AudioChunk, AudioChunk] | None:
    frames = len(chunk.pcm_i16)
    min_frames = MIN_RETRY_SECONDS * SAMPLE_RATE
    if frames < min_frames * 2:
        return None
    center = frames // 2
    search_radius = SAMPLE_RATE
    lo = max(min_frames, center - search_radius)
    hi = min(frames - min_frames, center + search_radius)
    segment = chunk.pcm_i16[lo:hi]
    if segment:
        near = _choose_quiet_boundary(segment, search_frames=len(segment)) + lo
        split_at = min(max(min_frames, near), frames - min_frames)
    else:
        split_at = center
    absolute = chunk.start_frame + split_at
    return (
        AudioChunk(chunk.start_frame, absolute, array.array("h", chunk.pcm_i16[:split_at]),
                   array.array("f", chunk.pcm_f32[:split_at])),
        AudioChunk(absolute, chunk.end_frame, array.array("h", chunk.pcm_i16[split_at:]),
                   array.array("f", chunk.pcm_f32[split_at:])),
    )


def _is_output_limit_error(exc: BaseException, transcribe_cpp: Any) -> bool:
    error_types = tuple(
        candidate for candidate in (getattr(transcribe_cpp, "OutputTruncated", None),
                                    getattr(transcribe_cpp, "OutputRepetition", None))
        if isinstance(candidate, type)
    )
    return bool(error_types) and isinstance(exc, error_types)


def _native_transcribe(model_path: Path, audio_path: Path, device: str, index: int, threads: int,
                       language: str | None, verify_gpu: bool) -> dict[str, Any]:
    configure_runtime("native", verify_gpu=verify_gpu)
    import transcribe_cpp

    if getattr(transcribe_cpp, "__version__", None) != "0.4.1":
        raise WorkerError(f"Expected transcribe-cpp 0.4.1, found {getattr(transcribe_cpp, '__version__', 'unknown')}")
    selected = _select_native_device(transcribe_cpp, device, index)
    placement = QwenPlacementDiagnostics(device) if verify_gpu and device != "cpu" else None
    if placement:
        transcribe_cpp.set_log_callback(placement)
    duration = _wav_duration(audio_path)
    started = time.perf_counter()
    load_start = started
    records: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    timings = Counter()
    try:
        with transcribe_cpp.Model(str(model_path), device=selected) as model:
            load_seconds = time.perf_counter() - load_start
            actual = model.device
            if not _is_native_gpu(actual, device) or (
                getattr(actual, "device_id", None) != getattr(selected, "device_id", None)
                or getattr(actual, "name", None) != getattr(selected, "name", None)
            ):
                raise WorkerError(f"Native model resolved to unexpected device {actual!r}; no fallback is allowed")
            with model.session(n_threads=threads) as session:
                processed = 0
                total_chunks = math.ceil(duration / MAX_CHUNK_SECONDS)
                for chunk_index, chunk in enumerate(iter_audio_chunks(audio_path), start=1):
                    if chunk.is_silent:
                        emit("progress", "Тишина пропущена без распознавания.", processed=chunk.end, total=duration)
                        processed = chunk.end_frame
                        continue

                    def infer(current: AudioChunk) -> list[dict[str, Any]]:
                        inference_start = time.perf_counter()
                        try:
                            result = session.run(current.pcm_f32, language=language)
                        except Exception as exc:
                            timings["inference"] += time.perf_counter() - inference_start
                            if not _is_output_limit_error(exc, transcribe_cpp):
                                raise
                            children = _split_retry_chunk(current)
                            if children is None:
                                raise WorkerError(
                                    f"{type(exc).__name__} persists for a {current.duration:.2f}s chunk; "
                                    f"minimum retry chunk is {MIN_RETRY_SECONDS}s"
                                ) from exc
                            return infer(children[0]) + infer(children[1])
                        timings["inference"] += time.perf_counter() - inference_start
                        native_timing = _native_timings(result)
                        for key, value in native_timing.items():
                            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                                timings[f"native_{key}"] = float(value)
                        language_code = getattr(result, "language", None)
                        text = str(getattr(result, "text", "") or "").strip()
                        if language_code and text:
                            observations.append({"start": current.start, "end": current.end,
                                                 "language": str(language_code), "probability": None})
                        return _result_segments(result, current)

                    rows = infer(chunk)
                    records.extend(rows)
                    processed = chunk.end_frame
                    emit("progress", f"Обработан аудиофрагмент {chunk_index} из примерно {total_chunks}.",
                         processed=chunk.end, total=duration)
                native_device = _native_descriptor(actual)
            # The v0.4.1 ASR Session has no timings property. Timings are
            # attached to each native result and collected in `timings` above.
            session_timings = None
        text = "\n".join(item["text"] for item in records).strip()
        timing_detail = {"load": load_seconds, "inference": timings.get("inference", 0.0),
                         "wall": time.perf_counter() - started}
        native_timings = {key.removeprefix("native_"): value for key, value in timings.items()
                          if key.startswith("native_")}
        diagnostics = {
            "runtime": "transcribe-cpp",
            "package_version": transcribe_cpp.__version__,
            "native_version": transcribe_cpp.native_version(),
            "native_commit": transcribe_cpp.native_commit(),
            "native_library": transcribe_cpp.library_path(),
            "model_device": native_device,
            "native_timings": session_timings or native_timings,
            "precision": "F16 GGUF",
        }
        if placement:
            placement_result = placement.finish()
            native_timings_for_verification = session_timings or native_timings
            native_encode = native_timings_for_verification.get("encode_ms", 0.0)
            native_decode = native_timings_for_verification.get("decode_ms", 0.0)
            placement_result["gpu_verified"] = bool(
                placement_result["gpu_verified"] and float(native_encode) > 0 and float(native_decode) > 0
            )
            diagnostics.update(placement_result)
        else:
            diagnostics.update({"gpu_verified": False, "verification_basis": "GPU verification was not requested."})
        return {
            "ok": True, "text": text, "segments": records,
            "language_observations": observations,
            "language": observations[-1]["language"] if observations else None,
            "language_probability": None,
            "duration": duration,
            "effective_device": device,
            "effective_backend": f"transcribe.cpp {device} (F16)",
            "precision": "F16",
            "timing_seconds": timing_detail,
            "diagnostics": diagnostics,
        }
    finally:
        if placement:
            transcribe_cpp.set_log_callback(None)


def _onnx_create_model(model_dir: Path, device: str, device_index: int, threads: int,
                       *, verify_gpu: bool):
    import onnx_asr
    import onnxruntime as ort

    provider = PROVIDERS.get(device)
    available = list(ort.get_available_providers())
    if provider is None or provider not in available:
        raise WorkerError(f"Requested ONNX provider {provider or device} is unavailable; available providers: {available}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_profiling = bool(verify_gpu and device != "cpu")
    profile_dir = APP_ROOT / ".runtime" / "asr-worker" / "parakeet" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    if options.enable_profiling:
        options.profile_file_prefix = str(profile_dir / f"parakeet-{device}")
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL if device == "directml" else ort.ExecutionMode.ORT_PARALLEL
    if device == "directml":
        options.enable_mem_pattern = False
    providers: list[Any] = [(provider, {"device_id": str(device_index)})] if device in {"directml", "cuda"} else [provider]
    cpu_options = ort.SessionOptions()
    cpu_options.intra_op_num_threads = threads
    cpu_options.inter_op_num_threads = 1
    started = time.perf_counter()
    model = onnx_asr.load_model(
        PARAKEET_ONNX_NAME,
        str(model_dir),
        sess_options=options,
        providers=providers,
        preprocessor_config={"use_numpy_preprocessors": True},
        resampler_config={"sess_options": cpu_options, "providers": [PROVIDERS["cpu"]]},
    ).with_timestamps()
    if device != "cpu":
        asr_sessions = _onnx_asr_sessions(model)
        if set(asr_sessions) != {"encoder", "decoder"}:
            raise WorkerError("ONNX ASR encoder and decoder sessions could not be verified")
        for role, session in asr_sessions.items():
            if provider not in session.get_providers():
                raise WorkerError(f"ONNX ASR {role} session did not select requested GPU provider {provider}")
            disable = getattr(session, "disable_fallback", None)
            if not callable(disable):
                raise WorkerError(f"ONNX ASR {role} session cannot disable CPU fallback")
            disable()
    return model, ort, time.perf_counter() - started, profile_dir


def _find_onnx_sessions(root: Any) -> list[Any]:
    found: list[Any] = []
    seen: set[int] = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if callable(getattr(value, "get_providers", None)) and callable(getattr(value, "end_profiling", None)):
            found.append(value)
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (tuple, list, set)):
            stack.extend(value)
        elif hasattr(value, "__dict__") and not isinstance(value, (str, bytes, Path, type)):
            stack.extend(vars(value).values())
    return found


def _onnx_asr_sessions(root: Any) -> dict[str, Any]:
    """Find the encoder and decoder sessions, excluding auxiliary sessions."""
    found: dict[str, Any] = {}
    seen: set[int] = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        for attr, role in (("_encoder", "encoder"), ("_decoder_joint", "decoder")):
            session = getattr(value, attr, None)
            if session is not None:
                found[role] = session
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (tuple, list, set)):
            stack.extend(value)
        elif hasattr(value, "__dict__") and not isinstance(value, (str, bytes, Path, type)):
            stack.extend(vars(value).values())
    return found


def _onnx_preprocessor_owner(root: Any) -> Any:
    seen: set[int] = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if hasattr(value, "_preprocessor"):
            return value
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (tuple, list, set)):
            stack.extend(value)
        elif hasattr(value, "__dict__") and not isinstance(value, (str, bytes, Path, type)):
            stack.extend(vars(value).values())
    raise WorkerError("ONNX ASR preprocessor owner could not be found")


def _onnx_preprocessor_classes() -> tuple[Any, Any]:
    """Resolve preprocessor implementations from the pinned onnx-asr package."""
    return (
        importlib.import_module("onnx_asr.preprocessors.preprocessor").OnnxPreprocessor,
        importlib.import_module("onnx_asr.preprocessors.numpy_preprocessor").NemoPreprocessorNumpy,
    )


def _configure_onnx_preprocessor(model: Any, ort: Any, device: str, index: int,
                                 threads: int, profile_dir: Path, *, verify_gpu: bool = False) -> dict[str, Any]:
    """Use NeMo's convolutional ONNX preprocessor on GPU with lazy NumPy fallback."""
    state: dict[str, Any] = {"device": "cpu", "backend": "numpy", "fallback_reason": None}
    if device == "cpu":
        return state
    owner = _onnx_preprocessor_owner(model)
    original = owner._preprocessor
    provider = PROVIDERS[device]
    try:
        onnx_cls, numpy_cls = _onnx_preprocessor_classes()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.enable_profiling = bool(verify_gpu)
        profile_dir.mkdir(parents=True, exist_ok=True)
        options.profile_file_prefix = str(profile_dir / f"parakeet-preprocessor-{device}")
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL if device == "directml" else ort.ExecutionMode.ORT_PARALLEL
        if device == "directml":
            options.enable_mem_pattern = False
        gpu_preprocessor = onnx_cls("nemo128_conv", {
            "sess_options": options,
            "providers": [(provider, {"device_id": str(index)})],
        })
        session = [gpu_preprocessor._preprocessor]
        if not session:
            raise WorkerError("GPU preprocessor did not expose its ONNX session")
        if any(provider not in item.get_providers() for item in session):
            raise WorkerError(f"GPU preprocessor did not select requested provider {provider}")
        for item in session:
            disable = getattr(item, "disable_fallback", None)
            if not callable(disable):
                raise WorkerError("GPU preprocessor cannot disable CPU fallback")
            disable()
        state.update({"device": device, "backend": provider, "session": session[0] if len(session) == 1 else None,
                      "profile_counts": {}})
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        state.update({"device": "cpu", "backend": "numpy", "fallback_reason": reason})
        emit("warning", f"GPU preprocessing unavailable; using CPU preprocessing: {reason}", stage="preprocessing")
        owner._preprocessor = original
        return state

    numpy_preprocessor: list[Any] = []
    def preprocess(*args: Any, **kwargs: Any) -> Any:
        try:
            return gpu_preprocessor(*args, **kwargs)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if not numpy_preprocessor:
                numpy_preprocessor.append(numpy_cls("nemo128"))
            state.update({"device": "cpu", "backend": "numpy", "fallback_reason": reason})
            emit("warning", f"GPU preprocessing failed; using CPU preprocessing: {reason}", stage="preprocessing")
            owner._preprocessor = numpy_preprocessor[0]
            return owner._preprocessor(*args, **kwargs)
    owner._preprocessor = preprocess
    return state


def _finish_preprocessor_profile(preprocessing: dict[str, Any], *, verify_gpu: bool) -> None:
    session = preprocessing.pop("session", None)
    if not verify_gpu or session is None:
        return
    profile_path: Path | None = None
    try:
        raw_path = session.end_profiling()
        if isinstance(raw_path, (str, os.PathLike)) and str(raw_path).strip():
            profile_path = Path(raw_path)
            preprocessing["profile_counts"] = dict(_profile_counts(profile_path))
    except Exception as exc:
        preprocessing["profile_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if profile_path is not None:
            try:
                profile_path.unlink(missing_ok=True)
            except OSError as exc:
                preprocessing["profile_error"] = f"cleanup {type(exc).__name__}: {exc}"


def _profile_counts(path: Path) -> Counter[tuple[str, str, str]]:
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return Counter()
    counts: Counter[tuple[str, str, str]] = Counter()
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict) or event.get("cat") != "Node":
                continue
            args = event.get("args") or {}
            if not isinstance(args, dict) or not args.get("provider"):
                continue
            counts[(str(args["provider"]), str(args.get("op_name", "Unknown")),
                    str(event.get("name", "")))] += 1
    return counts


def _onnx_transcribe(model_dir: Path, audio_path: Path, device: str, index: int,
                     threads: int, verify_gpu: bool) -> dict[str, Any]:
    configure_runtime("parakeet")
    import numpy as np
    import onnx_asr
    import onnxruntime as ort

    duration = _wav_duration(audio_path)
    model, _ort, load_seconds, profile_dir = _onnx_create_model(
        model_dir, device, index, threads, verify_gpu=verify_gpu
    )
    preprocessing_start = time.perf_counter()
    preprocessing = _configure_onnx_preprocessor(model, _ort, device, index, threads, profile_dir,
                                                 verify_gpu=verify_gpu)
    preprocessing_seconds = time.perf_counter() - preprocessing_start
    inference_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    timestamp_warnings: list[str] = []
    processed = 0.0
    for chunk_index, chunk in enumerate(iter_audio_chunks(audio_path), start=1):
        if chunk.is_silent:
            emit("progress", "Тишина пропущена без распознавания.", processed=chunk.end, total=duration)
            continue
        waveform = np.asarray(chunk.pcm_f32, dtype=np.float32)
        recognized, native_durations = recognize_with_native_durations(
            model, lambda: model.recognize(waveform, sample_rate=SAMPLE_RATE)
        )
        text = str(getattr(recognized, "text", "") or "").strip()
        if text:
            asr = model.asr
            try:
                frame_seconds = float(asr.window_step) * float(asr._subsampling_factor)
            except (AttributeError, TypeError, ValueError, OverflowError):
                frame_seconds = None
            words, warning = words_from_timestamped_result(
                recognized, native_durations, frame_seconds=frame_seconds,
                chunk_start=chunk.start, chunk_duration=chunk.duration,
            )
            row = {"start": chunk.start, "end": chunk.end, "text": text, "words": words}
            rows.append(row)
            if warning:
                warning_row = f"Chunk {chunk_index}: native word times unavailable: {warning}"
                timestamp_warnings.append(warning_row)
                emit("progress", f"Во фрагменте {chunk_index} не удалось определить время слов. "
                     "Их текст сохранён без временных меток.", processed=chunk.end, total=duration)
        processed = chunk.end
        emit("progress", f"Обработан аудиофрагмент {chunk_index}.", processed=processed, total=duration)
    inference_seconds = time.perf_counter() - inference_start
    _finish_preprocessor_profile(preprocessing, verify_gpu=verify_gpu)
    preprocessor_summary: dict[str, dict[str, int]] = {}
    for (provider_name, op, _node), count in preprocessing.get("profile_counts", {}).items():
        summary = preprocessor_summary.setdefault(provider_name, {})
        summary[op] = summary.get(op, 0) + count
    preprocessing.pop("profile_counts", None)

    sessions = _find_onnx_sessions(model)
    asr_sessions = _onnx_asr_sessions(model)
    operator_counts: Counter[tuple[str, str, str]] = Counter()
    session_info = []
    profile_errors = []
    cleanup: list[Path] = []
    asr_profiles: dict[str, Counter[tuple[str, str, str]]] = {}
    for session_number, session in enumerate(sessions):
        roles = [role for role, candidate in asr_sessions.items() if candidate is session]
        session_details = {"index": session_number, "roles": roles, "providers": session.get_providers(),
                           "provider_options": session.get_provider_options()}
        if verify_gpu and device != "cpu":
            try:
                raw_profile_path = session.end_profiling()
                if isinstance(raw_profile_path, (str, os.PathLike)) and str(raw_profile_path).strip():
                    profile_path = Path(raw_profile_path)
                    if profile_path.is_file():
                        counts = _profile_counts(profile_path)
                        if roles:
                            operator_counts.update(counts)
                        for role in roles:
                            asr_profiles[role] = counts
                        cleanup.append(profile_path)
                        session_details["profile_events"] = sum(counts.values())
                    else:
                        session_details["profile_events"] = 0
            except Exception as exc:
                if roles:
                    profile_errors.append({"session": session_number,
                                           "error": f"{type(exc).__name__}: {exc}"})
        session_info.append(session_details)
    for profile_path in cleanup:
        try:
            profile_path.unlink(missing_ok=True)
        except OSError as exc:
            profile_errors.append({"profile": str(profile_path),
                                   "error": f"cleanup {type(exc).__name__}: {exc}"})

    provider = PROVIDERS[device]
    gpu_ops = sum(count for (name, _op, _node), count in operator_counts.items() if name == provider) if device != "cpu" else 0
    cpu_ops = sum(count for (name, _op, _node), count in operator_counts.items() if name == PROVIDERS["cpu"])
    cpu_heavy = {
        op: sum(count for (name, actual_op, _node), count in operator_counts.items()
                if name == PROVIDERS["cpu"] and actual_op == op)
        for op in CPU_HEAVY_OPS
        if any(name == PROVIDERS["cpu"] and actual_op == op for name, actual_op, _node in operator_counts)
    }
    heavy_profiles = [asr_profiles.get(role, Counter()) for role in ("encoder", "decoder")]
    gpu_verified = bool(verify_gpu and device != "cpu" and set(asr_profiles) == {"encoder", "decoder"}
        and all(any(name == provider and op in CPU_HEAVY_OPS for (name, op, _node), count in prof.items() if count)
                for prof in heavy_profiles)
        and not profile_errors and gpu_ops > cpu_ops and not cpu_heavy)
    providers_summary: dict[str, dict[str, int]] = {}
    for (provider_name, op, _node), count in operator_counts.items():
        providers_summary.setdefault(provider_name, {})[op] = providers_summary.setdefault(provider_name, {}).get(op, 0) + count
    details = {
        "runtime": "onnxruntime",
        "available_providers": list(ort.get_available_providers()),
        "sessions": session_info,
        "actual_operator_counts_by_provider": providers_summary,
        "gpu_operator_count": gpu_ops,
        "cpu_operator_count": cpu_ops,
        "cpu_heavy_neural_operator_counts": cpu_heavy,
        "profiling_errors": profile_errors,
        "gpu_verified": gpu_verified,
        "verification_basis": (
            "Actual ONNX Runtime profile records GPU operators for the requested provider, with more GPU than CPU nodes and no CPU heavy neural operators."
            if gpu_verified else "Provider selection alone is not evidence of operator execution."
        ),
        "precision": "FP32",
        "preprocessing": preprocessing,
        "preprocessor_operator_counts_by_provider": preprocessor_summary,
        "native_timestamp_warnings": timestamp_warnings,
    }
    return {
        "ok": True, "text": "\n".join(row["text"] for row in rows).strip(), "segments": rows,
        "language_observations": [], "language": None, "language_probability": None,
        "duration": duration, "effective_device": device,
        "effective_backend": f"ONNX Runtime {provider} (FP32)", "precision": "FP32",
        "timing_seconds": {"load": load_seconds, "inference": inference_seconds,
                           "preprocessing_setup": preprocessing_seconds,
                           "wall": load_seconds + preprocessing_seconds + inference_seconds},
        "diagnostics": details,
    }


def _native_language_id(model_path: Path, audio_path: Path, device: str,
                        index: int, threads: int) -> dict[str, Any]:
    configure_runtime("native")
    import transcribe_cpp

    selected = _select_native_device(transcribe_cpp, device, index)
    duration = _wav_duration(audio_path)
    load_start = time.perf_counter()
    observations: list[dict[str, Any]] = []
    with transcribe_cpp.Model(str(model_path), device=selected) as model:
        load_seconds = time.perf_counter() - load_start
        actual = model.device
        if not _is_native_gpu(actual, device) or getattr(actual, "name", None) != getattr(selected, "name", None):
            raise WorkerError(f"Language-ID model resolved to unexpected device {actual!r}; no fallback is allowed")
        with model.langid_session(n_threads=threads) as session:
            info = model.langid_info
            inference_start = time.perf_counter()
            for chunk_index, chunk in enumerate(iter_audio_chunks(audio_path), start=1):
                if chunk.is_silent:
                    emit("progress", "Тишина пропущена без распознавания.", processed=chunk.end, total=duration)
                    continue
                result = session.run(chunk.pcm_f32)
                candidates = list(getattr(result, "candidates", ()) or ())
                if candidates:
                    candidates_json = [_jsonable(candidate) for candidate in candidates]
                    top = candidates[0]
                    scored_seconds = min(chunk.duration, float(info.max_audio_ms) / 1000.0) if info.max_audio_ms else chunk.duration
                    observations.append({"start": chunk.start, "end": chunk.start + scored_seconds,
                                         "duration": scored_seconds, "code": getattr(top, "code", None),
                                         "probability": float(getattr(top, "p")),
                                         "allowed_mass": float(result.allowed_mass), "candidates": candidates_json})
                emit("progress", f"Распознан язык во фрагменте {chunk_index}.", processed=chunk.end, total=duration)
            inference_seconds = time.perf_counter() - inference_start
            timings = _jsonable(session.timings)
            device_info = _native_descriptor(actual)
    return {
        "ok": True, "language_observations": observations,
        "language": observations[-1]["code"] if observations else None,
        "language_probability": observations[-1]["probability"] if observations else None,
        "allowed_mass": observations[-1]["allowed_mass"] if observations else None,
        "duration": duration, "effective_device": device,
        "effective_backend": f"transcribe.cpp language-id {device} (F32)",
        "precision": "F32", "timing_seconds": {"load": load_seconds, "inference": inference_seconds,
                                                "wall": load_seconds + inference_seconds},
        "diagnostics": {"runtime": "transcribe-cpp", "native_version": transcribe_cpp.native_version(),
                        "model_device": device_info, "native_timings": timings},
    }


def _alignment_language(code: str) -> str:
    aliases = {
        "ru": "Russian", "en": "English", "zh": "Chinese", "yue": "Cantonese",
        "de": "German", "fr": "French", "it": "Italian", "pt": "Portuguese",
        "es": "Spanish", "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    }
    language = aliases.get(code.casefold())
    if language is None:
        raise WorkerError(f"Forced alignment is unsupported for language {code!r}; supported codes: {', '.join(aliases)}")
    return language


def _raw_alignment_units(text: str, processor: Any, language: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Preserve whitespace tokens while allowing the aligner to split compounds."""
    units: list[dict[str, Any]] = []
    spans = list(re.finditer(r"\s*\S+", text))
    pending = ""
    for index, match in enumerate(spans):
        raw = match.group(0)
        if index == len(spans) - 1:
            raw += text[match.end():]
        normalized = processor.split_words_for_alignment(raw, language)
        if isinstance(normalized, str):
            normalized = [normalized] if normalized else []
        normalized = list(normalized)
        if not normalized:
            if units:
                units[-1]["text"] += raw
            else:
                pending += raw
            continue
        units.append({"text": pending + raw, "alignment_texts": [str(item) for item in normalized]})
        pending = ""
    if pending:
        if units:
            units[-1]["text"] += pending
        else:
            return [], []
    if "".join(unit["text"] for unit in units) != text:
        raise WorkerError("Forced-aligner tokenization did not preserve the source transcript")
    return units, [word for unit in units for word in unit["alignment_texts"]]


def _read_wav_range(path: Path, start: float, end: float):
    import numpy as np
    with wave.open(str(path), "rb") as wav:
        start_frame = max(0, int(round(start * SAMPLE_RATE)))
        end_frame = min(wav.getnframes(), int(round(end * SAMPLE_RATE)))
        if end_frame <= start_frame:
            return np.zeros(0, dtype=np.float32)
        wav.setpos(start_frame)
        raw = wav.readframes(end_frame - start_frame)
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return np.asarray(samples, dtype=np.float32) / 32768.0


def _align_segments(audio_path: Path, segments_path: Path, language_code: str,
                    device: str, threads: int) -> dict[str, Any]:
    configure_runtime("aligner")
    import numpy as np
    import torch
    from transformers import AutoModelForTokenClassification, AutoProcessor

    torch.set_num_threads(threads)
    language = _alignment_language(language_code)
    try:
        if device == "cuda":
            if not torch.cuda.is_available():
                raise WorkerError("PyTorch CUDA is unavailable in the isolated aligner runtime")
            torch_device = torch.device("cuda")
        elif device == "directml":
            try:
                import torch_directml
            except ImportError as exc:
                raise WorkerError("PyTorch DirectML bridge is not installed in the isolated aligner runtime") from exc
            torch_device = torch_directml.device(0)
        elif device == "cpu":
            torch_device = torch.device("cpu")
        else:
            raise WorkerError(f"Unsupported aligner device {device!r}")

        model_dir = alignment_model_dir()
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        model = AutoModelForTokenClassification.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False, dtype=torch.float32
        )
        model.to(torch_device).eval()
        load_seconds = time.perf_counter() - started
        input_segments = json.loads(segments_path.read_text(encoding="utf-8"))
        duration = _wav_duration(audio_path)
        aligned_rows = []
        inference_start = time.perf_counter()
        for segment_index, segment in enumerate(input_segments):
            text = str(segment.get("text", ""))
            segment_start = float(segment.get("start", 0.0))
            segment_end = float(segment.get("end", 0.0))
            if not text.strip():
                aligned_rows.append({**segment, "words": []})
                continue
            if not all(math.isfinite(value) for value in (segment_start, segment_end)) or not (0 <= segment_start < segment_end <= duration + 0.02):
                raise WorkerError(f"ASR segment {segment_index} has invalid alignment bounds")
            waveform = _read_wav_range(audio_path, segment_start, min(duration, segment_end))
            if not len(waveform):
                raise WorkerError(f"ASR segment {segment_index} has no audio samples")
            units, expected_words = _raw_alignment_units(text, processor, language)
            if not units:
                aligned_rows.append({**segment, "words": [{"text": text, "start": 0.0, "end": 0.0}]})
                continue
            inputs, word_lists = processor.prepare_forced_aligner_inputs(
                audio=waveform, transcript=text, language=language
            )
            if len(word_lists) != 1 or list(word_lists[0]) != expected_words:
                raise WorkerError(f"Aligner input words diverged for segment {segment_index}")
            inputs = {key: value.to(torch_device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
            if not torch.isfinite(outputs.logits).all():
                raise WorkerError(f"Forced aligner returned non-finite logits for segment {segment_index}")
            decoded = processor.decode_forced_alignment(
                logits=outputs.logits, input_ids=inputs["input_ids"], word_lists=word_lists,
                timestamp_token_id=model.config.timestamp_token_id,
            )[0]
            if len(decoded) != len(expected_words):
                raise WorkerError(f"Aligner returned {len(decoded)} words for {len(expected_words)} normalized units")
            words = []
            previous_start = previous_end = -math.inf
            local_duration = len(waveform) / SAMPLE_RATE
            decoded_index = 0
            for unit in units:
                spans: list[tuple[float, float]] = []
                all_valid = True
                for _alignment_text in unit["alignment_texts"]:
                    item = decoded[decoded_index]
                    decoded_index += 1
                    raw_start, raw_end = item.get("start_time"), item.get("end_time")
                    try:
                        start_local, end_local = float(raw_start), float(raw_end)
                    except (TypeError, ValueError):
                        start_local = end_local = math.nan
                    valid = (
                        math.isfinite(start_local) and math.isfinite(end_local)
                        and 0 <= start_local <= end_local <= local_duration + 0.08
                        and start_local >= previous_start and end_local >= previous_end
                        and end_local > start_local
                    )
                    if valid:
                        previous_start, previous_end = start_local, end_local
                        start_global = segment_start + max(0.0, start_local)
                        end_global = min(segment_end, segment_start + min(local_duration, end_local))
                        valid = math.isfinite(start_global) and math.isfinite(end_global) and end_global > start_global
                    if valid:
                        spans.append((start_global, end_global))
                    else:
                        all_valid = False
                if all_valid and spans:
                    word_start = min(start for start, _ in spans)
                    word_end = max(end for _, end in spans)
                else:
                    # A compound remains one raw transcript token. If any of its
                    # normalized units cannot be timed, preserve it as unknown.
                    word_start = word_end = 0.0
                words.append({"text": unit["text"], "start": word_start, "end": word_end})
            if "".join(word["text"] for word in words) != text:
                raise WorkerError(f"Aligned raw word text did not reconstruct segment {segment_index}")
            aligned_rows.append({**segment, "words": words})
            emit("progress", f"Выровнены слова фрагмента {segment_index + 1}.", stage="alignment")
        inference_seconds = time.perf_counter() - inference_start
        return {
            "ok": True, "segments": aligned_rows,
            "timing_seconds": {"load": load_seconds, "inference": inference_seconds,
                               "wall": load_seconds + inference_seconds},
            "effective_device": str(torch_device),
            "effective_backend": f"Forced Aligner {device} (FP32)", "precision": "FP32",
            "diagnostics": {"runtime": "transformers", "model_dir": str(model_dir),
                            "device": str(torch_device), **peak_memory_bytes()},
        }
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError(f"Forced alignment failed: {type(exc).__name__}: {exc}") from exc


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "devices":
        return list_devices(args.model)
    if not args.audio:
        raise WorkerError("--audio is required for this worker mode")
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        raise WorkerError(f"Audio file does not exist: {audio_path}")
    if args.mode == "transcribe":
        if args.model == NATIVE_NAME:
            model_file = model_paths(args.model)["model"]
            if args.device not in {"cpu", "vulkan", "cuda"}:
                raise WorkerError(f"Qwen ASR cannot use {args.device!r}; select cpu or vulkan")
            return _native_transcribe(model_file, audio_path, args.device, args.device_index,
                                      args.threads, args.language, args.verify_gpu)
        if args.model == PARAKEET_NAME:
            model_dir = model_paths(args.model)["model_dir"]
            if args.device not in {"cpu", "cuda", "directml"}:
                raise WorkerError(f"Parakeet ASR cannot use {args.device!r}; select cpu, cuda, or directml")
            return _onnx_transcribe(model_dir, audio_path, args.device, args.device_index,
                                    args.threads, args.verify_gpu)
        raise WorkerError(f"Unsupported ASR model id: {args.model}")
    if args.mode == "language-id":
        language_model = model_paths(PARAKEET_NAME)["language_model"]
        return _native_language_id(language_model, audio_path, args.device, args.device_index, args.threads)
    if args.mode == "align":
        if not args.segments_json:
            raise WorkerError("--segments-json is required for alignment")
        if not args.language:
            raise WorkerError("--language is required for alignment")
        return _align_segments(audio_path, Path(args.segments_json).expanduser().resolve(),
                               args.language, args.device, args.threads)
    raise WorkerError(f"Unsupported worker mode: {args.mode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("transcribe", "language-id", "align", "devices"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda", "directml", "vulkan"))
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--audio")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--language")
    parser.add_argument("--segments-json")
    parser.add_argument("--verify-gpu", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    try:
        if args.threads < 1 or args.device_index < 0:
            raise WorkerError("--threads must be positive and --device-index must be nonnegative")
        result = run(args)
    except Exception as exc:
        result = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)},
                  "effective_device": args.device, "diagnostics": peak_memory_bytes()}
        try:
            _write_json(output, result)
        except OSError as write_error:
            print(f"Could not write worker result: {write_error}", file=sys.stderr)
        emit("log", f"{type(exc).__name__}: {exc}")
        return 1
    try:
        if "diagnostics" in result:
            result["diagnostics"].update(peak_memory_bytes())
        _write_json(output, result)
    except Exception as exc:
        print(f"Could not write worker result: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
