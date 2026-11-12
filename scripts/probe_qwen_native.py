#!/usr/bin/env python3
"""Isolated Qwen3-ASR native backend probe for transcribe-cpp 0.4.1."""

from __future__ import annotations

import os
from pathlib import Path

# Set scheduler diagnostics and project-local scratch/cache locations before
# importing the native binding (which loads libtranscribe at import time).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import argparse
import array
import dataclasses
import hashlib
import json
import re
import struct
import sys
import threading
import time
import wave
from collections import Counter
from typing import Any, Iterable


class ProbeError(Exception):
    """An input, selection, or runtime error suitable for JSON reporting."""


def configure_runtime() -> None:
    """Configure private cache and scratch paths before native imports."""
    runtime = PROJECT_ROOT / ".runtime" / "asr-evaluation" / "qwen-native-probe"
    cache = runtime / "cache"
    scratch = runtime / "tmp"
    cache.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["GGML_SCHED_DEBUG"] = "2"
    os.environ["XDG_CACHE_HOME"] = str(cache)
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["TMP"] = str(scratch)
    os.environ["TEMP"] = str(scratch)
    os.environ["TMPDIR"] = str(scratch)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _read_pcm16_mono_16k(path: Path) -> tuple[array.array, float]:
    if not path.is_file():
        raise ProbeError(f"Audio file does not exist: {path}")
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            rate = wav.getframerate()
            width = wav.getsampwidth()
            comptype = wav.getcomptype()
            frames = wav.getnframes()
            if channels != 1 or rate != 16000 or width != 2 or comptype != "NONE":
                raise ProbeError(
                    "Audio must be uncompressed PCM16 mono 16000 Hz WAV "
                    f"(got channels={channels}, rate={rate}, sample_width={width}, "
                    f"compression={comptype})"
                )
            raw = wav.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ProbeError(f"Invalid PCM WAV audio: {exc}") from exc
    if not raw:
        raise ProbeError("Audio WAV contains no samples")
    samples_i16 = array.array("h")
    samples_i16.frombytes(raw)
    if sys.byteorder != "little":
        samples_i16.byteswap()
    pcm = array.array("f", (sample / 32768.0 for sample in samples_i16))
    return pcm, len(samples_i16) / 16000.0


_GGUF_VALUE_FORMATS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}


class _GGUFReader:
    def __init__(self, stream):
        self.stream = stream

    def read(self, count: int) -> bytes:
        data = self.stream.read(count)
        if len(data) != count:
            raise ProbeError("GGUF file ended unexpectedly while reading its header")
        return data

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        size = self.unpack("<Q")
        if size > 1_000_000_000:
            raise ProbeError("GGUF header contains an unreasonable string length")
        return self.read(size).decode("utf-8", "replace")

    def value(self, kind: int) -> None:
        if kind == 8:
            self.string()
        elif kind == 9:
            element_kind = self.unpack("<I")
            count = self.unpack("<Q")
            if count > 100_000_000:
                raise ProbeError("GGUF header contains an unreasonable metadata array")
            for _ in range(count):
                self.value(element_kind)
        elif kind in _GGUF_VALUE_FORMATS:
            fmt, _ = _GGUF_VALUE_FORMATS[kind]
            self.unpack(fmt)
        else:
            raise ProbeError(f"GGUF uses unsupported metadata value type {kind}")


def _validate_f16_gguf(path: Path) -> None:
    if not path.is_file():
        raise ProbeError(f"Model file does not exist: {path}")
    try:
        with path.open("rb") as stream:
            reader = _GGUFReader(stream)
            if reader.read(4) != b"GGUF":
                raise ProbeError("Model must be a GGUF file")
            version = reader.unpack("<I")
            if version not in (2, 3):
                raise ProbeError(f"Unsupported GGUF version {version}; expected version 2 or 3")
            tensor_count = reader.unpack("<Q")
            metadata_count = reader.unpack("<Q")
            if tensor_count == 0 or tensor_count > 10_000_000 or metadata_count > 10_000_000:
                raise ProbeError("GGUF header contains invalid tensor or metadata counts")
            for _ in range(metadata_count):
                reader.string()
                reader.value(reader.unpack("<I"))
            tensor_types: set[int] = set()
            for _ in range(tensor_count):
                reader.string()
                dimensions = reader.unpack("<I")
                if dimensions > 16:
                    raise ProbeError("GGUF tensor has an invalid number of dimensions")
                for _ in range(dimensions):
                    reader.unpack("<Q")
                tensor_types.add(reader.unpack("<I"))
                reader.unpack("<Q")  # tensor data offset
    except OSError as exc:
        raise ProbeError(f"Cannot read GGUF model: {exc}") from exc
    quantized = sorted(kind for kind in tensor_types if kind not in (0, 1))
    if quantized:
        raise ProbeError(
            "Model contains quantized tensors; this probe requires F16 GGUF "
            f"and does not allow quantization types {quantized}"
        )
    if 1 not in tensor_types:
        raise ProbeError(
            "Model must contain at least one F16 tensor; "
            f"tensor type IDs found: {sorted(tensor_types)}"
        )


def select_device(devices: Iterable[Any], backend: str, device_id: str | None = None):
    """Select one registered device by class/backend and optional stable ID."""
    devices = list(devices)
    if backend == "cpu":
        candidates = [d for d in devices if getattr(d, "device_type", None) == "cpu"]
    else:
        candidates = [
            d for d in devices
            if str(getattr(d, "kind", "")).casefold() == backend
            and getattr(d, "device_type", None) in ("gpu", "igpu")
        ]
    if device_id is not None:
        candidates = [d for d in candidates if getattr(d, "device_id", None) == device_id]
    if not candidates:
        suffix = f" with device id {device_id!r}" if device_id is not None else ""
        raise ProbeError(f"No registered {backend} device{suffix}; automatic fallback is disabled")
    if len(candidates) != 1:
        descriptions = [
            f"{getattr(d, 'name', '?')} (id={getattr(d, 'device_id', None)!r})"
            for d in candidates
        ]
        raise ProbeError(
            f"Multiple {backend} devices match{(' id ' + repr(device_id)) if device_id else ''}; "
            f"specify --device-id. Matches: {descriptions}"
        )
    return candidates[0]


class NativeLog:
    """Thread-safe native diagnostic sink with GGML scheduler-line counts."""

    # transcribe.cpp v0.4.1's pinned ggml prints operation assignments as:
    # node #  1 (     MUL_MAT): tensor-name (  2K) [Vulka          ] use=1,c=1:
    # These are scheduler assignments, which may also be emitted during graph
    # reservation. They are diagnostic only and never establish verification.
    _node_line = re.compile(
        r"^\s*node #\s*(?P<index>\d+)\s+\((?P<op>[^)]*)\):\s*"
        r"(?P<name>.*?)\s+\([^)]*\)\s+\[(?P<backend>[^\s\]]+)\s+"
        r"[^\]]*\]\s+use=\d+,c=(?P<compute>[01]):"
    )
    _gpu_backend = re.compile(r"^(?:cuda|rocm|metal|vulk|gpu|sycl)", re.IGNORECASE)

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.gpu_backends: Counter[str] = Counter()
        self.cpu_assignments = 0
        self.total_lines = 0
        self._file = path.open("w", encoding="utf-8", buffering=1)

    def __call__(self, level: int, message: str) -> None:
        line = f"[{level}] {message.rstrip()}"
        with self.lock:
            self._file.write(line + "\n")
            self.total_lines += 1
            match = self._node_line.match(message)
            if match and match.group("compute") == "1":
                backend = match.group("backend")
                if backend.casefold() == "cpu":
                    self.cpu_assignments += 1
                elif self._gpu_backend.match(backend):
                    self.gpu_backends[backend] += 1

    def summary(self) -> dict[str, Any]:
        with self.lock:
            self._file.flush()
            return {
                "line_count": self.total_lines,
                "assignment_count_semantics": (
                    "GGML scheduler planning lines; may include graph reservation/load and do not "
                    "prove executed operations"
                ),
                "scheduler_compute_node_assignments": {
                    "gpu_by_backend": dict(self.gpu_backends),
                    "cpu": self.cpu_assignments,
                },
            }

    def close(self) -> None:
        with self.lock:
            self._file.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    audio_path = Path(args.audio).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve() if args.log else output_path.with_suffix(".native.log")
    pcm, duration_s = _read_pcm16_mono_16k(audio_path)
    _validate_f16_gguf(model_path)

    configure_runtime()
    import transcribe_cpp

    devices = transcribe_cpp.backends()
    selected = select_device(devices, args.device, args.device_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = NativeLog(log_path)
    transcribe_cpp.set_log_callback(log)
    wall_started = time.perf_counter()
    load_started = wall_started
    result = None
    actual_device = None
    load_s = None
    inference_s = None
    try:
        with transcribe_cpp.Model(str(model_path), device=selected) as model:
            load_s = time.perf_counter() - load_started
            actual_device = model.device
            if actual_device.device_type not in (("cpu",) if args.device == "cpu" else ("gpu", "igpu")):
                raise ProbeError(
                    f"Model resolved to non-{args.device} device {actual_device!r}; refusing fallback"
                )
            if args.device != "cpu" and str(actual_device.kind).casefold() != args.device:
                raise ProbeError(
                    f"Model resolved to backend {actual_device.kind!r}, expected {args.device!r}"
                )
            if args.device_id is not None and actual_device.device_id != args.device_id:
                raise ProbeError(
                    f"Model resolved to device id {actual_device.device_id!r}, "
                    f"expected exact id {args.device_id!r}"
                )
            with model.session(n_threads=args.threads) as session:
                inference_started = time.perf_counter()
                result = session.run(pcm)
                inference_s = time.perf_counter() - inference_started
                timings = _jsonable(result.timings)
                model_descriptor = _jsonable(actual_device)
        placement = log.summary()
        if args.device == "cpu":
            verification = "verified"
            verification_reason = "CPU was selected and confirmed as the model device."
        else:
            verification = "unverified"
            verification_reason = (
                "The selected device is a GPU, but v0.4.1 exposes scheduler assignment diagnostics, "
                "not counters proving that encoder and language-model operations executed on it. "
                "The recorded assignment lines are diagnostic only."
            )
        payload = {
            "ok": True,
            "status": verification,
            "verification_reason": verification_reason,
            "text": result.text,
            "language": result.language,
            "duration_seconds": duration_s,
            "timing_seconds": {
                "load": load_s,
                "inference": inference_s,
                "wall": time.perf_counter() - wall_started,
            },
            "versions": {
                "python_package": getattr(transcribe_cpp, "__version__", "unknown"),
                "native": transcribe_cpp.native_version(),
            },
            "model": {"path": str(model_path), "size_bytes": model_path.stat().st_size, "sha256": _sha256(model_path)},
            "requested_device": args.device,
            "selected_device": _jsonable(selected),
            "model_device": model_descriptor,
            "result_timings": timings,
            "segments": _jsonable(result.segments),
            "words": _jsonable(result.words),
            "tokens": _jsonable(result.tokens),
            "native_log": {"path": str(log_path), **placement},
        }
        return payload
    finally:
        transcribe_cpp.set_log_callback(None)
        log.close()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="PCM16 mono 16 kHz WAV")
    parser.add_argument("--model", required=True, help="F16 GGUF model file")
    parser.add_argument("--device", required=True, choices=("cpu", "vulkan", "cuda"))
    parser.add_argument("--device-id", help="Exact stable backend device id")
    parser.add_argument("--threads", type=_positive_int, default=8)
    parser.add_argument("--output", required=True, help="JSON result path")
    parser.add_argument("--log", help="Native diagnostic log path (default: beside JSON output)")
    args = parser.parse_args(argv)
    output_path = Path(args.output).expanduser().resolve()
    try:
        payload = run_probe(args)
        _write_json(output_path, payload)
        return 0
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        payload = {"ok": False, "status": "error", "error": error}
        try:
            _write_json(output_path, payload)
        except OSError as write_exc:
            print(f"Could not write result JSON: {write_exc}", file=sys.stderr)
        print(f"Probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
