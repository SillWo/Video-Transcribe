#!/usr/bin/env python3
"""Isolated VoxLingua107 language identification probe for transcribe-cpp 0.4.1."""

from __future__ import annotations

import os
from pathlib import Path

import argparse
import array
import dataclasses
import hashlib
import json
import struct
import sys
import time
import wave
from typing import Any, Iterable


EXPECTED_MODEL_SHA256 = "d0420e485112ca05e31aa06e1615556102083a9725e89b55a1f0963c0c0ead66"
EXPECTED_MODEL_SIZE = 84_993_184


class ProbeError(Exception):
    """An input, selection, or model validation error for JSON reporting."""


def configure_runtime() -> None:
    """Set project-local caches and scratch paths before native imports."""
    runtime = Path(__file__).resolve().parents[1] / ".runtime" / "asr-evaluation" / "langid-native-probe"
    cache = runtime / "cache"
    scratch = runtime / "tmp"
    cache.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
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


_GGUF_SCALAR_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
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

    def skip_value(self, kind: int) -> None:
        if kind == 8:
            self.string()
        elif kind == 9:
            element_kind = self.unpack("<I")
            count = self.unpack("<Q")
            if count > 100_000_000:
                raise ProbeError("GGUF header contains an unreasonable metadata array")
            for _ in range(count):
                self.skip_value(element_kind)
        elif kind in _GGUF_SCALAR_FORMATS:
            self.unpack(_GGUF_SCALAR_FORMATS[kind])
        else:
            raise ProbeError(f"GGUF uses unsupported metadata value type {kind}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_f32_gguf(path: Path, *, verify_identity: bool = True) -> dict[str, Any]:
    """Validate that GGUF tensors are F32 and, by default, match the target asset."""
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
                reader.skip_value(reader.unpack("<I"))
            tensor_types: set[int] = set()
            for _ in range(tensor_count):
                reader.string()
                dimensions = reader.unpack("<I")
                if dimensions > 16:
                    raise ProbeError("GGUF tensor has an invalid number of dimensions")
                for _ in range(dimensions):
                    reader.unpack("<Q")
                tensor_types.add(reader.unpack("<I"))
                reader.unpack("<Q")
    except OSError as exc:
        raise ProbeError(f"Cannot read GGUF model: {exc}") from exc
    if tensor_types != {0}:
        quantized = sorted(kind for kind in tensor_types if kind != 0)
        raise ProbeError(
            "Model must contain only F32 tensors (GGML type 0); "
            f"tensor type IDs found: {sorted(tensor_types)}; non-F32 types: {quantized}"
        )

    size_bytes = path.stat().st_size
    digest = _sha256(path)
    if verify_identity and (size_bytes != EXPECTED_MODEL_SIZE or digest != EXPECTED_MODEL_SHA256):
        raise ProbeError(
            "Model file does not match the expected VoxLingua107 F32 asset "
            f"(size={size_bytes}, sha256={digest})"
        )
    return {"size_bytes": size_bytes, "sha256": digest, "tensor_type_ids": [0]}


def select_device(devices: Iterable[Any], backend: str):
    """Select the single live CPU/backend device requested; never choose auto."""
    devices = list(devices)
    if backend == "cpu":
        candidates = [device for device in devices if getattr(device, "device_type", None) == "cpu"]
    else:
        candidates = [
            device for device in devices
            if str(getattr(device, "kind", "")).casefold() == backend
            and getattr(device, "device_type", None) in ("gpu", "igpu")
        ]
    if not candidates:
        raise ProbeError(f"No registered {backend} device; automatic fallback is disabled")
    if len(candidates) != 1:
        names = [f"{getattr(device, 'name', '?')} id={getattr(device, 'device_id', None)!r}" for device in candidates]
        raise ProbeError(f"Multiple {backend} devices are registered; exact selection is ambiguous: {names}")
    return candidates[0]


def _same_device(actual: Any, selected: Any, backend: str) -> bool:
    allowed_types = ("cpu",) if backend == "cpu" else ("gpu", "igpu")
    if getattr(actual, "device_type", None) not in allowed_types:
        return False
    if backend != "cpu" and str(getattr(actual, "kind", "")).casefold() != backend:
        return False
    for attribute in ("name", "description", "kind", "device_type", "device_id"):
        if getattr(actual, attribute, None) != getattr(selected, attribute, None):
            return False
    return True


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    audio_path = Path(args.audio).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    pcm, duration_s = _read_pcm16_mono_16k(audio_path)
    model_fingerprint = validate_f32_gguf(model_path)

    configure_runtime()
    import transcribe_cpp

    selected = select_device(transcribe_cpp.backends(), args.device)
    wall_started = time.perf_counter()
    load_started = wall_started
    with transcribe_cpp.Model(str(model_path), device=selected) as model:
        load_s = time.perf_counter() - load_started
        actual_device = model.device
        if not _same_device(actual_device, selected, args.device):
            raise ProbeError(
                f"Model resolved to unexpected device {actual_device!r}; "
                f"requested exact {args.device} selection {selected!r}"
            )
        with model.langid_session(n_threads=args.threads) as session:
            inference_started = time.perf_counter()
            result = session.run(pcm)
            inference_s = time.perf_counter() - inference_started
            native_timings = _jsonable(session.timings)
            model_device = _jsonable(actual_device)

    candidates = _jsonable(result.candidates)
    return {
        "ok": True,
        "status": "ok",
        "code": result.code,
        "candidates": candidates,
        "allowed_mass": result.allowed_mass,
        "duration_seconds": duration_s,
        "timing_seconds": {
            "load": load_s,
            "inference": inference_s,
            "wall": time.perf_counter() - wall_started,
        },
        "native_timings": native_timings,
        "versions": {
            "python_package": getattr(transcribe_cpp, "__version__", "unknown"),
            "native": transcribe_cpp.native_version(),
        },
        "audio": {"path": str(audio_path), "size_bytes": audio_path.stat().st_size, "sha256": _sha256(audio_path)},
        "model": {"path": str(model_path), **model_fingerprint},
        "requested_device": args.device,
        "selected_device": _jsonable(selected),
        "model_device": model_device,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="PCM16 mono 16 kHz WAV")
    parser.add_argument("--model", required=True, help="F32 VoxLingua107 GGUF model")
    parser.add_argument("--device", required=True, choices=("cpu", "vulkan", "cuda"))
    parser.add_argument("--output", required=True, help="JSON result path")
    parser.add_argument("--threads", type=_positive_int, default=8)
    args = parser.parse_args(argv)
    output_path = Path(args.output).expanduser().resolve()
    try:
        payload = run_probe(args)
        _write_json(output_path, payload)
        return 0
    except Exception as exc:
        try:
            _write_json(output_path, {"ok": False, "status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}})
        except OSError as write_exc:
            print(f"Could not write result JSON: {write_exc}", file=sys.stderr)
        print(f"Probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
