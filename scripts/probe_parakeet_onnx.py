#!/usr/bin/env python3
"""Read-only, local ONNX Runtime probe for Parakeet-TDT-0.6B-v3 FP32."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / ".runtime" / "asr-evaluation" / "parakeet-onnx-probe"
CACHE_DIR = RUNTIME_DIR / "cache"
TMP_DIR = RUNTIME_DIR / "tmp"


def prepare_local_environment() -> None:
    """Set offline mode and project-local caches before inference imports."""
    for name, child in {
        "HF_HOME": "huggingface",
        "HF_HUB_CACHE": "huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
        "TRANSFORMERS_CACHE": "huggingface/transformers",
        "TORCH_HOME": "torch",
        "XDG_CACHE_HOME": "xdg",
    }.items():
        path = CACHE_DIR / child
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(TMP_DIR)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
MODEL_REPOSITORY = "istupakov/parakeet-tdt-0.6b-v3-onnx"
REQUIRED_MODEL_FILES = (
    "config.json",
    "encoder-model.onnx",
    "encoder-model.onnx.data",
    "decoder_joint-model.onnx",
    "nemo128.onnx",
    "vocab.txt",
)
PROVIDERS = {
    "directml": "DmlExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
CPU_HEAVY_NEURAL_OPS = {
    "Conv", "ConvTranspose", "MatMul", "Gemm", "LSTM", "Attention",
    "MultiHeadAttention", "GRU", "RNN",
}


class ProbeError(RuntimeError):
    """An input or runtime problem that should be reported without fallback."""


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be zero or a positive integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="Complete local FP32 ONNX model directory")
    parser.add_argument("--audio", required=True, type=Path, help="Local uncompressed PCM WAV")
    parser.add_argument("--output", required=True, type=Path, help="JSON result path")
    parser.add_argument("--device", choices=("cpu", "directml", "cuda"), default="cpu")
    parser.add_argument("--device-index", type=_nonnegative_int, default=0)
    parser.add_argument("--threads", type=_positive_int, default=8)
    return parser.parse_args(argv)


def validate_model_dir(model_dir: Path) -> list[str]:
    if not model_dir.is_dir():
        raise ProbeError(f"Model directory does not exist: {model_dir}")
    return [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]


def artifact_provenance(model_dir: Path) -> dict[str, dict[str, Any]]:
    """Hash and size every required artifact; no downloads or model rewriting."""
    missing = validate_model_dir(model_dir)
    if missing:
        raise ProbeError("Local model is incomplete; no download was attempted. Missing: " + ", ".join(missing))
    artifacts: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_MODEL_FILES:
        path = model_dir / name
        stat = path.stat()
        # Probe runs are short-lived; stat-keyed cache avoids hashing large files twice.
        cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        digest = _HASH_CACHE.get(cache_key)
        if digest is None:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            _HASH_CACHE[cache_key] = digest
        artifacts[name] = {"path": str(path.resolve()), "size_bytes": stat.st_size, "sha256": digest}
    return artifacts


_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def read_wav_info(path: Path) -> tuple[float, dict[str, Any]]:
    if not path.is_file():
        raise ProbeError(f"Audio file does not exist: {path}")
    try:
        with wave.open(str(path), "rb") as wav:
            info = {
                "channels": wav.getnchannels(),
                "sample_rate_hz": wav.getframerate(),
                "sample_width_bytes": wav.getsampwidth(),
                "frames": wav.getnframes(),
                "compression": wav.getcomptype(),
            }
    except (OSError, EOFError, wave.Error) as exc:
        raise ProbeError(f"Cannot read WAV file {path}: {exc}") from exc
    if info["compression"] != "NONE":
        raise ProbeError(f"Only uncompressed PCM WAV is supported (got {info['compression']}).")
    if info["sample_rate_hz"] <= 0 or info["channels"] <= 0 or info["frames"] <= 0:
        raise ProbeError("WAV must contain non-empty audio frames with valid sample rate and channel count.")
    if info["sample_width_bytes"] not in (1, 2, 3, 4):
        raise ProbeError("WAV must use PCM U8, 16-bit, 24-bit, or 32-bit samples.")
    duration = info["frames"] / info["sample_rate_hz"]
    return duration, info


def hardware_inventory(timeout_seconds: float = 10.0) -> dict[str, Any]:
    """Query Windows display adapters without opening a visible console."""
    if platform.system() != "Windows":
        return {"status": "unavailable", "reason": "Win32_VideoController inventory is Windows-only", "adapters": []}
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress"
    )
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "reason": str(exc), "adapters": []}
    if completed.returncode != 0:
        return {"status": "error", "reason": completed.stderr.strip() or f"PowerShell exit {completed.returncode}", "adapters": []}
    try:
        parsed = json.loads(completed.stdout) if completed.stdout.strip() else []
        if isinstance(parsed, dict):
            parsed = [parsed]
        adapters = [
            {key: item.get(key) for key in ("Name", "DriverVersion", "PNPDeviceID")}
            for item in parsed if isinstance(item, dict)
        ]
    except (json.JSONDecodeError, TypeError) as exc:
        return {"status": "error", "reason": f"Invalid inventory JSON: {exc}", "adapters": []}
    return {"status": "ok", "adapters": adapters}


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return {str(k): jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def find_sessions(root: Any) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    seen: set[int] = set()
    stack = [(root, "model")]
    while stack:
        value, label = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if callable(getattr(value, "get_providers", None)) and callable(getattr(value, "end_profiling", None)):
            found.append((label, value))
        elif isinstance(value, dict):
            stack.extend((item, f"{label}.{key}") for key, item in value.items())
        elif isinstance(value, (tuple, list, set)):
            stack.extend((item, f"{label}[{index}]") for index, item in enumerate(value))
        elif hasattr(value, "__dict__") and not isinstance(value, (str, bytes, Path, type)):
            stack.extend((item, f"{label}.{key}") for key, item in vars(value).items())
    return found


def read_profile(path: str) -> Counter[tuple[str, str]]:
    try:
        events = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return Counter()
    counts: Counter[tuple[str, str]] = Counter()
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict) or event.get("cat") != "Node":
                continue
            args = event.get("args") or {}
            if isinstance(args, dict) and args.get("provider"):
                counts[(str(args["provider"]), str(args.get("op_name", "Unknown")))] += 1
    return counts


def summarize_profile(counts: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = {}
    for (provider, op), count in counts.items():
        grouped.setdefault(provider, Counter())[op] += count
    return {provider: dict(ops) for provider, ops in grouped.items()}


def count_by_provider(counts: Counter[tuple[str, str]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for (provider, _), count in counts.items():
        totals[provider] += count
    return dict(totals)


def package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _provider_counts_for_device(device: str, counts: Counter[tuple[str, str]]) -> tuple[int, int, dict[str, int]]:
    requested = PROVIDERS[device]
    gpu_ops = sum(n for (provider, _), n in counts.items() if provider == requested) if device != "cpu" else 0
    cpu_ops = sum(n for (provider, _), n in counts.items() if provider == PROVIDERS["cpu"])
    cpu_heavy = {
        op: n for (provider, op), n in counts.items()
        if provider == PROVIDERS["cpu"] and op in CPU_HEAVY_NEURAL_OPS
    }
    return gpu_ops, cpu_ops, cpu_heavy


def verify_actual_placement(device: str, counts: Counter[tuple[str, str]], device_index: int) -> dict[str, Any]:
    if device == "cpu":
        return {"status": "not_applicable", "provider": PROVIDERS["cpu"]}
    gpu_ops, cpu_ops, cpu_heavy = _provider_counts_for_device(device, counts)
    provider = PROVIDERS[device]
    if gpu_ops == 0:
        status = "unsupported"
    elif cpu_heavy:
        status = "partial"
    else:
        status = "verified"
    return {
        "status": status,
        "provider": provider,
        "device_index": device_index,
        "gpu_operator_count": gpu_ops,
        "cpu_operator_count": cpu_ops,
        "cpu_heavy_neural_operator_counts": cpu_heavy,
        "basis": "Actual ONNX Runtime operator profiling; provider availability alone is not evidence of execution.",
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads < 1 or args.device_index < 0:
        raise ProbeError("threads must be positive and device index must be zero or greater")
    audio_path = Path(args.audio).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    duration, wav_info = read_wav_info(audio_path)
    artifacts = artifact_provenance(model_dir)

    # These are the first imports of inference packages; offline cache settings
    # were configured at module load and are reaffirmed immediately beforehand.
    prepare_local_environment()
    try:
        import onnx_asr
        import onnxruntime as ort
    except Exception as exc:
        raise ProbeError(f"Could not import the requested ONNX runtime packages: {type(exc).__name__}: {exc}") from exc

    available = list(ort.get_available_providers())
    selected_provider = PROVIDERS[args.device]
    if selected_provider not in available:
        raise ProbeError(
            f"Requested provider {selected_provider} is unavailable; available providers: {available}. "
            "Automatic provider fallback and CPU retry are disabled."
        )

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.enable_profiling = True
    profile_dir = CACHE_DIR / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    options.profile_file_prefix = str(profile_dir / f"parakeet-{args.device}")
    providers: list[Any]
    if args.device == "directml":
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False
        providers = [(selected_provider, {"device_id": str(args.device_index)})]
    elif args.device == "cuda":
        providers = [(selected_provider, {"device_id": str(args.device_index)})]
    else:
        providers = [selected_provider]

    # Resampling is utility work on CPU; the model graph receives only the
    # explicitly selected provider. NumPy preprocessing avoids an extra session.
    cpu_options = ort.SessionOptions()
    cpu_options.intra_op_num_threads = args.threads
    cpu_options.inter_op_num_threads = 1
    load_start = time.perf_counter()
    model = onnx_asr.load_model(
        MODEL_NAME,
        str(model_dir),
        sess_options=options,
        providers=providers,
        preprocessor_config={"use_numpy_preprocessors": True},
        resampler_config={"sess_options": cpu_options, "providers": [PROVIDERS["cpu"]]},
    ).with_timestamps()
    load_seconds = time.perf_counter() - load_start
    inference_start = time.perf_counter()
    recognition = model.recognize(str(audio_path))
    inference_seconds = time.perf_counter() - inference_start

    sessions = find_sessions(model)
    aggregate: Counter[tuple[str, str]] = Counter()
    session_details = []
    cleanup_paths = []
    try:
        for index, (label, session) in enumerate(sessions):
            profile_path = session.end_profiling()
            cleanup_paths.append(Path(profile_path))
            counts = read_profile(profile_path)
            aggregate.update(counts)
            session_details.append({
                "index": index,
                "label": label,
                "providers": session.get_providers(),
                "provider_options": session.get_provider_options(),
                "profile_operator_counts": summarize_profile(counts),
            })
    finally:
        for profile_path in cleanup_paths:
            try:
                profile_path.unlink(missing_ok=True)
            except OSError:
                pass

    verification = verify_actual_placement(args.device, aggregate, args.device_index)
    ok = args.device == "cpu" or verification["status"] == "verified"
    if args.device != "cpu" and not ok:
        if verification["status"] == "partial":
            verification["note"] = (
                "GPU operators did execute, and CPU heavy neural operators also executed; "
                "this is partial GPU execution, not complete GPU unavailability or full verification."
            )
        else:
            verification["note"] = (
                "No model operators were recorded on the requested GPU provider; "
                "recognized output is retained but GPU execution is unsupported."
            )
    return {
        "ok": ok,
        "model": MODEL_NAME,
        "model_repository": MODEL_REPOSITORY,
        "precision": "FP32",
        "model_dir": str(model_dir),
        "model_artifacts": artifacts,
        "audio": str(audio_path),
        "audio_info": {**wav_info, "duration_seconds": duration},
        "device_requested": args.device,
        "device_index": args.device_index,
        "threads": args.threads,
        "hardware_inventory": hardware_inventory(),
        "available_providers": available,
        "runtime_versions": {
            "onnx-asr": package_version("onnx-asr"),
            "onnxruntime": package_version("onnxruntime"),
            "onnxruntime-directml": package_version("onnxruntime-directml"),
            "onnxruntime-gpu": package_version("onnxruntime-gpu"),
            "numpy": package_version("numpy"),
        },
        "text": getattr(recognition, "text", None),
        "tokens": jsonable(getattr(recognition, "tokens", None)),
        "timestamps": jsonable(getattr(recognition, "timestamps", None)),
        "recognition": jsonable(recognition),
        "timing": {
            "model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "audio_duration_seconds": duration,
            "rtf": inference_seconds / duration,
        },
        "sessions": session_details,
        "actual_operator_provider_counts": count_by_provider(aggregate),
        "actual_operator_counts_by_provider_and_op": summarize_profile(aggregate),
        "gpu_verification": verification,
    }


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_probe(args)
    except Exception as exc:
        payload = {
            "ok": False,
            "model": MODEL_NAME,
            "model_repository": MODEL_REPOSITORY,
            "device_requested": args.device,
            "device_index": args.device_index,
            "hardware_inventory": hardware_inventory(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "gpu_verification": {"status": "unverified" if args.device != "cpu" else "not_applicable"},
        }
    try:
        write_result(args.output.expanduser().resolve(), payload)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Cannot write JSON output: {exc}"}), file=sys.stderr)
        return 2
    if not payload["ok"]:
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "output": str(args.output), "gpu_verification": payload["gpu_verification"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
