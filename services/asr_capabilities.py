"""Cached, model-specific readiness for the simple transcription profiles."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from services import asr_resources
from services.runtime_devices import APP_ROOT, asr_gpu_candidate, get_device_details


CAPABILITY_PATH = APP_ROOT / ".runtime" / "asr" / "capabilities.json"
_lock = threading.RLock()
_checking: set[tuple[Any, ...]] = set()


def expected_precision(model_id: str) -> str:
    if model_id == "parakeet-tdt-0.6b-v3":
        return "FP32"
    if model_id == "qwen3-asr-1.7b":
        return "F16"
    return "model-default"


def _runtime_version(model_id: str, backend: str) -> str:
    if model_id == "small":
        from services.runtime_devices import find_whisper_cpp_executable
        executable = find_whisper_cpp_executable()
        if not executable:
            return "whisper.cpp:missing"
        path = Path(executable)
        try:
            stat = path.stat()
            return f"whisper.cpp:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            return f"whisper.cpp:{path.resolve()}:unavailable"
    kind = "native" if model_id == "qwen3-asr-1.7b" else "parakeet"
    versions = asr_resources.PACKAGE_VERSIONS.get((kind, backend), {})
    return json.dumps(versions, sort_keys=True, separators=(",", ":"))


def resource_fingerprint(model_id: str, backend: str, diarization: bool = False) -> str:
    """Return manifest/runtime fingerprint without hashing large local weights."""
    if model_id == "small":
        from services.whisper_cpp import MODEL_CACHE, model_filename
        filename = model_filename("small")
        artifacts = []
        try:
            for path in MODEL_CACHE.rglob(filename):
                try:
                    stat = path.stat()
                    artifacts.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
                except OSError:
                    continue
        except OSError:
            pass
        from hashlib import sha256
        raw = json.dumps({"model": model_id, "backend": backend, "artifacts": sorted(artifacts)}, sort_keys=True)
        return sha256(raw.encode("utf-8")).hexdigest()
    return asr_resources._fingerprint(model_id, backend, diarization)


def _candidate_key(candidate: dict[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        str(candidate.get("backend") or ""),
        str(candidate.get("identity") or candidate.get("name") or ""),
        str(candidate.get("driverVersion") or ""),
        int(candidate.get("deviceIndex") if candidate.get("deviceIndex") is not None else -1),
        str(candidate.get("name") or ""),
    )


def _identity(model_id: str, candidate: dict[str, Any], fingerprint: str | None = None) -> dict[str, Any]:
    backend = str(candidate.get("backend") or "")
    return {
        "verificationVersion": 2,
        "model": model_id,
        "requestedDevice": "gpu",
        "gpuIdentity": str(candidate.get("identity") or candidate.get("name") or "unknown"),
        "deviceIndex": int(candidate.get("deviceIndex") if candidate.get("deviceIndex") is not None else -1),
        "backend": backend,
        "driverVersion": candidate.get("driverVersion"),
        "runtimeVersion": _runtime_version(model_id, backend),
        "resourceFingerprint": fingerprint or resource_fingerprint(model_id, backend),
        "precision": expected_precision(model_id),
    }


def _read_cache() -> dict[str, Any]:
    try:
        payload = json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))
        if payload.get("schemaVersion") == 1 and isinstance(payload.get("records"), list):
            return payload
    except (OSError, ValueError, AttributeError):
        pass
    return {"schemaVersion": 1, "records": []}


def _write_cache(payload: dict[str, Any]) -> None:
    CAPABILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="capabilities-", suffix=".tmp", dir=CAPABILITY_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, CAPABILITY_PATH)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _matching_record(model_id: str, candidate: dict[str, Any]) -> dict[str, Any] | None:
    expected = _identity(model_id, candidate)
    cache = _read_cache()
    matches = [row for row in cache["records"] if isinstance(row, dict)
               and all(row.get(key) == value for key, value in expected.items())]
    return matches[-1] if matches else None


def _resources_ready(model_id: str, backend: str) -> tuple[bool, str | None]:
    if model_id == "small":
        from services.model_management import get_model_status
        from services.runtime_devices import find_whisper_cpp_executable
        if not find_whisper_cpp_executable():
            return False, "Vulkan Whisper runtime is missing."
        try:
            status = get_model_status("small")
        except Exception as exc:
            return False, f"Whisper model readiness could not be checked: {exc}"
        if status.get("status") != "downloaded":
            return False, "Whisper small model is not prepared yet."
        return True, None
    try:
        status = asr_resources.asr_resource_status(model_id, backend)
    except Exception as exc:
        return False, f"ASR resource readiness could not be checked: {exc}"
    if not status.get("ready"):
        missing = status.get("missing") or []
        detail = ", ".join(str(item) for item in missing[:3])
        return False, f"ASR resources are not prepared yet{': ' + detail if detail else '.'}"
    return True, None


def start_gpu_check(model_id: str, candidate: dict[str, Any]) -> None:
    """Mark an owned GPU smoke as checking for live options polling."""
    with _lock:
        _checking.add((model_id, *_candidate_key(candidate)))


def finish_gpu_check(model_id: str, candidate: dict[str, Any]) -> None:
    with _lock:
        _checking.discard((model_id, *_candidate_key(candidate)))


def record_gpu_probe(model_id: str, candidate: dict[str, Any], result: dict[str, Any],
                     *, fingerprint: str | None = None) -> dict[str, Any]:
    """Persist successful GPU evidence only for the exact smoke configuration."""
    verification = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    if not verification:
        verification = result.get("gpu_verification") if isinstance(result.get("gpu_verification"), dict) else {}
    gpu_verified = bool(verification.get("gpu_verified")) or verification.get("status") == "verified"
    text = result.get("text")
    language = result.get("language") or result.get("detected_language")
    valid_smoke = bool(result.get("ok", True)) and isinstance(text, str) and bool(text.strip()) \
        and isinstance(language, str) and language.casefold().startswith("en") and gpu_verified
    record = _identity(model_id, candidate, fingerprint)
    record.update({
        "verified": valid_smoke,
        "textNonempty": bool(isinstance(text, str) and text.strip()),
        "language": language,
        "error": result.get("error") if isinstance(result.get("error"), str) else (
            result.get("error", {}).get("message") if isinstance(result.get("error"), dict) else None
        ),
        "verificationBasis": verification.get("verification_basis") or verification.get("basis"),
        "diagnostics": verification,
    })
    with _lock:
        cache = _read_cache()
        rows = [row for row in cache["records"] if not (
            isinstance(row, dict) and row.get("model") == model_id
            and row.get("requestedDevice") == "gpu"
            and row.get("gpuIdentity") == record["gpuIdentity"]
            and row.get("backend") == record["backend"]
        )]
        rows.append(record)
        _write_cache({"schemaVersion": 1, "records": rows})
    return record


def gpu_capability(model_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate.get("canPrepare") or not candidate.get("backend"):
        return {
            "available": False,
            "name": candidate.get("name") or "GPU",
            "reason": candidate.get("reason") or "Совместимый GPU не обнаружен.",
            "backend": candidate.get("backend"),
            "deviceIndex": candidate.get("deviceIndex"),
            "state": "unavailable",
            "canPrepare": False,
        }
    with _lock:
        checking = (model_id, *_candidate_key(candidate)) in _checking
    if checking:
        return {
            "available": False,
            "name": candidate.get("name") or "GPU",
            "reason": "Проверка GPU выполняется.",
            "backend": candidate["backend"],
            "deviceIndex": candidate.get("deviceIndex"),
            "state": "checking",
            "canPrepare": True,
        }
    record = _matching_record(model_id, candidate)
    if record and record.get("verified"):
        resources_ready, resource_reason = _resources_ready(model_id, candidate["backend"])
        if not resources_ready:
            return {
                "available": False,
                "name": candidate.get("name") or "GPU",
                "reason": resource_reason or "GPU resources are no longer ready.",
                "backend": candidate["backend"],
                "deviceIndex": candidate.get("deviceIndex"),
                "state": "needs_preparation",
                "canPrepare": True,
            }
        return {
            "available": True,
            "name": candidate.get("name") or "GPU",
            "reason": None,
            "backend": candidate["backend"],
            "deviceIndex": candidate.get("deviceIndex"),
            "state": "ready",
            "canPrepare": True,
        }
    if record and not record.get("verified"):
        reason = record.get("error") or record.get("reason")
        if not reason:
            diagnostics = record.get("diagnostics") or {}
            reason = diagnostics.get("reason") or diagnostics.get("verification_basis") or diagnostics.get("basis")
        return {
            "available": False,
            "name": candidate.get("name") or "GPU",
            "reason": str(reason or "Проверка GPU завершилась без подтверждения выполнения модели на видеокарте."),
            "backend": candidate["backend"],
            "deviceIndex": candidate.get("deviceIndex"),
            "state": "unavailable",
            "canPrepare": bool(candidate.get("canPrepare")),
        }
    reason = "Работа этой модели на видеокарте будет проверена при первом запуске."
    if candidate.get("reason"):
        reason = f"{reason} {candidate['reason']}"
    return {
        "available": False,
        "name": candidate.get("name") or "GPU",
        "reason": reason,
        "backend": candidate["backend"],
        "deviceIndex": candidate.get("deviceIndex"),
        "state": "checking" if checking else "needs_preparation",
        "canPrepare": True,
    }


def profile_device_options(profiles: dict[str, tuple[str, str]], max_cpu_threads: int) -> dict[str, Any]:
    cpu_rows = {item["id"]: item for item in get_device_details()}
    rows = []
    for profile_id, (label, model_id) in profiles.items():
        candidate = asr_gpu_candidate(model_id)
        devices = {
            "cpu": {"available": True, "name": cpu_rows.get("cpu", {}).get("name", "CPU"),
                    "reason": None, "backend": "cpu", "state": "ready", "canPrepare": True},
            "gpu": gpu_capability(model_id, candidate),
        }
        rows.append({"id": profile_id, "label": label, "model": model_id,
                     "devices": devices,
                     "defaultDevice": "gpu" if devices["gpu"]["available"] or devices["gpu"]["canPrepare"] else "cpu"})
    default = next((row for row in rows if row["id"] == "balanced"), rows[0] if rows else None)
    top_devices = default["devices"] if default else {
        "cpu": {"available": True, "name": "CPU"},
        "gpu": {"available": False, "name": "GPU", "reason": "Нет профилей распознавания."},
    }
    return {"profiles": rows, "devices": top_devices,
            "defaultDevice": default["defaultDevice"] if default else "cpu",
            "maxCpuThreads": max_cpu_threads}
