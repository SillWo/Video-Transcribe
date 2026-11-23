"""Cached, model-specific readiness for the simple transcription profiles."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from services import asr_resources
from services.runtime_devices import APP_ROOT, asr_gpu_candidate, get_device_details


CAPABILITY_PATH = APP_ROOT / ".runtime" / "asr" / "capabilities.json"
ASR_BENCHMARK_RECEIPT = APP_ROOT / ".runtime" / "asr" / "whisper-large-v3-auto-receipt.json"
ASR_BENCHMARK_WORDS_RECEIPT = APP_ROOT / ".runtime" / "asr" / "whisper-large-v3-auto-receipt-words.json"
ASR_BENCHMARK_VERSION = 1
_lock = threading.RLock()
_checking: set[tuple[Any, ...]] = set()


def expected_precision(model_id: str, backend: str | None = None) -> str:
    if model_id == "parakeet-tdt-0.6b-v3":
        return "FP32"
    if model_id == "qwen3-asr-1.7b":
        return "F16"
    if model_id == "large-v3":
        return {"cuda": "float16", "vulkan": "F16"}.get(backend, "int8")
    return "model-default"


def _runtime_version(model_id: str, backend: str) -> str:
    if model_id in {"small", "large-v3"}:
        from services.runtime_devices import find_whisper_cpp_executable
        if backend == "vulkan":
            executable = find_whisper_cpp_executable()
            if not executable:
                return "whisper.cpp:missing"
            path = Path(executable)
            try:
                stat = path.stat()
                return f"whisper.cpp:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
            except OSError:
                return f"whisper.cpp:{path.resolve()}:unavailable"
        try:
            import faster_whisper
            return f"faster-whisper:{getattr(faster_whisper, '__version__', 'unknown')}"
        except ImportError:
            return "faster-whisper:missing"
    kind = "native" if model_id == "qwen3-asr-1.7b" else "parakeet"
    versions = asr_resources.PACKAGE_VERSIONS.get((kind, backend), {})
    return json.dumps(versions, sort_keys=True, separators=(",", ":"))


def resource_fingerprint(model_id: str, backend: str, diarization: bool = False) -> str:
    """Return manifest/runtime fingerprint without hashing large local weights."""
    if model_id in {"small", "large-v3"} and backend == "vulkan":
        from services.whisper_cpp import MODEL_CACHE, model_filename
        filename = model_filename(model_id)
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
    if model_id in {"small", "large-v3"}:
        from services.model_management import get_model_status
        from hashlib import sha256
        try:
            location = get_model_status(model_id).get("cacheLocation")
            root = Path(location) if location else None
            artifacts = []
            if root and root.exists():
                for path in root.rglob("*"):
                    if path.is_file():
                        stat = path.stat()
                        artifacts.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
            raw = json.dumps({"model": model_id, "backend": backend, "artifacts": sorted(artifacts)}, sort_keys=True)
            return sha256(raw.encode("utf-8")).hexdigest()
        except OSError:
            pass
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
        "precision": expected_precision(model_id, backend),
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
    if model_id in {"small", "large-v3"}:
        from services.model_management import get_model_status
        if backend == "vulkan":
            from services.prepare_models import _vulkan_cached
            from services.runtime_devices import find_whisper_cpp_executable
            from services.whisper_cpp import cached_tokenizer_path
            if not find_whisper_cpp_executable():
                return False, "Vulkan Whisper runtime is missing."
            if not _vulkan_cached(model_id):
                return False, f"Whisper {model_id} Vulkan model is not prepared yet."
            if not cached_tokenizer_path(model_id):
                return False, f"Whisper {model_id} tokenizer for dictionary hints is not prepared yet."
        else:
            try:
                status = get_model_status(model_id)
            except Exception as exc:
                return False, f"Whisper model readiness could not be checked: {exc}"
            if status.get("status") != "downloaded":
                return False, f"Whisper {model_id} model is not prepared yet."
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


def asr_benchmark_identity(model_id: str, candidate: dict[str, Any], *, word_timestamps: bool = False) -> dict[str, Any]:
    """Identity for paired Whisper speed/quality evidence; OCR receipts remain separate."""
    backend = str(candidate.get("backend") or "")
    precision = expected_precision(model_id, backend)
    return {
        "benchmarkVersion": ASR_BENCHMARK_VERSION,
        "model": model_id,
        "backend": backend,
        "gpuIdentity": str(candidate.get("identity") or candidate.get("name") or "unknown"),
        "deviceIndex": int(candidate.get("deviceIndex") if candidate.get("deviceIndex") is not None else -1),
        "driverVersion": candidate.get("driverVersion"),
        "runtimeVersion": _runtime_version(model_id, backend),
        "resourceFingerprint": resource_fingerprint(model_id, backend),
        "computePrecision": precision,
        "wordTimestamps": bool(word_timestamps),
        "hintAdapterVersion": 1,
    }


def _qualified_asr_receipt(identity: dict[str, Any], receipt_path: Path = ASR_BENCHMARK_RECEIPT) -> dict[str, Any] | None:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    assessment = receipt.get("assessment") if isinstance(receipt.get("assessment"), dict) else {}
    try:
        speedup = float(assessment.get("speedup", 0))
        agreement = float(assessment.get("qualityAgreement", 0))
        mismatches = int(assessment.get("numericIdMismatches", 1))
    except (TypeError, ValueError):
        return None
    if (receipt.get("verified") is not True or receipt.get("identity") != identity
            or not math.isfinite(speedup) or not math.isfinite(agreement)
            or speedup < 1.5 or agreement < .95 or mismatches != 0):
        return None
    if (assessment.get("actualGpuVerified") is not True
            or assessment.get("criticalIdsPresent") is not True
            or assessment.get("gatesPassed") is not True
            or assessment.get("timedRunsPerBackend") != 3
            or assessment.get("warmupRunsPerBackend") != 1
            or assessment.get("quickRun") is True):
        return None
    return receipt


def auto_whisper_candidate(model_id: str, *, word_timestamps: bool = False) -> dict[str, Any] | None:
    """Return GPU only when a current paired ASR receipt clears all automatic gates."""
    if model_id != "large-v3":
        return None
    candidate = asr_gpu_candidate(model_id)
    if not candidate.get("canPrepare") or candidate.get("backend") not in {"cuda", "vulkan"}:
        return None
    identity = asr_benchmark_identity(model_id, candidate, word_timestamps=word_timestamps)
    receipt_path = ASR_BENCHMARK_WORDS_RECEIPT if word_timestamps else ASR_BENCHMARK_RECEIPT
    if _qualified_asr_receipt(identity, receipt_path) is None:
        return None
    ready, _reason = _resources_ready(model_id, candidate["backend"])
    return candidate if ready else None


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
        auto_candidate = auto_whisper_candidate(model_id)
        devices["auto"] = {
            "available": auto_candidate is not None,
            "name": auto_candidate.get("name") if auto_candidate else "CPU",
            "backend": auto_candidate.get("backend") if auto_candidate else "cpu",
            "device": "gpu" if auto_candidate else "cpu",
            "state": "verified" if auto_candidate else "unverified_cpu",
            "reason": None if auto_candidate else "Автоматический режим использует CPU: сопоставимый замер GPU скорости и качества отсутствует.",
        }
        rows.append({"id": profile_id, "label": label, "model": model_id,
                     "devices": devices,
                     "defaultDevice": devices["auto"]["device"], "defaultExecutionMode": "auto"})
    default = next((row for row in rows if row["id"] == "balanced"), rows[0] if rows else None)
    top_devices = default["devices"] if default else {
        "cpu": {"available": True, "name": "CPU"},
        "gpu": {"available": False, "name": "GPU", "reason": "Нет профилей распознавания."},
    }
    return {"profiles": rows, "devices": top_devices,
            "defaultDevice": default["defaultDevice"] if default else "cpu",
            "maxCpuThreads": max_cpu_threads}
