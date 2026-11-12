"""Pinned NeMo-Speech.cpp runtime and device discovery for diarization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from services.model_paths import MODEL_STORAGE_ROOT
from services.task_errors import TaskFailure

APP_ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.3.0"
MODEL_REPO = "nvidia/Nemotron-3-Diarization"
MODEL_REVISION = "52a432672732425e4c19765b9fec9b5810e6f4ba"
MODEL_FILENAME = "Nemotron-3-Diarization.q8_0.gguf"
MODEL_SIZE = 107012128
MODEL_SHA256 = "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1"
RUNTIMES = {
    "cpu": ("4ed9740e361e4653f56c3dce7d1e81883a1c8b831c1bd7d01567fff1130b5b05", "cpu"),
    "vulkan": ("72b3b8b41bfe2be0320eff4e9d310594e4f99deefcde4a157799a039656963ce", "vulkan"),
    "cuda": ("d0d32ae1bf6ee0534767a47b2aa0e9e2ff6fe39643465c0baae9301d04f6ed42", "cuda"),
}
_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_DOCTOR_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _runtime_root() -> Path:
    return APP_ROOT / ".runtime" / "nemo-speech"


def _assert_runtime_contained(path: Path) -> Path:
    root = _runtime_root().resolve()
    runtime_parent = (APP_ROOT / ".runtime").resolve()
    if root != runtime_parent and runtime_parent not in root.parents:
        raise ValueError(f"Runtime root resolves outside .runtime: {root}")
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Runtime path resolves outside nemo-speech: {resolved}")
    return root


def _safe_remove_runtime_tree(path: Path) -> None:
    """Remove a runtime staging/backup path only when it resolves inside nemo-speech."""
    root = _assert_runtime_contained(path)
    if path.is_symlink():
        lexical = Path(os.path.abspath(path))
        if lexical != _runtime_root() and _runtime_root() not in lexical.parents:
            raise ValueError(f"Refusing to remove a link outside the runtime root: {path}")
        path.unlink()
        return
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to remove a path outside the runtime root: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def _runtime_executable_candidates(backend: str) -> list[Path]:
    root = _runtime_root() / backend
    if not root.exists():
        return []
    names = ("nemo-speech.exe", "nemo-speech-cli.exe", "nemo-speech")
    found = [p for name in names for p in root.rglob(name) if p.is_file()]
    return sorted(found)


def find_diarization_executable(backend: str) -> str | None:
    """Return an installed executable for the requested backend."""
    candidates = _runtime_executable_candidates(backend)
    return str(candidates[0].resolve()) if candidates else None


def _sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _HASH_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _HASH_CACHE[key] = value
    return value


def resolve_diarization_model() -> Path:
    """Validate the pinned model without downloading it."""
    path = MODEL_STORAGE_ROOT / "diarization" / MODEL_FILENAME
    try:
        if path.is_file() and path.stat().st_size == MODEL_SIZE and _sha256(path) == MODEL_SHA256:
            return path
    except OSError:
        pass
    raise TaskFailure(
        f"Diarization model is missing or failed SHA-256 validation: {path}. Run start.ps1 to prepare it.",
        "model_unavailable", stage="diarization",
    )


def _download(url: str, destination: Path, expected_hash: str, *, expected_size: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(url, timeout=60) as response, temp_path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != expected_hash or (expected_size is not None and size != expected_size):
            raise ValueError(f"SHA-256/size mismatch (sha256={digest.hexdigest()}, bytes={size})")
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def _safe_extract(archive: Path, target: Path) -> None:
    base = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            name = entry.filename.replace("\\", "/")
            relative = PurePosixPath(name)
            if relative.is_absolute() or any(part in ("..", "") for part in relative.parts):
                raise ValueError(f"Unsafe path in runtime archive: {entry.filename}")
            out = (target / Path(*relative.parts)).resolve()
            if out != base and base not in out.parents:
                raise ValueError(f"Runtime archive path escapes target: {entry.filename}")
        bundle.extractall(target)


def _install_runtime(backend: str) -> str:
    root = _runtime_root()
    _assert_runtime_contained(root)
    installed = find_diarization_executable(backend)
    if installed and _runtime_valid(_doctor(installed, refresh=True), backend):
        return installed
    expected_hash, flavor = RUNTIMES[backend]
    filename = f"nemo-speech-{VERSION}-windows-x86_64-{flavor}.zip"
    url = f"https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v{VERSION}/{filename}"
    downloads = root / ".downloads"
    if downloads.is_symlink():
        raise ValueError("Refusing to use a .downloads link outside the NeMo-Speech runtime directory.")
    downloads.mkdir(parents=True, exist_ok=True)
    _assert_runtime_contained(downloads)
    archive = downloads / filename
    if not archive.is_file() or _sha256(archive) != expected_hash:
        _download(url, archive, expected_hash)
    staging = root / f".{backend}.extracting"
    _safe_remove_runtime_tree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    _safe_extract(archive, staging)
    matches = [p for n in ("nemo-speech.exe", "nemo-speech-cli.exe") for p in staging.rglob(n) if p.is_file()]
    if not matches:
        raise ValueError(f"Pinned {backend} archive contains no NeMo-Speech executable")
    target = root / backend
    old = root / f".{backend}.previous"
    _safe_remove_runtime_tree(old)
    if target.exists():
        _assert_runtime_contained(target)
        os.replace(target, old)
    try:
        os.replace(staging, target)
    except Exception:
        if old.exists() and not target.exists():
            os.replace(old, target)
        raise
    _safe_remove_runtime_tree(old)
    return str(next(p for n in ("nemo-speech.exe", "nemo-speech-cli.exe") for p in target.rglob(n) if p.is_file()).resolve())


def _install_model() -> Path:
    path = MODEL_STORAGE_ROOT / "diarization" / MODEL_FILENAME
    try:
        if path.is_file() and path.stat().st_size == MODEL_SIZE and _sha256(path) == MODEL_SHA256:
            return path
    except OSError:
        pass
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILENAME}"
    _download(url, path, MODEL_SHA256, expected_size=MODEL_SIZE)
    _HASH_CACHE.clear()
    return path


def _windows_inventory() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    script = ("Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID "
              "| ConvertTo-Json -Compress")
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", script], stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=8,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        rows = json.loads(result.stdout or "[]") if result.returncode == 0 else []
        if isinstance(rows, dict):
            rows = [rows]
        found = []
        for row in rows:
            name = str(row.get("Name") or "").strip()
            pnp = str(row.get("PNPDeviceID") or "").upper()
            vendor = next((v for marker, v in (("VEN_10DE", 0x10DE), ("VEN_1002", 0x1002), ("VEN_8086", 0x8086)) if marker in pnp), None)
            if name and vendor and not _software_name(name):
                found.append({"name": name, "vendorId": vendor, "pnpDeviceId": pnp})
        return found
    except Exception:
        return []


def _software_name(name: str) -> bool:
    return any(token in name.casefold() for token in ("microsoft basic", "render-only", "software rasterizer", "swiftshader", "lavapipe", "llvmpipe", "virtual display"))


def _vulkan_inventory() -> list[dict[str, Any]]:
    try:
        from services.runtime_devices import enumerate_vulkan_gpus
        return [dict(item) for item in enumerate_vulkan_gpus() if not _software_name(str(item.get("name", "")))]
    except Exception:
        return []


def _cuda_inventory() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name,index", "--format=csv,noheader,nounits"],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            return []
        rows = []
        for line in result.stdout.splitlines():
            cols = [col.strip() for col in line.split(",")]
            if len(cols) >= 4:
                rows.append({"uuid": cols[0], "pciBusId": cols[1].lower(), "name": cols[2], "cudaIndex": int(cols[3]), "vendorId": 0x10DE})
        return rows
    except Exception:
        return []


def _stable_id(gpu: dict[str, Any]) -> str:
    pnp = str(gpu.get("pnpDeviceId") or "").upper()
    if gpu.get("uuid"):
        fingerprint = str(gpu["uuid"]).upper()
    elif gpu.get("deviceUUID"):
        fingerprint = str(gpu["deviceUUID"]).upper()
    elif pnp:
        fingerprint = pnp
    else:
        fingerprint = f"{gpu.get('vendorId')}|{gpu.get('deviceId', '')}|{gpu.get('name', '').casefold()}|{gpu.get('duplicate', 0)}"
    return "gpu-" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20]


def _inventory() -> list[dict[str, Any]]:
    windows = _windows_inventory()
    vk = _vulkan_inventory()
    cuda = _cuda_inventory()
    physical: list[dict[str, Any]] = []
    # NVIDIA's UUID is the authoritative identity for CUDA devices; its ordinal
    # is refreshed from nvidia-smi each inventory pass.
    for row in cuda:
        item = dict(row)
        item["name"] = item.get("name") or "NVIDIA GPU"
        physical.append(item)

    # Vulkan identity comes from VkPhysicalDeviceIDProperties, never from the
    # enumeration position or adapter name. Ignore NVIDIA Vulkan duplicates:
    # diarization dispatches NVIDIA through CUDA.
    vulkan_physical = []
    for row in vk:
        if row.get("vendorId") not in (0x1002, 0x8086):
            continue
        item = dict(row)
        item["name"] = item.get("name") or "GPU"
        item["vulkanIndex"] = item.pop("runtimeIndex", item.pop("deviceIndex", None))
        item["integrated"] = item.get("type") == 1
        vulkan_physical.append(item)
        physical.append(item)

    # Keep adapters seen by Windows but not exposed by the runtime inventory.
    # When identical devices are ambiguous, leave them unattached and unavailable.
    runtime_counts = Counter((g.get("vendorId"), str(g.get("name", "")).casefold()) for g in physical)
    seen_windows: Counter[tuple[int, str]] = Counter()
    for row in windows:
        signature = (row.get("vendorId"), str(row.get("name", "")).casefold())
        seen_windows[signature] += 1
        if seen_windows[signature] > runtime_counts[signature]:
            physical.append(dict(row))
    counts: dict[str, int] = {}
    for item in physical:
        base = f"{item.get('vendorId')}|{item.get('deviceId', '')}|{item.get('name', '').casefold()}"
        duplicate = counts.get(base, 0)
        counts[base] = duplicate + 1
        item["duplicate"] = duplicate
        item["id"] = _stable_id(item)
        item["integrated"] = bool(item.get("integrated") or item.get("type") == 1 or
                                   (item.get("type") is None and re.search(r"(uhd|iris|radeon \d{3,4}m|graphics)", str(item.get("name", "")), re.I)))
    return sorted(physical, key=lambda g: (g.get("vendorId", 0), g["id"]))


def _doctor(executable: str, *, refresh: bool = False) -> dict[str, Any]:
    import time
    now = time.monotonic()
    cached = _DOCTOR_CACHE.get(executable)
    if not refresh and cached and now - cached[0] < 5:
        return cached[1]
    try:
        result = subprocess.run([executable, "--json", "doctor"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=12,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        payload = json.loads(result.stdout)
        if result.returncode:
            payload["_error"] = result.stderr.strip() or f"doctor exited {result.returncode}"
    except Exception as exc:
        payload = {"_error": str(exc)}
    _DOCTOR_CACHE[executable] = (now, payload)
    return payload


def _runtime_valid(doctor: dict[str, Any], backend: str) -> bool:
    if doctor.get("version") != VERSION or not doctor.get("features", {}).get("diarization"):
        return False
    feature = {"cuda": "backend_cuda", "vulkan": "backend_vulkan"}.get(backend)
    return not feature or bool(doctor.get("features", {}).get(feature))


def _ordered_devices(inventory: list[dict[str, Any]], doctor: dict[str, Any], backends: set[str]) -> list[dict[str, Any]]:
    features = doctor.get("features", {})
    rows = doctor.get("devices", [])
    devices = []
    for gpu in inventory:
        backend = "cuda" if gpu.get("vendorId") == 0x10DE else "vulkan"
        runtime_supported = backend in backends
        index_key = "cudaIndex" if backend == "cuda" else "vulkanIndex"
        physical_index = gpu.get(index_key)
        prefix = "CUDA" if backend == "cuda" else "Vulkan"
        matching = next((row for row in rows
                         if re.fullmatch(prefix + r"(\d+)", str(row.get("name", "")), re.I)
                         and int(re.fullmatch(prefix + r"(\d+)", str(row.get("name", "")), re.I).group(1)) == physical_index
                         and str(row.get("description", "")).strip().casefold() == str(gpu.get("name", "")).strip().casefold()
                         and ("gpu" in str(row.get("type", "")).casefold() or "cuda" in str(row.get("type", "")).casefold())
                         and "cpu" not in str(row.get("type", "")).casefold()), None)
        # Backend ordinals are encoded in the device name. The generic doctor
        # index also counts CPU entries on some builds, so it is not dispatch-safe.
        index = int(re.fullmatch(prefix + r"(\d+)", str(matching.get("name", "")), re.I).group(1)) if matching else None
        backend_feature = bool(features.get("backend_cuda" if backend == "cuda" else "backend_vulkan"))
        available = bool(runtime_supported and matching is not None and features.get("diarization") and backend_feature and
                         _runtime_valid(doctor, backend) and doctor.get("accelerator_available") and
                         doctor.get("driver_runtime_compatible") and not doctor.get("_error"))
        reason = None if available else ("NeMo-Speech runtime is not installed." if not runtime_supported else "GPU driver/runtime is unavailable or incompatible." if not doctor.get("driver_runtime_compatible", False) or not doctor.get("accelerator_available", False) else "Diarization backend is unavailable in this runtime." if not backend_feature or not features.get("diarization") else "GPU is not exposed by NeMo-Speech doctor.")
        devices.append({"id": gpu["id"], "name": gpu["name"], "backend": backend, "available": available,
                        "reason": reason, "deviceIndex": index, "description": matching.get("description") if matching else None})
    return sorted(devices, key=lambda d: (0 if d["backend"] == "cuda" else 1, 1 if next((g.get("integrated", False) for g in inventory if g["id"] == d["id"]), False) else 0, d["id"]))


def _prepared(backends: set[str], *, refresh: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = _inventory()
    devices: list[dict[str, Any]] = []
    doctor_by_backend: dict[str, dict[str, Any]] = {}
    for backend in sorted(backends):
        executable = find_diarization_executable(backend)
        doctor = _doctor(executable, refresh=refresh) if executable else {"_error": "NeMo-Speech executable is missing", "features": {}, "devices": []}
        doctor_by_backend[backend] = doctor
        matching_gpu_rows = [g for g in inventory if ("cuda" if g.get("vendorId") == 0x10DE else "vulkan") == backend]
        if matching_gpu_rows:
            devices.extend(_ordered_devices(matching_gpu_rows, doctor, {backend}))
    cpu_executable = find_diarization_executable("cpu")
    cpu_doctor = _doctor(cpu_executable, refresh=refresh) if cpu_executable else {"_error": "CPU runtime is missing", "features": {}, "devices": []}
    cpu_available = bool("cpu" in backends and _runtime_valid(cpu_doctor, "cpu"))
    devices.append({"id": "cpu", "name": "CPU", "backend": "cpu", "available": cpu_available,
                    "reason": None if cpu_available else cpu_doctor.get("_error") or "CPU diarization runtime is not ready.",
                    "deviceIndex": 0})
    ordered = sorted(devices, key=lambda d: (0 if d["backend"] == "cuda" else 1,
                                              1 if next((g.get("integrated", False) for g in inventory if g["id"] == d["id"]), False) else 0,
                                              d["id"]))
    # Return a doctor report for callers that need CPU readiness when no GPUs exist.
    primary = doctor_by_backend.get("cuda") or doctor_by_backend.get("vulkan") or cpu_doctor
    return ordered, primary


def prepare_diarization(check_only: bool = False) -> dict[str, Any]:
    """Prepare CPU plus each detected vendor runtime, model, and report all failures."""
    inventory = _inventory()
    backends = {"cpu"}
    backends.update("cuda" if gpu.get("vendorId") == 0x10DE else "vulkan" for gpu in inventory if gpu.get("vendorId") in (0x10DE, 0x1002, 0x8086))
    failures: list[dict[str, str]] = []
    prepared_backends: set[str] = set()
    for backend in sorted(backends):
        executable = find_diarization_executable(backend)
        try:
            if not check_only:
                executable = _install_runtime(backend)
            if not executable:
                failures.append({"backend": backend, "message": "Runtime is not installed (check-only mode)."})
                continue
            doctor = _doctor(executable, refresh=True)
            if doctor.get("_error") or not _runtime_valid(doctor, backend):
                failures.append({"backend": backend, "message": doctor.get("_error") or f"NeMo-Speech doctor is not a compatible v{VERSION} {backend} diarization runtime."})
            else:
                prepared_backends.add(backend)
        except Exception as exc:
            failures.append({"backend": backend, "message": str(exc)})
    try:
        if not check_only:
            _install_model()
        resolve_diarization_model()
    except Exception as exc:
        failures.append({"backend": "model", "message": str(exc)})
    devices, _doctor_data = _prepared(backends)
    for backend in sorted({"cuda" if gpu.get("vendorId") == 0x10DE else "vulkan" for gpu in inventory}):
        if not any(device["available"] for device in devices if device["backend"] == backend):
            failures.append({"backend": backend, "message": f"Detected GPU hardware, but no compatible {backend} diarization runtime is ready."})
    return {"ready": not failures and bool(prepared_backends), "failures": failures, "devices": devices}


def diarization_options() -> dict[str, Any]:
    inventory = _inventory()
    backends = {"cpu"}
    backends.update("cuda" if gpu.get("vendorId") == 0x10DE else "vulkan" for gpu in inventory)
    devices, doctor = _prepared(backends)
    model_ready = True
    try:
        resolve_diarization_model()
    except TaskFailure:
        model_ready = False
    auto_target = next((d for d in devices if d["backend"] in (("cuda", "vulkan") if inventory else ("cpu",))), None)
    auto_ready = bool(auto_target and auto_target["available"])
    devices.insert(0, {"id": "auto", "name": "Automatic", "backend": "auto", "available": auto_ready and model_ready,
                       "reason": None if auto_ready and model_ready else "No usable diarization device is ready."})
    if not model_ready:
        for device in devices:
            if device["available"]:
                device["available"] = False
                device["reason"] = "Diarization model is not ready."
    return {"enabledByDefault": False, "defaultDevice": "auto", "devices": devices,
            "ready": model_ready and any(d["available"] for d in devices)}


def resolve_diarization_device(requested: str = "auto") -> dict[str, Any]:
    """Resolve a stable hardware ID to a fresh runtime index; never silently fallback."""
    try:
        model = resolve_diarization_model()
    except TaskFailure as exc:
        raise TaskFailure(str(exc), "model_unavailable", stage="diarization") from exc
    inventory = _inventory()
    backends = {"cpu"}
    backends.update("cuda" if g.get("vendorId") == 0x10DE else "vulkan" for g in inventory)
    devices, _doctor_data = _prepared(backends, refresh=True)
    if requested in (None, "", "auto"):
        eligible = [d for d in devices if d["backend"] in ("cuda", "vulkan")] if inventory else [d for d in devices if d["backend"] == "cpu"]
        selected = eligible[0] if eligible else None
        if not selected or not selected["available"]:
            message = selected.get("reason") if selected else None
            raise TaskFailure(message or "No diarization device is ready.", "device_unavailable", stage="diarization")
    else:
        selected = next((d for d in devices if d["id"] == requested), None)
        if selected is None:
            raise TaskFailure(f"Selected diarization device is not available: {requested}", "device_unavailable", stage="diarization")
        if not selected["available"]:
            raise TaskFailure(selected.get("reason") or "Selected diarization device is not ready.", "device_unavailable", stage="diarization")
    executable = find_diarization_executable(selected["backend"])
    if not executable:
        raise TaskFailure(f"{selected['backend']} diarization runtime is missing.", "device_unavailable", stage="diarization")
    return {"id": selected["id"], "name": selected["name"], "backend": selected["backend"],
            "deviceIndex": selected.get("deviceIndex", 0), "executable": executable, "modelPath": model}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Prepare Nemotron-3-Diarization runtime and model")
    parser.add_argument("--check-only", action="store_true", help="Check existing files without downloading")
    args = parser.parse_args(argv)
    report = prepare_diarization(check_only=args.check_only)
    print("Диаризация готова." if report["ready"] else "Диаризация не готова.")
    for device in report["devices"]:
        state = "готово" if device.get("available") else "недоступно"
        print(f"  {device.get('name')} [{device.get('backend')}]: {state}" + (f" — {device['reason']}" if device.get("reason") else ""))
    for failure in report["failures"]:
        print(f"Ошибка {failure['backend']}: {failure['message']}", file=sys.stderr)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
