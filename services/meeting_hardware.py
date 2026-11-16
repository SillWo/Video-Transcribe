"""Meeting-stage device selection and the isolated screen DirectML runtime.

This module is intentionally lightweight at import time: probing a screen
runtime never imports OpenCV, RapidOCR, or ONNX Runtime in the server process.
"""
from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

APP_ROOT = Path(__file__).resolve().parents[1]
GPU_ROOT = APP_ROOT / ".runtime" / "meeting-ml" / "directml"
GPU_PYTHON = GPU_ROOT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
GPU_REQUIREMENTS = APP_ROOT / "requirements-meetings-gpu.txt"
RECEIPT = APP_ROOT / ".runtime" / "meeting-ml" / "screen-directml-receipt.json"
WORKLOAD = "rapidocr-ppocrv5-mobile-cyrillic;screen-roi-v7-native-decoder-rec-variants"
AV_WHEEL_BYTES = 27_595_679
_GPU_COMPONENT_PATTERNS = (
    ("rapidocr-*.dist-info", None),
    ("onnxruntime_directml-*.dist-info", None),
    ("numpy-*.dist-info", None),
    ("opencv_python_headless-*.dist-info", None),
    ("pillow-*.dist-info", None),
    ("tokenizers-*.dist-info", None),
    ("av-18.1.0.dist-info", "av-18.1.0.dist-info"),
)
_MODEL_HASH_CACHE: tuple[tuple, str] | None = None


def _gpu_inventory() -> list[dict]:
    try:
        from services.runtime_devices import (
            _dxgi_gpu_inventory, _windows_gpu_inventory, enumerate_vulkan_gpus,
        )
        dxgi = _dxgi_gpu_inventory()
        wmi = _windows_gpu_inventory()
        vulkan = enumerate_vulkan_gpus()
        supported = {0x10DE, 0x1002, 0x8086}

        def matches(left: dict, right: dict) -> bool:
            return (left.get("vendorId") == right.get("vendorId")
                    and str(left.get("name") or "").strip().casefold()
                    == str(right.get("name") or "").strip().casefold()
                    and (left.get("deviceId") is None or right.get("deviceId") is None
                         or left.get("deviceId") == right.get("deviceId")))

        # Only DXGI supplies DirectML's adapter ordinal. Attach a WMI driver
        # fingerprint only when exactly one WMI row describes this adapter.
        rows = []
        for adapter in dxgi:
            if adapter.get("vendorId") not in supported:
                continue
            driver_rows = [row for row in wmi if matches(adapter, row)]
            row = {**adapter, "source": "dxgi", "driverIdentityVerified": False}
            if len(driver_rows) == 1:
                driver = driver_rows[0]
                row.update(driverVersion=driver.get("driverVersion"),
                           pnpDeviceId=driver.get("pnpDeviceId"),
                           driverIdentityVerified=bool(driver.get("driverVersion") and driver.get("pnpDeviceId")))
            rows.append(row)

        # Other APIs can establish that a supported GPU exists, but their
        # ordinals are never copied into deviceIndex or used for DirectML.
        for source, adapters in (("vulkan", vulkan), ("wmi", wmi)):
            for adapter in adapters:
                if adapter.get("vendorId") not in supported:
                    continue
                row = {**adapter, "source": source, "deviceIndex": None,
                       "driverIdentityVerified": False}
                if not any(matches(row, existing) for existing in rows):
                    rows.append(row)
        return rows
    except Exception:
        return []


def _identity() -> dict:
    adapters = _gpu_inventory()
    # Prefer CUDA for NVIDIA; the screen adapter implemented here is DML.
    candidate = None
    for vendor in (0x1002, 0x8086, 0x10DE):
        vendor_adapters = [g for g in adapters if g.get("vendorId") == vendor]
        candidate = next((g for g in vendor_adapters if g.get("source") == "dxgi"), None)
        if candidate is None:
            candidate = next(iter(vendor_adapters), None)
        if candidate is not None:
            break
    if not candidate:
        return {"name": None, "vendor_id": None, "device_id": None,
                "driver": None, "identity": None, "device_index": None,
                "driver_identity_verified": False}
    return {"name": candidate.get("name"), "vendor_id": candidate.get("vendorId"),
            "device_id": candidate.get("deviceId"), "driver": candidate.get("driverVersion"),
            "pnp_device_id": candidate.get("pnpDeviceId"),
            "identity": candidate.get("luid") or candidate.get("pnpDeviceId"),
            "device_index": candidate.get("deviceIndex"),
            "driver_identity_verified": bool(candidate.get("driverIdentityVerified"))}


def _has_dml_target(identity: dict) -> bool:
    index = identity.get("device_index")
    return (identity.get("vendor_id") in {0x1002, 0x8086, 0x10DE}
            and isinstance(index, int) and not isinstance(index, bool) and index >= 0)


def _has_receipt_identity(identity: dict) -> bool:
    return (_has_dml_target(identity) and bool(identity.get("identity"))
            and bool(identity.get("driver_identity_verified"))
            and bool(identity.get("driver")) and bool(identity.get("pnp_device_id")))


def _python_utf8_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _stock_ocr_env() -> dict[str, str]:
    """Clear inherited model overrides for CPU and unverified GPU runs."""
    return {
        "VT_MEETING_OCR_REC_PRECISION": "fp32",
        "VT_MEETING_OCR_BATCHING": "stock",
        "VT_MEETING_OCR_REC_MODEL_PATH": "",
        "VT_MEETING_OCR_REC_MODEL_SHA256": "",
    }


def _receipt_matches(identity: dict) -> bool:
    if not _has_receipt_identity(identity):
        return False
    try:
        data = json.loads(RECEIPT.read_text(encoding="utf-8"))
        variant = data.get("selected_variant") or {}
        if not isinstance(variant, dict):
            return False
        # FP16 remains an explicit experiment, but the short canvas receipt is
        # not representative enough to authorize automatic FP16 recognition.
        variant_valid = variant.get("precision") == "fp32" and variant.get("batching") in {
            "stock", "equal-width"
        }
        return bool(data.get("schema") == 2 and data.get("verified")
                    and data.get("identity") == identity
                    and data.get("python") == str(GPU_PYTHON.resolve())
                    and data.get("requirements_hash") == _requirements_hash()
                    and data.get("model_manifest") == _model_manifest()
                    and data.get("model_hash") == _model_hash()
                    and data.get("runtime_signature") == _runtime_signature()
                    and data.get("workload") == WORKLOAD
                    and variant_valid)
    except (OSError, ValueError, TypeError):
        return False


def _requirements_hash() -> str:
    return hashlib.sha256(GPU_REQUIREMENTS.read_bytes()).hexdigest()


def _model_manifest() -> list[list]:
    root = APP_ROOT / ".runtime" / "models" / "ocr"
    try:
        return [list(item) for item in sorted((str(path.relative_to(root)), path.stat().st_size,
                                               path.stat().st_mtime_ns)
                                              for path in root.rglob("*") if path.is_file())]
    except OSError:
        return []


def _model_hash() -> str:
    global _MODEL_HASH_CACHE
    manifest = tuple(_model_manifest())
    if _MODEL_HASH_CACHE and _MODEL_HASH_CACHE[0] == manifest:
        return _MODEL_HASH_CACHE[1]
    digest = hashlib.sha256()
    root = APP_ROOT / ".runtime" / "models" / "ocr"
    try:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            digest.update(path.relative_to(root).as_posix().encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    except OSError:
        return ""
    value = digest.hexdigest()
    _MODEL_HASH_CACHE = (manifest, value)
    return value


def _runtime_signature() -> dict:
    try:
        python_stat = GPU_PYTHON.stat()
        site = GPU_PYTHON.parent.parent / "Lib" / "site-packages" if os.name == "nt" else GPU_PYTHON.parent.parent / "lib"
        site_stat = site.stat()
        packages = []
        for pattern, _ in _GPU_COMPONENT_PATTERNS:
            dist_infos = list(site.glob(pattern))
            if not dist_infos:
                packages.append([pattern, None])
            for dist_info in dist_infos:
                metadata = dist_info / "METADATA"
                try:
                    stat = metadata.stat()
                    packages.append([dist_info.name, stat.st_size, stat.st_mtime_ns,
                                     hashlib.sha256(metadata.read_bytes()).hexdigest()])
                except OSError:
                    packages.append([dist_info.name, None])
        return {"python_size": python_stat.st_size, "python_mtime_ns": python_stat.st_mtime_ns,
                "site_mtime_ns": site_stat.st_mtime_ns, "packages": sorted(packages)}
    except OSError:
        return {}


def _gpu_component_state() -> tuple[bool, list[str]]:
    """Check installed distribution metadata without importing GPU packages."""
    site = (GPU_PYTHON.parent.parent / "Lib" / "site-packages" if os.name == "nt"
            else GPU_PYTHON.parent.parent / "lib")
    missing = []
    for pattern, exact_name in _GPU_COMPONENT_PATTERNS:
        matches = [site / exact_name] if exact_name else list(site.glob(pattern))
        if not any(path.is_dir() for path in matches):
            missing.append(pattern)
    return not missing, missing


def _component_missing_reason(missing: list[str]) -> str:
    if "av-18.1.0.dist-info" in missing:
        return "Не подготовлен компонент чтения видео. Подготовьте среду GPU в настройках."
    return "В изолированной DirectML-среде отсутствуют необходимые компоненты."


def get_hardware_status() -> dict:
    """Describe hardware and stage readiness without importing ML packages."""
    identity = _identity()
    python_ok = GPU_PYTHON.is_file()
    components_ready, missing_components = _gpu_component_state() if python_ok else (False, ["runtime"])
    prepared = python_ok and components_ready
    receipt_ok = _receipt_matches(identity) if prepared else False
    directml_possible = _has_dml_target(identity)
    try:
        required_bytes = sum(Path(p).stat().st_size for p in [GPU_REQUIREMENTS] if Path(p).exists())
    except OSError:
        required_bytes = 0
    try:
        from services.meeting_settings import load_settings
        settings = load_settings(APP_ROOT)
    except Exception:
        settings = {"execution_mode": "auto", "device": "cpu"}
    model_ready = bool(_model_manifest())
    av_ready = python_ok and "av-18.1.0.dist-info" not in missing_components
    from services.meeting_model_variants import conversion_status

    conversion = conversion_status()
    estimated_bytes = ((0 if python_ok else 160 * 1024 * 1024)
                       + (0 if model_ready else 24 * 1024 * 1024)
                       + (0 if av_ready else AV_WHEEL_BYTES)
                       + conversion["estimated_download_bytes"])
    return {
        "execution_mode": settings.get("execution_mode", "auto"),
        "device": settings.get("device", "cpu"),
        "hardware": {"cpu": platform.processor() or "CPU", "gpu": identity,
                     "directml_registered": directml_possible,
                     "intel_arc_verified": False},
        "stages": {
            "screen": {"cpu": {"available": True, "verified": True},
                       "gpu": {"available": receipt_ok, "backend": "directml",
                               "prepared": prepared, "verified": receipt_ok,
                               "reason": None if receipt_ok else (
                                   _component_missing_reason(missing_components) if python_ok and not components_ready else
                                   "DirectML экранный OCR ещё не прошёл локальный замер CPU/GPU качества и скорости." if prepared
                                   else "Изолированная DirectML-среда ещё не подготовлена."),
                               "model_variants": conversion,
                               "can_prepare": directml_possible and (not receipt_ok or not prepared)}},
            "audio": {"auto_device": "cpu", "reason": "Нет сохранённого сопоставимого замера скорости и качества."},
            "diarization": {"auto_device": "cpu",
                            "reason": "Нет сохранённого сопоставимого замера скорости и качества диаризации."},
            "index": {"auto_device": "cpu",
                      "query_reason": "В коротком профиле запроса CPU показал меньшую задержку.",
                      "bulk_reason": "Нет сопоставимого замера пакетной индексации; используется CPU."},
        },
        "gpu_runtime": {"python": str(GPU_PYTHON), "requirements_bytes": required_bytes,
                        "prepared": prepared, "estimated_download_bytes": estimated_bytes,
                        "receipt": str(RECEIPT) if receipt_ok else None},
    }


def plan_stage(stage: str, execution_mode: str = "auto", requested_device: str = "cpu",
               profile: str = "balanced") -> dict:
    """Resolve CPU/GPU runtime for meeting workers, with no implicit fallback."""
    stage = str(stage).casefold()
    mode = str(execution_mode).casefold()
    requested = str(requested_device).casefold()
    if stage not in {"screen", "audio", "index"}:
        return {"device": "cpu", "python": sys.executable, "env": {}, "verified": False,
                "reason": "Неизвестный этап обработки.", "error": "unknown_stage"}
    if mode not in {"auto", "manual"} or requested not in {"cpu", "gpu", "directml"}:
        return {"device": "cpu", "python": sys.executable, "env": {}, "verified": False,
                "reason": "Некорректный режим или устройство.", "error": "invalid_selection"}
    if stage == "audio":
        return {"device": "cpu", "python": sys.executable,
                "env": {"PYTHONPATH": str(APP_ROOT), **_stock_ocr_env()},
                "verified": True,
                "reason": "Сжатие Opus выполняется на CPU; GPU для этого этапа не используется.",
                "error": None}
    want_gpu = requested in {"gpu", "directml"} if mode == "manual" else False
    if mode == "auto":
        # Only screen OCR currently has a device-specific receipt. No ASR GPU
        # selection is made without paired quality and performance evidence.
        want_gpu = stage == "screen" and _receipt_matches(_identity())
    if not want_gpu:
        return {"device": "cpu", "python": sys.executable,
                "env": {"PYTHONPATH": str(APP_ROOT), **_stock_ocr_env()},
                "verified": True, "reason": "CPU выбран как доступный проверенный путь.", "error": None}
    identity = _identity()
    if not _has_dml_target(identity):
        return {"device": "directml", "python": str(GPU_PYTHON),
                "env": {"PYTHONPATH": str(APP_ROOT)}, "verified": False,
                "reason": "Не удалось определить индекс GPU для DirectML.",
                "error": "gpu_device_index_unavailable" if mode == "manual" else None}
    if stage != "screen":
        return {"device": "directml", "python": sys.executable, "env": {"PYTHONPATH": str(APP_ROOT)},
                "verified": False, "reason": "Для этого этапа GPU путь не подтверждён.",
                "error": "gpu_stage_unsupported" if mode == "manual" else None}
    if not GPU_PYTHON.is_file():
        return {"device": "directml", "python": str(GPU_PYTHON), "env": {"PYTHONPATH": str(APP_ROOT)},
                "verified": False, "reason": "Изолированная DirectML-среда не подготовлена.",
                "error": "gpu_runtime_missing" if mode == "manual" else None}
    components_ready, missing_components = _gpu_component_state()
    if not components_ready:
        return {"device": "directml", "python": str(GPU_PYTHON), "env": {"PYTHONPATH": str(APP_ROOT)},
                "verified": False, "reason": _component_missing_reason(missing_components),
                "error": "gpu_runtime_missing" if mode == "manual" else None}
    verified = _receipt_matches(identity)
    env = {"PYTHONPATH": str(APP_ROOT),
           "VT_MEETING_DML_DEVICE_ID": str(identity["device_index"]),
           **_stock_ocr_env()}
    if verified:
        env.update(_selected_variant_env())
    return {"device": "directml", "python": str(GPU_PYTHON), "env": env,
            "verified": verified,
            "reason": None if verified else "Ручной DirectML запуск разрешён без актуального receipt качества/скорости; результат будет экспериментальным.",
            "error": None}


def _selected_variant_env() -> dict[str, str]:
    try:
        variant = json.loads(RECEIPT.read_text(encoding="utf-8")).get("selected_variant") or {}
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(variant, dict):
        return {}
    env = {
        "VT_MEETING_OCR_REC_PRECISION": str(variant.get("precision", "fp32")),
        "VT_MEETING_OCR_BATCHING": str(variant.get("batching", "stock")),
        "VT_MEETING_OCR_REC_MODEL_PATH": "",
        "VT_MEETING_OCR_REC_MODEL_SHA256": "",
    }
    if variant.get("precision") == "fp16":
        env["VT_MEETING_OCR_REC_MODEL_PATH"] = str(variant.get("model_path", ""))
        env["VT_MEETING_OCR_REC_MODEL_SHA256"] = str(variant.get("model_sha256", ""))
    return env


def prepare_gpu_runtime(progress: Callable[[dict], None] | None = None) -> dict:
    """Create/install the isolated screen environment; calibration is a separate gate."""
    def report(stage: str, **items):
        if progress:
            progress({"stage": stage, **items})
    if not _identity().get("vendor_id"):
        return {"ok": False, "error": "compatible_gpu_missing", "prepared": False}
    if os.name != "nt":
        return {"ok": False, "error": "directml_requires_windows", "prepared": False}
    if sys.version_info[:2] != (3, 13):
        return {"ok": False, "error": "python_313_required", "prepared": False,
                "reason": "Screen DirectML runtime requires the project Python 3.13 environment."}
    if not GPU_REQUIREMENTS.is_file():
        return {"ok": False, "error": "requirements_missing", "prepared": False}
    try:
        GPU_ROOT.parent.mkdir(parents=True, exist_ok=True)
        if not GPU_PYTHON.is_file():
            report("environment", status="creating")
            subprocess.run([sys.executable, "-m", "venv", str(GPU_ROOT)], cwd=APP_ROOT,
                           check=True, timeout=300, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           env=_python_utf8_env())
        report("packages", status="installing")
        proc = subprocess.Popen([str(GPU_PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
                                 "--progress-bar", "off", "-r", str(GPU_REQUIREMENTS)], cwd=APP_ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=_python_utf8_env(), shell=False)
        output_tail = []
        assert proc.stdout is not None
        for line in proc.stdout:
            output_tail.append(line.rstrip())
            output_tail = output_tail[-30:]
            report("packages", status="installing", detail=line.strip()[-180:])
        returncode = proc.wait(timeout=3600)
        if returncode:
            return {"ok": False, "error": "package_install_failed", "prepared": False,
                    "log_tail": "\n".join(output_tail)[-4000:]}
        report("packages", status="installed")
        # Import and provider checks run in the isolated process, never here.
        check = "import av, cv2, numpy, onnxruntime, rapidocr; print(onnxruntime.__version__, onnxruntime.get_available_providers())"
        probe = subprocess.run([str(GPU_PYTHON), "-c", check], cwd=APP_ROOT, check=False,
                               timeout=120, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               encoding="utf-8", errors="replace", env=_python_utf8_env())
        probe_stdout, probe_stderr = probe.stdout or "", probe.stderr or ""
        if probe.returncode or "DmlExecutionProvider" not in probe_stdout:
            return {"ok": False, "error": "directml_provider_unavailable", "prepared": True,
                    "log_tail": (probe_stderr or probe_stdout or "Runtime probe produced no output.")[-4000:]}
        from services.meeting_model_variants import prepare_variants_and_calibrate

        variant, calibration = prepare_variants_and_calibrate(
            GPU_PYTHON, calibrate_screen_runtime, progress
        )
        return {"ok": bool(calibration.get("verified")), "prepared": True,
                "verified": bool(calibration.get("verified")),
                "reason": calibration.get("reason"), "python": str(GPU_PYTHON),
                "providers": probe_stdout.strip(), "fp16_variant": variant,
                "fp16_candidate_unavailable_reason": calibration.get("fp16_unavailable_reason"),
                "calibration": calibration}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": type(exc).__name__, "prepared": GPU_PYTHON.is_file(),
                "reason": str(exc)}


def calibrate_screen_runtime(progress: Callable[[dict], None] | None = None,
                             fp16_variant: dict | None = None) -> dict:
    """Run paired CPU/DML OCR in the isolated interpreter and persist a strict receipt."""
    identity_for_run = _identity()
    if not _has_dml_target(identity_for_run):
        return {"verified": False, "reason": "Не удалось определить индекс GPU для DirectML."}
    if not _has_receipt_identity(identity_for_run):
        return {"verified": False,
                "reason": "Не удалось однозначно подтвердить идентичность драйвера GPU."}
    if not GPU_PYTHON.is_file():
        return {"verified": False, "reason": "DirectML runtime is not prepared."}
    script = r'''import json, os, tempfile, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from services import meeting_screen as screen
from services.meeting_hardware import APP_ROOT
import rapidocr.inference_engine.onnxruntime.main as ort_main
from onnxruntime import InferenceSession as OriginalSession

profile_dir = Path(tempfile.gettempdir()) / "meeting-ocr-profile"
profile_dir.mkdir(exist_ok=True)
sessions = []
def profiled_session(*args, **kwargs):
    opts = kwargs.get("sess_options")
    if opts is not None:
        opts.enable_profiling = True
        opts.profile_file_prefix = str(profile_dir / ("ocr-" + str(time.time_ns())))
    session = OriginalSession(*args, **kwargs)
    sessions.append(session)
    return session
ort_main.InferenceSession = profiled_session

canvas = Image.new("RGB", (1440, 1100), "white")
draw = ImageDraw.Draw(canvas)
font = None
for candidate in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
    try:
        font = ImageFont.truetype(candidate, 32); break
    except OSError:
        pass
if font is None: font = ImageFont.load_default()
canvas_lines = [
    "ARCH-427", "OPS-318", "INV-7421", "Build 59104", "Count 3", "Limit 25",
    "Value 27", "Rate 0.95", "Team EDU", "State OK", "ID 103", "Q2026Q4",
    "Демонстрация Jira / Jira screen", "Лимит повторов: 25",
    "Status: In Progress", "SQL: SELECT 25 AS value", "Queue: OPEN / OPS",
    "Approval: Pending",
]
for row, text in enumerate(canvas_lines):
    draw.text((70, 35 + row * 58), text, fill="#111111", font=font)
image = np.asarray(canvas)[:, :, ::-1].copy()
changed_canvas = canvas.copy()
changed_draw = ImageDraw.Draw(changed_canvas)
prefix = "Лимит повторов: "
changed_y = 35 + canvas_lines.index(prefix + "25") * 58
number_x = 70 + draw.textlength(prefix, font=font)
number_box = draw.textbbox((number_x, changed_y), "25", font=font)
changed_draw.rectangle((number_box[0]-2, number_box[1]-2,
                        number_box[2]+2, number_box[3]+2), fill="white")
changed_draw.text((number_x, changed_y), "27", fill="#111111", font=font)
changed_image = np.asarray(changed_canvas)[:, :, ::-1].copy()

def run(device, precision="fp32", batching="stock", model_path=None, model_sha=None):
    global sessions
    sessions = []
    for key in ("VT_MEETING_OCR_REC_PRECISION", "VT_MEETING_OCR_REC_MODEL_PATH",
                "VT_MEETING_OCR_REC_MODEL_SHA256", "VT_MEETING_OCR_BATCHING"):
        os.environ.pop(key, None)
    if precision == "fp16":
        os.environ["VT_MEETING_OCR_REC_PRECISION"] = "fp16"
        os.environ["VT_MEETING_OCR_REC_MODEL_PATH"] = model_path
        os.environ["VT_MEETING_OCR_REC_MODEL_SHA256"] = model_sha
    if batching == "equal-width":
        os.environ["VT_MEETING_OCR_BATCHING"] = "equal-width"
    engine = screen._load_ocr(device)
    rec_model = engine._load_rec_model()
    stock_rec_batch_num = int(rec_model.rec_batch_num)
    session_batch_sizes = []
    original_rec_session = rec_model.session
    from services.meeting_ocr_batches import DEFAULT_EQUAL_WIDTH_MAX_BATCH
    configured_max_batch = (DEFAULT_EQUAL_WIDTH_MAX_BATCH if batching == "equal-width"
                            else stock_rec_batch_num)
    class SessionBatchRecorder:
        def __call__(self, batch, *args, **kwargs):
            session_batch_sizes.append(int(batch.shape[0]))
            return original_rec_session(batch, *args, **kwargs)
        def __getattr__(self, name):
            return getattr(original_rec_session, name)
    rec_model.session = SessionBatchRecorder()
    engine(image)  # warm-up, also constructs all three ONNX sessions
    full_changed_reference = screen._parse_ocr(changed_image, engine)
    anchored_stats = {}
    samples = []
    delta_samples = []
    base_regions = []
    anchored_regions = []
    session_batch_sizes.clear()
    for _ in range(3):
        started = time.perf_counter()
        base_regions = screen._parse_ocr(image, engine)
        rois = screen._expand_rois_to_regions(
            screen._changed_rois(image, changed_image), base_regions, image.shape[1], image.shape[0])
        delta_started = time.perf_counter()
        anchored_regions = screen._incremental_regions(
            changed_image, base_regions, rois, engine, 1.0,
            previous_image=image, stats=anchored_stats)
        delta_samples.append(time.perf_counter() - delta_started)
        samples.append(time.perf_counter() - started)
    providers = [p for s in sessions for p in s.get_providers()]
    dml_ops = 0
    profiles = []
    for s in sessions:
        try:
            profile = Path(s.end_profiling())
            profiles.append(profile)
            events = json.loads(profile.read_text(encoding="utf-8"))
            dml_ops += sum(1 for event in events
                           if (event.get("args") or {}).get("provider") == "DmlExecutionProvider")
        except Exception:
            pass
    for path in profiles:
        try: path.unlink()
        except OSError: pass
    batch_proof = {
        "stock_rec_batch_num": stock_rec_batch_num,
        "max_batch": configured_max_batch,
        "session_batch_sizes": session_batch_sizes,
        "max_session_batch_size": max(session_batch_sizes, default=0),
        "equal_width_merge_exercised": any(
            size > stock_rec_batch_num for size in session_batch_sizes),
    }
    return {"regions": base_regions or [], "anchored_regions": anchored_regions or [],
            "full_changed_reference": full_changed_reference or [],
            "anchored_stats": anchored_stats,
            "samples": samples, "delta_samples": delta_samples,
            "elapsed_seconds": sorted(samples)[len(samples)//2],
            "providers": providers, "operator_profile": {"dml_operator_count": dml_ops},
            "batch_proof": batch_proof,
            "variant": {"precision": precision, "batching": batching}}

cpu = run("cpu")
cpu_control = {"regions": cpu["anchored_regions"],
               "full_reference_regions": cpu["full_changed_reference"],
               "anchored_rec_regions": cpu["anchored_stats"].get("anchored_rec_regions", 0)}
cpu["anchored_control"] = cpu_control
candidate_specs = [
    ("fp32", "stock", None),
    ("fp32", "equal-width", None),
]
fp16_unavailable = None
fp16_variant = json.loads(os.environ.get("VT_MEETING_FP16_VARIANT", "null"))
if fp16_variant and fp16_variant.get("ready"):
    candidate_specs.extend([
        ("fp16", "stock", fp16_variant),
        ("fp16", "equal-width", fp16_variant),
    ])
else:
    fp16_unavailable = (fp16_variant or {}).get("reason", "conversion_unavailable")

candidates = []
candidate_failures = []
for precision, batching, variant_info in candidate_specs:
    try:
        result = run(
            "directml", precision, batching,
            (variant_info or {}).get("model_path"),
            (variant_info or {}).get("model_sha256"),
        )
        result["anchored_control"] = {
            "regions": result["anchored_regions"],
            "full_reference_regions": cpu["full_changed_reference"],
            "anchored_rec_regions": result["anchored_stats"].get("anchored_rec_regions", 0),
        }
        if precision == "fp16":
            result["variant"].update(
                model_path=variant_info["model_path"],
                model_sha256=variant_info["model_sha256"],
            )
        candidates.append(result)
    except Exception as exc:
        candidate_failures.append({
            "variant": {"precision": precision, "batching": batching},
            "reason": type(exc).__name__ + ": " + str(exc),
        })
        if precision == "fp16":
            fp16_unavailable = candidate_failures[-1]["reason"]
print("MEETING_OCR_CALIBRATION=" + json.dumps(
    {"cpu": cpu, "candidates": candidates, "candidate_failures": candidate_failures,
     "fp16_unavailable": fp16_unavailable},
    ensure_ascii=False))
'''
    if progress:
        progress({"stage": "calibration", "status": "cpu_reference"})
    try:
        child_env = _python_utf8_env(os.environ)
        child_env["VT_MEETING_DML_DEVICE_ID"] = str(identity_for_run["device_index"])
        child_env["VT_MEETING_FP16_VARIANT"] = json.dumps(fp16_variant or {
            "ready": False, "reason": "conversion_unavailable"
        })
        result = subprocess.run([str(GPU_PYTHON), "-c", script], cwd=APP_ROOT, env=child_env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace",
                                timeout=900, check=False)
        stdout, stderr = result.stdout or "", result.stderr or ""
        payload_line = next((line for line in reversed(stdout.splitlines())
                             if line.startswith("MEETING_OCR_CALIBRATION=")), None)
        if result.returncode or not payload_line:
            return {"verified": False, "reason": "Calibration subprocess failed.",
                    "error": (stderr or stdout or "Calibration subprocess returned no JSON output.")[-3000:]}
        output = json.loads(payload_line.split("=", 1)[1])
        if progress:
            progress({"stage": "calibration", "status": "comparing"})
        # Use a content hash for the actual OCR weights and exact runtime/workload
        # signatures for later cache invalidation.
        model_hash = _model_hash()
        identity = identity_for_run
        assessed = []
        for candidate_index, candidate in enumerate(output["candidates"]):
            result_assessment = _assess_screen_candidate(identity, output["cpu"], candidate)
            assessed.append({
                "candidate_index": candidate_index,
                "variant": candidate["variant"],
                "elapsed_seconds": candidate.get("elapsed_seconds"),
                "samples": candidate.get("samples", []),
                "delta_samples": candidate.get("delta_samples", []),
                "regions": candidate.get("regions", []),
                "anchored_control": candidate.get("anchored_control", {}),
                "providers": candidate.get("providers", []),
                "operator_profile": candidate.get("operator_profile", {}),
                "batch_proof": candidate.get("batch_proof", {}),
                "assessment": result_assessment,
            })
        candidate_records = [*assessed]
        candidate_failures = output.get("candidate_failures") or []
        candidate_records.extend({
            "variant": failure.get("variant"),
            "available": False,
            "reason": failure.get("reason"),
        } for failure in candidate_failures)
        if (output.get("fp16_unavailable")
                and not any((failure.get("variant") or {}).get("precision") == "fp16"
                            for failure in candidate_failures)):
            candidate_records.append({
                "variant": {"precision": "fp16", "batching": "stock/equal-width"},
                "available": False,
                "reason": output["fp16_unavailable"],
            })
        verified_candidates = [item for item in assessed if item["assessment"]["verified"]]
        selected = min(verified_candidates,
                       key=lambda item: item["assessment"]["gpu_seconds"]) if verified_candidates else None
        selected_variant = selected["variant"] if selected else None
        selected_result = (output["candidates"][selected["candidate_index"]]
                           if selected else
                           min(output["candidates"],
                               key=lambda item: item.get("elapsed_seconds", float("inf"))))
        receipt = write_screen_receipt(
            identity=identity, model_hash=model_hash,
            runtime=json.dumps(_runtime_signature(), sort_keys=True),
            workload=WORKLOAD, cpu_result=output["cpu"],
            gpu_result=selected_result, variant=selected_variant,
            candidates=candidate_records,
        )
        return {"verified": receipt["verified"], "receipt": receipt,
                "cpu_seconds": receipt["cpu_seconds"], "gpu_seconds": receipt["gpu_seconds"],
                "selected_variant": receipt.get("selected_variant"),
                "fp16_unavailable_reason": output.get("fp16_unavailable"),
                "reason": None if receipt["verified"] else (
                    "Anchored recognition was not exercised or its numeric/ID results differed from full CPU OCR."
                    if not receipt.get("anchored_control_passed") else
                    "OCR quality, speed, or DirectML operator evidence did not meet the acceptance gate.")}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        return {"verified": False, "reason": "Calibration could not be completed.",
                "error": f"{type(exc).__name__}: {exc}"}


def _assess_screen_candidate(identity: dict, cpu_result: dict, gpu_result: dict) -> dict:
    """Apply the existing strict text, numeric, anchored, speed, and DML gates."""
    cpu_text = cpu_result.get("regions") or []
    gpu_text = gpu_result.get("regions") or []
    cpu_literal = [(str(r.get("text") or ""), r.get("bbox") or []) for r in cpu_text]
    gpu_literal = [(str(r.get("text") or ""), r.get("bbox") or []) for r in gpu_text]
    identifiers = lambda rows: sorted(text for text, _bbox in rows if any(c.isdigit() for c in text))
    operator = gpu_result.get("operator_profile") or {}
    dml_ops = int(operator.get("dml_operator_count", 0) or 0)

    def critical_numeric_texts(regions):
        return sorted(str(region.get("text") or "").strip() for region in regions
                      if any(char.isdigit() for char in str(region.get("text") or "")))

    cpu_control = cpu_result.get("anchored_control") or {}
    gpu_control = gpu_result.get("anchored_control") or {}
    full_reference = cpu_control.get("full_reference_regions") or []
    cpu_anchored = cpu_control.get("regions") or []
    gpu_anchored = gpu_control.get("regions") or []
    control_numeric = critical_numeric_texts(full_reference)
    anchored_control_ok = bool(
        cpu_control.get("anchored_rec_regions", 0) > 0
        and gpu_control.get("anchored_rec_regions", 0) > 0
        and control_numeric
        and critical_numeric_texts(cpu_anchored) == control_numeric
        and critical_numeric_texts(gpu_anchored) == control_numeric
    )
    count = max(1, len(cpu_literal))
    available = set(range(len(gpu_literal)))
    matched = 0
    for cpu_text_value, cpu_box in cpu_literal:
        if len(cpu_box) != 4:
            continue
        candidates = []
        for index in available:
            gpu_text_value, gpu_box = gpu_literal[index]
            if gpu_text_value != cpu_text_value or len(gpu_box) != 4:
                continue
            # Geometry may drift by a couple of pixels across providers, while
            # still requiring each recognized line to occupy the same area.
            if all(abs(float(a) - float(b)) <= 8 for a, b in zip(cpu_box, gpu_box)):
                candidates.append(index)
        if candidates:
            available.remove(candidates[0])
            matched += 1
    accuracy = matched / count
    gpu_seconds = float(gpu_result.get("elapsed_seconds", 0) or 0)
    cpu_seconds = float(cpu_result.get("elapsed_seconds", 0) or 0)
    variant = gpu_result.get("variant") or {}
    batch_proof = gpu_result.get("batch_proof") or {}
    equal_width_merge_exercised = bool(
        batch_proof.get("equal_width_merge_exercised")
        and batch_proof.get("max_session_batch_size", 0)
        > batch_proof.get("stock_rec_batch_num", float("inf"))
        and batch_proof.get("max_session_batch_size", 0)
        <= batch_proof.get("max_batch", float("inf"))
    )
    batching_gate = (variant.get("batching") != "equal-width"
                     or equal_width_merge_exercised)
    # Auto is allowed to choose DML only when a paired probe demonstrates an
    # actual advantage, not merely a successful GPU session.
    verified = (_has_receipt_identity(identity)
                and accuracy >= .95 and identifiers(cpu_literal) == identifiers(gpu_literal) and dml_ops > 0
                and gpu_seconds > 0 and cpu_seconds > gpu_seconds and anchored_control_ok
                and batching_gate)
    experimental = variant.get("precision") == "fp16"
    reason = ("FP16 requires representative critical-value acceptance; real pilot found numeric regressions"
              if experimental else None)
    return {"verified": bool(verified and not experimental), "accuracy": accuracy,
            "experimental": experimental, "reason": reason, "cpu_seconds": cpu_seconds,
            "gpu_seconds": gpu_seconds, "dml_operator_count": dml_ops,
            "equal_width_merge_exercised": equal_width_merge_exercised,
            "literal_regions_equal": cpu_literal == gpu_literal,
            "identifiers_and_numbers_equal": identifiers(cpu_literal) == identifiers(gpu_literal),
            "anchored_control_passed": anchored_control_ok,
            "anchored_control_cpu_regions": len(cpu_anchored),
            "anchored_control_gpu_regions": len(gpu_anchored),
            "anchored_control_critical_values_equal": bool(
                control_numeric and critical_numeric_texts(cpu_anchored) == control_numeric
                and critical_numeric_texts(gpu_anchored) == control_numeric)}


def write_screen_receipt(*, identity: dict, model_hash: str, runtime: str,
                         workload: str, cpu_result: dict, gpu_result: dict,
                         output_path: Path | None = None,
                         variant: dict | None = None,
                         candidates: list[dict] | None = None) -> dict:
    """Persist a verified comparison only when accuracy and operator evidence pass."""
    if variant is None:
        variant = {"precision": "fp32", "batching": "stock"}
    assessment = _assess_screen_candidate(identity, cpu_result, gpu_result)
    selected_fp16 = variant.get("precision") == "fp16"
    if selected_fp16:
        assessment = {**assessment, "verified": False, "experimental": True,
                      "reason": "FP16 requires representative critical-value acceptance; real pilot found numeric regressions"}
    verified = bool(assessment["verified"] and variant is not None and not selected_fp16)
    runtime_signature = _runtime_signature()
    model_manifest = _model_manifest()
    receipt = {"schema": 2, "verified": verified, "identity": identity,
               "model_hash": model_hash, "runtime": runtime, "workload": workload,
               "python": str(GPU_PYTHON.resolve()),
               "requirements_hash": _requirements_hash() if GPU_REQUIREMENTS.exists() else None,
               "runtime_signature": runtime_signature, "model_manifest": model_manifest,
               "selected_variant": variant if verified else None,
               "candidate_results": candidates or [],
               **{key: assessment[key] for key in (
                   "accuracy", "cpu_seconds", "gpu_seconds", "dml_operator_count",
                   "equal_width_merge_exercised",
                   "literal_regions_equal", "identifiers_and_numbers_equal",
                   "anchored_control_passed", "anchored_control_cpu_regions",
                   "anchored_control_gpu_regions", "anchored_control_critical_values_equal")},
               "created_at": time.time()}
    destination = output_path or RECEIPT
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, destination)
    return receipt
