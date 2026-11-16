"""Optional, receipt-backed OCR model conversion helpers.

Importing this module is lightweight. ONNX and ml_dtypes are installed into a
separate target and imported only by the explicitly invoked preparation
subprocess; they never enter the DirectML environment's site-packages.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = APP_ROOT / ".runtime" / "models" / "ocr" / "cyrillic_PP-OCRv5_rec_mobile.onnx"
VARIANTS_ROOT = APP_ROOT / ".runtime" / "model-variants" / "ocr-rec-fp16"
CONVERTER_LIBS = APP_ROOT / ".runtime" / "meeting-ml" / "converter-libs"
CONVERTER_REQUIREMENTS = APP_ROOT / "requirements-meetings-conversion.txt"
CONVERTER_REQUIREMENTS_BYTES = 16_389_207 + 212_224


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def requirements_sha256() -> str:
    return sha256_file(CONVERTER_REQUIREMENTS)


def variant_paths(source_sha256: str) -> tuple[Path, Path]:
    return (VARIANTS_ROOT / f"{source_sha256}.onnx",
            VARIANTS_ROOT / f"{source_sha256}.json")


def conversion_status() -> dict[str, Any]:
    """Read-only readiness and download estimate; never imports or downloads."""
    try:
        requirements_hash = requirements_sha256()
        source_hash = sha256_file(SOURCE_MODEL) if SOURCE_MODEL.is_file() else None
    except OSError:
        requirements_hash, source_hash = None, None
    converter_ready = _converter_ready()
    if source_hash:
        model_path, receipt_path = variant_paths(source_hash)
        ready = _receipt_valid(model_path, receipt_path, source_hash, requirements_hash)
    else:
        model_path, receipt_path, ready = None, None, False
    return {
        "source_available": SOURCE_MODEL.is_file(),
        "source_sha256": source_hash,
        "ready": ready,
        "model_path": str(model_path) if model_path else None,
        "model_sha256": _receipt_sha(receipt_path) if ready and receipt_path else None,
        "converter_ready": converter_ready,
        "estimated_download_bytes": 0 if converter_ready else CONVERTER_REQUIREMENTS_BYTES,
        "reason": None if ready else (
            "Source recognition model is unavailable; FP16 conversion can run after OCR model preparation."
            if not SOURCE_MODEL.is_file() else "FP16 recognition variant is not prepared."
        ),
    }


def validate_fp16_variant(model_path: str | Path, expected_sha256: str) -> bool:
    """Validate a selected variant's location, source, receipt, and content hash."""
    try:
        candidate = Path(model_path).resolve(strict=True)
        root = VARIANTS_ROOT.resolve(strict=True)
        candidate.relative_to(root)
        source_hash = sha256_file(SOURCE_MODEL)
        expected_path, receipt_path = variant_paths(source_hash)
        return (candidate == expected_path.resolve(strict=True)
                and _receipt_valid(candidate, receipt_path, source_hash, requirements_sha256())
                and sha256_file(candidate) == expected_sha256)
    except (OSError, ValueError):
        return False


def prepare_fp16_variant(gpu_python: str | Path) -> dict[str, Any]:
    """Install isolated converters and atomically prepare the FP16 recognizer."""
    if not SOURCE_MODEL.is_file():
        return {"ready": False, "reason": "source_model_missing"}
    try:
        source_hash = sha256_file(SOURCE_MODEL)
        requirements_hash = requirements_sha256()
        model_path, receipt_path = variant_paths(source_hash)
        if _receipt_valid(model_path, receipt_path, source_hash, requirements_hash):
            return {"ready": True, "model_path": str(model_path),
                    "model_sha256": _receipt_sha(receipt_path),
                    "source_sha256": source_hash}
        if not _converter_ready():
            CONVERTER_LIBS.mkdir(parents=True, exist_ok=True)
            install = subprocess.run(
                [str(gpu_python), "-m", "pip", "install", "--disable-pip-version-check",
                 "--no-deps", "--upgrade", "--target", str(CONVERTER_LIBS), "-r",
                 str(CONVERTER_REQUIREMENTS)],
                cwd=APP_ROOT, check=False, timeout=900,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            if install.returncode:
                return {"ready": False, "reason": "converter_install_failed",
                        "detail": (install.stdout or "")[-2500:]}
            if not _converter_ready():
                return {"ready": False, "reason": "converter_install_incomplete"}

        model_path.parent.mkdir(parents=True, exist_ok=True)
        conversion_script = r'''import os, sys
from pathlib import Path
libs, source, destination = map(Path, sys.argv[1:])
sys.path.insert(0, str(libs))
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.onnx_model import OnnxModel
model = onnx.load(str(source))
model = convert_float_to_float16(model, keep_io_types=True)
OnnxModel(model).topological_sort()
onnx.checker.check_model(model)
temporary = destination.with_suffix(destination.suffix + ".tmp")
onnx.save(model, str(temporary))
os.replace(temporary, destination)
'''
        converted = subprocess.run(
            [str(gpu_python), "-c", conversion_script, str(CONVERTER_LIBS),
             str(SOURCE_MODEL), str(model_path)],
            cwd=APP_ROOT, check=False, timeout=900,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        if converted.returncode or not model_path.is_file():
            try:
                model_path.with_suffix(model_path.suffix + ".tmp").unlink()
            except OSError:
                pass
            return {"ready": False, "reason": "conversion_failed",
                    "detail": (converted.stdout or "Conversion produced no model.")[-2500:]}

        candidate_hash = sha256_file(model_path)
        receipt = {
            "schema": 1,
            "source_path": _recorded_path(SOURCE_MODEL),
            "source_sha256": source_hash,
            "model_path": _recorded_path(model_path),
            "model_sha256": candidate_hash,
            "converter_requirements_sha256": requirements_hash,
        }
        temporary_receipt = receipt_path.with_suffix(".json.tmp")
        temporary_receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        os.replace(temporary_receipt, receipt_path)
        return {"ready": True, "model_path": str(model_path),
                "model_sha256": candidate_hash, "source_sha256": source_hash}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        candidate = locals().get("model_path")
        receipt = locals().get("receipt_path")
        for temporary in (
            candidate.with_suffix(candidate.suffix + ".tmp") if candidate else None,
            receipt.with_suffix(".json.tmp") if receipt else None,
        ):
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass
        return {"ready": False, "reason": "conversion_failed",
                "detail": f"{type(exc).__name__}: {exc}"}


def preload_rec_model(gpu_python: str | Path) -> dict[str, Any]:
    """Ask RapidOCR in the isolated interpreter to download/load its official Rec model."""
    if SOURCE_MODEL.is_file():
        return {"ready": True, "source_sha256": sha256_file(SOURCE_MODEL), "already_available": True}
    script = (
        "from services.meeting_screen import _load_ocr; "
        "engine = _load_ocr('cpu'); engine._load_rec_model()"
    )
    try:
        result = subprocess.run(
            [str(gpu_python), "-c", script], cwd=APP_ROOT, check=False, timeout=900,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
        )
        if result.returncode or not SOURCE_MODEL.is_file():
            return {"ready": False, "reason": "source_model_preload_failed",
                    "detail": (result.stdout or "RapidOCR did not produce the expected Rec model.")[-2500:]}
        return {"ready": True, "source_sha256": sha256_file(SOURCE_MODEL)}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ready": False, "reason": "source_model_preload_failed",
                "detail": f"{type(exc).__name__}: {exc}"}


def prepare_variants_and_calibrate(
    gpu_python: str | Path,
    calibrate,
    progress=None,
    *,
    preload_fn=None,
    convert_fn=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preload a cold Rec model, prepare its optional variant, then calibrate."""
    preload_fn = preload_fn or preload_rec_model
    convert_fn = convert_fn or prepare_fp16_variant

    def report(stage: str, **items):
        if progress:
            progress({"stage": stage, **items})

    if not SOURCE_MODEL.is_file():
        report("models", status="preparing", model="recognition")
        try:
            preload = preload_fn(gpu_python)
        except Exception as exc:
            preload = {"ready": False, "reason": "source_model_preload_failed",
                       "detail": f"{type(exc).__name__}: {exc}"}
        if not preload.get("ready") or not SOURCE_MODEL.is_file():
            variant = {"ready": False,
                       "reason": preload.get("reason", "source_model_preload_failed"),
                       "detail": preload.get("detail")}
        else:
            variant = convert_fn(gpu_python)
    else:
        variant = convert_fn(gpu_python)

    report("calibration", status="running")
    calibration = calibrate(progress=progress, fp16_variant=variant)
    return variant, calibration


def _receipt_sha(receipt_path: Path) -> str | None:
    try:
        return json.loads(receipt_path.read_text(encoding="utf-8")).get("model_sha256")
    except (OSError, ValueError, TypeError):
        return None


def _converter_ready() -> bool:
    return all((CONVERTER_LIBS / marker).is_dir() for marker in (
        "onnx", "onnx-1.20.1.dist-info", "ml_dtypes", "ml_dtypes-0.5.4.dist-info",
    ))


def _recorded_path(path: Path) -> str:
    try:
        return path.relative_to(APP_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _receipt_valid(model_path: Path, receipt_path: Path, source_hash: str,
                   requirements_hash: str | None) -> bool:
    try:
        model_path.resolve(strict=True).relative_to(VARIANTS_ROOT.resolve(strict=True))
        receipt_path.resolve(strict=True).relative_to(VARIANTS_ROOT.resolve(strict=True))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        return (model_path.is_file()
                and receipt.get("schema") == 1
                and receipt.get("source_sha256") == source_hash
                and receipt.get("model_sha256") == sha256_file(model_path)
                and receipt.get("converter_requirements_sha256") == requirements_hash)
    except (OSError, ValueError, TypeError):
        return False
