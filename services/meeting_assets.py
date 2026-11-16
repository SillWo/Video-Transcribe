"""Explicitly prepare the pinned local multilingual-e5-small ONNX assets.

No model code is imported here and no download runs on module import or status
reads. Public helpers: ``asset_status(project_root=None)`` and
``prepare_semantic_search(project_root=None, progress=None)``.
"""
from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from pathlib import Path
from typing import Any, Callable


MODEL_NAME = "multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DOWNLOAD_BYTES = 470268510 + 17082730 + 653
_FILES = (
    {"path": "onnx/model.onnx", "size": 470268510,
     "sha256": "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"},
    {"path": "onnx/tokenizer.json", "size": 17082730,
     "sha256": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"},
    {"path": "onnx/config.json", "size": 653,
     "sha256": "bbb7c1333fc4b3e27fbc9cd5d2070aabcc1d4dfb99917c3633e772f97545a6b6"},
)
_URL = f"https://huggingface.co/intfloat/{MODEL_NAME}/resolve/{REVISION}/{{relative}}?download=true"
_LOCK = threading.RLock()
_VERIFIED: dict[str, tuple[int, int, bool]] = {}


def model_root(project_root: str | Path | None = None) -> Path:
    project = Path(project_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    return project / ".runtime" / "models" / MODEL_NAME


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _valid(path: Path, item: dict[str, Any], *, force_hash: bool = False) -> bool:
    try:
        stat = path.stat()
    except OSError:
        _VERIFIED.pop(str(path), None)
        return False
    if stat.st_size != item["size"]:
        _VERIFIED.pop(str(path), None)
        return False
    key = str(path.resolve())
    cached = _VERIFIED.get(key)
    signature = (stat.st_mtime_ns, stat.st_size)
    if not force_hash and cached and cached[:2] == signature:
        return cached[2]
    valid = _digest(path) == item["sha256"]
    _VERIFIED[key] = (*signature, valid)
    return valid


def asset_status(project_root: str | Path | None = None) -> dict[str, Any]:
    """Report exact pinned model availability without loading ONNX or tokenizer."""
    root = model_root(project_root)
    entries = []
    for item in _FILES:
        path = root / item["path"]
        entries.append({"name": Path(item["path"]).name, "available": _valid(path, item),
                        "expected_bytes": item["size"], "downloaded_bytes": path.stat().st_size if path.is_file() else 0})
    return {"available": all(entry["available"] for entry in entries), "version": REVISION,
            "download_bytes": DOWNLOAD_BYTES, "files": entries,
            "path": str(root / "onnx")}


def _download_one(path: Path, item: dict[str, Any], progress: Callable[[dict], None] | None,
                  completed_before: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    url = _URL.format(relative=item["path"])
    digest = hashlib.sha256()
    received = 0
    try:
        with urllib.request.urlopen(url, timeout=45) as response, partial.open("wb") as target:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                received += len(block)
                if received > item["size"]:
                    raise ValueError(f"Downloaded {Path(item['path']).name} is larger than its pinned size")
                target.write(block)
                digest.update(block)
                if progress:
                    progress({"file": Path(item["path"]).name,
                              "completed_bytes": completed_before + received,
                              "file_completed_bytes": received, "file_total_bytes": item["size"],
                              "total_bytes": DOWNLOAD_BYTES,
                              "percent": min(99, int((completed_before + received) * 100 / DOWNLOAD_BYTES))})
            target.flush()
            os.fsync(target.fileno())
        if received != item["size"] or digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Pinned checksum/size mismatch for {Path(item['path']).name}")
        os.replace(partial, path)
        stat = path.stat()
        _VERIFIED[str(path.resolve())] = (stat.st_mtime_ns, stat.st_size, True)
    except Exception:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise


def prepare_semantic_search(project_root: str | Path | None = None,
                            progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Download only missing or invalid pinned files, atomically and with SHA checks."""
    root = model_root(project_root)
    completed = 0
    with _LOCK:
        for item in _FILES:
            path = root / item["path"]
            if _valid(path, item):
                completed += item["size"]
                if progress:
                    progress({"file": Path(item["path"]).name, "completed_bytes": completed,
                              "file_completed_bytes": item["size"], "file_total_bytes": item["size"],
                              "total_bytes": DOWNLOAD_BYTES, "percent": int(completed * 100 / DOWNLOAD_BYTES),
                              "skipped": True})
                continue
            _download_one(path, item, progress, completed)
            completed += item["size"]
    result = asset_status(project_root)
    if not result["available"]:
        raise RuntimeError("Model files did not pass the pinned integrity check")
    if progress:
        progress({"file": None, "completed_bytes": DOWNLOAD_BYTES, "total_bytes": DOWNLOAD_BYTES,
                  "percent": 100, "ready": True})
    return result
