"""HTTP adapter for the meeting hardware status and runtime preparation UI."""
from __future__ import annotations

import threading
import uuid
from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/api/meetings/hardware", tags=["meetings"])
_lock = threading.RLock()
_job: dict[str, Any] | None = None


def _status_payload() -> dict:
    from services.meeting_hardware import get_hardware_status
    status = get_hardware_status()
    status["preparation"] = dict(_job) if _job else None
    return status


def _estimated_download_bytes() -> int:
    try:
        return int(_status_payload()["gpu_runtime"].get("estimated_download_bytes", 0))
    except (KeyError, TypeError, ValueError):
        return 184 * 1024 * 1024


@router.get("")
def get_hardware():
    return _status_payload()


@router.post("/prepare", status_code=202)
def prepare_hardware():
    """Start the pinned DirectML install in the background and return its ID."""
    global _job
    with _lock:
        if _job and _job.get("status") == "running":
            return {"job_id": _job["id"], "status": "running",
                    "estimated_download_bytes": _estimated_download_bytes()}
        job_id = uuid.uuid4().hex
        _job = {"id": job_id, "status": "running", "stage": "queued", "progress": []}

    def update(item: dict):
        with _lock:
            if _job and _job.get("id") == job_id:
                _job["stage"] = item.get("stage")
                _job["progress"] = (_job.get("progress", []) + [item])[-50:]

    def worker():
        global _job
        from services.meeting_hardware import prepare_gpu_runtime
        result = prepare_gpu_runtime(update)
        with _lock:
            if _job and _job.get("id") == job_id:
                _job["status"] = "completed" if result.get("ok") else "failed"
                _job["result"] = result

    threading.Thread(target=worker, name="meeting-gpu-prepare", daemon=True).start()
    return {"job_id": job_id, "status": "running",
            "estimated_download_bytes": _estimated_download_bytes()}


@router.get("/prepare")
def preparation_status():
    return {"preparation": dict(_job) if _job else None}

