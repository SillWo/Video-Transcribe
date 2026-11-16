"""Loopback HTTP API for explicitly preparing optional local semantic search."""
from __future__ import annotations

import threading
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from services.meeting_assets import DOWNLOAD_BYTES, asset_status, prepare_semantic_search


router = APIRouter(prefix="/api/meetings/assets", tags=["meeting-assets"])
_LOCK = threading.RLock()
_JOB: dict | None = None


class PrepareRequest(BaseModel):
    component: str = "semantic_search"


def _loopback(request: Request) -> None:
    host = request.client.host if request.client else ""
    if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="Asset preparation is available only from this computer")


def _status() -> dict:
    return asset_status()


@router.get("")
def get_assets(request: Request):
    _loopback(request)
    return {"semantic_search": _status()}


@router.get("/prepare")
def get_prepare_status(request: Request):
    _loopback(request)
    with _LOCK:
        state = dict(_JOB) if _JOB else None
    return {"preparation": state, "semantic_search": _status()}


@router.post("/prepare", status_code=202)
def prepare_assets(payload: PrepareRequest, request: Request):
    global _JOB
    _loopback(request)
    if payload.component != "semantic_search":
        raise HTTPException(status_code=400, detail="Only semantic_search can be prepared")
    current = _status()
    if current["available"]:
        return {"job_id": None, "status": "completed", "estimated_download_bytes": 0,
                "semantic_search": current}
    with _LOCK:
        if _JOB and _JOB.get("status") == "running":
            return {"job_id": _JOB["id"], "status": "running",
                    "estimated_download_bytes": DOWNLOAD_BYTES, "semantic_search": current}
        job_id = uuid.uuid4().hex
        _JOB = {"id": job_id, "status": "running", "stage": "queued", "progress": [],
                "estimated_download_bytes": DOWNLOAD_BYTES}

    def update(item: dict) -> None:
        global _JOB
        with _LOCK:
            if _JOB and _JOB.get("id") == job_id:
                _JOB["stage"] = item.get("file") or ("complete" if item.get("ready") else "downloading")
                _JOB["progress"] = (_JOB.get("progress", []) + [dict(item)])[-40:]

    def worker() -> None:
        global _JOB
        try:
            result = prepare_semantic_search(progress=update)
            outcome = {"status": "completed", "result": result}
        except Exception as exc:
            outcome = {"status": "failed", "error": str(exc)}
        with _LOCK:
            if _JOB and _JOB.get("id") == job_id:
                _JOB.update(outcome)

    threading.Thread(target=worker, name="meeting-semantic-assets", daemon=True).start()
    return {"job_id": job_id, "status": "running", "estimated_download_bytes": DOWNLOAD_BYTES,
            "semantic_search": current}
