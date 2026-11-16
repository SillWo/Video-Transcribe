"""Loopback-only HTTP routes for local meeting recording.

Public contract: GET ``/api/meetings/recorder`` returns a RecorderStatus;
POST ``/recorder/prepare`` starts the private OBS process only;
POST ``/recorder/configure`` accepts monitor_id/browser_id/microphone_id;
POST ``/recorder/start`` accepts title; POST ``/recorder/stop`` imports and
returns ``{meeting, recorder}``; GET ``/recorder/preview`` returns JPEG bytes.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from services.meeting_recorder import MeetingRecorder, RecorderError, get_default_recorder

router = APIRouter(prefix="/api/meetings/recorder", tags=["meeting-recorder"])


class ConfigureRequest(BaseModel):
    monitor_id: str
    browser_id: str
    microphone_id: str


class StartRequest(BaseModel):
    title: str = "Meeting"


def _local_only(request: Request) -> None:
    host = request.client.host if request.client else ""
    # Starlette's in-process test client uses this synthetic hostname.
    if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="Recorder is available only from this computer")


def _service():
    return get_default_recorder()


def _error(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


@router.get("")
def recorder_status(request: Request):
    _local_only(request)
    try:
        status = _service().status()
        status["download"] = MeetingRecorder.expected_download() if not status.get("available") else None
        return status
    except Exception as exc:
        raise _error(exc) from exc


@router.post("/prepare")
def prepare_recorder(request: Request):
    _local_only(request)
    try:
        return _service().prepare()
    except Exception as exc:
        raise _error(exc) from exc


@router.post("/configure")
def configure_recorder(payload: ConfigureRequest, request: Request):
    _local_only(request)
    try:
        return _service().configure(payload.model_dump() if hasattr(payload, "model_dump") else payload.dict())
    except Exception as exc:
        raise _error(exc) from exc


@router.post("/start")
def start_recorder(payload: StartRequest, request: Request):
    _local_only(request)
    try:
        return _service().start(payload.title)
    except Exception as exc:
        raise _error(exc) from exc


@router.post("/stop")
def stop_recorder(request: Request):
    _local_only(request)
    try:
        return _service().stop()
    except Exception as exc:
        raise _error(exc) from exc


@router.get("/preview")
def recorder_preview(request: Request):
    _local_only(request)
    try:
        return Response(content=_service().preview(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except Exception as exc:
        raise _error(exc) from exc

