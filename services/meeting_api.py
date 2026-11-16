"""REST API for local meeting archive operations; heavy processing uses the shared FIFO."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import re
import os
from difflib import SequenceMatcher
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/meetings", tags=["meetings"])


def _model_values(model: BaseModel) -> dict:
    dump = getattr(model, "model_dump", None)
    return dump(exclude_unset=True) if callable(dump) else model.dict(exclude_unset=True)


class ImportRequest(BaseModel):
    paths: list[str]
    started_at: str | None = None


class ProcessRequest(BaseModel):
    meeting_ids: list[str]
    profile: str | None = None
    device: str = "cpu"
    execution_mode: str | None = None
    cpu_threads: int = 8
    diarization: bool = True
    stages: list[str] = Field(default_factory=lambda: ["speech", "screen", "audio", "index"])


class SegmentEdit(BaseModel):
    text: str
    remember: bool = False


class SpeakerEdit(BaseModel):
    name: str


class GlossaryAdd(BaseModel):
    meeting_id: str
    term: str
    definition: str = ""


class MeetingSettingsUpdate(BaseModel):
    archive_root: str | None = None
    backup_root: str | None = None
    auto_cleanup: bool | None = None
    original_retention_days: int | None = None
    execution_mode: str | None = None
    device: str | None = None
    profile: str | None = None


def _root() -> Path:
    from services.meeting_settings import load_settings
    import web_api
    return Path(load_settings(web_api.APP_ROOT)["archive_root"])


def _archive():
    from services.meeting_archive import MeetingArchive
    return MeetingArchive(_root())


def _safe_meeting(archive, meeting_id: str) -> dict:
    try:
        return archive.get_meeting(meeting_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Meeting not found") from exc


def _short_job_error(value: Any) -> str | None:
    if not value:
        return None
    text = " ".join(str(value).split())
    return text[:500] if text else None


def _history_job_summaries(history_store, meeting_id: str | None = None) -> list[dict[str, Any]]:
    list_summaries = getattr(history_store, "list_meeting_jobs", None)
    if callable(list_summaries):
        return list_summaries(meeting_id)
    records = history_store.list()
    return [record for record in records if meeting_id is None or record.get("meetingId") == meeting_id]


def _job_sort_key(job: dict) -> tuple[bool, datetime, str]:
    value = job.get("createdAt")
    created_at = None
    if value:
        try:
            created_at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            created_at = created_at.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            created_at = None
    return (created_at is not None, created_at or datetime.min.replace(tzinfo=timezone.utc), str(job.get("id", "")))


@router.get("")
def list_meetings():
    archive = _archive()
    current_root = Path(archive.root).expanduser().resolve()

    def belongs_to_current_root(value: Any) -> bool:
        if not value:
            return True
        try:
            return Path(value).expanduser().resolve() == current_root
        except (OSError, RuntimeError, TypeError, ValueError):
            return False

    import web_api
    items = []
    history = [record for record in _history_job_summaries(web_api.history_store)
               if belongs_to_current_root(record.get("meetingRoot"))]
    history_by_id = {record.get("id"): record for record in history if record.get("id")}
    for meeting in archive.list_meetings():
        item = dict(meeting)
        item["stage_state"] = item.get("stage_state", {})
        jobs = [job for job in web_api.jobs.values()
                if job.settings.get("_meetingId") == item.get("meeting_id")
                and belongs_to_current_root(job.settings.get("_meetingRoot"))]
        serialized = []
        for job in jobs:
            record = history_by_id.get(job.id, {})
            serialized.append({
                "id": job.id,
                "status": job.status,
                "stage": job.stage,
                "error": _short_job_error(getattr(job, "error", None) or record.get("error")),
                "errorCode": getattr(job, "errorCode", None) or record.get("errorCode"),
                "createdAt": getattr(job, "created_at", None) or record.get("createdAt"),
                "updatedAt": getattr(job, "updated_at", None) or record.get("updatedAt"),
            })
        known = {entry["id"] for entry in serialized}
        serialized.extend({
            "id": record["id"], "status": record.get("status"), "stage": record.get("stage"),
            "error": _short_job_error(record.get("error")), "errorCode": record.get("errorCode"),
            "createdAt": record.get("createdAt"), "updatedAt": record.get("updatedAt"),
        } for record in history if record.get("meetingId") == item.get("meeting_id")
          and record.get("id") not in known)
        serialized.sort(key=_job_sort_key)
        item["jobs"] = serialized
        item["jobStatus"] = serialized[-1]["status"] if serialized else None
        items.append(item)
    return {"items": items}


@router.post("/import")
def import_meetings(payload: ImportRequest):
    archive = _archive()
    items, rejected = [], []
    for raw_path in payload.paths:
        try:
            path = Path(raw_path).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError("not_a_file")
            items.append(archive.import_file(path, started_at=payload.started_at))
        except (OSError, ValueError) as exc:
            rejected.append({"path": raw_path, "reason": str(exc)})
    return {"items": items, "rejected": rejected}


@router.post("/process", status_code=202)
def process_meetings(payload: ProcessRequest):
    if not payload.meeting_ids:
        raise HTTPException(status_code=400, detail="meeting_ids must not be empty")
    allowed = {"speech", "screen", "audio", "index"}
    if not payload.stages or any(stage not in allowed for stage in payload.stages):
        raise HTTPException(status_code=400, detail="Invalid meeting processing stages")
    archive = _archive()
    from services.meeting_settings import load_settings
    import web_api
    configured = load_settings(web_api.APP_ROOT)
    profile = payload.profile or configured.get("profile", "balanced")
    execution_mode = payload.execution_mode or configured.get("execution_mode", "auto")
    device = payload.device if payload.execution_mode is not None else configured.get("device", payload.device)
    if profile not in {"balanced", "quality", "economy"}:
        raise HTTPException(status_code=400, detail="Invalid meeting ASR profile")
    if execution_mode not in {"auto", "manual"} or device not in {"cpu", "gpu"}:
        raise HTTPException(status_code=400, detail="Invalid execution mode or device")
    sources = []
    for meeting_id in payload.meeting_ids:
        _safe_meeting(archive, meeting_id)
        source = archive.original_path(meeting_id)
        if set(payload.stages) & {"speech", "screen", "audio"} and source is None:
            raise HTTPException(status_code=409, detail=f"Original source unavailable for {meeting_id}")
        sources.append((str(source or ""), "file"))
    diarization_device = "auto" if execution_mode == "manual" and device == "gpu" else "cpu"
    records = web_api._enqueue_simple_jobs(
        sources, profile, device if execution_mode == "manual" else "cpu", payload.cpu_threads, False,
        payload.diarization, diarization_device, meeting_ids=payload.meeting_ids,
        stages=payload.stages, meeting_root=str(_root()), execution_mode=execution_mode,
    )
    return {"jobs": records}


@router.get("/{meeting_id}/timeline")
def meeting_timeline(meeting_id: str, start: float = 0, end: float | None = None,
                     limit: int = 100, offset: int = 0, include_geometry: bool = True):
    archive = _archive()
    _safe_meeting(archive, meeting_id)
    result = archive.timeline(meeting_id, start=start, end=end, limit=max(1, min(500, limit)), offset=max(0, offset))
    if not include_geometry:
        result["items"] = [{key: value for key, value in item.items() if key not in {"regions", "tables"}}
                           for item in result["items"]]
    return result


@router.get("/{meeting_id}/revision")
def meeting_revision(meeting_id: str):
    archive = _archive()
    _safe_meeting(archive, meeting_id)
    folder = archive.meeting_dir(meeting_id)
    digest = hashlib.sha256()
    for name in ("meeting.json", "transcript.json", "speakers.json", "screen-events.json",
                 "glossary.json", "manifest.json"):
        try:
            stat = (folder / name).stat()
            value = f"{name}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        except FileNotFoundError:
            value = f"{name}\0missing\n"
        digest.update(value.encode("utf-8"))
    return {"revision": digest.hexdigest()}


@router.get("/{meeting_id}/frames/{evidence_id}")
def meeting_frame(meeting_id: str, evidence_id: str, crop: str | None = None):
    archive = _archive()
    _safe_meeting(archive, meeting_id)
    events = archive.read_screen_events(meeting_id).get("events", [])
    event = next((item for item in events if isinstance(item, dict)
                  and str(item.get("id")) == evidence_id), None)
    if event is None or not event.get("frame"):
        raise HTTPException(status_code=404, detail="Retained frame not found")
    parsed_crop = None
    if crop:
        try:
            parsed_crop = [int(value) for value in crop.split(",")]
            if len(parsed_crop) != 4 or min(parsed_crop) < 0:
                raise ValueError
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="crop must be x,y,width,height") from exc
    try:
        from services.meeting_images import frame_image
        content = frame_image(archive.meeting_dir(meeting_id), event, crop=parsed_crop)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Retained frame is missing") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=content, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=60"})


@router.get("/{meeting_id:uuid}")
def get_meeting(meeting_id: UUID, include_timeline: bool = True):
    meeting_id = str(meeting_id)
    archive = _archive()
    meeting = _safe_meeting(archive, meeting_id)
    folder = archive.meeting_dir(meeting_id)
    import json
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    timeline = archive.timeline(meeting_id, limit=100, offset=0) if include_timeline else None
    import web_api
    current_root = Path(archive.root).expanduser().resolve()

    def belongs_to_current_root(value: Any) -> bool:
        if not value:
            return True
        try:
            return Path(value).expanduser().resolve() == current_root
        except (OSError, RuntimeError, TypeError, ValueError):
            return False

    history = [{"id": row["id"], "status": row.get("status"), "stage": row.get("stage"),
                "createdAt": row.get("createdAt"), "updatedAt": row.get("updatedAt"),
                "error": row.get("error")}
               for row in _history_job_summaries(web_api.history_store, meeting_id)
               if belongs_to_current_root(row.get("meetingRoot"))]
    result = {"meeting": meeting, "manifest": manifest, "jobs": history}
    if include_timeline:
        result["timeline"] = timeline
    return result


@router.put("/{meeting_id}/segments/{segment_id:path}")
def edit_segment(meeting_id: str, segment_id: str, payload: SegmentEdit):
    archive = _archive()
    _safe_meeting(archive, meeting_id)
    try:
        result = archive.edit_segment(meeting_id, segment_id, payload.text)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Segment not found") from exc
    remembered_id = None
    if payload.remember:
        old_text = str(result.get("original_text", ""))
        new_text = payload.text
        old_tokens = re.findall(r"[^\W_]+(?:[-.][^\W_]+)*", old_text, flags=re.UNICODE)
        new_tokens = re.findall(r"[^\W_]+(?:[-.][^\W_]+)*", new_text, flags=re.UNICODE)
        matcher = SequenceMatcher(a=[t.casefold() for t in old_tokens], b=[t.casefold() for t in new_tokens], autojunk=False)
        candidates = []
        aliases = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag not in {"replace", "insert"}:
                continue
            candidate = " ".join(new_tokens[j1:j2]).strip()
            prior = " ".join(old_tokens[i1:i2]).strip()
            if not candidate or len(candidate) > 64 or len(new_tokens[j1:j2]) > 3:
                continue
            if re.search(r"\d|https?://|www\.|\b(select|insert|update|delete|drop|join|where)\b", candidate, flags=re.IGNORECASE):
                continue
            candidates.append((candidate, prior))
            if prior:
                aliases.append(prior)
        if candidates:
            term, old_term = candidates[0]
            remembered_id = archive.add_glossary_entry(
                meeting_id, term, f"Corrected from: {old_term}" if old_term else "Remembered correction",
                aliases=aliases,
                provenance={"kind": "segment_correction", "segment_id": segment_id,
                            "original_text": old_text, "edited_text": new_text},
            )
    archive.refresh_editorial_exports(meeting_id)
    return {"segment": result, "remembered_term_undo_id": remembered_id}


@router.put("/{meeting_id}/speakers/{speaker_id}")
def edit_speaker(meeting_id: str, speaker_id: str, payload: SpeakerEdit):
    archive = _archive()
    _safe_meeting(archive, meeting_id)
    result = archive.confirm_speaker(meeting_id, speaker_id, payload.name)
    archive.refresh_editorial_exports(meeting_id)
    return result


@router.get("/{meeting_id}/evidence")
def meeting_evidence(meeting_id: str, t: float = 0):
    _safe_meeting(_archive(), meeting_id)
    from services.meeting_links import open_evidence_url
    return {"meeting_id": meeting_id, "time": max(0.0, t), "url": open_evidence_url(_root(), meeting_id, max(0.0, t))}


@router.get("/settings")
def get_meeting_settings():
    from services.meeting_settings import load_settings
    import web_api
    return {"settings": load_settings(web_api.APP_ROOT)}


@router.put("/settings")
def update_meeting_settings(payload: MeetingSettingsUpdate):
    from services.meeting_settings import load_settings, save_settings
    import web_api
    values = _model_values(payload)
    current = load_settings(web_api.APP_ROOT)
    next_root = Path(values.get("archive_root", current["archive_root"])).expanduser().resolve()
    if ("archive_root" in values and os.environ.get("VT_MEETING_ROOT")
            and str(next_root) != str(Path(current["archive_root"]).resolve())):
        raise HTTPException(status_code=409, detail="Archive root is controlled by VT_MEETING_ROOT")
    if str(next_root) != str(Path(current["archive_root"]).resolve()):
        with web_api.jobs_lock:
            busy = any(job.settings.get("_meetingId") and job.status in {"queued", "running", "cancelling"}
                       for job in web_api.jobs.values())
        if not busy:
            try:
                from services.meeting_recorder import recording_session_active
                busy = recording_session_active()
            except Exception:
                busy = False
        if not busy:
            busy = any(record.get("meetingId") and record.get("status") in {"queued", "running", "cancelling"}
                       for record in web_api.history_store.list())
        if busy:
            raise HTTPException(status_code=409, detail="Archive root cannot change while meeting jobs are queued or active")
    try:
        return {"settings": save_settings(values, web_api.APP_ROOT)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/pick-directory")
def pick_meeting_directory():
    from services import native_file_picker
    picker = getattr(native_file_picker, "pick_directory", None)
    if picker is None:
        raise HTTPException(status_code=501, detail="Directory picker is unavailable")
    try:
        return {"path": picker()}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/glossary")
def get_glossary(meeting_id: str | None = None):
    archive = _archive()
    if meeting_id:
        _safe_meeting(archive, meeting_id)
        return {"meeting_id": meeting_id, "items": archive.glossary(meeting_id)}
    items = []
    for meeting in archive.list_meetings(limit=10000):
        mid = meeting["meeting_id"]
        items.extend({"meeting_id": mid, **entry} for entry in archive.glossary(mid))
    return {"items": items}


@router.post("/glossary")
def add_glossary(payload: GlossaryAdd):
    archive = _archive()
    _safe_meeting(archive, payload.meeting_id)
    entry_id = archive.add_glossary_entry(payload.meeting_id, payload.term, payload.definition)
    archive.refresh_editorial_exports(payload.meeting_id)
    return {"id": entry_id, "meeting_id": payload.meeting_id}


@router.delete("/glossary/{meeting_id}/{entry_id}")
def delete_glossary(meeting_id: str, entry_id: str):
    archive = _archive()
    _safe_meeting(archive, meeting_id)
    try:
        undo_id = archive.remove_glossary_entry(meeting_id, entry_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Glossary entry not found") from exc
    archive.refresh_editorial_exports(meeting_id)
    return {"undo_id": undo_id}


@router.post("/glossary/{meeting_id}/{entry_id}/undo")
def undo_glossary(meeting_id: str, entry_id: str):
    archive = _archive()
    _safe_meeting(archive, meeting_id)
    try:
        entry = archive.restore_glossary_entry(meeting_id, entry_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Undo entry not found") from exc
    archive.refresh_editorial_exports(meeting_id)
    return {"item": entry}


@router.post("/reindex")
def reindex_meetings():
    import web_api
    archive = _archive()
    meetings = archive.list_meetings(limit=10**9)
    if not meetings:
        return {"jobs": []}
    records = web_api._enqueue_simple_jobs(
        [("", "file") for _ in meetings], "balanced", "cpu", min(8, web_api.logical_cpu_count()), False,
        diarization_enabled=False, meeting_ids=[row["meeting_id"] for row in meetings],
        stages=["index"], meeting_root=str(_root()), execution_mode="auto",
    )
    return {"jobs": records}


@router.get("/connector/status")
def connector_status():
    from services.meeting_connector import get_status
    return get_status()


@router.post("/connector/connect")
def connector_connect():
    from services.meeting_connector import register
    return register()


@router.post("/connector/check")
def connector_check():
    from services.meeting_connector import MeetingConnector
    return MeetingConnector().verify_protocol()


@router.post("/connector/disconnect")
def connector_disconnect():
    from services.meeting_connector import disconnect
    return disconnect()
