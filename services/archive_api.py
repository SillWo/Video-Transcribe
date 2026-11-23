"""HTTP access to the unified portable archive."""

from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from services.archive_catalog import (
    _checked_folder,
    _safe_id,
    archive_lock,
    list_items,
    mark_deleted,
    read_record,
    replace_portable_text,
    sync_history,
)

router = APIRouter(prefix="/api/archive", tags=["archive"])


class TextEdit(BaseModel):
    text: str
    baseRevision: int
    expectedArchiveRoot: str | None = None


def _root():
    from services.meeting_settings import load_settings
    import web_api

    return Path(load_settings(web_api.APP_ROOT)["archive_root"])


def _store():
    import web_api

    return web_api.history_store


def _row(record, kind="audio", archive_root=None):
    return {
        "id": record.get("id"),
        "kind": kind,
        "archive_root": str(Path(archive_root).resolve()) if archive_root else None,
        "title": record.get("title")
        or Path(str(record.get("source") or record.get("id") or "Transcription")).name,
        "created_at": record.get("createdAt") or record.get("created_at"),
        "started_at": record.get("startedAt"),
        "status": record.get("status"),
        "source": record.get("source"),
        "job_id": record.get("id"),
        "meeting_id": record.get("meetingId"),
        "speech_availability": record.get("speechAvailability"),
    }


@router.get("")
def list_archive(kind: str = "all", query: str = "", limit: int = 50, offset: int = 0):
    kind = {"meeting": "video", "transcription": "audio"}.get(kind, kind)
    if kind not in {"all", "audio", "video"}:
        raise HTTPException(400, "Invalid kind")
    limit = max(1, min(200, limit))
    offset = max(0, offset)
    root = _root()
    store = _store()
    sync_history(root, store)
    rows = []
    if kind in {"all", "audio"}:
        rows.extend(_row(r, archive_root=root) for r in list_items(root))
    if kind in {"all", "video"}:
        from services.meeting_archive import MeetingArchive

        rows.extend(
            {
                "id": m.get("meeting_id") or m.get("id"),
                "kind": "video",
                "archive_root": str(root.resolve()),
                "title": m.get("title") or m.get("name") or "Meeting",
                "created_at": m.get("created_at"),
                "started_at": m.get("started_at"),
                "status": m.get("status"),
                "meeting_id": m.get("meeting_id") or m.get("id"),
            }
            for m in MeetingArchive(root, read_only=True).list_meetings(limit=10**9)
        )
    q = query.casefold().strip()
    if q:
        rows = [
            r
            for r in rows
            if q in str(r.get("title", "")).casefold()
            or q in str(r.get("source", "")).casefold()
        ]
    rows.sort(
        key=lambda r: (str(r.get("created_at") or ""), str(r["id"])), reverse=True
    )
    page = rows[offset : offset + limit]
    nxt = offset + len(page) if offset + len(page) < len(rows) else None
    return {
        "items": page,
        "total": len(rows),
        "next_offset": nxt,
        "archive_root": str(root.resolve()),
    }


@router.get("/{kind}/{item_id}")
def get_item(kind: str, item_id: str):
    root = _root()
    kind = {"meeting": "video", "transcription": "audio"}.get(kind, kind)
    if kind == "video":
        from services.meeting_archive import MeetingArchive

        try:
            meeting = MeetingArchive(root, read_only=True).get_meeting(item_id)
        except (KeyError, ValueError) as e:
            raise HTTPException(404, "Meeting not found") from e
        if meeting.get("provisional") or meeting.get("archivePublication") == "pending":
            raise HTTPException(404, "Archive item not found")
        normalized = {
            "id": item_id,
            "kind": "video",
            "archive_root": str(root.resolve()),
            "title": meeting.get("title") or meeting.get("name") or "Meeting",
            "created_at": meeting.get("created_at"),
            "started_at": meeting.get("started_at"),
            "status": meeting.get("status"),
            "meeting_id": item_id,
        }
        return {
            "item": normalized,
            "record": meeting,
            "text": None,
            "revision": None,
            "archive_root": str(root.resolve()),
        }
    if kind != "audio":
        raise HTTPException(400, "Invalid kind")
    try:
        _safe_id(item_id)
    except ValueError as e:
        raise HTTPException(400, "Invalid archive item id") from e
    store = _store()
    sync_history(root, store)
    record = store.get(item_id)
    if record and (
        record.get("meetingId")
        or record.get("archivePublication") == "pending"
        or record.get("mediaClassification") == "pending"
        or Path(record.get("archiveRoot", "")).expanduser().resolve() != root
    ):
        record = None
    portable = read_record(root, item_id)
    if record and store.read_text(item_id) is None and portable:
        record = None
    if record and not record.get("meetingId"):
        text = store.read_text(item_id)
        if text is not None:
            item = record
        elif portable:
            item, text = portable["item"], portable["text"]
        else:
            raise HTTPException(404, "Archive item not found")
    elif portable:
        item, text = portable["item"], portable["text"]
    else:
        raise HTTPException(404, "Archive item not found")
    return {
        "item": _row(item, archive_root=root),
        "record": item,
        "text": text,
        "revision": item.get("revision", 0),
        "archive_root": str(root.resolve()),
    }


@router.put("/audio/{item_id}/text")
@router.put("/transcription/{item_id}/text")
def edit_text(item_id: str, payload: TextEdit):
    root = _root()
    resolved_root = root.resolve()
    if payload.expectedArchiveRoot is not None:
        try:
            expected_root = Path(payload.expectedArchiveRoot).expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as e:
            raise HTTPException(409, "Archive root changed; reload before saving") from e
        if expected_root != resolved_root:
            raise HTTPException(409, "Archive root changed; reload before saving")
    store = _store()
    try:
        _safe_id(item_id)
    except ValueError as e:
        raise HTTPException(400, "Invalid archive item id") from e
    sync_history(root, store)
    record = store.get(item_id)
    if record and (
        record.get("meetingId")
        or record.get("archivePublication") == "pending"
        or record.get("mediaClassification") == "pending"
        or Path(record.get("archiveRoot", "")).expanduser().resolve() != root
    ):
        record = None
    portable = read_record(root, item_id)
    learned = []
    try:
        if record and not record.get("meetingId"):
            updated = store.replace_text(item_id, payload.text, payload.baseRevision)
            revision = updated.get("revision", 0)
        elif portable:
            updated, old = replace_portable_text(
                root, item_id, payload.text, payload.baseRevision
            )
            revision = updated["revision"]
            from services.personal_dictionary import remember_corrections

            learned = remember_corrections(
                old, payload.text, {"job_id": item_id, "archive_root": str(root)}
            )
        else:
            raise LookupError(item_id)
    except PermissionError as e:
        raise HTTPException(409, str(e)) from e
    except LookupError as e:
        raise HTTPException(404, "Archive item not found") from e
    item = updated if isinstance(updated, dict) else store.get(item_id)
    if record:
        learned = updated.get("rememberedTerms", [])
    return {
        "item": _row(item, archive_root=root),
        "record": item,
        "text": payload.text,
        "revision": revision,
        "remembered_terms": learned,
        "archive_root": str(root.resolve()),
    }


@router.delete("/audio/{item_id}")
@router.delete("/transcription/{item_id}")
def delete_transcription(item_id: str):
    root = _root()
    store = _store()
    try:
        _safe_id(item_id)
    except ValueError as e:
        raise HTTPException(400, "Invalid archive item id") from e
    sync_history(root, store)
    record = store.get(item_id)
    if record and (
        record.get("meetingId")
        or record.get("archivePublication") == "pending"
        or record.get("mediaClassification") == "pending"
        or Path(record.get("archiveRoot", "")).expanduser().resolve() != root
    ):
        record = None
    try:
        if record:
            store.delete(item_id)
        else:
            import shutil

            with archive_lock(root):
                path = _checked_folder(root, item_id)
                if not path.is_dir():
                    raise HTTPException(404, "Archive item not found")
                mark_deleted(root, item_id)
                shutil.rmtree(path)
    except PermissionError as e:
        raise HTTPException(409, str(e)) from e
    return {"deleted": True, "id": item_id}


@router.get("/audio/{item_id}/download")
@router.get("/transcription/{item_id}/download")
def download_transcription(
    item_id: str, format: str = Query("txt", pattern="^(txt|md)$")
):
    from fastapi.responses import PlainTextResponse

    root = _root()
    store = _store()
    try:
        _safe_id(item_id)
    except ValueError as e:
        raise HTTPException(400, "Invalid archive item id") from e
    sync_history(root, store)
    record = store.get(item_id)
    if record and (
        record.get("meetingId")
        or Path(record.get("archiveRoot", "")).expanduser().resolve() != root
    ):
        record = None
    text = store.read_text(item_id) if record else None
    if text is None:
        found = read_record(root, item_id)
        text = found["text"] if found else None
    if text is None:
        raise HTTPException(404, "Transcript not found")
    return PlainTextResponse(
        text, media_type="text/markdown" if format == "md" else "text/plain"
    )
