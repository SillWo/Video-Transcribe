"""Personal dictionary HTTP API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.personal_dictionary import get_dictionary

router = APIRouter(prefix="/api/dictionary", tags=["dictionary"])


class Entry(BaseModel):
    term: str
    definition: str | None = None


class Settings(BaseModel):
    auto_learn: bool


def _dict():
    return get_dictionary()


@router.get("")
def list_entries(query: str = ""):
    from services.meeting_settings import load_settings
    from services.personal_dictionary import migrate_legacy_meetings
    import web_api

    root = load_settings(web_api.APP_ROOT)["archive_root"]
    migrate_legacy_meetings(root)
    dictionary = _dict()
    return {
        "items": dictionary.list(query),
        "version": dictionary.snapshot()["version"],
        "auto_learn": dictionary.settings().get("auto_learn", True),
    }


@router.post("")
def add_entry(payload: Entry):
    try:
        item = _dict().add(payload.term, payload.definition)
        return {"item": item}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/settings")
def get_settings():
    return _dict().settings()


@router.put("/settings")
def put_settings(payload: Settings):
    return _dict().set_auto_learn(payload.auto_learn)


@router.put("/{entry_id}")
def update_entry(entry_id: str, payload: Entry):
    try:
        item = _dict().update(entry_id, payload.term, payload.definition)
        return {"item": item}
    except KeyError as exc:
        raise HTTPException(404, "Dictionary entry not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/{entry_id}")
def delete_entry(entry_id: str):
    try:
        return {"undo_id": _dict().delete(entry_id)}
    except KeyError as exc:
        raise HTTPException(404, "Dictionary entry not found") from exc


@router.post("/undo/{undo_id}")
def undo_entry(undo_id: str):
    try:
        return {"item": _dict().undo(undo_id)}
    except KeyError as exc:
        raise HTTPException(404, "Undo entry not found") from exc
