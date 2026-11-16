"""Portable, file based archive for recorded meetings.

Public API: ``MeetingArchive(root, read_only=False)``, ``import_file``,
``list_meetings``, ``get_meeting``, ``meeting_dir``, ``original_path``,
``write_transcript``, ``edit_segment``, ``confirm_speaker``,
``write_screen_events``, ``timeline``, glossary methods, ``update_meeting``
and ``finalize``. Meeting content lives in one folder and remains readable
without an index database.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


_NAMESPACE = uuid.UUID("f2e7a39c-97be-4c5b-a8c3-6a11fe7d0637")
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_read(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


@contextlib.contextmanager
def _archive_lock(root: Path) -> Iterator[None]:
    key = str(root.resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    with lock:
        root.mkdir(parents=True, exist_ok=True)
        lock_path = root / ".archive.lock"
        with lock_path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class MeetingArchive:
    """A local archive whose meeting folders are the authoritative storage."""

    def __init__(self, root: str | Path, read_only: bool = False):
        self.root = Path(root).expanduser()
        self.read_only = read_only
        if not read_only:
            self.root.mkdir(parents=True, exist_ok=True)

    def _require_write(self) -> None:
        if self.read_only:
            raise PermissionError("MeetingArchive is read-only")

    @staticmethod
    def _valid_id(meeting_id: str) -> str:
        try:
            parsed = uuid.UUID(str(meeting_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid meeting_id") from exc
        canonical = str(parsed)
        if canonical != str(meeting_id).lower():
            raise ValueError("Invalid meeting_id")
        return canonical

    def meeting_dir(self, meeting_id: str) -> Path:
        candidate = self.root / "meetings" / self._valid_id(meeting_id)
        archive_root = self.root.resolve()
        meeting_root = (self.root / "meetings").resolve()
        resolved = candidate.resolve()
        if not resolved.is_relative_to(archive_root) or not resolved.is_relative_to(meeting_root):
            raise ValueError("Meeting directory resolves outside the archive")
        return resolved

    def original_path(self, meeting_id: str) -> Path | None:
        meeting = self.get_meeting(meeting_id)
        rel = meeting.get("original_rel")
        if not rel:
            return None
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root.resolve()):
            return None
        return path if path.is_file() else None

    def _meeting_meta_path(self, meeting_id: str) -> Path:
        return self.meeting_dir(meeting_id) / "meeting.json"

    def get_meeting(self, meeting_id: str) -> dict:
        meeting_id = self._valid_id(meeting_id)
        data = _json_read(self._meeting_meta_path(meeting_id), None)
        if not isinstance(data, dict):
            raise KeyError(meeting_id)
        return data

    def import_file(self, path: str | Path, title: str | None = None, started_at: str | None = None) -> dict:
        self._require_write()
        source = Path(path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("Source must be a file")
        digest = hashlib.sha256()
        with source.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
        with _archive_lock(self.root):
            for entry in (self.root / "meetings").glob("*/meeting.json"):
                meta = _json_read(entry, {})
                if meta.get("sha256") == sha:
                    return meta
            meeting_id = str(uuid.uuid5(_NAMESPACE, sha))
            folder = self.meeting_dir(meeting_id)
            folder.mkdir(parents=True, exist_ok=True)
            original_name = source.name or "original"
            target = self.root / "originals" / meeting_id / original_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            meta = {
                "meeting_id": meeting_id,
                "title": title or source.stem,
                "started_at": started_at,
                "created_at": _now(),
                "source_modified_at": datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                "source_filename": original_name,
                "source_available": True,
                "original_rel": target.relative_to(self.root).as_posix(),
                "sha256": sha,
                "status": "imported",
                "job_id": None,
            }
            _json_write(folder / "meeting.json", meta)
            _json_write(folder / "transcript.json", {"raw_segments": [], "raw_text": None, "overrides": {}})
            _json_write(folder / "speakers.json", {"confirmed": {}})
            _json_write(folder / "screen-events.json", {"events": [], "status": "pending", "limitations": []})
            _json_write(folder / "glossary.json", {"entries": []})
            return meta

    def list_meetings(self, query: str = "", limit: int = 50, offset: int = 0) -> list[dict]:
        query_fold = (query or "").casefold()
        rows = []
        meetings = self.root / "meetings"
        if meetings.is_dir():
            for entry in meetings.glob("*/meeting.json"):
                item = _json_read(entry, None)
                if isinstance(item, dict) and (not query_fold or query_fold in json.dumps(item, ensure_ascii=False).casefold()):
                    rows.append(item)
        rows.sort(key=lambda m: (m.get("started_at") or m.get("created_at") or "", m.get("meeting_id", "")), reverse=True)
        return rows[max(0, offset):max(0, offset) + max(0, limit)]

    def _read_transcript(self, meeting_id: str) -> dict:
        return _json_read(self.meeting_dir(meeting_id) / "transcript.json", {"raw_segments": [], "raw_text": None, "overrides": {}})

    def read_screen_events(self, meeting_id: str) -> dict:
        """Read retained screen events and their status without loading timeline data."""
        folder = self.meeting_dir(meeting_id)
        self.get_meeting(meeting_id)
        data = _json_read(folder / "screen-events.json", {"events": [], "status": "pending", "limitations": []})
        if not isinstance(data, dict):
            data = {"events": [], "status": "pending", "limitations": []}
        result = dict(data)
        if not isinstance(result.get("events"), list):
            result["events"] = []
        result.setdefault("status", "pending")
        if not isinstance(result.get("limitations"), list):
            result["limitations"] = []
        return result

    @staticmethod
    def _effective_segments(data: dict) -> list[dict]:
        overrides = data.get("overrides", {})
        out = []
        for raw in data.get("raw_segments", []):
            seg = dict(raw)
            override = overrides.get(str(seg.get("id")))
            if isinstance(override, dict):
                same_source = (override.get("original_start") == seg.get("start")
                               and override.get("original_end") == seg.get("end"))
                if same_source:
                    seg["text"] = override.get("text", seg.get("text", ""))
                    seg["edited"] = True
            elif override is not None:  # Older archives stored just the corrected string.
                seg["text"] = override
                seg["edited"] = True
            out.append(seg)
        return out

    @staticmethod
    def _override_matches(raw: dict, override: Any) -> bool:
        if not isinstance(override, dict):
            return True
        return (override.get("original_start") == raw.get("start")
                and override.get("original_end") == raw.get("end"))

    def write_source_artifacts(self, meeting_id: str, *, raw_asr: dict | None = None,
                               speaker_turns: list[dict] | None = None) -> None:
        """Persist exact pipeline source output separately from editable turns."""
        self._require_write()
        with _archive_lock(self.root):
            self.get_meeting(meeting_id)
            folder = self.meeting_dir(meeting_id)
            if raw_asr is not None:
                _json_write(folder / "raw-asr.json", raw_asr)
            if speaker_turns is not None:
                _json_write(folder / "speaker-turns.json", {"turns": speaker_turns})

    def write_transcript(self, meeting_id: str, segments: list[dict], raw_text: str | None = None) -> dict:
        self._require_write()
        meeting_id = self._valid_id(meeting_id)
        with _archive_lock(self.root):
            self.get_meeting(meeting_id)
            data = self._read_transcript(meeting_id)
            data["raw_segments"] = [dict(s) for s in segments]
            data["raw_text"] = raw_text
            _json_write(self.meeting_dir(meeting_id) / "transcript.json", data)
            return {"raw_segments": data["raw_segments"], "segments": self._effective_segments(data), "raw_text": raw_text}

    def edit_segment(self, meeting_id: str, segment_id: str, text: str) -> dict:
        self._require_write()
        with _archive_lock(self.root):
            self.get_meeting(meeting_id)
            data = self._read_transcript(meeting_id)
            raw = next((s for s in data.get("raw_segments", []) if str(s.get("id")) == str(segment_id)), None)
            if raw is None:
                raise KeyError(segment_id)
            data.setdefault("overrides", {})[str(segment_id)] = {
                "text": text,
                "original_start": raw.get("start"),
                "original_end": raw.get("end"),
                "original_text": raw.get("text"),
            }
            _json_write(self.meeting_dir(meeting_id) / "transcript.json", data)
            updated = dict(raw)
            updated["original_text"] = raw.get("text", "")
            updated["text"] = text
            return updated

    def confirm_speaker(self, meeting_id: str, speaker_id: str, name: str) -> dict:
        self._require_write()
        with _archive_lock(self.root):
            self.get_meeting(meeting_id)
            path = self.meeting_dir(meeting_id) / "speakers.json"
            data = _json_read(path, {"confirmed": {}})
            data.setdefault("confirmed", {})[str(speaker_id)] = {"speaker_id": str(speaker_id), "name": name, "confirmed_at": _now()}
            _json_write(path, data)
            return data["confirmed"][str(speaker_id)]

    def write_screen_events(self, meeting_id: str, events: list[dict], status: str = "ready", limitations: list | None = None) -> dict:
        self._require_write()
        with _archive_lock(self.root):
            self.get_meeting(meeting_id)
            data = {"events": [dict(e) for e in events], "status": status, "limitations": list(limitations or [])}
            _json_write(self.meeting_dir(meeting_id) / "screen-events.json", data)
            return data

    def timeline(self, meeting_id: str, start: float = 0, end: float | None = None, limit: int = 100, offset: int = 0) -> dict:
        folder = self.meeting_dir(meeting_id)
        self.get_meeting(meeting_id)
        transcript = self._read_transcript(meeting_id)
        screen = _json_read(folder / "screen-events.json", {"events": [], "status": "pending", "limitations": []})
        confirmed = _json_read(folder / "speakers.json", {"confirmed": {}}).get("confirmed", {})
        items = []
        for seg in self._effective_segments(transcript):
            speaker_cluster = str(seg.get("speaker")) if seg.get("speaker") is not None else None
            item = {**seg, "kind": "transcript", "evidence_id": str(seg.get("id", "")), "original_text": next((s.get("text") for s in transcript.get("raw_segments", []) if str(s.get("id")) == str(seg.get("id"))), seg.get("text", "")), "speaker_cluster": speaker_cluster, "confirmed_speaker": confirmed.get(speaker_cluster, {}).get("name") if speaker_cluster else None}
            items.append(item)
        present_ids = {str(s.get("id")) for s in transcript.get("raw_segments", [])}
        raw_by_id = {str(s.get("id")): s for s in transcript.get("raw_segments", [])}
        for orphan_id, override in transcript.get("overrides", {}).items():
            if str(orphan_id) not in present_ids or not self._override_matches(raw_by_id[str(orphan_id)], override):
                orphan_text = override.get("text") if isinstance(override, dict) else override
                original = override.get("original_text") if isinstance(override, dict) else None
                items.append({"id": str(orphan_id), "kind": "orphan_edit", "evidence_id": str(orphan_id), "start": override.get("original_start") if isinstance(override, dict) else None, "end": override.get("original_end") if isinstance(override, dict) else None, "text": orphan_text, "original_text": original, "reason": "segment_missing_or_changed_after_rerun"})
        for event in screen.get("events", []):
            items.append({**event, "kind": "screen", "evidence_id": str(event.get("id", "")), "original_text": event.get("text", "")})
        def time_key(it):
            try: return float(it.get("start", 0))
            except (ValueError, TypeError): return 0.0
        def overlaps(item):
            try:
                item_start = float(item.get("start", 0) or 0)
                item_end = float(item.get("end", item_start) or item_start)
            except (ValueError, TypeError):
                return False
            return item_end >= start and (end is None or item_start <= end)
        items = [i for i in items if overlaps(i)]
        items.sort(key=lambda i: (time_key(i), i["kind"], i["evidence_id"]))
        total = len(items)
        page = items[max(0, offset):max(0, offset) + max(0, limit)]
        next_offset = offset + len(page)
        return {"items": page, "total": total, "next_offset": next_offset if next_offset < total else None, "complete": next_offset >= total, "screen_ready": screen.get("status") == "ready", "screen_status": screen.get("status", "pending"), "limitations": screen.get("limitations", [])}

    def glossary(self, meeting_id: str) -> list[dict]:
        return _json_read(self.meeting_dir(meeting_id) / "glossary.json", {"entries": []}).get("entries", [])

    def add_glossary_entry(self, meeting_id: str, term: str, definition: str = "", entry_id: str | None = None,
                           aliases: list[str] | None = None, provenance: dict | None = None) -> str:
        self._require_write()
        with _archive_lock(self.root):
            self.get_meeting(meeting_id)
            data = _json_read(self.meeting_dir(meeting_id) / "glossary.json", {"entries": []})
            entry_id = entry_id or str(uuid.uuid4())
            data["entries"] = [e for e in data.get("entries", []) if e.get("id") != entry_id]
            entry = {"id": entry_id, "term": term, "definition": definition}
            if aliases:
                entry["aliases"] = list(dict.fromkeys(aliases))
            if provenance:
                entry["provenance"] = dict(provenance)
            data["entries"].append(entry)
            _json_write(self.meeting_dir(meeting_id) / "glossary.json", data)
            return entry_id

    def remove_glossary_entry(self, meeting_id: str, entry_id: str) -> str:
        self._require_write()
        with _archive_lock(self.root):
            self.get_meeting(meeting_id)
            path = self.meeting_dir(meeting_id) / "glossary.json"
            data = _json_read(path, {"entries": []})
            found = any(e.get("id") == entry_id for e in data.get("entries", []))
            if not found:
                raise KeyError(entry_id)
            removed = next(e for e in data["entries"] if e.get("id") == entry_id)
            data["entries"] = [e for e in data["entries"] if e.get("id") != entry_id]
            data.setdefault("removed", {})[entry_id] = removed
            _json_write(path, data)
            return entry_id

    def restore_glossary_entry(self, meeting_id: str, entry_id: str) -> dict:
        self._require_write()
        with _archive_lock(self.root):
            self.get_meeting(meeting_id)
            path = self.meeting_dir(meeting_id) / "glossary.json"
            data = _json_read(path, {"entries": [], "removed": {}})
            removed = data.get("removed", {}).pop(entry_id, None)
            if removed is None:
                raise KeyError(entry_id)
            data["entries"] = [e for e in data.get("entries", []) if e.get("id") != entry_id]
            data["entries"].append(removed)
            _json_write(path, data)
            return removed

    def update_meeting(self, meeting_id: str, **fields) -> dict:
        self._require_write()
        with _archive_lock(self.root):
            meta = self.get_meeting(meeting_id)
            meta.update(fields)
            meta["meeting_id"] = self._valid_id(meeting_id)
            _json_write(self._meeting_meta_path(meeting_id), meta)
            return meta

    def _write_editorial_exports(self, meeting_id: str):
        """Refresh transcript and glossary exports from authoritative JSON."""
        meta = self.get_meeting(meeting_id)
        folder = self.meeting_dir(meeting_id)
        transcript = self._read_transcript(meeting_id)
        speakers = _json_read(folder / "speakers.json", {"confirmed": {}})
        glossary_data = _json_read(folder / "glossary.json", {"entries": []})
        effective = self._effective_segments(transcript)
        raw_by_id = {str(s.get("id")): s for s in transcript.get("raw_segments", [])}
        orphan_overrides = {key: value for key, value in transcript.get("overrides", {}).items()
                            if key not in raw_by_id or not self._override_matches(raw_by_id[key], value)}
        exported_transcript = {**transcript, "segments": effective, "orphan_overrides": orphan_overrides}
        if transcript != exported_transcript:
            _json_write(folder / "transcript.json", exported_transcript)
        (folder / "transcript.md").write_text(
            "\n".join(
                f"[{s.get('start', 0)}] "
                f"{speakers.get('confirmed', {}).get(str(s.get('speaker')), {}).get('name') or ('Unresolved speaker' if s.get('speaker_ambiguous') else s.get('speaker', ''))}: "
                f"{s.get('text', '')}".strip()
                for s in effective) + "\n", encoding="utf-8")
        glossary = glossary_data.get("entries", [])
        (folder / "glossary.md").write_text(
            "\n".join(f"- **{e.get('term', '')}**: {e.get('definition', '')}"
                      for e in glossary) + "\n", encoding="utf-8")
        (folder / "meeting.md").write_text(
            f"# {meta.get('title') or 'Meeting'}\n\n- ID: `{meeting_id}`\n"
            f"- Started: {meta.get('started_at') or 'Unknown'}\n"
            f"- Source: {meta.get('source_filename') or 'Unavailable'}\n", encoding="utf-8")
        return meta, transcript, speakers, glossary_data, effective

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def refresh_editorial_exports(self, meeting_id: str) -> dict:
        """Refresh editorial markdown and manifest without rehashing media."""
        self._require_write()
        meeting_id = self._valid_id(meeting_id)
        with _archive_lock(self.root):
            meta, transcript, _speakers, _glossary_data, effective = self._write_editorial_exports(meeting_id)
            folder = self.meeting_dir(meeting_id)
            previous = _json_read(folder / "manifest.json", None)
            has_previous = isinstance(previous, dict)
            previous = previous if has_previous else {}
            files = dict(previous.get("files") or {}) if isinstance(previous.get("files"), dict) else {}

            # These are the only local exports whose content this operation updates.
            small_files = ("meeting.md", "transcript.md", "transcript.json", "speakers.json",
                           "glossary.json", "glossary.md", "meeting.json")
            for name in small_files:
                path = folder / name
                if path.is_file():
                    files[name] = self._sha256_file(path)

            screen_path = folder / "screen-events.json"
            prior_screen_hash = files.get("screen-events.json")
            current_screen_hash = self._sha256_file(screen_path) if screen_path.is_file() else None
            screen_changed = has_previous and current_screen_hash != prior_screen_hash
            if current_screen_hash is None:
                files.pop("screen-events.json", None)
            else:
                files["screen-events.json"] = current_screen_hash

            prior_readiness = previous.get("readiness") if isinstance(previous.get("readiness"), dict) else {}
            readiness = {key: prior_readiness.get(key, "pending")
                         for key in ("source", "transcript", "screen", "audio")}
            active_stages = {
                stage for stage, state in (meta.get("stage_state") or {}).items()
                if isinstance(state, dict) and state.get("status") == "active"
            }
            media_active = bool(active_stages & {"screen", "audio"})
            any_active = media_active or bool(active_stages) or str(meta.get("status", "")).endswith("_active")
            if "speech" in active_stages or str(meta.get("status", "")).startswith("speech_") and str(meta.get("status", "")).endswith("_active"):
                readiness["transcript"] = "pending"
            if "screen" in active_stages or screen_changed:
                readiness["screen"] = "pending"
            if "audio" in active_stages:
                readiness["audio"] = "pending"

            source_path = self.original_path(meeting_id)
            source_availability = dict(previous.get("source_availability") or {})
            source_availability.update({
                "source": bool(source_path),
                "transcript": bool(effective) or transcript.get("raw_text") is not None,
            })
            status_previous = previous.get("status") if isinstance(previous.get("status"), dict) else {}
            overall = ("ready" if has_previous and status_previous.get("overall") == "ready"
                       and readiness.get("transcript") == "ready"
                       and readiness.get("screen") == "ready"
                       and not any_active and not screen_changed else "partial")
            status = {
                "speech": readiness["transcript"], "screen": readiness["screen"],
                "audio": readiness["audio"],
                "index": "ready" if (self.root / "meeting-index.sqlite3").is_file() else "not_built",
                "overall": overall, "backup": "unverified",
            }
            manifest = {
                "meeting_id": meeting_id, "generated_at": _now(), "files": files,
                "readiness": readiness, "status": status,
                "source_availability": source_availability,
                "limitations": previous.get("limitations", []),
            }
            _json_write(folder / "manifest.json", manifest)
            return manifest

    def finalize(self, meeting_id: str) -> dict:
        self._require_write()
        with _archive_lock(self.root):
            meta, transcript, speakers, glossary_data, effective = self._write_editorial_exports(meeting_id)
            folder = self.meeting_dir(meeting_id)
            screen = _json_read(folder / "screen-events.json", {"events": [], "status": "pending", "limitations": []})
            timeline_rows = sorted(screen.get("events", []), key=lambda e: (float(e.get("start", 0) or 0), str(e.get("id", ""))))
            (folder / "screen-timeline.md").write_text("\n".join(f"[{e.get('start', 0)}] {e.get('text', '')}" for e in timeline_rows) + "\n", encoding="utf-8")
            transcript_present = bool(effective) or transcript.get("raw_text") is not None
            speech_ready = meta.get("speech_status") in {"ready", "completed"} or bool(effective)
            availability = {"source": bool(self.original_path(meeting_id)), "transcript": transcript_present, "screen": screen.get("status") == "ready", "audio": any((folder / "audio").glob("*.opus")) if (folder / "audio").is_dir() else False}
            readiness = {"source": "ready" if availability["source"] else "unavailable", "transcript": "ready" if speech_ready else "pending", "screen": screen.get("status", "pending"), "audio": "ready" if availability["audio"] else "unavailable"}
            generated = ["meeting.md", "transcript.md", "screen-timeline.md", "transcript.json", "raw-asr.json", "speaker-turns.json", "speakers.json", "glossary.json", "glossary.md", "screen-events.json", "meeting.json"]
            hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in generated if (folder / name).is_file()}
            audio_dir = folder / "audio"
            if audio_dir.is_dir():
                for audio_file in audio_dir.glob("*.opus"):
                    hashes[audio_file.relative_to(folder).as_posix()] = hashlib.sha256(audio_file.read_bytes()).hexdigest()
            frames_dir = folder / "frames"
            if frames_dir.is_dir():
                for frame_file in frames_dir.rglob("*"):
                    if frame_file.is_file():
                        hashes[frame_file.relative_to(folder).as_posix()] = hashlib.sha256(frame_file.read_bytes()).hexdigest()
            overall = "ready" if readiness["transcript"] == "ready" and readiness["screen"] == "ready" else "partial"
            status = {"speech": readiness["transcript"], "screen": readiness["screen"], "audio": readiness["audio"], "index": "ready" if (self.root / "meeting-index.sqlite3").is_file() else "not_built", "overall": overall, "backup": "unverified"}
            manifest = {"meeting_id": meeting_id, "generated_at": _now(), "files": hashes, "readiness": readiness, "status": status, "source_availability": availability, "limitations": screen.get("limitations", [])}
            _json_write(folder / "manifest.json", manifest)
            return manifest
