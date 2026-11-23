"""Portable catalog for ordinary transcription records."""

from __future__ import annotations
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

_INDEX_CONDITION = threading.Condition()
_INDEX_PENDING = {}
_INDEX_WORKER = None
_INDEX_ACTIVE = 0
_INDEX_STOP = False
_INDEX_SHUTDOWN = False
_LOGGER = logging.getLogger(__name__)


@contextmanager
def archive_index_publish_guard():
    """Serialize a short index publication against application shutdown."""
    with _INDEX_CONDITION:
        yield not _INDEX_SHUTDOWN


@contextmanager
def archive_lock(root):
    from services.meeting_archive import _archive_lock

    with _archive_lock(Path(root).expanduser().resolve()):
        yield


def _safe_id(value):
    value = str(value or "")
    if not value or not re.fullmatch(r"[0-9a-fA-F-]{1,80}", value):
        raise ValueError("Invalid job id")
    return value


def folder(root, job_id):
    return Path(root).expanduser().resolve() / "transcriptions" / _safe_id(job_id)


def _checked_folder(root, job_id, create=False):
    root = Path(root).expanduser().resolve()
    base = root / "transcriptions"
    path = base / _safe_id(job_id)
    if create:
        base.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(base.resolve(strict=False))
    except ValueError as exc:
        raise ValueError("Archive path escapes transcriptions directory") from exc
    if path.is_symlink() or base.is_symlink():
        raise ValueError("Archive path contains a symbolic link")
    return resolved


def _deleted_path(root):
    return Path(root).expanduser().resolve() / "transcriptions" / "deleted.json"


def _deleted_ids(root):
    try:
        data = json.loads(_deleted_path(root).read_text(encoding="utf-8"))
        return set(data if isinstance(data, list) else [])
    except (OSError, ValueError):
        return set()


def mark_deleted(root, job_id):
    job_id = _safe_id(job_id)
    path = _deleted_path(root)
    values = _deleted_ids(root)
    if job_id not in values:
        values.add(job_id)
        _atomic(path, sorted(values))


def _atomic(path: Path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    if path.is_file() and path.read_text(encoding="utf-8") == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def catalog_summary(record):
    return {
        key: record.get(key)
        for key in (
            "id",
            "title",
            "createdAt",
            "updatedAt",
            "startedAt",
            "status",
            "revision",
            "archiveRoot",
            "mediaClassification",
            "archivePublication",
            "archiveKind",
            "archiveItemId",
            "speechAvailability",
            "source",
            "meetingId",
        )
    }


def read_catalog_summary(root, job_id):
    try:
        directory = _checked_folder(root, job_id)
        data = json.loads((directory / "catalog.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and not _is_provisional(data) else None
    except (OSError, ValueError):
        return None


def write_record(root, record, text=None):
    if record.get("meetingId") or _is_provisional(record):
        return None
    with archive_lock(root):
        return _write_record_unlocked(root, record, text)


def _write_record_unlocked(root, record, text=None):
    if record.get("meetingId") or _is_provisional(record):
        return None
    job_id = _safe_id(record.get("id"))
    if job_id in _deleted_ids(root):
        return None
    directory = _checked_folder(root, job_id, create=True)
    directory.mkdir(parents=True, exist_ok=True)
    portable = dict(record)
    _atomic(directory / "item.json", portable)
    _atomic(directory / "catalog.json", catalog_summary(portable))
    if text is not None:
        for ext in ("txt", "md"):
            target = directory / f"transcript.{ext}"
            tmp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            if target.is_file() and target.read_text(encoding="utf-8") == text:
                continue
            try:
                with tmp.open("w", encoding="utf-8", newline="") as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)
    return portable


def replace_portable_text(root, job_id, text, base_revision):
    with archive_lock(root):
        found = read_record(root, job_id)
        if not found:
            raise LookupError(job_id)
        item = found["item"]
        revision = int(item.get("revision", 0))
        if revision != base_revision:
            raise PermissionError(f"Conflict. Current revision: {revision}.")
        if item.get("status") != "completed":
            raise PermissionError("Only completed transcriptions can be edited.")
        updated = {**item, "revision": revision + 1}
        _write_record_unlocked(root, updated, text)
        previous_text = found["text"] or ""
    _index_completed_transcription(root, updated, text)
    return updated, previous_text


def read_record(root, job_id):
    if _safe_id(job_id) in _deleted_ids(root):
        return None
    directory = _checked_folder(root, job_id)
    path = directory / "item.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("meetingId") or _is_provisional(value):
        return None
    textpath = directory / "transcript.txt"
    text = textpath.read_text(encoding="utf-8") if textpath.is_file() else None
    return {"item": value, "text": text, "directory": directory}


def read_text(root, job_id):
    if _safe_id(job_id) in _deleted_ids(root):
        return None
    directory = _checked_folder(root, job_id)
    try:
        item = json.loads((directory / "item.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(item, dict) or _is_provisional(item):
        return None
    path = directory / "transcript.txt"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def sync_history(root, history_store):
    root = Path(root).expanduser().resolve()
    result = {"synced": 0, "skipped_meeting": 0, "skipped_without_text": 0}
    meeting_summaries = getattr(history_store, "list_meeting_jobs", None)
    if callable(meeting_summaries):
        result["skipped_meeting"] = len(meeting_summaries())
    summaries_method = getattr(history_store, "list_ordinary_summaries", None)
    if callable(summaries_method):
        summaries = summaries_method()
    else:
        summaries = [catalog_summary(record) for record in history_store.list()]
    for summary in summaries:
        if summary.get("meetingId"):
            result["skipped_meeting"] += 1
            continue
        if _is_provisional(summary):
            _mark_existing_pending(root, summary.get("id"), summary)
            continue
        if summary.get("id") in _deleted_ids(root):
            continue
        bound = summary.get("archiveRoot")
        claimed = False
        if not bound:
            claim_archive_root = getattr(history_store, "claim_archive_root", None)
            if not callable(claim_archive_root):
                raise TypeError("history store does not support archive root claims")
            record = claim_archive_root(summary["id"], str(root))
            if record is None:
                continue
            summary = catalog_summary(record)
            bound = str(root)
            claimed = True
        if Path(bound).expanduser().resolve() != root:
            continue
        if read_catalog_summary(root, summary["id"]) == catalog_summary(summary):
            if claimed:
                result["synced"] += 1
            continue
        with archive_lock(root):
            record = history_store.get(summary["id"])
            if not record:
                continue
            text = history_store.read_text(summary["id"])
            _write_record_unlocked(root, record, text)
        result["synced"] += 1
    return result


def _mark_existing_pending(root, job_id, summary):
    """Keep an older published copy on disk while hiding its pending job."""
    if not job_id:
        return
    try:
        directory = _checked_folder(root, job_id)
        item_path = directory / "item.json"
        if not item_path.is_file():
            return
        item = json.loads(item_path.read_text(encoding="utf-8"))
        if not isinstance(item, dict):
            return
        item.update({
            key: summary.get(key)
            for key in ("archivePublication", "archiveKind", "archiveItemId", "speechAvailability", "mediaClassification")
            if key in summary
        })
        item["archivePublication"] = "pending"
        _atomic(item_path, item)
        _atomic(directory / "catalog.json", catalog_summary(item))
    except (OSError, ValueError):
        return


def update_from_history(
    history_store,
    job_id,
    record=None,
    text=None,
    root=None,
    index_transcription=True,
):
    record = record or history_store.get(job_id)
    if not record or record.get("meetingId"):
        return
    archive_root = record.get("archiveRoot") or root
    if not archive_root:
        return
    if _is_provisional(record):
        _mark_existing_pending(archive_root, job_id, record)
        return
    with archive_lock(archive_root):
        current = history_store.get(job_id) or record
        if text is None:
            text = history_store.read_text(job_id)
        _write_record_unlocked(archive_root, current, text)
    if index_transcription:
        _index_completed_transcription(archive_root, current, text)


def _index_completed_transcription(root, record, text):
    if not record or record.get("meetingId") or _is_provisional(record) or record.get("status") != "completed":
        return
    if text is None:
        return
    enqueue_transcription_index(root, record, text)


def remove_pending_ordinary_transcription(root, job_id):
    """Remove only an active ordinary portable stub when a job becomes a meeting."""
    with archive_lock(root):
        found = read_record(root, job_id)
        if not found:
            return False
        item = found["item"]
        if item.get("meetingId") or item.get("status") not in {"queued", "running", "cancelling"}:
            return False
        folder = found["directory"]
        shutil.rmtree(folder)
        return True


def enqueue_archive_transcription_reindex(root):
    count = 0
    for record in list_items(root):
        text = read_text(root, str(record.get("id") or ""))
        if text is None:
            continue
        enqueue_transcription_index(root, record, text)
        count += 1
    return count


def enqueue_transcription_index(root, record, text):
    global _INDEX_WORKER, _INDEX_STOP
    key = (str(Path(root).expanduser().resolve()), str(record.get("id")))
    with _INDEX_CONDITION:
        if _INDEX_SHUTDOWN:
            return
        _INDEX_PENDING[key] = (key[0], dict(record), text)
        if _INDEX_WORKER is None or not _INDEX_WORKER.is_alive():
            _INDEX_STOP = False
            _INDEX_WORKER = threading.Thread(
                target=_transcription_index_worker,
                name="portable-transcription-indexer",
                daemon=True,
            )
            _INDEX_WORKER.start()
        _INDEX_CONDITION.notify()


def _transcription_index_worker():
    global _INDEX_WORKER, _INDEX_ACTIVE
    while True:
        with _INDEX_CONDITION:
            while not _INDEX_PENDING and not _INDEX_STOP:
                _INDEX_CONDITION.wait()
            if _INDEX_STOP and not _INDEX_PENDING:
                _INDEX_WORKER = None
                _INDEX_CONDITION.notify_all()
                return
            root, record, text = _INDEX_PENDING.pop(next(iter(_INDEX_PENDING)))
            _INDEX_ACTIVE += 1
        try:
            from services.meeting_index import MeetingIndex

            result = MeetingIndex(root).index_transcription(
                record,
                text,
                cancellation_check=_archive_index_is_shutdown,
                publish_guard=archive_index_publish_guard,
            )
            status = result.get("status")
            if status not in {
                "ready",
                "text_indexed_embeddings_unavailable",
                "cancelled",
            }:
                _LOGGER.warning(
                    "Portable transcript index update for %s returned %s",
                    record.get("id"),
                    status,
                )
        except Exception:
            _LOGGER.exception(
                "Portable transcript index update failed for %s", record.get("id")
            )
        finally:
            with _INDEX_CONDITION:
                _INDEX_ACTIVE -= 1
                _INDEX_CONDITION.notify_all()


def _archive_index_is_shutdown():
    with _INDEX_CONDITION:
        return _INDEX_SHUTDOWN


def wait_archive_indexing_idle(timeout=10):
    deadline = time.monotonic() + max(0.0, float(timeout))
    with _INDEX_CONDITION:
        while _INDEX_PENDING or _INDEX_ACTIVE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _INDEX_CONDITION.wait(remaining)
    return True


def reset_archive_indexing_for_tests():
    global _INDEX_SHUTDOWN, _INDEX_STOP
    with _INDEX_CONDITION:
        _INDEX_SHUTDOWN = False
        _INDEX_STOP = False
        _INDEX_CONDITION.notify_all()


def shutdown_archive_indexing(timeout=2):
    global _INDEX_STOP, _INDEX_SHUTDOWN
    with _INDEX_CONDITION:
        _INDEX_SHUTDOWN = True
        _INDEX_STOP = True
        _INDEX_PENDING.clear()
        _INDEX_CONDITION.notify_all()
        worker = _INDEX_WORKER
    if worker is not None and worker is not threading.current_thread():
        worker.join(timeout=max(0.0, float(timeout)))


def list_items(root):
    base = Path(root).expanduser().resolve() / "transcriptions"
    rows = []
    if not base.is_dir():
        return rows
    for p in base.iterdir():
        if p.name == "deleted.json":
            continue
        try:
            summary = read_catalog_summary(root, p.name)
            if summary:
                rows.append(summary)
            else:
                found = read_record(root, p.name)
                if found:
                    rows.append(catalog_summary(found["item"]))
        except (OSError, ValueError):
            pass
    return rows


def _is_provisional(record):
    """New media jobs become visible only after successful publication.

    The older mediaClassification marker is retained for records created by
    earlier builds. Records without either marker remain visible for backwards
    compatibility.
    """
    return bool(record) and (
        record.get("archivePublication") == "pending"
        or record.get("mediaClassification") == "pending"
    )
