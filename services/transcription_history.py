"""Durable metadata and artifact storage for simple transcription jobs."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


STAGES = ("preparation", "source", "audio", "recognition", "postprocess", "result")
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}


def initial_checkpoints() -> list[dict[str, Any]]:
    return [{"stage": stage, "status": "pending", "startedAt": None, "completedAt": None}
            for stage in STAGES]


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _duration_seconds(created_at: Any, finished_at: Any) -> float | None:
    created = _parse_time(created_at)
    finished = _parse_time(finished_at)
    if created is None or finished is None or finished < created:
        return None
    return max(0.0, (finished - created).total_seconds())


def normalize_terminal_timing(record: dict[str, Any]) -> dict[str, Any]:
    """Return stable timing fields, recovering legacy terminal rows from checkpoints only."""
    normalized = dict(record)
    if normalized.get("status") not in TERMINAL_STATUSES:
        normalized["finishedAt"] = None
        normalized["durationSeconds"] = None
        return normalized

    created_at = normalized.get("createdAt")
    finished_at = normalized.get("finishedAt")
    created_time = _parse_time(created_at)
    finished_time = _parse_time(finished_at)
    if finished_time is not None and created_time is not None and finished_time < created_time:
        finished_at = None
        finished_time = None
    if finished_time is None:
        candidates = []
        for checkpoint in normalized.get("checkpoints", []):
            if not isinstance(checkpoint, dict):
                continue
            value = checkpoint.get("completedAt")
            parsed = _parse_time(value)
            elapsed = _duration_seconds(created_at, value)
            if parsed is not None and elapsed is not None:
                candidates.append((parsed, value))
        finished_at = max(candidates, default=(None, None), key=lambda pair: pair[0])[1]

    elapsed = _duration_seconds(created_at, finished_at)
    normalized["finishedAt"] = finished_at if finished_time is not None or elapsed is not None else None
    normalized["durationSeconds"] = elapsed
    return normalized


class TranscriptionHistory:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.db_path = self.root / "history.sqlite3"
        self.transcriptions_dir = self.root / "transcriptions"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.transcriptions_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                status TEXT NOT NULL, stage TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL)""")
        self.recover_interrupted()

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, metadata: dict[str, Any]) -> dict[str, Any]:
        record = dict(metadata)
        record.update(id=record.get("id") or uuid.uuid4().hex, createdAt=now(), updatedAt=now(),
                      status="queued", stage="source", revision=0, detectedLanguage=None,
                      error=None, notice=None, progress=None,
                      finishedAt=None, durationSeconds=None,
                      checkpoints=record.get("checkpoints") or initial_checkpoints())
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)", (
                record["id"], record["createdAt"], record["updatedAt"], record["status"],
                record["stage"], record["revision"], json.dumps(record, ensure_ascii=False)))
        self.job_dir(record["id"]).mkdir(parents=True, exist_ok=True)
        return record

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT metadata FROM jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def update(self, job_id: str, **changes) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT metadata FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            record = json.loads(row[0])
            previous_status = record.get("status")
            prior_timing = normalize_terminal_timing(record) if previous_status in TERMINAL_STATUSES else None
            record.update(changes)
            if previous_status in TERMINAL_STATUSES:
                record["finishedAt"] = prior_timing["finishedAt"]
                record["durationSeconds"] = prior_timing["durationSeconds"]
            elif record.get("status") in TERMINAL_STATUSES:
                record["finishedAt"] = now()
                record["durationSeconds"] = _duration_seconds(record.get("createdAt"), record["finishedAt"])
            else:
                record["finishedAt"] = None
                record["durationSeconds"] = None
            record["updatedAt"] = now()
            db.execute("UPDATE jobs SET updated_at=?, status=?, stage=?, revision=?, metadata=? WHERE id=?",
                       (record["updatedAt"], record["status"], record["stage"], record["revision"],
                        json.dumps(record, ensure_ascii=False), job_id))
            return record

    def list(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT metadata FROM jobs ORDER BY created_at DESC, id DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def latest(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT metadata FROM jobs ORDER BY created_at DESC, id DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

    def job_dir(self, job_id: str) -> Path:
        # IDs are generated internally, but also reject malformed IDs at the boundary.
        if not job_id or any(ch not in "0123456789abcdef-" for ch in job_id.lower()):
            raise ValueError("Invalid job id")
        path = (self.transcriptions_dir / job_id).resolve()
        path.relative_to(self.transcriptions_dir.resolve())
        return path

    def artifact(self, job_id: str, extension: str) -> Path:
        if extension not in {"txt", "md"}:
            raise ValueError("Unsupported artifact format")
        return self.job_dir(job_id) / f"transcript.{extension}"

    def write_text(self, job_id: str, text: str, revision: int) -> None:
        directory = self.job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        temps = []
        try:
            for ext in ("txt", "md"):
                target = self.artifact(job_id, ext)
                temp = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
                with temp.open("w", encoding="utf-8", newline="") as stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                temps.append((temp, target))
            with self._lock:
                prior = [(target, target.read_bytes() if target.exists() else None) for _, target in temps]
                replaced = []
                try:
                    for temp, target in temps:
                        os.replace(temp, target)
                        replaced.append(target)
                except Exception:
                    for target, content in prior:
                        if target in replaced:
                            if content is None:
                                target.unlink(missing_ok=True)
                            else:
                                rollback = target.with_name(target.name + f".{uuid.uuid4().hex}.rollback")
                                rollback.write_bytes(content)
                                os.replace(rollback, target)
                    raise
        finally:
            for temp, _ in temps:
                temp.unlink(missing_ok=True)

    def read_text(self, job_id: str) -> str | None:
        with self._lock:
            path = self.artifact(job_id, "txt")
            return path.read_text(encoding="utf-8") if path.is_file() else None

    def replace_text(self, job_id: str, text: str, base_revision: int) -> dict[str, Any]:
        with self._lock:
            record = self.get(job_id)
            if not record:
                raise LookupError(job_id)
            if record["status"] != "completed":
                raise PermissionError("Изменять можно только завершённую расшифровку.")
            if record.get("revision", 0) != base_revision:
                raise PermissionError(f"Конфликт версии. Текущая версия: {record.get('revision', 0)}.")
            if not self.artifact(job_id, "txt").is_file() or not self.artifact(job_id, "md").is_file():
                raise LookupError(job_id)
            revision = base_revision + 1
            self.write_text(job_id, text, revision)
            updated = self.update(job_id, revision=revision)
            assert updated is not None
            return updated

    def recover_interrupted(self) -> None:
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT id, metadata FROM jobs WHERE status IN ('queued','running','cancelling')").fetchall()
            for row in rows:
                record = json.loads(row["metadata"])
                timestamp = now()
                for checkpoint in record.get("checkpoints", []):
                    if checkpoint.get("status") == "active":
                        checkpoint.update(status="failed", completedAt=timestamp)
                record.update(status="interrupted", error="Приложение было закрыто до завершения расшифровки.",
                              updatedAt=timestamp, finishedAt=timestamp,
                              durationSeconds=_duration_seconds(record.get("createdAt"), timestamp))
                db.execute("UPDATE jobs SET status=?, updated_at=?, metadata=? WHERE id=?",
                           (record["status"], record["updatedAt"], json.dumps(record, ensure_ascii=False), row["id"]))

    def delete(self, job_id: str) -> bool:
        with self._lock:
            record = self.get(job_id)
            if not record:
                return False
            if record["status"] in {"queued", "running", "cancelling"}:
                raise PermissionError("Cannot delete an active transcription.")
            path = self.job_dir(job_id)
            if path.exists():
                shutil.rmtree(path)
            with self._connect() as db:
                db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            return True
