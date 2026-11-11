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


STAGES = ("preparation", "source", "audio", "recognition", "diarization", "postprocess", "result")
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
    interrupted_unknown = (normalized.get("status") == "interrupted"
                           and "finishedAt" in normalized and "durationSeconds" in normalized)
    if finished_time is None and not interrupted_unknown:
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
            db.execute("""CREATE TABLE IF NOT EXISTS queue_entries (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE)""")
            db.execute("""CREATE TABLE IF NOT EXISTS queue_state (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                paused INTEGER NOT NULL DEFAULT 0,
                reason_code TEXT,
                reason_message TEXT)""")
            db.execute("INSERT OR IGNORE INTO queue_state(singleton, paused) VALUES (1, 0)")
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
                      startedAt=None, queueWaitSeconds=None, processingDurationSeconds=None,
                      checkpoints=record.get("checkpoints") or initial_checkpoints())
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)", (
                record["id"], record["createdAt"], record["updatedAt"], record["status"],
                record["stage"], record["revision"], json.dumps(record, ensure_ascii=False)))
        self.job_dir(record["id"]).mkdir(parents=True, exist_ok=True)
        return record

    def create_queued(self, metadata_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Create history rows and FIFO entries in one SQLite transaction."""
        records = []
        for metadata in metadata_items:
            record = dict(metadata)
            timestamp = now()
            record.update(id=record.get("id") or uuid.uuid4().hex,
                          createdAt=timestamp, updatedAt=timestamp, status="queued",
                          stage="source", revision=0, detectedLanguage=None, error=None,
                          notice=None, progress=None, finishedAt=None, durationSeconds=None,
                          startedAt=None, queueWaitSeconds=None, processingDurationSeconds=None,
                          checkpoints=record.get("checkpoints") or initial_checkpoints())
            records.append(record)
        with self._lock, self._connect() as db:
            for record in records:
                db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)", (
                    record["id"], record["createdAt"], record["updatedAt"], record["status"],
                    record["stage"], record["revision"], json.dumps(record, ensure_ascii=False)))
                db.execute("INSERT INTO queue_entries(job_id) VALUES (?)", (record["id"],))
        return records

    def queue_ids(self) -> list[str]:
        with self._lock, self._connect() as db:
            rows = db.execute("""SELECT queue_entries.job_id FROM queue_entries
                JOIN jobs ON jobs.id=queue_entries.job_id WHERE jobs.status='queued'
                ORDER BY queue_entries.sequence""").fetchall()
        return [row[0] for row in rows]

    def queued_records(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            rows = db.execute("""SELECT jobs.metadata FROM queue_entries
                JOIN jobs ON jobs.id=queue_entries.job_id WHERE jobs.status='queued'
                ORDER BY queue_entries.sequence""").fetchall()
        return [json.loads(row[0]) for row in rows]

    def queued_records(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            rows = db.execute("""SELECT jobs.metadata FROM queue_entries
                JOIN jobs ON jobs.id=queue_entries.job_id ORDER BY queue_entries.sequence""").fetchall()
        return [json.loads(row[0]) for row in rows]

    def dequeue_next(self) -> str | None:
        with self._lock, self._connect() as db:
            while True:
                row = db.execute("""SELECT queue_entries.sequence, queue_entries.job_id, jobs.status
                    FROM queue_entries LEFT JOIN jobs ON jobs.id=queue_entries.job_id
                    ORDER BY queue_entries.sequence LIMIT 1""").fetchone()
                if row is None:
                    return None
                if row["status"] == "queued":
                    return row["job_id"]
                db.execute("DELETE FROM queue_entries WHERE sequence=?", (row["sequence"],))

    def mark_started(self, job_id: str) -> dict[str, Any] | None:
        """Atomically claim the FIFO head after the worker owns the runtime slot."""
        with self._lock, self._connect() as db:
            row = db.execute("SELECT metadata, status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["status"] != "queued":
                return None
            record = json.loads(row["metadata"])
            started_at = now()
            record.update(status="running", startedAt=started_at,
                          queueWaitSeconds=_duration_seconds(record.get("createdAt"), started_at),
                          stage="preparation", updatedAt=started_at)
            db.execute("DELETE FROM queue_entries WHERE job_id=?", (job_id,))
            db.execute("UPDATE jobs SET updated_at=?, status='running', stage='preparation', metadata=? WHERE id=?",
                       (started_at, json.dumps(record, ensure_ascii=False), job_id))
            return record

    def remove_queued(self, job_id: str) -> bool:
        with self._lock, self._connect() as db:
            cursor = db.execute("DELETE FROM queue_entries WHERE job_id=?", (job_id,))
            return cursor.rowcount > 0

    def queue_position(self, job_id: str) -> int | None:
        with self._lock, self._connect() as db:
            row = db.execute("""SELECT sequence FROM queue_entries JOIN jobs ON jobs.id=queue_entries.job_id
                WHERE job_id=? AND jobs.status='queued'""", (job_id,)).fetchone()
            if row is None:
                return None
            rank = db.execute("""SELECT COUNT(*) FROM queue_entries JOIN jobs ON jobs.id=queue_entries.job_id
                WHERE sequence<=? AND jobs.status='queued'""", (row[0],)).fetchone()[0]
        return int(rank)

    def queue_state(self) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT paused, reason_code, reason_message FROM queue_state WHERE singleton=1").fetchone()
        return {"paused": bool(row[0]),
                "pauseReason": ({"code": row[1], "message": row[2]} if row[1] else None)}

    def set_queue_state(self, paused: bool, reason: dict[str, str] | None = None) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            db.execute("UPDATE queue_state SET paused=?, reason_code=?, reason_message=? WHERE singleton=1",
                       (int(paused), (reason or {}).get("code"), (reason or {}).get("message")))
        return self.queue_state()

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
            prior_elapsed_fields = {key: record.get(key) for key in
                                    ("startedAt", "queueWaitSeconds", "processingDurationSeconds")}
            record.update(changes)
            if previous_status == "queued" and record.get("status") == "running":
                started_at = now()
                record["startedAt"] = started_at
                record["queueWaitSeconds"] = _duration_seconds(record.get("createdAt"), started_at)
            if previous_status in TERMINAL_STATUSES:
                record["finishedAt"] = prior_timing["finishedAt"]
                record["durationSeconds"] = prior_timing["durationSeconds"]
                record.update(prior_elapsed_fields)
            elif record.get("status") in TERMINAL_STATUSES:
                record["finishedAt"] = now()
                record["durationSeconds"] = _duration_seconds(record.get("createdAt"), record["finishedAt"])
                started_at = _parse_time(record.get("startedAt"))
                finished_at = _parse_time(record.get("finishedAt"))
                created_at = _parse_time(record.get("createdAt"))
                if started_at:
                    record["queueWaitSeconds"] = _duration_seconds(record.get("createdAt"), started_at.isoformat())
                elif record.get("startedAt") is None and previous_status in {"queued", "cancelling"}:
                    record["queueWaitSeconds"] = _duration_seconds(record.get("createdAt"), record["finishedAt"])
                else:
                    record["queueWaitSeconds"] = None
                record["processingDurationSeconds"] = (
                    max(0.0, (finished_at - started_at).total_seconds())
                    if started_at and finished_at and finished_at >= started_at
                    else (0.0 if (started_at is None and record.get("startedAt") is None
                                  and previous_status in {"queued", "cancelling"} and created_at and finished_at)
                          else None))
            else:
                record["finishedAt"] = None
                record["durationSeconds"] = None
                if record.get("status") == "queued":
                    record["startedAt"] = None
                    record["queueWaitSeconds"] = None
                    record["processingDurationSeconds"] = None
            record["updatedAt"] = now()
            db.execute("UPDATE jobs SET updated_at=?, status=?, stage=?, revision=?, metadata=? WHERE id=?",
                       (record["updatedAt"], record["status"], record["stage"], record["revision"],
                        json.dumps(record, ensure_ascii=False), job_id))
            if record.get("status") in TERMINAL_STATUSES:
                db.execute("DELETE FROM queue_entries WHERE job_id=?", (job_id,))
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

    def write_diarization(self, job_id: str, value: dict[str, Any]) -> Path:
        """Atomically persist structured diarization separately from editable transcripts."""
        directory = self.job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "diarization.json"
        temp = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
        try:
            with temp.open("w", encoding="utf-8", newline="") as stream:
                json.dump(value, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return target

    def read_diarization(self, job_id: str) -> dict[str, Any] | None:
        path = self.job_dir(job_id) / "diarization.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

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
            rows = db.execute("""SELECT id, metadata, status FROM jobs
                WHERE status IN ('queued','running','cancelling')""").fetchall()
            for row in rows:
                record = json.loads(row["metadata"])
                if row["status"] == "queued":
                    if db.execute("SELECT 1 FROM queue_entries WHERE job_id=?", (row["id"],)).fetchone():
                        continue
                timestamp = now()
                for checkpoint in record.get("checkpoints", []):
                    if checkpoint.get("status") == "active":
                        checkpoint.update(status="failed", completedAt=timestamp)
                record.update(status="interrupted", error="Приложение было закрыто до завершения расшифровки.",
                              updatedAt=timestamp, finishedAt=None, durationSeconds=None,
                              queueWaitSeconds=(record.get("queueWaitSeconds")
                                                if record.get("startedAt") else None),
                              processingDurationSeconds=None)
                db.execute("UPDATE jobs SET status=?, updated_at=?, metadata=? WHERE id=?",
                           (record["status"], record["updatedAt"], json.dumps(record, ensure_ascii=False), row["id"]))
                db.execute("DELETE FROM queue_entries WHERE job_id=?", (row["id"],))
            pending = db.execute("SELECT 1 FROM queue_entries LIMIT 1").fetchone() is not None
            if pending:
                db.execute("UPDATE queue_state SET paused=1, "
                           "reason_code=COALESCE(reason_code, 'recovered_pending'), "
                           "reason_message=COALESCE(reason_message, 'После перезапуска очередь приостановлена.') "
                           "WHERE singleton=1")
            else:
                state = db.execute("SELECT reason_code FROM queue_state WHERE singleton=1").fetchone()
                if state is not None and state[0] in {None, "recovered_pending"}:
                    db.execute("UPDATE queue_state SET paused=0, reason_code=NULL, reason_message=NULL "
                               "WHERE singleton=1")

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
                db.execute("DELETE FROM queue_entries WHERE job_id=?", (job_id,))
                db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            return True
