"""Background HTTP API for verified meeting archive backups and cleanup."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/meetings/backup", tags=["meetings"])
_lock = threading.RLock()
_jobs: dict[str, dict[str, Any]] = {}
_maintenance_lock = threading.RLock()
_maintenance_controller = None
_MAINTENANCE_STATE_NAME = ".meeting-cleanup-maintenance.json"
_MAINTENANCE_MIN_ATTEMPT_INTERVAL = 24 * 60 * 60


class SnapshotRequest(BaseModel):
    snapshot_id: str


class RestoreRequest(BaseModel):
    snapshot_id: str
    target: str


def _settings() -> dict:
    from services.meeting_settings import load_settings
    from services.meeting_hardware import APP_ROOT
    return load_settings(APP_ROOT)


def _configured_roots() -> tuple[str, str | None]:
    settings = _settings()
    return settings["archive_root"], settings.get("backup_root")


def _utc_now(clock: Callable[[], datetime] | None = None) -> datetime:
    value = clock() if clock else datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _maintenance_state_path(settings: dict) -> Path | None:
    archive_root = settings.get("archive_root")
    backup_root = settings.get("backup_root")
    if not archive_root or not backup_root:
        return None
    from services.meeting_backup import _archive_root, _backup_root
    archive = _archive_root(archive_root)
    backup = _backup_root(backup_root, archive, create=False)
    if not backup.is_dir():
        return None
    return backup / _MAINTENANCE_STATE_NAME


def maintenance_tick(*, activity_probe: Callable[[], dict[str, Any]] | None,
                     now: datetime | None = None, clock: Callable[[], datetime] | None = None,
                     min_attempt_interval_seconds: int = _MAINTENANCE_MIN_ATTEMPT_INTERVAL) -> dict[str, Any]:
    """Run one safe maintenance check; no work is done while disabled or busy."""
    settings = _settings()
    if not settings.get("auto_cleanup", False):
        return {"status": "disabled", "attempted": False}
    if activity_probe is None:
        return {"status": "skipped", "attempted": False,
                "reason": "Состояние очереди и записи недоступно; очистка отложена."}
    try:
        activity = activity_probe() or {}
    except Exception as exc:
        return {"status": "skipped", "attempted": False,
                "reason": "Не удалось проверить очередь и запись.", "detail": type(exc).__name__}
    if activity.get("queue_active") or activity.get("recording"):
        return {"status": "skipped", "attempted": False,
                "reason": "Очистка отложена: встреча записывается или очередь обработки занята."}
    state_path = _maintenance_state_path(settings)
    if state_path is None:
        return {"status": "blocked", "attempted": False,
                "reason": "Папка резервных копий не настроена или недоступна."}
    from services.meeting_backup import _json_read, _atomic_json, run_cleanup, BackupError
    timestamp = now.astimezone(timezone.utc) if now and now.tzinfo else (
        now.replace(tzinfo=timezone.utc) if now else _utc_now(clock))
    state = _json_read(state_path, {})
    previous = state.get("last_attempt_at") if isinstance(state, dict) else None
    if previous:
        try:
            last_attempt = datetime.fromisoformat(previous)
            if last_attempt.tzinfo is None:
                last_attempt = last_attempt.replace(tzinfo=timezone.utc)
            if (timestamp - last_attempt.astimezone(timezone.utc)).total_seconds() < max(1, int(min_attempt_interval_seconds)):
                return {"status": "not_due", "attempted": False,
                        "last_attempt_at": previous}
        except (TypeError, ValueError):
            pass
    new_state = {"schema": 1, "last_attempt_at": timestamp.isoformat(timespec="seconds"),
                 "status": "running"}
    try:
        _atomic_json(state_path, new_state)
    except OSError as exc:
        return {"status": "blocked", "attempted": False,
                "reason": "Не удалось сохранить время последней попытки очистки.",
                "detail": type(exc).__name__}
    try:
        result = run_cleanup(settings["archive_root"], settings.get("backup_root"))
        final = {**new_state, "status": "completed", "result": result}
        _atomic_json(state_path, final)
        return {"status": "completed", "attempted": True, "result": result,
                "last_attempt_at": new_state["last_attempt_at"]}
    except BackupError as exc:
        final = {**new_state, "status": "failed", "error": {"code": exc.code, "message": str(exc)}}
        _atomic_json(state_path, final)
        return {"status": "failed", "attempted": True, "error": final["error"],
                "last_attempt_at": new_state["last_attempt_at"]}
    except Exception as exc:
        final = {**new_state, "status": "failed",
                 "error": {"code": "maintenance_failed", "message": type(exc).__name__}}
        _atomic_json(state_path, final)
        return {"status": "failed", "attempted": True, "error": final["error"],
                "last_attempt_at": new_state["last_attempt_at"]}


class CleanupMaintenance:
    """Explicitly started hourly guard loop for the main application only."""

    def __init__(self, activity_probe: Callable[[], dict[str, Any]], interval_seconds: float,
                 clock: Callable[[], datetime] | None):
        self.activity_probe = activity_probe
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.clock = clock
        self.stop_event = threading.Event()
        self.last_result: dict[str, Any] | None = None
        self.thread = threading.Thread(target=self._run, name="meeting-cleanup-maintenance", daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.last_result = maintenance_tick(activity_probe=self.activity_probe, clock=self.clock)
            except Exception as exc:
                self.last_result = {"status": "failed", "attempted": False,
                                    "error": {"code": "maintenance_failed", "message": type(exc).__name__}}
            if self.stop_event.wait(self.interval_seconds):
                break

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        global _maintenance_controller
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=max(0.0, timeout))
        with _maintenance_lock:
            if _maintenance_controller is self and not self.thread.is_alive():
                _maintenance_controller = None


def start_cleanup_maintenance(*, activity_probe: Callable[[], dict[str, Any]] | None,
                              interval_seconds: float = 60 * 60,
                              clock: Callable[[], datetime] | None = None):
    """Start only when explicitly called by the main app and cleanup is enabled.

    This module does not start work on import, when the API router is mounted,
    or from the MCP server. The main application lifespan owns this handle.
    """
    global _maintenance_controller
    settings = _settings()
    if not settings.get("auto_cleanup", False) or activity_probe is None:
        return None
    with _maintenance_lock:
        if _maintenance_controller is not None and _maintenance_controller.thread.is_alive():
            return _maintenance_controller
        controller = CleanupMaintenance(activity_probe, interval_seconds, clock)
        _maintenance_controller = controller
        controller.start()
        return controller


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    return {"job_id": job["job_id"], "operation": job["operation"],
            "status": job["status"], "stage": job.get("stage"),
            "progress": list(job.get("progress", [])),
            "result": job.get("result"), "error": job.get("error")}


def _start_job(operation: str, action: Callable[[Callable[[dict], None]], dict]) -> dict:
    job_id = uuid.uuid4().hex
    job = {"job_id": job_id, "operation": operation, "status": "running",
           "stage": "queued", "progress": [], "result": None, "error": None}
    with _lock:
        _jobs[job_id] = job
        for old_id in list(_jobs)[:-20]:
            if _jobs[old_id].get("status") != "running":
                _jobs.pop(old_id, None)

    def progress(item: dict) -> None:
        with _lock:
            current = _jobs.get(job_id)
            if current is not None:
                current["stage"] = item.get("stage")
                current["progress"] = (current["progress"] + [item])[-100:]

    def work() -> None:
        with _lock:
            current = _jobs.get(job_id)
            if current is None:
                return
            current["stage"] = "starting"
        try:
            result = action(progress)
            with _lock:
                current = _jobs.get(job_id)
                if current is not None:
                    current["result"] = result
                    current["status"] = "completed"
                    current["stage"] = "completed"
        except Exception as exc:
            from services.meeting_backup import BackupError
            if isinstance(exc, BackupError):
                error = {"code": exc.code, "message": str(exc)}
            else:
                error = {"code": "backup_operation_failed",
                         "message": "Операция резервного копирования не выполнена.",
                         "detail": f"{type(exc).__name__}: {exc}"}
            with _lock:
                current = _jobs.get(job_id)
                if current is not None:
                    current["status"] = "failed"
                    current["stage"] = "failed"
                    current["error"] = error

    threading.Thread(target=work, name=f"meeting-backup-{operation}", daemon=True).start()
    return {"job_id": job_id, "operation": operation, "status": "running"}


@router.get("/status")
def get_backup_status(job_id: str | None = None):
    from services.meeting_backup import backup_status, BackupError
    archive_root, backup_root = _configured_roots()
    try:
        status = backup_status(archive_root, backup_root)
    except BackupError as exc:
        status = {"configured": False, "error": {"code": exc.code, "message": str(exc)}}
    with _lock:
        selected = _jobs.get(job_id) if job_id else None
        jobs = [_job_view(row) for row in list(_jobs.values())[-10:]]
    return {"storage": status, "job": _job_view(selected) if selected else None,
            "jobs": jobs}


@router.post("/backup", status_code=202)
def post_backup():
    from services.meeting_backup import create_backup
    archive_root, backup_root = _configured_roots()
    if not backup_root:
        return {"status": "failed", "error": {"code": "backup_unconfigured",
                "message": "Сначала выберите папку резервных копий в настройках встреч."}}
    return _start_job("backup", lambda progress: create_backup(archive_root, backup_root, progress=progress))


@router.post("/verify", status_code=202)
def post_verify(payload: SnapshotRequest):
    from services.meeting_backup import verify_backup, BackupError
    archive_root, backup_root = _configured_roots()
    if not backup_root:
        return {"status": "failed", "error": {"code": "backup_unconfigured",
                "message": "Сначала выберите папку резервных копий в настройках встреч."}}

    def action(progress):
        result = verify_backup(backup_root, payload.snapshot_id, progress=progress)
        if not result.get("verified"):
            error = result.get("error") or {}
            raise BackupError(error.get("code", "verification_failed"),
                              error.get("message", "Проверка резервной копии не прошла."))
        return {key: value for key, value in result.items() if key != "manifest"}

    return _start_job("verify", action)


@router.post("/restore-test", status_code=202)
def post_restore_test(payload: SnapshotRequest):
    from services.meeting_backup import restore_test
    archive_root, backup_root = _configured_roots()
    if not backup_root:
        return {"status": "failed", "error": {"code": "backup_unconfigured",
                "message": "Сначала выберите папку резервных копий в настройках встреч."}}
    return _start_job("restore_test", lambda progress: restore_test(
        archive_root, backup_root, payload.snapshot_id, progress=progress))


@router.post("/restore", status_code=202)
def post_restore(payload: RestoreRequest):
    from services.meeting_backup import restore_backup
    archive_root, backup_root = _configured_roots()
    if not backup_root:
        return {"status": "failed", "error": {"code": "backup_unconfigured",
                "message": "Сначала выберите папку резервных копий в настройках встреч."}}
    return _start_job("restore", lambda progress: restore_backup(
        archive_root, backup_root, payload.snapshot_id, payload.target, progress=progress))


@router.post("/cleanup/preview", status_code=202)
def post_cleanup_preview():
    from services.meeting_backup import cleanup_preview
    archive_root, backup_root = _configured_roots()
    return _start_job("cleanup_preview", lambda progress: cleanup_preview(
        archive_root, backup_root, progress=progress))


@router.post("/cleanup/run", status_code=202)
def post_cleanup_run():
    from services.meeting_backup import run_cleanup
    archive_root, backup_root = _configured_roots()
    return _start_job("cleanup", lambda progress: run_cleanup(
        archive_root, backup_root, progress=progress))

