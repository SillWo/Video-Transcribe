import json
import importlib.util
import re
import os
import shutil
import subprocess
import sys
import threading
import uuid
import ipaddress
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from services.model_paths import MODEL_STORAGE_ROOT
import psutil
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from services.transcription_history import TranscriptionHistory
from services import native_file_picker, simple_profiles
from services.resource_preparation import PREPARATION_EVENT_PREFIX, ResourcePreparationError
from services.job_processes import JobCancelled, JobProcessController

from services.model_management import (
    ModelCacheInspectionError,
    ModelDeleteValidationError,
    ModelDownloadValidationError,
    delete_registered_model,
    download_registered_model,
    get_model_panel_payload,
    get_model_status,
    list_models_with_status,
)
from services.model_registry import (
    ModelRegistryLookupError,
    get_model_registry_entry,
    list_enabled_backend_values,
    list_model_catalog,
)
from services.runtime_devices import VULKAN_MODELS, get_device_details
from source_check import check_source

APP_ROOT = Path(__file__).resolve().parent
RUNS_DIR = APP_ROOT / "web_runs"
SIMPLE_RUNTIME_DIR = APP_ROOT / ".runtime"
_history_store: TranscriptionHistory | None = None
_history_store_init_lock = threading.Lock()


def get_history_store() -> TranscriptionHistory:
    global _history_store
    if _history_store is None:
        with _history_store_init_lock:
            if _history_store is None:
                configured = os.environ.get("VT_SIMPLE_RUNTIME_DIR")
                _history_store = TranscriptionHistory(Path(configured) if configured else SIMPLE_RUNTIME_DIR)
    return _history_store


class _LazyHistoryStore:
    def __getattr__(self, name):
        return getattr(get_history_store(), name)


history_store = _LazyHistoryStore()
JSON_EVENT_PREFIX = "__VT_JSON__ "


class SourceCheckRequest(BaseModel):
    url: str


class ModelDownloadRequest(BaseModel):
    modelId: str


class ModelDeleteRequest(BaseModel):
    modelId: str


class SimpleJobRequest(BaseModel):
    sourceType: str
    url: str | None = None
    path: str | None = None
    profile: str
    device: str
    cpuThreads: int
    restorePunctuation: bool


class SimpleTextRequest(BaseModel):
    text: str
    baseRevision: int


@dataclass
class ModelDownloadProgressState:
    percent: int = 0
    label: str = "Queued"


@dataclass
class ModelOperationJobState:
    job_id: str
    model_id: str
    status: str = "queued"
    progress: ModelDownloadProgressState = field(default_factory=ModelDownloadProgressState)
    message: str = "Download queued."
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ActiveModelOperationState:
    operation_type: str
    job_id: str
    model_id: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def safe_filename(name: str | None, fallback: str = "upload.bin") -> str:
    if not name:
        return fallback
    clean_name = Path(name).name.strip()
    for invalid_char in '<>:"/\\|?*':
        clean_name = clean_name.replace(invalid_char, "_")
    return clean_name or fallback


def logical_cpu_count() -> int:
    return psutil.cpu_count(logical=True) or os.cpu_count() or 1


def parse_int(value: str | int | None, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default

    return max(minimum, min(parsed, maximum))


@dataclass
class JobState:
    id: str
    settings: dict[str, Any]
    status: str = "queued"
    stage: str = "source"
    logs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    source_name: str | None = None
    output_path: str | None = None
    output_format: str | None = None
    result_text: str | None = None
    rendered_result: str | None = None
    detected_language: str | None = None
    error: str | None = None
    saved_audio: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    effective_device: str | None = None
    effective_backend: str | None = None


jobs: dict[str, JobState] = {}
job_process_controllers: dict[str, JobProcessController] = {}
simple_worker_states: dict[str, str] = {}
jobs_lock = threading.Lock()
transcription_semaphore = threading.Semaphore(1)
model_download_jobs: dict[str, ModelOperationJobState] = {}
model_delete_jobs: dict[str, ModelOperationJobState] = {}
model_operation_jobs_lock = threading.Lock()
active_model_operation: ActiveModelOperationState | None = None

app = FastAPI(title="Video Transcribe Local API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_management_router = APIRouter(prefix="/api/models", tags=["model-management"])


def get_job(job_id: str) -> JobState:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


def update_job(job_id: str, **changes) -> None:
    with jobs_lock:
        job = jobs[job_id]
        if job.settings.get("_simple") and job.status in {"cancelling", "cancelled", "completed", "failed", "interrupted"}:
            if changes.get("status") not in {None, job.status}:
                return
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = utc_now()
        is_simple = bool(job.settings.get("_simple"))
        simple_changes = dict(changes)
        simple_changes["updatedAt"] = job.updated_at
        if is_simple:
            if "status" in changes:
                simple_changes["status"] = changes["status"]
            if "stage" in changes:
                simple_changes["stage"] = changes["stage"]
            if "detected_language" in changes:
                simple_changes["detectedLanguage"] = changes["detected_language"]
            if "error" in changes:
                simple_changes["error"] = concise_simple_error(changes["error"])
            if "notice" in changes:
                simple_changes["notice"] = changes["notice"]
            allowed = {"status", "stage", "detectedLanguage", "error", "notice"}
            history_store.update(job_id, **{k: v for k, v in simple_changes.items() if k in allowed})


def concise_simple_error(message: str | None) -> str | None:
    if not message:
        return None
    text = str(message).strip()
    lower = text.lower()
    if "start.ps1" in lower or "runstart.ps1" in lower or "runstart" in lower:
        return "Не удалось подготовить ресурсы. Повторите попытку."
    if any(marker in lower for marker in (
        "out of memory", "cuda_error_out_of_memory", "cudnn_status_alloc_failed",
        "cublas_status_alloc_failed", "alloc_failed", "failed to allocate", "allocation failed", "not enough memory",
    )):
        return "Недостаточно памяти видеокарты. Выберите «Экономный» или процессор и повторите запуск."
    if any(marker in lower for marker in (
        "ffmpeg", "audio extraction", "invalid data found", "invaliddata", "no audio stream",
        "could not find codec parameters",
    )):
        return "Не удалось прочитать звуковую дорожку. Выберите доступный аудио- или видеофайл."
    if any(marker in lower for marker in (
        "playlist", "live stream", "livestream", "upcoming", "multiple video",
        "more than one video", "not a single finished", "broadcast is live",
    )):
        return "Поддерживаются только публичные ссылки на одиночные завершённые записи."
    if any(marker in lower for marker in (
        "model not found", "model file", "could not download model", "model is not installed",
        "no such file or directory", "hugging face did not return",
    )):
        return "Не удалось загрузить модель. Проверьте подключение к интернету и повторите попытку."
    if any(marker in lower for marker in (
        "cuda", "cudnn", "cublas", "vulkan", "whisper-cli", "gpu runtime", "dll",
        "driver", "runtime is not installed",
    )):
        return "Среда GPU недоступна. Выберите процессор или обновите драйвер видеокарты."
    if any(marker in lower for marker in (
        "timed out", "timeout", "connection refused", "connection reset", "urlopen error",
        "http error", "yt-dlp", "could not download", "unable to download", "network",
    )):
        return "Проверьте публичную ссылку и подключение к интернету, затем повторите попытку."

    technical_markers = (
        "traceback", "exception", "error:", "failed with", " at ", "file \"", "--enable-",
        "--disable-", "\n", "\r",
    )
    has_cyrillic = any("а" <= character.lower() <= "я" or character.lower() == "ё" for character in text)
    if has_cyrillic and len(text) <= 180 and not any(marker in lower for marker in technical_markers):
        return text
    return "Не удалось завершить расшифровку. Проверьте ссылку или файл и попробуйте снова."


def verify_simple_model_cache(settings: dict[str, Any]) -> str | None:
    """Read-only preflight for simple jobs; never lets the CLI download missing weights."""
    try:
        if settings.get("device") == "vulkan":
            from services.prepare_models import _vulkan_cached

            model_ready = _vulkan_cached(settings["model"])
        else:
            model_ready = get_model_status(settings["model"]).get("status") == "downloaded"
    except Exception:
        model_ready = False
    if not model_ready:
        return "Не удалось загрузить модель. Проверьте подключение к интернету и повторите попытку."
    if settings.get("restorePunctuation"):
        try:
            from services.prepare_models import _punctuation_cached

            punctuation_ready = _punctuation_cached()
        except Exception:
            punctuation_ready = False
        if not punctuation_ready:
            return "Отключите пунктуацию и повторите попытку."
    return None


def append_log(job_id: str, line: str) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job.logs.append(line)
        if len(job.logs) > 500:
            job.logs = job.logs[-500:]
        job.updated_at = utc_now()


def _checkpoint_update(job_id: str, stage: str, status: str) -> None:
    if stage not in {"preparation", "source", "audio", "recognition", "postprocess", "result"}:
        return
    record = history_store.get(job_id)
    if not record:
        return
    checkpoints = [dict(item) for item in record.get("checkpoints", [])]
    checkpoint = next((item for item in checkpoints if item.get("stage") == stage), None)
    if checkpoint is None:
        return
    stamp = utc_now()
    if status == "active":
        if checkpoint.get("status") in {"pending", "active"}:
            checkpoint.update(status="active", startedAt=checkpoint.get("startedAt") or stamp)
    elif status in {"completed", "skipped", "failed", "cancelled"}:
        if status in {"completed", "failed"} and not checkpoint.get("startedAt"):
            checkpoint["startedAt"] = stamp
        checkpoint.update(status=status, completedAt=checkpoint.get("completedAt") or stamp)
    history_store.update(job_id, checkpoints=checkpoints, stage=stage)


def _simple_progress(job_id: str, stage: str, payload: dict[str, Any]) -> None:
    record = history_store.get(job_id)
    if not record:
        return
    previous = record.get("progress") or {}
    same_stage = previous.get("stage") == stage
    percent = payload.get("percent", previous.get("percent") if same_stage else None)
    if percent is not None:
        try:
            percent = max(0, min(100, int(percent)))
        except (TypeError, ValueError):
            percent = None
    def bounded(value):
        try:
            return max(0, int(value)) if value is not None else None
        except (TypeError, ValueError):
            return None
    completed = bounded(payload.get("completedBytes", previous.get("completedBytes") if same_stage else None))
    total = bounded(payload.get("totalBytes", previous.get("totalBytes") if same_stage else None))
    if completed is not None and total is not None:
        completed = min(completed, total)
    seconds = payload.get("processedSeconds", previous.get("processedSeconds") if same_stage else None)
    duration = payload.get("totalSeconds", previous.get("totalSeconds") if same_stage else None)
    def finite(value):
        try:
            number = float(value)
            return max(0.0, number) if number == number and number != float("inf") else None
        except (TypeError, ValueError):
            return None
    seconds, duration = finite(seconds), finite(duration)
    safe_messages = {
        "source": "Загрузка источника", "audio": "Подготовка аудио",
        "recognition": "Распознавание речи", "postprocess": "Обработка текста",
        "result": "Подготовка результата",
    }
    if stage == "preparation":
        allowed_prep = {"Загрузка модели распознавания", "Загрузка модели пунктуации",
                        "Подготовка ресурсов", "Подготовка завершена"}
        requested_message = payload.get("message")
        if payload.get("ready"):
            message = "Подготовка завершена"
        elif requested_message in allowed_prep:
            message = requested_message
        else:
            message = "Подготовка ресурсов"
    else:
        message = safe_messages.get(stage)
        requested_message = payload.get("message")
        if stage == "recognition" and requested_message in {
            "Загружаем модель в память", "Загружаем модель в память."
        }:
            message = "Загружаем модель в память"
        elif stage == "postprocess" and isinstance(requested_message, str) and re.fullmatch(
            r"Обработано фрагментов \d+ из \d+\.?", requested_message
        ):
            message = requested_message
    progress = {"stage": stage, "percent": percent, "completedBytes": completed,
                "totalBytes": total, "processedSeconds": seconds, "totalSeconds": duration,
                "message": message, "updatedAt": utc_now()}
    changed_meaningfully = (previous.get("stage") != stage or previous.get("percent") != percent
                            or previous.get("message") != message)
    if (not changed_meaningfully and previous.get("completedBytes") is not None and completed is not None
            and abs(completed - previous["completedBytes"]) >= 1024 * 1024):
        changed_meaningfully = True
    try:
        previous_at = datetime.fromisoformat(previous.get("updatedAt", ""))
        elapsed = (datetime.now(timezone.utc) - previous_at).total_seconds()
    except (TypeError, ValueError):
        elapsed = 2
    if changed_meaningfully or elapsed >= 1:
        history_store.update(job_id, stage=stage, progress=progress)


def _simple_stage_event(job_id: str, stage: str, status: str | None) -> None:
    if status not in {"completed", "skipped", "failed", "active", "in_progress"}:
        status = "active"
    if status == "in_progress":
        status = "active"
    order = ["preparation", "source", "audio", "recognition", "postprocess", "result"]
    record = history_store.get(job_id)
    if record and stage in order:
        target_index = order.index(stage)
        for checkpoint in record.get("checkpoints", []):
            old_stage = checkpoint.get("stage")
            if checkpoint.get("status") == "active" and old_stage in order and order.index(old_stage) < target_index:
                _checkpoint_update(job_id, old_stage, "completed")
    if status == "active":
        _checkpoint_update(job_id, stage, "active")
    else:
        _checkpoint_update(job_id, stage, status)


def _finish_simple_checkpoints(job_id: str) -> None:
    record = history_store.get(job_id)
    if not record:
        return
    for checkpoint in record.get("checkpoints", []):
        if checkpoint.get("status") == "active":
            _checkpoint_update(job_id, checkpoint["stage"], "completed")
        elif checkpoint.get("status") == "pending":
            _checkpoint_update(job_id, checkpoint["stage"], "skipped")


def _simple_controller(job_id: str) -> JobProcessController:
    with jobs_lock:
        controller = job_process_controllers.get(job_id)
        if controller is None:
            controller = JobProcessController()
            job_process_controllers[job_id] = controller
        return controller


def _finalize_simple_job(job_id: str, status: str, *, error: str | None = None,
                         transcript: str | None = None,
                         require_worker_finished: bool = False) -> bool:
    """Commit a terminal simple-job state without racing a cancellation request."""
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None or not job.settings.get("_simple"):
            return False
        if require_worker_finished and simple_worker_states.get(job_id) in {"not_started", "running"}:
            return False
        if job.status in {"completed", "failed", "cancelled", "interrupted"}:
            return False
        controller = job_process_controllers.get(job_id)
        if status in {"completed", "failed"} and (job.status == "cancelling" or (controller and controller.cancelled)):
            return False
        if status == "cancelled" and job.status != "cancelling" and not (controller and controller.cancelled):
            return False
        final_stage = "result" if status == "completed" else job.stage
        final_error = None if status in {"completed", "cancelled"} else concise_simple_error(error)
        record = history_store.get(job_id)
        if record is None:
            return False
        checkpoints = [dict(item) for item in record.get("checkpoints", [])]
        prior_artifacts = None
        if status == "completed":
            for checkpoint in checkpoints:
                if checkpoint.get("status") == "active":
                    checkpoint.update(status="completed", completedAt=utc_now())
                elif checkpoint.get("status") == "pending":
                    checkpoint.update(status="skipped", completedAt=utc_now())
                if checkpoint.get("stage") == "result":
                    checkpoint.update(status="completed", startedAt=checkpoint.get("startedAt") or utc_now(),
                                      completedAt=checkpoint.get("completedAt") or utc_now())
            # Hold the job lock through artifact publication and final status so
            # cancellation either wins first or observes a completed job.
            prior_artifacts = {}
            for extension in ("txt", "md"):
                artifact = history_store.artifact(job_id, extension)
                prior_artifacts[extension] = artifact.read_bytes() if artifact.exists() else None
            history_store.write_text(job_id, transcript or "", revision=0)
        elif status == "cancelled":
            current = next((item for item in checkpoints if item.get("status") == "active"), None)
            if current is None:
                current = next((item for item in checkpoints if item.get("status") == "pending"), None)
            if current is None:
                current = next((item for item in checkpoints if item.get("stage") == job.stage), None)
            for checkpoint in checkpoints:
                if checkpoint.get("status") == "pending":
                    checkpoint.update(status="skipped", completedAt=utc_now())
            if current is not None:
                stamp = utc_now()
                current.update(status="cancelled", startedAt=current.get("startedAt") or stamp,
                               completedAt=stamp)
                final_stage = current.get("stage") or final_stage
        elif status == "failed":
            for checkpoint in checkpoints:
                if checkpoint.get("status") == "active":
                    checkpoint.update(status="failed", completedAt=utc_now())
        progress = record.get("progress")
        if status == "completed":
            progress = {"stage": "result", "percent": 100, "completedBytes": None,
                        "totalBytes": None, "processedSeconds": None, "totalSeconds": None,
                        "message": "Расшифровка завершена", "updatedAt": utc_now()}
        try:
            updated = history_store.update(job_id, status=status, stage=final_stage,
                                           error=final_error, checkpoints=checkpoints, progress=progress)
            if updated is None:
                raise LookupError(job_id)
        except Exception:
            if status == "completed" and prior_artifacts is not None:
                for extension, content in prior_artifacts.items():
                    artifact = history_store.artifact(job_id, extension)
                    if content is None:
                        artifact.unlink(missing_ok=True)
                    else:
                        artifact.write_bytes(content)
            raise
        job.status = status
        job.stage = final_stage
        job.error = final_error
        job.updated_at = utc_now()
        return True


def _leave_simple_job_cancelling(job_id: str) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None or job.status in {"completed", "failed", "cancelled", "interrupted"}:
            return
        job.status = "cancelling"
        job.error = "Не удалось остановить обработку. Повторите отмену."
        job.updated_at = utc_now()
        history_store.update(job_id, status="cancelling", error=job.error)


def _cancel_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        record = history_store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Расшифровка не найдена.")
        status = record.get("status")
        if status in {"completed", "failed", "interrupted"}:
            raise HTTPException(status_code=409, detail="Эту расшифровку уже нельзя отменить.")
        if status == "cancelled":
            return record
        if status not in {"queued", "running", "cancelling"}:
            raise HTTPException(status_code=409, detail="Эту расшифровку уже нельзя отменить.")
        if job is not None and job.status not in {"queued", "running", "cancelling"}:
            raise HTTPException(status_code=409, detail="Эту расшифровку уже нельзя отменить.")
        controller = job_process_controllers.get(job_id)
        if controller is None:
            controller = JobProcessController()
            job_process_controllers[job_id] = controller
        if status != "cancelling":
            if job is not None:
                job.status = "cancelling"
                job.error = None
                job.updated_at = utc_now()
            record = history_store.update(job_id, status="cancelling", error=None) or record
    # Process-tree termination can take time; never hold the global job lock.
    try:
        controller.cancel()
        with jobs_lock:
            job = jobs.get(job_id)
            if job is not None and job.status == "cancelling" and job.error == "Не удалось остановить обработку. Повторите отмену.":
                job.error = None
                job.updated_at = utc_now()
                record = history_store.update(job_id, error=None) or record
    except Exception:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is not None and job.status == "cancelling":
                job.error = "Не удалось остановить обработку. Повторите отмену."
                job.updated_at = utc_now()
                record = history_store.update(job_id, error=job.error) or record
        raise HTTPException(status_code=503, detail="Не удалось остановить обработку. Повторите отмену.")
    if _finalize_simple_job(job_id, "cancelled", require_worker_finished=True):
        with jobs_lock:
            job_process_controllers.pop(job_id, None)
    return history_store.get(job_id) or record


class OwnedProcessStopError(RuntimeError):
    """A child process tree could not be confirmed stopped after worker failure."""


def _run_owned_process(controller: JobProcessController, command: list[str], *, on_line,
                       env: dict[str, str] | None = None) -> int:
    controller.check_cancelled()
    process = subprocess.Popen(
        command, cwd=APP_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env or {**os.environ,
        "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}, bufsize=1,
    )
    registered = False
    stopped = False
    try:
        controller.register(process)
        registered = True
        assert process.stdout is not None
        for raw_line in process.stdout:
            controller.check_cancelled()
            on_line(raw_line.rstrip())
        result = process.wait()
        stopped = True
        return result
    except Exception as exc:
        try:
            controller.stop_owned()
            stopped = True
            process.wait()
        except Exception as stop_error:
            raise OwnedProcessStopError("The owned process tree could not be stopped.") from stop_error
        raise
    finally:
        if process.stdout is not None and hasattr(process.stdout, "close"):
            process.stdout.close()
        if registered and (stopped or process.poll() is not None):
            controller.unregister(process)


def _run_simple_job(job_id: str, source_path: str) -> None:
    acquired = False
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None or job.status in {"completed", "failed", "cancelled", "interrupted"}:
            return
        simple_worker_states[job_id] = "running"
        controller = job_process_controllers.get(job_id)
        if controller is None:
            controller = JobProcessController()
            job_process_controllers[job_id] = controller
    try:
        while not transcription_semaphore.acquire(timeout=0.1):
            controller.check_cancelled()
        acquired = True
        controller.check_cancelled()
        job = get_job(job_id)
        update_job(job_id, status="running", stage="preparation")
        controller.check_cancelled()
        _checkpoint_update(job_id, "preparation", "active")

        preparation_error: str | None = None

        def on_preparation_line(line: str) -> None:
            nonlocal preparation_error
            if line.startswith(PREPARATION_EVENT_PREFIX):
                payload = json.loads(line[len(PREPARATION_EVENT_PREFIX):])
                if payload.get("type") == "progress":
                    _simple_stage_event(job_id, "preparation", payload.get("status", "active"))
                    _simple_progress(job_id, "preparation", payload)
                elif payload.get("type") == "error":
                    preparation_error = payload.get("message")
            elif line:
                append_log(job_id, line)

        prepare_command = [sys.executable, "-m", "services.resource_preparation",
                           "--model", job.settings["model"], "--device", job.settings["device"]]
        if job.settings.get("restorePunctuation"):
            prepare_command.append("--restore-punctuation")
        prepare_env = os.environ.copy()
        prepare_env.pop("HF_HUB_OFFLINE", None)
        prepare_env.pop("TRANSFORMERS_OFFLINE", None)
        prepare_env["PYTHONIOENCODING"] = "utf-8"
        prepare_env.setdefault("PYTHONUTF8", "1")
        prep_code = _run_owned_process(controller, prepare_command, on_line=on_preparation_line,
                                        env=prepare_env)
        controller.check_cancelled()
        if prep_code != 0:
            raise ResourcePreparationError(preparation_error or "Не удалось подготовить ресурсы. Повторите попытку.")
        _checkpoint_update(job_id, "preparation", "completed")
        _simple_progress(job_id, "preparation", {"percent": 100, "ready": True})

        controller.check_cancelled()
        command = build_command(job, source_path)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("PYTHONUTF8", "1")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        gpu_index = job.settings.get("_gpuIndex")
        if gpu_index is None:
            env.pop("VT_VULKAN_DEVICE_INDEX", None)
        else:
            env["VT_VULKAN_DEVICE_INDEX"] = str(gpu_index)
        update_job(job_id, status="running", stage="source")
        controller.check_cancelled()
        _simple_stage_event(job_id, "source", "active")
        _simple_progress(job_id, "source", {})
        append_log(job_id, f"$ {' '.join(command)}")

        pipeline_error: str | None = None

        def on_pipeline_line(line: str) -> None:
            nonlocal pipeline_error
            if not line:
                return
            if not line.startswith(JSON_EVENT_PREFIX):
                append_log(job_id, line)
                return
            controller.check_cancelled()
            payload = json.loads(line[len(JSON_EVENT_PREFIX):])
            if payload["type"] == "stage":
                stage = payload.get("stage", "recognition")
                update_job(job_id, stage=stage)
                _simple_stage_event(job_id, stage, payload.get("status"))
                _simple_progress(job_id, stage, {
                    "percent": 100 if payload.get("status") == "completed" else None,
                    "message": payload.get("message"),
                })
            elif payload["type"] == "progress":
                stage = payload.get("stage", "recognition")
                update_job(job_id, stage=stage)
                _simple_stage_event(job_id, stage, payload.get("status", "active"))
                _simple_progress(job_id, stage, payload)
            elif payload["type"] == "result":
                apply_result(job_id, payload["result"])
                stage = payload.get("stage", "result")
                update_job(job_id, stage=stage)
                _simple_stage_event(job_id, stage, payload.get("status", "completed"))
                _simple_progress(job_id, stage, {"percent": 100})
            elif payload["type"] == "complete":
                apply_complete(job_id, payload.get("results", []))
                update_job(job_id, stage=payload.get("stage", "result"))
                _finish_simple_checkpoints(job_id)
                _simple_stage_event(job_id, "result", "completed")
            elif payload["type"] == "error":
                pipeline_error = payload.get("message")
                stage = payload.get("stage", "recognition")
                update_job(job_id, stage=stage, error=pipeline_error)
                _simple_stage_event(job_id, stage, "failed")
                append_log(job_id, f"[error] {pipeline_error or 'Unknown error'}")

        return_code = _run_owned_process(controller, command, on_line=on_pipeline_line, env=env)
        controller.check_cancelled()
        if return_code != 0:
            raise RuntimeError(pipeline_error or f"whisper-gpu.py exited with code {return_code}")
        current = get_job(job_id)
        if current.output_path is None:
            raise RuntimeError("Transcription finished without an output file")
        if not current.result_text:
            raise RuntimeError("Расшифровка завершилась без текста.")
        controller.check_cancelled()
        if not _finalize_simple_job(job_id, "completed", transcript=current.result_text):
            controller.check_cancelled()
            raise JobCancelled("This job was cancelled before results were published.")
    except JobCancelled:
        try:
            controller.stop_owned()
        except Exception:
            _leave_simple_job_cancelling(job_id)
        else:
            _finalize_simple_job(job_id, "cancelled")
    except OwnedProcessStopError:
        _leave_simple_job_cancelling(job_id)
    except Exception as exc:
        with jobs_lock:
            cancellation_pending = controller.cancelled or (jobs.get(job_id) is not None
                                                              and jobs[job_id].status == "cancelling")
        if cancellation_pending:
            try:
                controller.stop_owned()
            except Exception:
                _leave_simple_job_cancelling(job_id)
            else:
                _finalize_simple_job(job_id, "cancelled")
        else:
            _finalize_simple_job(job_id, "failed", error=str(exc))
    finally:
        if acquired:
            transcription_semaphore.release()
        with jobs_lock:
            simple_worker_states[job_id] = "finished"
            job = jobs.get(job_id)
            if job is not None and job.status in {"completed", "failed", "cancelled"}:
                job_process_controllers.pop(job_id, None)


def serialize_job(job: JobState) -> dict[str, Any]:
    payload = asdict(job)
    payload["downloadUrl"] = f"/api/jobs/{job.id}/download" if job.output_path else None
    return payload


def get_model_download_job(job_id: str) -> ModelOperationJobState:
    with model_operation_jobs_lock:
        job = model_download_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Model download job not found")
        return job


def get_model_delete_job(job_id: str) -> ModelOperationJobState:
    with model_operation_jobs_lock:
        job = model_delete_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Model delete job not found")
        return job


def serialize_model_operation_job(job: ModelOperationJobState) -> dict[str, Any]:
    return {
        "jobId": job.job_id,
        "modelId": job.model_id,
        "status": job.status,
        "progress": {
            "percent": job.progress.percent,
            "label": job.progress.label,
        },
        "message": job.message,
        "error": job.error,
    }


def serialize_model_download_job(job: ModelOperationJobState) -> dict[str, Any]:
    return serialize_model_operation_job(job)


def serialize_model_delete_job(job: ModelOperationJobState) -> dict[str, Any]:
    return serialize_model_operation_job(job)


def update_model_download_job(
    job_id: str,
    *,
    status: str | None = None,
    percent: int | None = None,
    label: str | None = None,
    message: str | None = None,
    error: str | None = None,
) -> None:
    with model_operation_jobs_lock:
        job = model_download_jobs[job_id]
        if status is not None:
            job.status = status
        if percent is not None:
            job.progress.percent = max(0, min(100, percent))
        if label is not None:
            job.progress.label = label
        if message is not None:
            job.message = message
        job.error = error
        job.updated_at = utc_now()


def update_model_delete_job(
    job_id: str,
    *,
    status: str | None = None,
    percent: int | None = None,
    label: str | None = None,
    message: str | None = None,
    error: str | None = None,
) -> None:
    with model_operation_jobs_lock:
        job = model_delete_jobs[job_id]
        if status is not None:
            job.status = status
        if percent is not None:
            job.progress.percent = max(0, min(100, percent))
        if label is not None:
            job.progress.label = label
        if message is not None:
            job.message = message
        job.error = error
        job.updated_at = utc_now()


def get_running_model_operation() -> ActiveModelOperationState | None:
    with model_operation_jobs_lock:
        if active_model_operation is None:
            return None

        if active_model_operation.operation_type == "download":
            job = model_download_jobs.get(active_model_operation.job_id)
        else:
            job = model_delete_jobs.get(active_model_operation.job_id)

        if job is None or job.status not in {"queued", "running"}:
            return None

        return active_model_operation


def reserve_model_operation_slot(operation_type: str, model_id: str) -> str:
    global active_model_operation

    with model_operation_jobs_lock:
        if active_model_operation is not None:
            if active_model_operation.operation_type == "download":
                active_job = model_download_jobs.get(active_model_operation.job_id)
            else:
                active_job = model_delete_jobs.get(active_model_operation.job_id)

            if active_job is not None and active_job.status in {"queued", "running"}:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Model {active_model_operation.operation_type} for '{active_job.model_id}' is already in progress. "
                        "Only one model operation can run at a time."
                    ),
                )

        job_id = uuid.uuid4().hex
        active_model_operation = ActiveModelOperationState(
            operation_type=operation_type,
            job_id=job_id,
            model_id=model_id,
        )
        if operation_type == "download":
            model_download_jobs[job_id] = ModelOperationJobState(job_id=job_id, model_id=model_id)
        else:
            model_delete_jobs[job_id] = ModelOperationJobState(
                job_id=job_id,
                model_id=model_id,
                message="Delete queued.",
            )
        return job_id


def release_model_operation_slot(job_id: str, operation_type: str) -> None:
    global active_model_operation

    with model_operation_jobs_lock:
        if (
            active_model_operation is not None
            and active_model_operation.job_id == job_id
            and active_model_operation.operation_type == operation_type
        ):
            active_model_operation = None


def run_model_download_job(job_id: str) -> None:
    try:
        job = get_model_download_job(job_id)
        update_model_download_job(
            job_id,
            status="running",
            percent=5,
            label="Checking cache",
            message="Checking whether the model is already available locally.",
            error=None,
        )

        result = download_registered_model(
            job.model_id,
            progress_callback=lambda percent, label: update_model_download_job(
                job_id,
                status="running",
                percent=percent,
                label=label,
                message=label,
                error=None,
            ),
        )

        message = (
            "Model already exists in the local Hugging Face cache."
            if result["alreadyDownloaded"]
            else "Model downloaded and validated in the local Hugging Face cache."
        )
        update_model_download_job(
            job_id,
            status="success",
            percent=100,
            label="Complete",
            message=message,
            error=None,
        )
    except (ModelRegistryLookupError, ModelCacheInspectionError, ModelDownloadValidationError) as exc:
        update_model_download_job(
            job_id,
            status="error",
            label="Download failed",
            message="Model download failed.",
            error=str(exc),
        )
    except Exception as exc:
        update_model_download_job(
            job_id,
            status="error",
            label="Download failed",
            message="Model download failed.",
            error=str(exc),
        )
    finally:
        release_model_operation_slot(job_id, "download")


def run_model_delete_job(job_id: str) -> None:
    try:
        job = get_model_delete_job(job_id)
        update_model_delete_job(
            job_id,
            status="running",
            percent=5,
            label="Checking cache",
            message="Checking whether the model is available locally.",
            error=None,
        )

        result = delete_registered_model(
            job.model_id,
            progress_callback=lambda percent, label: update_model_delete_job(
                job_id,
                status="running",
                percent=percent,
                label=label,
                message=label,
                error=None,
            ),
        )

        message = (
            "Model is not present in the local Hugging Face cache. Nothing was removed."
            if result["alreadyDeleted"]
            else "Model removed from the local Hugging Face cache."
        )
        update_model_delete_job(
            job_id,
            status="success",
            percent=100,
            label="Complete",
            message=message,
            error=None,
        )
    except (ModelRegistryLookupError, ModelCacheInspectionError, ModelDeleteValidationError) as exc:
        update_model_delete_job(
            job_id,
            status="error",
            label="Delete failed",
            message="Model delete failed.",
            error=str(exc),
        )
    except Exception as exc:
        update_model_delete_job(
            job_id,
            status="error",
            label="Delete failed",
            message="Model delete failed.",
            error=str(exc),
        )
    finally:
        release_model_operation_slot(job_id, "delete")


def read_output_text(output_path: str | None) -> str | None:
    if not output_path:
        return None

    path = Path(output_path)
    if not path.exists() or not path.is_file():
        return None

    return path.read_text(encoding="utf-8-sig")


def apply_result(job_id: str, result: dict[str, Any]) -> None:
    rendered_result = read_output_text(result.get("output_path")) or result.get("rendered_result")
    update_job(
        job_id,
        output_path=result.get("output_path"),
        output_format=result.get("output_format"),
        result_text=result.get("text"),
        rendered_result=rendered_result,
        detected_language=result.get("detected_language"),
        notice=result.get("notice"),
        saved_audio=result.get("saved_audio"),
        effective_device=result.get("effective_device"),
        effective_backend=result.get("effective_backend"),
    )


def apply_complete(job_id: str, results: list[dict[str, Any]]) -> None:
    if not results:
        return

    combined_text = "\n\n".join(result.get("text", "").strip() for result in results if result.get("text")).strip()
    rendered_chunks = []
    saved_audio = next((result.get("saved_audio") for result in reversed(results) if result.get("saved_audio")), None)
    for result in results:
        rendered_result = read_output_text(result.get("output_path")) or result.get("rendered_result")
        if rendered_result:
            rendered_chunks.append(rendered_result.strip())

    update_job(
        job_id,
        results=results,
        result_text=combined_text or None,
        rendered_result="\n\n".join(rendered_chunks).strip() or None,
        output_path=results[-1].get("output_path"),
        output_format=results[-1].get("output_format"),
        detected_language=results[-1].get("detected_language"),
        notice=results[-1].get("notice"),
        saved_audio=saved_audio,
        effective_device=results[-1].get("effective_device"),
        effective_backend=results[-1].get("effective_backend"),
    )


def build_command(job: JobState, source_path: str) -> list[str]:
    settings = job.settings
    command = [
        sys.executable,
        "whisper-gpu.py",
        "-f",
        source_path,
        "-od",
        str(Path(job.settings.get("_outputDir") or (RUNS_DIR / job.id / "output"))),
        "-l",
        settings["language"],
        "-s",
        settings["model"],
        "-d",
        settings["device"],
        "--output_format",
        settings["outputFormat"],
        "--web_json",
    ]

    if settings["device"] == "cpu":
        command.extend(["-n", str(settings["nproc"])])

    if settings["saveAudio"]:
        command.append("-k")

    if not settings["useTimestamps"] and settings["outputFormat"] != "srt":
        command.append("--no_timestamps")

    if settings.get("restorePunctuation") and (settings["language"] == "ru" or settings.get("_simple")):
        command.append("--restore_punctuation")
    if settings.get("_simple"):
        command.append("--single-source")

    return command


def run_job(job_id: str, source_path: str) -> None:
    try:
        initial_job = get_job(job_id)
    except HTTPException:
        return
    if initial_job.settings.get("_simple"):
        _run_simple_job(job_id, source_path)
        return
    acquired = False
    try:
        job = get_job(job_id)
        append_log(job_id, "Waiting for the transcription runtime to become available…")
        transcription_semaphore.acquire()
        acquired = True
        job_dir = RUNS_DIR / job_id
        output_dir = Path(job.settings.get("_outputDir") or (job_dir / "output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        if job.settings.get("_simple"):
            update_job(job_id, status="running", stage="preparation")
            _checkpoint_update(job_id, "preparation", "active")

            def preparation_progress(payload: dict) -> None:
                total = payload.get("totalBytes")
                completed = payload.get("completedBytes")
                percent = min(99, int(completed * 100 / total)) if total and completed is not None else None
                _simple_progress(job_id, "preparation", {**payload, "percent": percent})

            ensure_selected_resources(job.settings, on_progress=preparation_progress)
            _checkpoint_update(job_id, "preparation", "completed")
            _simple_progress(job_id, "preparation", {"percent": 100, "ready": True})
        command = build_command(job, source_path)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("PYTHONUTF8", "1")
        if job.settings.get("_simple"):
            gpu_index = job.settings.get("_gpuIndex")
            if gpu_index is None:
                env.pop("VT_VULKAN_DEVICE_INDEX", None)
            else:
                env["VT_VULKAN_DEVICE_INDEX"] = str(gpu_index)
        update_job(job_id, status="running", stage="source")
        if job.settings.get("_simple"):
            _simple_stage_event(job_id, "source", "active")
            _simple_progress(job_id, "source", {})
        append_log(job_id, f"$ {' '.join(command)}")

        process = subprocess.Popen(
            command,
            cwd=APP_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue

            if line.startswith(JSON_EVENT_PREFIX):
                payload = json.loads(line[len(JSON_EVENT_PREFIX) :])
                if payload["type"] == "stage":
                    stage = payload.get("stage", "recognition")
                    update_job(job_id, stage=stage)
                    if job.settings.get("_simple"):
                        _simple_stage_event(job_id, stage, payload.get("status"))
                        _simple_progress(job_id, stage, {
                            "percent": 100 if payload.get("status") == "completed" else None,
                        })
                    if payload.get("message"):
                        append_log(job_id, f"[{payload.get('stage', 'stage')}] {payload['message']}")
                elif payload["type"] == "progress":
                    stage = payload.get("stage", "recognition")
                    update_job(job_id, stage=stage)
                    if job.settings.get("_simple"):
                        _simple_stage_event(job_id, stage, payload.get("status", "active"))
                        _simple_progress(job_id, stage, payload)
                elif payload["type"] == "result":
                    apply_result(job_id, payload["result"])
                    update_job(job_id, stage=payload.get("stage", "result"))
                    if job.settings.get("_simple"):
                        stage = payload.get("stage", "result")
                        _simple_stage_event(job_id, stage, payload.get("status", "completed"))
                        _simple_progress(job_id, stage, {"percent": 100})
                    if payload.get("message"):
                        append_log(job_id, f"[result] {payload['message']}")
                elif payload["type"] == "complete":
                    apply_complete(job_id, payload.get("results", []))
                    update_job(job_id, stage=payload.get("stage", "result"))
                    if job.settings.get("_simple"):
                        _finish_simple_checkpoints(job_id)
                        _simple_stage_event(job_id, "result", "completed")
                    if payload.get("message"):
                        append_log(job_id, f"[result] {payload['message']}")
                elif payload["type"] == "error":
                    update_job(job_id, stage=payload.get("stage", "result"), error=payload.get("message"))
                    if job.settings.get("_simple"):
                        _simple_stage_event(job_id, payload.get("stage", "recognition"), "failed")
                    append_log(job_id, f"[error] {payload.get('message', 'Unknown error')}")
                continue

            append_log(job_id, line)

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"whisper-gpu.py exited with code {return_code}")

        current = get_job(job_id)
        if current.output_path is None:
            raise RuntimeError("Transcription finished without an output file")

        if current.settings.get("_simple"):
            if not current.result_text:
                raise RuntimeError("Расшифровка завершилась без текста.")
            history_store.write_text(job_id, current.result_text, revision=0)
            update_job(job_id, stage="result")
            _simple_progress(job_id, "result", {"percent": 100, "message": "Расшифровка завершена"})
            _simple_stage_event(job_id, "result", "completed")
            _finish_simple_checkpoints(job_id)

        update_job(job_id, status="completed", stage="result")

    except Exception as exc:
        try:
            append_log(job_id, f"[error] {exc}")
            current_error = get_job(job_id).error
            job = get_job(job_id)
            message = (str(exc) if isinstance(exc, ResourcePreparationError) else current_error or str(exc))
            update_job(job_id, status="failed", stage=job.stage, error=message)
            if job.settings.get("_simple"):
                _simple_stage_event(job_id, job.stage, "failed")
                friendly = concise_simple_error(message)
                _simple_progress(job_id, job.stage, {"message": friendly})
        except (KeyError, HTTPException):
            pass
    finally:
        if acquired:
            transcription_semaphore.release()


@app.get("/api/health")
def health_check():
    return {"status": "ok", "appId": "video-transcribe"}


@app.get("/api/options")
def get_options():
    max_cpu_threads = logical_cpu_count()
    punctuation_available = bool(importlib.util.find_spec("sbert_punc_case_ru") and importlib.util.find_spec("razdel"))
    return {
        "models": list_enabled_backend_values(),
        "devices": ["auto", "cpu", "vulkan", "cuda"],
        "defaultDevice": "auto",
        "deviceDetails": get_device_details(),
        "vulkanModels": VULKAN_MODELS,
        "punctuationAvailable": punctuation_available,
        "modelsDirectory": str(MODEL_STORAGE_ROOT),
        "outputFormats": ["txt", "srt", "json"],
        "languages": ["auto", "ru", "en", "de", "fr", "es", "it", "pt", "uk", "ja", "ko", "zh"],
        "maxCpuThreads": max_cpu_threads,
    }


def _simple_public_url(raw: str) -> str:
    value = raw.strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="Введите полную публичную ссылку HTTP или HTTPS.")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".internal", ".localhost")):
        raise HTTPException(status_code=400, detail="Ссылка должна вести на публичный сайт.")
    try:
        address = ipaddress.ip_address(host)
        if not address.is_global:
            raise HTTPException(status_code=400, detail="Ссылка должна вести на публичный сайт.")
    except ValueError:
        if "." not in host or host.startswith("."):
            raise HTTPException(status_code=400, detail="Введите публичный домен в ссылке.")
    return value


def _simple_shape(record: dict[str, Any], *, with_text: bool) -> dict[str, Any]:
    files = []
    for ext in ("txt", "md"):
        path = history_store.artifact(record["id"], ext)
        if record["status"] == "completed" and path.is_file():
            files.append({"format": ext, "name": path.name,
                          "downloadUrl": f"/api/simple/jobs/{record['id']}/download?format={ext}"})
    text = history_store.read_text(record["id"]) if with_text and record["status"] == "completed" else None
    result = {
        "id": record["id"], "sourceType": record["sourceType"], "source": record["source"],
        "profile": record["profile"], "device": record["device"], "cpuThreads": record["cpuThreads"],
        "restorePunctuation": record["restorePunctuation"], "status": record["status"],
        "stage": record["stage"], "createdAt": record["createdAt"], "updatedAt": record["updatedAt"],
        "detectedLanguage": record.get("detectedLanguage"), "error": concise_simple_error(record.get("error")),
        "notice": record.get("notice"), "revision": record.get("revision", 0), "files": files,
        "progress": record.get("progress"), "checkpoints": record.get("checkpoints", []),
    }
    if with_text:
        result["text"] = text
    return result


@app.get("/api/simple/options")
def simple_options():
    return simple_profiles.options(logical_cpu_count())


@app.get("/api/simple/session")
def simple_session():
    record = history_store.latest()
    return {"job": _simple_shape(record, with_text=True) if record else None}


@app.get("/api/simple/jobs/{job_id}")
def get_simple_job(job_id: str):
    record = history_store.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Расшифровка не найдена.")
    return _simple_shape(record, with_text=True)


@app.get("/api/simple/history")
def simple_history():
    return {"items": [_simple_shape(record, with_text=False) for record in history_store.list()]}


@app.post("/api/simple/pick-file")
def simple_pick_file():
    try:
        selected = native_file_picker.pick_file()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not selected:
        return {"path": None, "name": None}
    path = Path(selected).expanduser().resolve()
    if not path.is_file():
        raise HTTPException(status_code=400, detail="Выбранный файл больше не существует.")
    return {"path": str(path), "name": path.name}


@app.post("/api/simple/jobs", status_code=202)
def create_simple_job(payload: SimpleJobRequest):
    if payload.sourceType not in {"url", "file"}:
        raise HTTPException(status_code=400, detail="sourceType должен быть url или file.")
    if payload.sourceType == "url":
        source = _simple_public_url(payload.url or "")
    else:
        if not payload.path:
            raise HTTPException(status_code=400, detail="Выберите локальный файл.")
        candidate = Path(payload.path).expanduser()
        if not candidate.is_absolute():
            raise HTTPException(status_code=400, detail="Путь к файлу должен быть абсолютным.")
        path = candidate.resolve()
        if not path.is_file():
            raise HTTPException(status_code=400, detail="Выбранный файл не найден.")
        source = str(path)
    maximum = logical_cpu_count()
    if payload.cpuThreads < 1 or payload.cpuThreads > maximum:
        raise HTTPException(status_code=400, detail=f"cpuThreads должен быть от 1 до {maximum}.")
    try:
        model, backend_device = simple_profiles.resolve(payload.profile, payload.device)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex
    settings = {"_simple": True, "_outputDir": str(history_store.job_dir(job_id) / "output"),
                "sourceType": payload.sourceType, "language": "auto", "model": model,
                "device": backend_device, "nproc": payload.cpuThreads, "outputFormat": "txt",
                "saveAudio": False, "useTimestamps": False,
                "restorePunctuation": payload.restorePunctuation}
    if backend_device == "vulkan":
        selected_gpu = simple_profiles.gpu_details()
        settings["_gpuIndex"] = selected_gpu["deviceIndex"]
    metadata = {"id": job_id, "sourceType": payload.sourceType, "source": source,
                "profile": payload.profile, "device": payload.device, "cpuThreads": payload.cpuThreads,
                "restorePunctuation": payload.restorePunctuation,
                "settings": {k: settings[k] for k in ("language", "model", "device", "nproc", "outputFormat", "saveAudio", "useTimestamps", "restorePunctuation")},
                "artifactPaths": {ext: str(history_store.artifact(job_id, ext)) for ext in ("txt", "md")}}
    with jobs_lock:
        if any(item.status in {"queued", "running", "cancelling"} for item in jobs.values()):
            raise HTTPException(status_code=409, detail="Другая расшифровка уже выполняется.")
        record = history_store.create(metadata)
        job = JobState(id=job_id, settings=settings, source_name=source)
        jobs[job_id] = job
        job_process_controllers[job_id] = JobProcessController()
        simple_worker_states[job_id] = "not_started"
    worker = threading.Thread(target=run_job, args=(job_id, source), daemon=True)
    worker.start()
    return _simple_shape(record, with_text=True)


@app.post("/api/simple/jobs/{job_id}/cancel", status_code=202)
def cancel_simple_job(job_id: str):
    record = _cancel_job(job_id)
    return _simple_shape(record, with_text=True)


@app.put("/api/simple/jobs/{job_id}/text")
def edit_simple_text(job_id: str, payload: SimpleTextRequest):
    try:
        record = history_store.replace_text(job_id, payload.text, payload.baseRevision)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Расшифровка не найдена.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"text": payload.text, "revision": record["revision"]}


@app.get("/api/simple/jobs/{job_id}/download")
def download_simple_text(job_id: str, format: str = "txt"):
    try:
        path = history_store.artifact(job_id, format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Формат должен быть txt или md.") from exc
    record = history_store.get(job_id)
    if not record or record["status"] != "completed" or not path.is_file():
        raise HTTPException(status_code=404, detail="Файл расшифровки не найден.")
    return FileResponse(path=path, filename=f"transcript.{format}", media_type="text/plain; charset=utf-8")


@app.delete("/api/simple/history/{job_id}", status_code=204)
def delete_simple_history(job_id: str):
    with jobs_lock:
        record = history_store.get(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="Расшифровка не найдена.")
        if record["status"] in {"queued", "running", "cancelling"}:
            raise HTTPException(status_code=409, detail="Нельзя удалить выполняющуюся расшифровку.")
        try:
            history_store.delete(job_id)
        except PermissionError as exc:
            raise HTTPException(status_code=409, detail="Нельзя удалить выполняющуюся расшифровку.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Расшифровка не найдена.") from exc
        jobs.pop(job_id, None)
        job_process_controllers.pop(job_id, None)
        simple_worker_states.pop(job_id, None)
    return Response(status_code=204)


@model_management_router.get("/catalog")
def get_models_catalog():
    return list_model_catalog()


@model_management_router.get("/status")
def get_models_status():
    try:
        return list_models_with_status()
    except ModelCacheInspectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@model_management_router.get("/panel")
def get_models_panel():
    try:
        return get_model_panel_payload()
    except ModelCacheInspectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@model_management_router.post("/download")
def start_model_download(payload: ModelDownloadRequest):
    try:
        entry = get_model_registry_entry(payload.modelId)
    except ModelRegistryLookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not entry.enabled:
        raise HTTPException(status_code=400, detail=f"Model '{payload.modelId}' is disabled.")

    try:
        current_status = get_model_status(payload.modelId)
    except ModelCacheInspectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    running_operation = get_running_model_operation()
    if running_operation is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Model {running_operation.operation_type} for '{running_operation.model_id}' is already in progress. "
                "Only one model operation can run at a time."
            ),
        )

    if current_status["status"] == "downloaded":
        job_id = uuid.uuid4().hex
        job = ModelOperationJobState(
            job_id=job_id,
            model_id=payload.modelId,
            status="success",
            progress=ModelDownloadProgressState(percent=100, label="Complete"),
            message="Model already exists in the local Hugging Face cache.",
            error=None,
        )
        with model_operation_jobs_lock:
            model_download_jobs[job_id] = job
        return serialize_model_download_job(job)

    job_id = reserve_model_operation_slot("download", payload.modelId)
    worker = threading.Thread(target=run_model_download_job, args=(job_id,), daemon=True)
    worker.start()
    return serialize_model_download_job(get_model_download_job(job_id))


@model_management_router.get("/download/{job_id}")
def get_model_download_status(job_id: str):
    return serialize_model_download_job(get_model_download_job(job_id))


@model_management_router.post("/delete")
def start_model_delete(payload: ModelDeleteRequest):
    try:
        entry = get_model_registry_entry(payload.modelId)
    except ModelRegistryLookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not entry.enabled:
        raise HTTPException(status_code=400, detail=f"Model '{payload.modelId}' is disabled.")

    with jobs_lock:
        active_same_model = False
        for job in jobs.values():
            if job.status not in {"queued", "running"}:
                continue
            try:
                active_entry = get_model_registry_entry(job.settings.get("model", ""))
            except ModelRegistryLookupError:
                continue
            if active_entry.hfRepoId == entry.hfRepoId:
                active_same_model = True
                break
    if active_same_model:
        raise HTTPException(status_code=409, detail=f"Model '{payload.modelId}' is in use by an active transcription.")

    try:
        current_status = get_model_status(payload.modelId)
    except ModelCacheInspectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    running_operation = get_running_model_operation()
    if running_operation is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Model {running_operation.operation_type} for '{running_operation.model_id}' is already in progress. "
                "Only one model operation can run at a time."
            ),
        )

    if current_status["status"] == "unknown":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Model '{payload.modelId}' has an unknown cache state. "
                "Refusing to delete a model that cannot be verified as downloaded."
            ),
        )

    if current_status["status"] == "not_downloaded":
        job_id = uuid.uuid4().hex
        job = ModelOperationJobState(
            job_id=job_id,
            model_id=payload.modelId,
            status="success",
            progress=ModelDownloadProgressState(percent=100, label="Complete"),
            message="Model is not present in the local Hugging Face cache. Nothing was removed.",
            error=None,
        )
        with model_operation_jobs_lock:
            model_delete_jobs[job_id] = job
        return serialize_model_delete_job(job)

    job_id = reserve_model_operation_slot("delete", payload.modelId)
    worker = threading.Thread(target=run_model_delete_job, args=(job_id,), daemon=True)
    worker.start()
    return serialize_model_delete_job(get_model_delete_job(job_id))


@model_management_router.get("/delete/{job_id}")
def get_model_delete_status(job_id: str):
    return serialize_model_delete_job(get_model_delete_job(job_id))


@app.post("/api/check-source")
def check_source_endpoint(payload: SourceCheckRequest):
    return check_source(payload.url)


@app.post("/api/transcriptions")
async def create_transcription(
    sourceType: str = Form(...),
    url: str | None = Form(None),
    file: UploadFile | None = File(None),
    language: str = Form("ru"),
    model: str = Form("small"),
    device: str = Form("auto"),
    nproc: str | None = Form(None),
    outputFormat: str = Form("srt"),
    saveAudio: str = Form("false"),
    useTimestamps: str = Form("true"),
    restorePunctuation: str = Form("false"),
):
    normalized_device = device.strip().lower()
    if normalized_device not in {"auto", "cpu", "vulkan", "cuda"}:
        raise HTTPException(status_code=400, detail="device must be one of: auto, cpu, vulkan, cuda")
    normalized_model = model.strip()
    try:
        model_entry = get_model_registry_entry(normalized_model)
    except ModelRegistryLookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    normalized_language = language.strip().lower()
    valid_languages = {"auto", "ru", "en", "de", "fr", "es", "it", "pt", "uk", "ja", "ko", "zh"}
    if normalized_language not in valid_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported language '{normalized_language}'.")
    if model_entry.languageScope == "english" and normalized_language not in {"auto", "en"}:
        raise HTTPException(status_code=400, detail=f"Model '{normalized_model}' supports English only; choose English or another model.")
    normalized_format = outputFormat.strip().lower()
    if normalized_format not in {"txt", "srt", "json"}:
        raise HTTPException(status_code=400, detail="outputFormat must be txt, srt, or json")
    if normalized_device == "vulkan" and normalized_model not in VULKAN_MODELS:
        raise HTTPException(status_code=400, detail=(
            f"Model '{normalized_model}' is not supported by Vulkan. Available Vulkan models: {', '.join(VULKAN_MODELS)}"
        ))
    if normalized_language == "ru" and parse_bool(restorePunctuation, default=False) and not (
        importlib.util.find_spec("sbert_punc_case_ru") and importlib.util.find_spec("razdel")
    ):
        raise HTTPException(status_code=400, detail="Russian punctuation restoration is unavailable. Install the optional punctuation package.")

    if sourceType not in {"url", "file"}:
        raise HTTPException(status_code=400, detail="sourceType must be 'url' or 'file'")

    if sourceType == "url" and not (url or "").strip():
        raise HTTPException(status_code=400, detail="URL is required")
    if sourceType == "url":
        parsed_url = urlsplit((url or "").strip())
        if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.hostname:
            raise HTTPException(status_code=400, detail="URL must be an absolute http or https URL")

    if sourceType == "file" and file is None:
        raise HTTPException(status_code=400, detail="File is required")

    job_id = uuid.uuid4().hex
    job_dir = RUNS_DIR / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    source_path: str
    source_name: str

    if sourceType == "file" and file is not None:
        source_name = safe_filename(file.filename, fallback="upload.bin")
        target_file = input_dir / source_name
        with target_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        await file.close()
        source_path = str(target_file)
    else:
        source_name = (url or "").strip()
        source_path = source_name

    max_cpu_threads = logical_cpu_count()
    settings = {
        "sourceType": sourceType,
        "url": (url or "").strip(),
        "language": normalized_language,
        "model": normalized_model,
        "device": normalized_device,
        "nproc": parse_int(nproc, default=max_cpu_threads, minimum=1, maximum=max_cpu_threads),
        "outputFormat": normalized_format,
        "saveAudio": parse_bool(saveAudio, default=False),
        "useTimestamps": parse_bool(useTimestamps, default=True),
        "restorePunctuation": parse_bool(restorePunctuation, default=False) and (normalized_language == "ru"),
    }

    with jobs_lock:
        if any(item.status in {"queued", "running", "cancelling"} for item in jobs.values()):
            raise HTTPException(status_code=409, detail="Another transcription is already running.")
        job = JobState(id=job_id, settings=settings, source_name=source_name)
        jobs[job_id] = job

    worker = threading.Thread(target=run_job, args=(job_id, source_path), daemon=True)
    worker.start()

    return serialize_job(job)


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    return serialize_job(get_job(job_id))


@app.get("/api/jobs/{job_id}/download")
def download_result(job_id: str):
    job = get_job(job_id)
    if not job.output_path:
        raise HTTPException(status_code=404, detail="Result file is not ready")

    output_path = Path(job.output_path)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Result file is missing")

    return FileResponse(path=output_path, filename=output_path.name, media_type="application/octet-stream")


app.include_router(model_management_router)


UI_DIST = APP_ROOT / "ui" / "dist"
if UI_DIST.is_dir() and (UI_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="ui-assets")


@app.get("/")
@app.get("/models")
def serve_ui():
    index = UI_DIST / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="UI build is unavailable")
    return FileResponse(index)


@app.get("/{asset_path:path}")
def serve_ui_asset(asset_path: str):
    # Keep unknown API endpoints as JSON 404s rather than falling through to SPA HTML.
    if asset_path == "api" or asset_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    candidate = (UI_DIST / asset_path).resolve()
    try:
        candidate.relative_to(UI_DIST.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found")
    if candidate.is_file():
        return FileResponse(candidate)
    # Browser-side routes use the built SPA entry point.
    index = UI_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="UI build is unavailable")
