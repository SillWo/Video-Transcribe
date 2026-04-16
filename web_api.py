import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

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
from source_check import check_source

APP_ROOT = Path(__file__).resolve().parent
RUNS_DIR = APP_ROOT / "web_runs"
JSON_EVENT_PREFIX = "__VT_JSON__ "


class SourceCheckRequest(BaseModel):
    url: str


class ModelDownloadRequest(BaseModel):
    modelId: str


class ModelDeleteRequest(BaseModel):
    modelId: str


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


jobs: dict[str, JobState] = {}
jobs_lock = threading.Lock()
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
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = utc_now()


def append_log(job_id: str, line: str) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job.logs.append(line)
        if len(job.logs) > 500:
            job.logs = job.logs[-500:]
        job.updated_at = utc_now()


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
        saved_audio=result.get("saved_audio"),
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
        saved_audio=saved_audio,
    )


def build_command(job: JobState, source_path: str) -> list[str]:
    settings = job.settings
    command = [
        sys.executable,
        "whisper-gpu.py",
        "-f",
        source_path,
        "-od",
        str(RUNS_DIR / job.id / "output"),
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

    return command


def run_job(job_id: str, source_path: str) -> None:
    job_dir = RUNS_DIR / job_id
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    job = get_job(job_id)
    command = build_command(job, source_path)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONUTF8", "1")

    update_job(job_id, status="running", stage="source")
    append_log(job_id, f"$ {' '.join(command)}")

    try:
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
                    update_job(job_id, stage=payload.get("stage", "recognition"))
                    if payload.get("message"):
                        append_log(job_id, f"[{payload.get('stage', 'stage')}] {payload['message']}")
                elif payload["type"] == "result":
                    apply_result(job_id, payload["result"])
                    update_job(job_id, stage=payload.get("stage", "result"))
                    if payload.get("message"):
                        append_log(job_id, f"[result] {payload['message']}")
                elif payload["type"] == "complete":
                    apply_complete(job_id, payload.get("results", []))
                    update_job(job_id, stage=payload.get("stage", "result"))
                    if payload.get("message"):
                        append_log(job_id, f"[result] {payload['message']}")
                elif payload["type"] == "error":
                    update_job(job_id, stage=payload.get("stage", "result"), error=payload.get("message"))
                    append_log(job_id, f"[error] {payload.get('message', 'Unknown error')}")
                continue

            append_log(job_id, line)

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"whisper-gpu.py exited with code {return_code}")

        current = get_job(job_id)
        if current.output_path is None:
            raise RuntimeError("Transcription finished without an output file")

        update_job(job_id, status="completed", stage="result")

    except Exception as exc:
        append_log(job_id, f"[error] {exc}")
        current_error = get_job(job_id).error
        update_job(job_id, status="failed", stage="result", error=current_error or str(exc))


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/options")
def get_options():
    max_cpu_threads = logical_cpu_count()
    return {
        "models": list_enabled_backend_values(),
        "devices": ["cpu", "cuda"],
        "outputFormats": ["txt", "srt", "json"],
        "languages": ["auto", "ru", "en", "de", "fr", "es", "it", "pt", "uk", "ja", "ko", "zh"],
        "maxCpuThreads": max_cpu_threads,
    }


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
    device: str = Form("cpu"),
    nproc: str | None = Form(None),
    outputFormat: str = Form("srt"),
    saveAudio: str = Form("false"),
    useTimestamps: str = Form("true"),
):
    if sourceType not in {"url", "file"}:
        raise HTTPException(status_code=400, detail="sourceType must be 'url' or 'file'")

    if sourceType == "url" and not (url or "").strip():
        raise HTTPException(status_code=400, detail="URL is required")

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
        "language": language.strip() or "ru",
        "model": model.strip() or "small",
        "device": device.strip() or "cpu",
        "nproc": parse_int(nproc, default=max_cpu_threads, minimum=1, maximum=max_cpu_threads),
        "outputFormat": outputFormat.strip() or "srt",
        "saveAudio": parse_bool(saveAudio, default=False),
        "useTimestamps": parse_bool(useTimestamps, default=True),
    }

    job = JobState(id=job_id, settings=settings, source_name=source_name)
    with jobs_lock:
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
