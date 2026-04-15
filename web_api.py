import json
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from faster_whisper.utils import available_models

APP_ROOT = Path(__file__).resolve().parent
RUNS_DIR = APP_ROOT / "web_runs"
JSON_EVENT_PREFIX = "__VT_JSON__ "


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

app = FastAPI(title="Video Transcribe Local API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def apply_result(job_id: str, result: dict[str, Any]) -> None:
    update_job(
        job_id,
        output_path=result.get("output_path"),
        output_format=result.get("output_format"),
        result_text=result.get("text"),
        rendered_result=result.get("rendered_result"),
        detected_language=result.get("detected_language"),
        saved_audio=result.get("saved_audio"),
    )


def apply_complete(job_id: str, results: list[dict[str, Any]]) -> None:
    if not results:
        return

    combined_text = "\n\n".join(result.get("text", "").strip() for result in results if result.get("text")).strip()
    rendered_chunks = [result.get("rendered_result", "").strip() for result in results if result.get("rendered_result")]
    update_job(
        job_id,
        results=results,
        result_text=combined_text or None,
        rendered_result="\n\n".join(rendered_chunks).strip() or None,
        output_path=results[-1].get("output_path"),
        output_format=results[-1].get("output_format"),
        detected_language=results[-1].get("detected_language"),
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
        update_job(job_id, status="failed", stage="result", error=str(exc))


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/options")
def get_options():
    return {
        "models": available_models(),
        "devices": ["cpu", "cuda"],
        "outputFormats": ["txt", "srt", "json"],
        "languages": ["ru", "en", "de", "fr", "es", "it", "pt", "uk", "ja", "ko", "zh"],
    }


@app.post("/api/transcriptions")
async def create_transcription(
    sourceType: str = Form(...),
    url: str | None = Form(None),
    file: UploadFile | None = File(None),
    language: str = Form("ru"),
    model: str = Form("small"),
    device: str = Form("cpu"),
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

    settings = {
        "sourceType": sourceType,
        "url": (url or "").strip(),
        "language": language.strip() or "ru",
        "model": model.strip() or "small",
        "device": device.strip() or "cpu",
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
