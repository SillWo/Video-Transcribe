"""Prepare only the resources selected by a simple transcription job."""

from __future__ import annotations

import importlib.util
import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Callable

from services.download_progress import create_download_progress_tqdm
from services.model_management import ModelDownloadValidationError, download_registered_model, get_model_status
from services.model_paths import HF_HUB_CACHE
from services.task_errors import TaskFailure
from huggingface_hub import snapshot_download

PUNCTUATION_REPO = "kontur-ai/sbert_punc_case_ru"
PUNCTUATION_ALLOW_PATTERNS = [
    "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.txt", "vocab.json", "merges.txt", "sentencepiece.bpe.model",
    "model.safetensors", "pytorch_model.bin", "model.bin",
]
Progress = Callable[[dict], None]


class ResourcePreparationError(TaskFailure):
    """A selected resource could not be prepared for this job."""

    def __init__(self, message: str, code: str = "model_download_failed"):
        super().__init__(message, code, stage="preparation")


def _emit(progress: Progress | None, *, completed: int | None = None,
          total: int | None = None, message: str) -> None:
    if progress:
        progress({"stage": "preparation", "completedBytes": completed,
                  "totalBytes": total, "message": message})


def _punctuation_dependencies_ready() -> bool:
    try:
        return all(importlib.util.find_spec(name) is not None
                   for name in ("razdel", "sbert_punc_case_ru", "torch", "transformers"))
    except (ImportError, ValueError):
        return False


def _download_punctuation(progress: Progress | None) -> None:
    if not _punctuation_dependencies_ready():
        raise ResourcePreparationError("Отключите пунктуацию и повторите попытку.", code="dependency_missing")

    from services.prepare_models import _punctuation_cached

    if _punctuation_cached():
        return
    try:
        plan = snapshot_download(PUNCTUATION_REPO, cache_dir=str(HF_HUB_CACHE),
                                 allow_patterns=PUNCTUATION_ALLOW_PATTERNS, dry_run=True)
    except Exception as exc:
        raise ResourcePreparationError(
            "Не удалось загрузить модель. Проверьте подключение к интернету и повторите попытку."
        ) from exc
    total = sum(max(int(item.file_size or 0), 0) for item in plan)
    cached = sum(max(int(item.file_size or 0), 0) for item in plan
                 if item.is_cached and not item.will_download)
    if total:
        _emit(progress, completed=cached, total=total, message="Загрузка модели пунктуации")
    lock = threading.Lock()
    latest = {"completed": cached}

    def bytes_progress(completed: int, reported_total: int | None) -> None:
        with lock:
            latest["completed"] = max(latest["completed"], max(0, completed))
            bounded = min(latest["completed"], reported_total) if reported_total else latest["completed"]
        _emit(progress, completed=bounded, total=reported_total or (total or None),
              message="Загрузка модели пунктуации")

    tqdm_class = create_download_progress_tqdm(
        total_bytes=total or None, cached_bytes=cached, initial_percent=0,
        byte_callback=bytes_progress,
    )
    try:
        snapshot_download(PUNCTUATION_REPO, cache_dir=str(HF_HUB_CACHE),
                          allow_patterns=PUNCTUATION_ALLOW_PATTERNS, tqdm_class=tqdm_class)
    except Exception as exc:
        raise ResourcePreparationError(
            "Не удалось загрузить модель. Проверьте подключение к интернету и повторите попытку."
        ) from exc
    if not _punctuation_cached():
        raise ResourcePreparationError(
            "Не удалось подготовить пунктуацию. Отключите пунктуацию и повторите попытку.",
            code="model_unavailable",
        )


def ensure_selected_resources(settings: dict, on_progress: Progress | None = None) -> None:
    """Validate and fetch only the model and optional punctuation chosen for a job."""
    model_id = settings["model"]
    device = settings["device"]
    if device == "vulkan":
        from services.prepare_models import _vulkan_cached

        if not _vulkan_cached(model_id):
            _emit(on_progress, message="Загрузка модели распознавания")

            def byte_progress(completed: int, total: int | None) -> None:
                _emit(on_progress, completed=max(0, completed), total=total,
                      message="Загрузка модели распознавания")

            from services.whisper_cpp import resolve_model_path
            try:
                resolve_model_path(model_id, byte_progress=byte_progress)
                if not _vulkan_cached(model_id):
                    raise ResourcePreparationError(
                        "Не удалось загрузить модель. Проверьте подключение к интернету и повторите попытку.",
                        code="model_unavailable",
                    )
            except TaskFailure:
                raise
            except Exception as exc:
                raise ResourcePreparationError(
                    "Не удалось загрузить модель. Проверьте подключение к интернету и повторите попытку."
                ) from exc
    else:
        try:
            ready = get_model_status(model_id).get("status") == "downloaded"
        except Exception:
            ready = False
        if not ready:
            _emit(on_progress, message="Загрузка модели распознавания")

            def byte_progress(completed: int, total: int | None) -> None:
                _emit(on_progress, completed=max(0, completed), total=total,
                      message="Загрузка модели распознавания")

            try:
                download_registered_model(model_id, byte_progress_callback=byte_progress)
            except TaskFailure:
                raise
            except ModelDownloadValidationError as exc:
                raise ResourcePreparationError(
                    "Не удалось проверить подготовленную модель. Выберите другую модель или повторите попытку.",
                    code="model_unavailable",
                ) from exc
            except Exception as exc:
                raise ResourcePreparationError(
                    "Не удалось загрузить модель. Проверьте подключение к интернету и повторите попытку."
                ) from exc

    if settings.get("restorePunctuation"):
        try:
            from services.prepare_models import _punctuation_cached
            punctuation_ready = _punctuation_cached()
        except Exception:
            punctuation_ready = False
        if not punctuation_ready:
            _download_punctuation(on_progress)

    _emit(on_progress, message="Подготовка завершена")


PREPARATION_EVENT_PREFIX = "__VT_PREP__ "


def _emit_preparation_event(payload: dict) -> None:
    print(PREPARATION_EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Prepare resources for one transcription")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--restore-punctuation", action="store_true")
    args = parser.parse_args(argv)

    def progress(payload: dict) -> None:
        completed = payload.get("completedBytes")
        total = payload.get("totalBytes")
        percent = min(99, int(completed * 100 / total)) if total and completed is not None else None
        _emit_preparation_event({"type": "progress", "stage": "preparation", "status": "active",
                                 "percent": percent, "completedBytes": completed,
                                 "totalBytes": total, "message": payload.get("message")})

    try:
        ensure_selected_resources({"model": args.model, "device": args.device,
                                   "restorePunctuation": args.restore_punctuation}, progress)
    except TaskFailure as exc:
        _emit_preparation_event({"type": "error", "stage": "preparation", "status": "failed",
                                 "message": str(exc), "code": exc.code})
        return 1
    except Exception:
        _emit_preparation_event({"type": "error", "stage": "preparation", "status": "failed",
                                 "message": "Не удалось подготовить ресурсы. Повторите попытку.",
                                 "code": "internal_error"})
        return 1
    _emit_preparation_event({"type": "progress", "stage": "preparation", "status": "completed",
                             "percent": 100, "completedBytes": None, "totalBytes": None,
                             "message": "Подготовка завершена"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
