from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from huggingface_hub import DryRunFileInfo, HFCacheInfo, scan_cache_dir, snapshot_download
from huggingface_hub.errors import CacheNotFound
from tqdm.auto import tqdm as base_tqdm

from .model_registry import (
    ModelRegistryEntry,
    get_model_registry,
    get_model_registry_entry,
)

ModelDownloadStatus = Literal["downloaded", "not_downloaded", "unknown"]
ModelDownloadProgressCallback = Callable[[int, str], None]

REQUIRED_MODEL_FILES = {"config.json", "model.bin", "tokenizer.json"}
MODEL_DOWNLOAD_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


class ModelCacheInspectionError(RuntimeError):
    """Raised when the local Hugging Face cache cannot be inspected at all."""


class ModelDownloadValidationError(RuntimeError):
    """Raised when a model download finishes without a valid local snapshot."""


class ModelDeleteValidationError(RuntimeError):
    """Raised when a model delete request cannot be executed safely."""


@dataclass(frozen=True)
class ModelStatusEntry:
    id: str
    displayName: str
    backendValue: str
    hfRepoId: str
    languageScope: str
    family: str
    enabled: bool
    isDownloaded: bool
    cacheLocation: str | None
    downloadedSizeBytes: int | None
    lastModified: str | None
    status: ModelDownloadStatus


def _to_iso_timestamp(value: float | int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _resolve_revision(repo_info: Any) -> Any | None:
    if not getattr(repo_info, "revisions", None):
        return None

    return max(repo_info.revisions, key=lambda revision: getattr(revision, "last_modified", 0.0) or 0.0)


def _has_required_model_files(snapshot_path: Path) -> bool:
    if not snapshot_path.exists() or not snapshot_path.is_dir():
        return False

    present_files = {item.name for item in snapshot_path.iterdir() if item.is_file()}
    if not REQUIRED_MODEL_FILES.issubset(present_files):
        return False

    return any(path_name.startswith("vocabulary.") for path_name in present_files)


def _build_unknown_repo_state(repo_info: Any) -> dict[str, Any]:
    revision = _resolve_revision(repo_info)
    cache_location = None
    downloaded_size_bytes = getattr(repo_info, "size_on_disk", None)
    last_modified = _to_iso_timestamp(getattr(repo_info, "last_modified", None))

    if revision is not None:
        cache_location = str(revision.snapshot_path)
        downloaded_size_bytes = getattr(revision, "size_on_disk", downloaded_size_bytes)
        last_modified = _to_iso_timestamp(getattr(revision, "last_modified", None)) or last_modified
    elif getattr(repo_info, "repo_path", None) is not None:
        cache_location = str(repo_info.repo_path)

    return {
        "cacheLocation": cache_location,
        "downloadedSizeBytes": downloaded_size_bytes,
        "lastModified": last_modified,
        "status": "unknown",
    }


def _scan_cache_info(cache_dir: str | Path | None = None) -> HFCacheInfo | None:
    try:
        return scan_cache_dir(cache_dir=cache_dir)
    except CacheNotFound:
        return None
    except Exception as exc:
        raise ModelCacheInspectionError(f"Unable to inspect Hugging Face cache: {exc}") from exc


def _scan_cache_index(cache_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    cache_info = _scan_cache_info(cache_dir=cache_dir)
    if cache_info is None:
        return {}

    repos: dict[str, dict[str, Any]] = {}
    for repo in cache_info.repos:
        try:
            revision = _resolve_revision(repo)
            if revision is None:
                repos[repo.repo_id] = _build_unknown_repo_state(repo)
                continue

            snapshot_path = Path(revision.snapshot_path)
            status: ModelDownloadStatus = "downloaded" if _has_required_model_files(snapshot_path) else "unknown"
            repos[repo.repo_id] = {
                "cacheLocation": str(snapshot_path),
                "downloadedSizeBytes": revision.size_on_disk,
                "lastModified": _to_iso_timestamp(revision.last_modified),
                "status": status,
            }
        except Exception:
            repos[repo.repo_id] = _build_unknown_repo_state(repo)

    return repos


def _build_model_status(entry: ModelRegistryEntry, cache_index: dict[str, dict[str, Any]]) -> ModelStatusEntry:
    cached_repo = cache_index.get(entry.hfRepoId)
    if cached_repo is None:
        return ModelStatusEntry(
            id=entry.id,
            displayName=entry.displayName,
            backendValue=entry.backendValue,
            hfRepoId=entry.hfRepoId,
            languageScope=entry.languageScope,
            family=entry.family,
            enabled=entry.enabled,
            isDownloaded=False,
            cacheLocation=None,
            downloadedSizeBytes=None,
            lastModified=None,
            status="not_downloaded",
        )

    status = cached_repo["status"]
    return ModelStatusEntry(
        id=entry.id,
        displayName=entry.displayName,
        backendValue=entry.backendValue,
        hfRepoId=entry.hfRepoId,
        languageScope=entry.languageScope,
        family=entry.family,
        enabled=entry.enabled,
        isDownloaded=status == "downloaded",
        cacheLocation=cached_repo["cacheLocation"],
        downloadedSizeBytes=cached_repo["downloadedSizeBytes"],
        lastModified=cached_repo["lastModified"],
        status=status,
    )


def get_model_status(model_id: str, cache_dir: str | Path | None = None) -> dict[str, Any]:
    entry = get_model_registry_entry(model_id)
    cache_index = _scan_cache_index(cache_dir=cache_dir)
    return asdict(_build_model_status(entry, cache_index))


def list_models_with_status(cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    cache_index = _scan_cache_index(cache_dir=cache_dir)
    return [
        asdict(_build_model_status(entry, cache_index))
        for entry in get_model_registry()
    ]


def get_model_panel_payload(cache_dir: str | Path | None = None) -> dict[str, Any]:
    catalog = list_models_with_status(cache_dir=cache_dir)
    downloaded_items = [item for item in catalog if item["enabled"] and item["status"] == "downloaded"]
    unique_downloaded_locations: set[str] = set()
    total_downloaded_size_bytes = 0

    for item in downloaded_items:
        storage_key = str(item["cacheLocation"] or item["hfRepoId"])
        if storage_key in unique_downloaded_locations:
            continue
        unique_downloaded_locations.add(storage_key)
        total_downloaded_size_bytes += int(item["downloadedSizeBytes"] or 0)

    available_count = sum(1 for item in catalog if item["enabled"])

    return {
        "catalog": catalog,
        "summary": {
            "downloadedCount": len(downloaded_items),
            "availableCount": available_count,
            "totalDownloadedSizeBytes": total_downloaded_size_bytes,
        },
    }


def _emit_progress(callback: ModelDownloadProgressCallback | None, percent: int, label: str) -> None:
    if callback is None:
        return

    callback(max(0, min(100, percent)), label)


class _SilentTqdm(base_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def _build_download_plan(
    repo_id: str,
    cache_dir: str | Path | None = None,
) -> list[DryRunFileInfo]:
    return snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        allow_patterns=MODEL_DOWNLOAD_ALLOW_PATTERNS,
        dry_run=True,
        tqdm_class=_SilentTqdm,
    )


def download_registered_model(
    model_id: str,
    progress_callback: ModelDownloadProgressCallback | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    entry = get_model_registry_entry(model_id)
    current_status = get_model_status(model_id, cache_dir=cache_dir)
    if current_status["status"] == "downloaded":
        _emit_progress(progress_callback, 100, "Model already present in cache.")
        return {
            "modelId": entry.id,
            "alreadyDownloaded": True,
            "snapshotPath": current_status["cacheLocation"],
            "status": current_status,
        }

    _emit_progress(progress_callback, 5, "Preparing download plan.")
    download_plan = _build_download_plan(entry.hfRepoId, cache_dir=cache_dir)
    total_bytes = sum(max(int(item.file_size or 0), 0) for item in download_plan)
    already_cached_bytes = sum(
        max(int(item.file_size or 0), 0)
        for item in download_plan
        if item.is_cached and not item.will_download
    )
    bytes_to_download = max(total_bytes - already_cached_bytes, 0)
    initial_percent = int((already_cached_bytes / total_bytes) * 100) if total_bytes > 0 else 10
    _emit_progress(progress_callback, max(10, initial_percent), "Downloading model files.")

    class _DownloadProgressTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            self._is_bytes_bar = kwargs.get("unit") == "B"
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            if self._is_bytes_bar:
                self._emit()

        def _emit(self) -> None:
            if not self._is_bytes_bar:
                return

            if total_bytes <= 0:
                _emit_progress(progress_callback, 90, "Finalizing downloaded files.")
                return

            downloaded_now = min(max(int(getattr(self, "n", 0)), 0), bytes_to_download)
            completed_bytes = min(already_cached_bytes + downloaded_now, total_bytes)
            percent = int((completed_bytes / total_bytes) * 100)
            label = getattr(self, "desc", None) or "Downloading model files."
            _emit_progress(progress_callback, percent, label)

        def update(self, n: int | float | None = 1):
            result = super().update(n)
            self._emit()
            return result

        def refresh(self, *args, **kwargs):
            result = super().refresh(*args, **kwargs)
            self._emit()
            return result

        def set_description(self, desc=None, refresh=True):
            result = super().set_description(desc, refresh)
            self._emit()
            return result

    snapshot_path = snapshot_download(
        entry.hfRepoId,
        cache_dir=cache_dir,
        allow_patterns=MODEL_DOWNLOAD_ALLOW_PATTERNS,
        tqdm_class=_DownloadProgressTqdm,
    )

    verified_status = get_model_status(model_id, cache_dir=cache_dir)
    if verified_status["status"] != "downloaded":
        raise ModelDownloadValidationError(
            "Download finished but the local model snapshot could not be validated."
        )

    _emit_progress(progress_callback, 100, "Model is ready for faster-whisper.")
    return {
        "modelId": entry.id,
        "alreadyDownloaded": False,
        "snapshotPath": str(snapshot_path),
        "status": verified_status,
    }


def delete_registered_model(
    model_id: str,
    progress_callback: ModelDownloadProgressCallback | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    entry = get_model_registry_entry(model_id)
    current_status = get_model_status(model_id, cache_dir=cache_dir)

    if current_status["status"] == "not_downloaded":
        _emit_progress(progress_callback, 100, "Model is not present in the local cache.")
        return {
            "modelId": entry.id,
            "alreadyDeleted": True,
            "status": current_status,
        }

    if current_status["status"] != "downloaded":
        raise ModelDeleteValidationError(
            "Model cache state is unknown. Refusing to delete the model without a verified local snapshot."
        )

    _emit_progress(progress_callback, 10, "Inspecting cached revisions.")
    cache_info = _scan_cache_info(cache_dir=cache_dir)
    if cache_info is None:
        _emit_progress(progress_callback, 100, "Model is not present in the local cache.")
        return {
            "modelId": entry.id,
            "alreadyDeleted": True,
            "status": get_model_status(model_id, cache_dir=cache_dir),
        }

    cached_repo = next((repo for repo in cache_info.repos if repo.repo_id == entry.hfRepoId), None)
    if cached_repo is None:
        _emit_progress(progress_callback, 100, "Model is not present in the local cache.")
        return {
            "modelId": entry.id,
            "alreadyDeleted": True,
            "status": get_model_status(model_id, cache_dir=cache_dir),
        }

    revision_hashes = tuple(sorted(revision.commit_hash for revision in cached_repo.revisions))
    if not revision_hashes:
        raise ModelDeleteValidationError(
            "Cached model repository has no deletable revisions. Refusing to delete an unresolved cache state."
        )

    _emit_progress(progress_callback, 35, "Preparing delete strategy.")
    delete_strategy = cache_info.delete_revisions(*revision_hashes)

    _emit_progress(progress_callback, 75, "Deleting cached model files.")
    delete_strategy.execute()

    _emit_progress(progress_callback, 90, "Verifying cache state.")
    verified_status = get_model_status(model_id, cache_dir=cache_dir)
    if verified_status["status"] == "downloaded":
        raise ModelDeleteValidationError(
            "Delete finished but the model still appears as downloaded in the local cache."
        )

    _emit_progress(progress_callback, 100, "Model removed from the local Hugging Face cache.")
    return {
        "modelId": entry.id,
        "alreadyDeleted": False,
        "status": verified_status,
    }
