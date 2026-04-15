from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound

from .model_registry import ModelRegistryEntry, get_model_registry

ModelDownloadStatus = Literal["downloaded", "not_downloaded", "unknown"]

REQUIRED_MODEL_FILES = {"config.json", "model.bin", "tokenizer.json"}


class ModelCacheInspectionError(RuntimeError):
    """Raised when the local Hugging Face cache cannot be inspected at all."""


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


def _scan_cache_index(cache_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    try:
        cache_info = scan_cache_dir(cache_dir=cache_dir)
    except CacheNotFound:
        return {}
    except Exception as exc:
        raise ModelCacheInspectionError(f"Unable to inspect Hugging Face cache: {exc}") from exc

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


def list_models_with_status(cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    cache_index = _scan_cache_index(cache_dir=cache_dir)
    return [
        asdict(_build_model_status(entry, cache_index))
        for entry in get_model_registry()
    ]


def get_model_panel_payload(cache_dir: str | Path | None = None) -> dict[str, Any]:
    catalog = list_models_with_status(cache_dir=cache_dir)
    downloaded_items = [item for item in catalog if item["enabled"] and item["status"] == "downloaded"]
    total_downloaded_size_bytes = sum(int(item["downloadedSizeBytes"] or 0) for item in downloaded_items)
    available_count = sum(1 for item in catalog if item["enabled"])

    return {
        "catalog": catalog,
        "summary": {
            "downloadedCount": len(downloaded_items),
            "availableCount": available_count,
            "totalDownloadedSizeBytes": total_downloaded_size_bytes,
        },
    }
