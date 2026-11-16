"""Persistent paths and user preferences for the local meeting archive."""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_DEFAULTS = {"archive_root": None, "backup_root": None, "auto_cleanup": False,
             "original_retention_days": 30, "execution_mode": "auto", "device": "cpu", "profile": "balanced"}


def settings_path() -> Path:
    configured = os.environ.get("VT_MEETING_SETTINGS")
    if configured:
        return Path(configured).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return (Path(local) / "VideoTranscribe" / "meetings-settings.json").resolve()
    return (Path.home() / ".video-transcribe" / "meetings-settings.json").resolve()


def default_archive_root(project_root: str | Path | None = None) -> Path:
    configured = os.environ.get("VT_MEETING_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return ((Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1])
            / ".runtime" / "meetings")


def load_settings(project_root: str | Path | None = None, path: str | Path | None = None) -> dict[str, Any]:
    settings_file = Path(path).expanduser().resolve() if path else settings_path()
    try:
        raw = json.loads(settings_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    data = {**_DEFAULTS, **(raw if isinstance(raw, dict) else {})}
    data["archive_root"] = str(default_archive_root(project_root) if os.environ.get("VT_MEETING_ROOT") else
                                (Path(data["archive_root"]).expanduser().resolve() if data.get("archive_root") else default_archive_root(project_root)))
    if data.get("backup_root"):
        data["backup_root"] = str(Path(data["backup_root"]).expanduser().resolve())
    else:
        data["backup_root"] = None
    data["auto_cleanup"] = bool(data.get("auto_cleanup", False))
    if data.get("execution_mode") not in {"auto", "manual"}:
        data["execution_mode"] = "auto"
    if data.get("device") not in {"cpu", "gpu"}:
        data["device"] = "cpu"
    if data.get("profile") not in {"balanced", "quality", "economy"}:
        data["profile"] = "balanced"
    try:
        data["original_retention_days"] = max(0, int(data.get("original_retention_days", 30)))
    except (TypeError, ValueError):
        data["original_retention_days"] = 30
    data["settings_path"] = str(settings_file)
    root = Path(data["archive_root"])
    try:
        probe = root
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        data["disk"] = {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
    except OSError:
        data["disk"] = None
    return data


def save_settings(values: dict[str, Any], project_root: str | Path | None = None,
                  path: str | Path | None = None) -> dict[str, Any]:
    current = load_settings(project_root, path)
    settings_file = Path(path).expanduser().resolve() if path else settings_path()
    allowed = {key: values[key] for key in _DEFAULTS if key in values}
    next_data = {key: current.get(key) for key in _DEFAULTS}
    next_data.update(allowed)
    next_data["archive_root"] = str(Path(next_data["archive_root"]).expanduser().resolve()) if next_data.get("archive_root") else str(default_archive_root(project_root))
    if next_data.get("backup_root"):
        next_data["backup_root"] = str(Path(next_data["backup_root"]).expanduser().resolve())
    next_data["auto_cleanup"] = bool(next_data.get("auto_cleanup", False))
    if next_data.get("execution_mode") not in {"auto", "manual"}:
        raise ValueError("execution_mode must be auto or manual")
    if next_data.get("device") not in {"cpu", "gpu"}:
        raise ValueError("device must be cpu or gpu")
    if next_data.get("profile") not in {"balanced", "quality", "economy"}:
        raise ValueError("profile must be balanced, quality, or economy")
    try:
        next_data["original_retention_days"] = max(0, int(next_data.get("original_retention_days", 30)))
    except (TypeError, ValueError) as exc:
        raise ValueError("original_retention_days must be a non-negative integer") from exc
    with _LOCK:
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        temp = settings_file.with_suffix(settings_file.suffix + ".tmp")
        temp.write_text(json.dumps(next_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, settings_file)
    return load_settings(project_root, settings_file)
