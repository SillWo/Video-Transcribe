"""Verified local snapshots for the portable meeting archive.

Snapshots contain regular files from the archive root (meeting folders and
originals included), a SHA-256 manifest, and an external verification receipt.
All destructive operations recompute hashes; a stored ``verified`` flag alone
never authorizes deletion.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from services.meeting_archive import MeetingArchive, _archive_lock, _json_read, _json_write

Progress = Callable[[dict[str, Any]], None]
SCHEMA = 1
_EXCLUDED_NAMES = {".archive.lock"}


class BackupError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _progress(callback: Progress | None, stage: str, **values: Any) -> None:
    if callback:
        callback({"stage": stage, **values})


def _reject_symlink_path(path: Path) -> None:
    absolute = Path(os.path.abspath(path.expanduser()))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise BackupError("symlink_path", f"Путь содержит символическую ссылку: {current}")


def _archive_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    _reject_symlink_path(root)
    resolved = root.resolve(strict=False)
    if not resolved.exists() or not resolved.is_dir():
        raise BackupError("archive_missing", "Папка архива встреч не найдена.")
    return resolved


def _backup_root(path: str | Path, archive_root: Path, *, create: bool) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    _reject_symlink_path(root)
    resolved = root.resolve(strict=False)
    if resolved == archive_root or resolved.is_relative_to(archive_root):
        raise BackupError("backup_inside_archive", "Папка резервных копий должна находиться вне архива встреч.")
    if resolved.exists() and not resolved.is_dir():
        raise BackupError("backup_root_not_directory", "Папка резервных копий занята файлом.")
    for child in (resolved / "snapshots", resolved / "receipts"):
        _reject_symlink_path(child)
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _existing_backup_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    _reject_symlink_path(root)
    resolved = root.resolve(strict=False)
    if not resolved.is_dir():
        raise BackupError("backup_root_missing", "Папка резервных копий не найдена.")
    return resolved


def _validate_snapshot_id(snapshot_id: str) -> str:
    try:
        parsed = str(uuid.UUID(str(snapshot_id)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise BackupError("invalid_snapshot_id", "Некорректный идентификатор снимка.") from exc
    if parsed != str(snapshot_id).lower():
        raise BackupError("invalid_snapshot_id", "Некорректный идентификатор снимка.")
    return parsed


def _safe_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise BackupError("invalid_manifest", "В манифесте найден небезопасный путь.")
    posix = PurePosixPath(value)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise BackupError("invalid_manifest", "В манифесте найден небезопасный путь.")
    path = Path(*posix.parts)
    if path.is_absolute() or ".." in path.parts:
        raise BackupError("invalid_manifest", "В манифесте найден небезопасный путь.")
    return path


def _walk_files(root: Path) -> list[Path]:
    files: list[Path] = []
    def on_error(error: OSError) -> None:
        raise BackupError("archive_read_failed", f"Не удалось прочитать папку архива: {error.filename or root}")

    for current, dirs, names in os.walk(root, followlinks=False, onerror=on_error):
        current_path = Path(current)
        kept_dirs = []
        for name in dirs:
            directory = current_path / name
            if directory.is_symlink():
                raise BackupError("symlink_in_archive", f"Архив содержит символическую ссылку: {directory.relative_to(root)}")
            if name == ".staging":
                continue
            kept_dirs.append(name)
        dirs[:] = kept_dirs
        for name in names:
            path = current_path / name
            if path.is_symlink():
                raise BackupError("symlink_in_archive", f"Архив содержит символическую ссылку: {path.relative_to(root)}")
            if name in _EXCLUDED_NAMES or name.endswith("-shm"):
                continue
            if path.is_file():
                files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entries_fingerprint(entries: list[dict[str, Any]]) -> str:
    normalized = [{"path": row["path"], "size": int(row["size"]), "sha256": row["sha256"]}
                  for row in sorted(entries, key=lambda item: item["path"])]
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _snapshot_paths(backup_root: Path, snapshot_id: str) -> tuple[Path, Path]:
    safe_id = _validate_snapshot_id(snapshot_id)
    snapshot = backup_root / "snapshots" / safe_id
    receipts = backup_root / "receipts" / f"{safe_id}.json"
    for path in (snapshot, receipts):
        _reject_symlink_path(path)
    if not snapshot.resolve().is_relative_to(backup_root.resolve()):
        raise BackupError("snapshot_path_escape", "Путь снимка выходит из папки резервных копий.")
    return snapshot, receipts


def _snapshot_reuse_index(backup_root: Path) -> dict[tuple[int, str], list[Path]]:
    """Index files from previously verified snapshots; never point at live data."""
    index: dict[tuple[int, str], list[Path]] = {}
    snapshots_root = backup_root / "snapshots"
    if not snapshots_root.is_dir():
        return index
    candidates = []
    for directory in snapshots_root.iterdir():
        if not directory.is_dir() or not _is_uuid(directory.name):
            continue
        try:
            _reject_symlink_path(directory)
            manifest = _read_manifest(directory)
            receipt = _load_receipt(backup_root, directory.name) or {}
            if (receipt.get("verified") is not True
                    or receipt.get("content_hash") != manifest.get("content_hash")
                    or _entries_fingerprint(manifest["files"]) != manifest.get("content_hash")):
                continue
            candidates.append((str(manifest.get("created_at") or ""), directory, manifest))
        except (BackupError, OSError, KeyError, TypeError):
            continue
    candidates.sort(key=lambda row: row[0], reverse=True)
    for _created, directory, manifest in candidates:
        for row in manifest.get("files", []):
            try:
                relative = _safe_relative(row["path"])
                size, digest = int(row["size"]), str(row["sha256"])
                if len(digest) != 64:
                    continue
                candidate = directory / "data" / relative
                _reject_symlink_path(candidate)
                if not candidate.is_file() or candidate.stat().st_size != size:
                    continue
                key = (size, digest)
                index.setdefault(key, []).append(candidate)
            except (BackupError, OSError, KeyError, TypeError, ValueError):
                continue
    return index


def _verified_reuse_candidate(index: dict[tuple[int, str], list[Path]], size: int,
                              digest: str) -> Path | None:
    for candidate in index.get((size, digest), []):
        try:
            if candidate.stat().st_size == size and _sha256(candidate) == digest:
                return candidate
        except OSError:
            continue
    return None


def _read_manifest(snapshot: Path, *, snapshot_id: str | None = None) -> dict[str, Any]:
    manifest_path = snapshot / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupError("manifest_missing", "Манифест снимка отсутствует или повреждён.") from exc
    if not isinstance(manifest, dict):
        raise BackupError("invalid_manifest", "Структура манифеста снимка некорректна.")
    if manifest.get("schema") != SCHEMA or manifest.get("snapshot_id") != (snapshot_id or snapshot.name):
        raise BackupError("invalid_manifest", "Версия или идентификатор манифеста не совпадает.")
    if not isinstance(manifest.get("files"), list):
        raise BackupError("invalid_manifest", "Список файлов в манифесте некорректен.")
    return manifest


def _verify_snapshot_contents(snapshot: Path, *, snapshot_id: str | None = None) -> dict[str, Any]:
    manifest = _read_manifest(snapshot, snapshot_id=snapshot_id)
    expected_id = snapshot_id or snapshot.name
    if manifest.get("snapshot_id") != expected_id:
        raise BackupError("invalid_manifest", "Версия или идентификатор манифеста не совпадает.")
    data_root = snapshot / "data"
    if not data_root.is_dir() or data_root.is_symlink():
        raise BackupError("snapshot_data_missing", "В снимке отсутствует папка данных.")
    expected: dict[str, dict[str, Any]] = {}
    normalized_paths: set[str] = set()
    for row in manifest["files"]:
        if not isinstance(row, dict):
            raise BackupError("invalid_manifest", "Запись файла в манифесте некорректна.")
        relative = str(row.get("path", ""))
        safe = _safe_relative(relative)
        if relative in expected or relative.casefold() in normalized_paths:
            raise BackupError("invalid_manifest", "В манифесте повторяется путь файла.")
        normalized_paths.add(relative.casefold())
        expected[relative] = row
        target = data_root / safe
        _reject_symlink_path(target)
        if not target.is_file():
            raise BackupError("snapshot_file_missing", f"В снимке отсутствует файл: {relative}")
        if target.stat().st_size != row.get("size") or _sha256(target) != row.get("sha256"):
            raise BackupError("snapshot_hash_mismatch", f"Контрольная сумма не совпадает: {relative}")
    actual = [path.relative_to(data_root).as_posix() for path in _walk_files(data_root)]
    if set(actual) != set(expected):
        raise BackupError("snapshot_file_set_mismatch", "Состав файлов снимка отличается от манифеста.")
    fingerprint = _entries_fingerprint(manifest["files"])
    if manifest.get("content_hash") != fingerprint:
        raise BackupError("manifest_hash_mismatch", "Контрольная сумма манифеста не совпадает.")
    return {"manifest": manifest, "content_hash": fingerprint,
            "file_count": len(expected), "total_bytes": sum(int(row["size"]) for row in expected.values())}


def _receipt_path(backup_root: Path, snapshot_id: str) -> Path:
    return backup_root / "receipts" / f"{_validate_snapshot_id(snapshot_id)}.json"


def _write_receipt(backup_root: Path, snapshot_id: str, receipt: dict[str, Any]) -> None:
    _atomic_json(_receipt_path(backup_root, snapshot_id), receipt)


def _load_receipt(backup_root: Path, snapshot_id: str) -> dict[str, Any] | None:
    path = _receipt_path(backup_root, snapshot_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def create_backup(archive_root: str | Path, backup_root: str | Path, *,
                  progress: Progress | None = None) -> dict[str, Any]:
    """Create an immutable snapshot through an isolated staging directory."""
    source = _archive_root(archive_root)
    target_root = _backup_root(backup_root, source, create=True)
    snapshot_id = str(uuid.uuid4())
    staging = target_root / f".staging-{snapshot_id}"
    final = target_root / "snapshots" / snapshot_id
    if final.exists():
        raise BackupError("snapshot_collision", "Не удалось зарезервировать уникальный снимок.")
    staging.mkdir(parents=True, exist_ok=False)
    entries = []
    total_bytes = 0
    reused_files = 0
    copied_files = 0
    try:
        reuse_index = _snapshot_reuse_index(target_root)
        with _archive_lock(source):
            sources = _walk_files(source)
            total_bytes = sum(path.stat().st_size for path in sources)
            _progress(progress, "copy", completed=0, total=len(sources), bytes_total=total_bytes)
            bytes_completed = 0
            for index, path in enumerate(sources, 1):
                relative = path.relative_to(source).as_posix()
                destination = staging / "data" / _safe_relative(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                before = path.stat()
                source_hash = _sha256(path)
                candidate = _verified_reuse_candidate(reuse_index, before.st_size, source_hash)
                linked = False
                if candidate is not None:
                    try:
                        os.link(candidate, destination)
                        linked = (_sha256(candidate) == source_hash
                                  and _sha256(destination) == source_hash)
                        if not linked:
                            destination.unlink()
                    except OSError:
                        with contextlib.suppress(OSError):
                            if destination.exists():
                                destination.unlink()
                if linked:
                    reused_files += 1
                else:
                    shutil.copy2(path, destination)
                    copied_files += 1
                    if _sha256(destination) != source_hash:
                        raise BackupError("copy_hash_mismatch", f"Контрольная сумма новой копии не совпадает: {relative}")
                after = path.stat()
                if (before.st_size != after.st_size
                        or before.st_mtime_ns != after.st_mtime_ns):
                    raise BackupError("archive_changed", f"Файл изменился во время копирования: {relative}")
                entries.append({"path": relative, "size": after.st_size, "sha256": source_hash})
                bytes_completed += int(after.st_size)
                _progress(progress, "copy", completed=index, total=len(sources),
                          bytes_completed=bytes_completed, bytes_total=total_bytes,
                          reused_files=reused_files, copied_files=copied_files)
        content_hash = _entries_fingerprint(entries)
        manifest = {"schema": SCHEMA, "snapshot_id": snapshot_id,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "files": entries, "content_hash": content_hash}
        _atomic_json(staging / "manifest.json", manifest)
        checked = _verify_snapshot_contents(staging, snapshot_id=snapshot_id)
        if checked["content_hash"] != content_hash:
            raise BackupError("staging_verification_failed", "Проверка временной копии не прошла.")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        receipt = {"schema": SCHEMA, "snapshot_id": snapshot_id,
                   "verified": True, "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "content_hash": content_hash, "file_count": checked["file_count"],
                   "total_bytes": checked["total_bytes"], "restore_test": None}
        _write_receipt(target_root, snapshot_id, receipt)
        return {"snapshot_id": snapshot_id, "status": "verified", "file_count": checked["file_count"],
                "total_bytes": checked["total_bytes"], "content_hash": content_hash,
                "reused_files": reused_files, "copied_files": copied_files}
    except Exception:
        with contextlib.suppress(OSError):
            if staging.exists():
                shutil.rmtree(staging)
        raise


def verify_backup(backup_root: str | Path, snapshot_id: str, *,
                  progress: Progress | None = None) -> dict[str, Any]:
    target_root = _existing_backup_root(backup_root)
    try:
        snapshot, _receipt = _snapshot_paths(target_root, snapshot_id)
    except BackupError as exc:
        return {"snapshot_id": str(snapshot_id), "verified": False,
                "error": {"code": exc.code, "message": str(exc)}}
    try:
        _progress(progress, "verify", status="running")
        checked = _verify_snapshot_contents(snapshot)
        old = _load_receipt(target_root, snapshot_id) or {}
        receipt = {**old, "schema": SCHEMA, "snapshot_id": snapshot_id,
                   "verified": True, "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "content_hash": checked["content_hash"], "file_count": checked["file_count"],
                   "total_bytes": checked["total_bytes"]}
        _write_receipt(target_root, snapshot_id, receipt)
        return {"snapshot_id": snapshot_id, "verified": True, **checked}
    except (BackupError, OSError) as exc:
        if not isinstance(exc, BackupError):
            exc = BackupError("backup_read_failed", "Не удалось прочитать все файлы резервной копии.")
        old = _load_receipt(target_root, snapshot_id) or {}
        _write_receipt(target_root, snapshot_id,
                       {**old, "schema": SCHEMA, "snapshot_id": snapshot_id, "verified": False,
                        "error": {"code": exc.code, "message": str(exc)}})
        return {"snapshot_id": snapshot_id, "verified": False,
                "error": {"code": exc.code, "message": str(exc)}}


def _target_path(target: str | Path, source_archive: Path, backup_root: Path) -> Path:
    path = Path(target).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    _reject_symlink_path(path)
    resolved = path.resolve(strict=False)
    if (resolved == source_archive or resolved.is_relative_to(source_archive)
            or resolved == backup_root or resolved.is_relative_to(backup_root)):
        raise BackupError("restore_target_overlap", "Папка восстановления пересекается с архивом или резервными копиями.")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise BackupError("restore_target_not_empty", "Восстановление разрешено только в новую или пустую папку.")
    return resolved


def _copy_snapshot_to(snapshot: Path, target: Path, manifest: dict[str, Any],
                      progress: Progress | None = None) -> dict[str, Any]:
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{target.name}.restore-{uuid.uuid4().hex}.tmp"
    if staging.exists():
        raise BackupError("restore_staging_collision", "Временная папка восстановления уже существует.")
    staging.mkdir(parents=False)
    entries = manifest["files"]
    committed = False
    existing_empty = target.exists()
    try:
        for index, row in enumerate(entries, 1):
            relative = _safe_relative(row["path"])
            source = snapshot / "data" / relative
            _reject_symlink_path(source)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if destination.stat().st_size != row["size"] or _sha256(destination) != row["sha256"]:
                raise BackupError("restore_hash_mismatch", f"Контрольная сумма восстановления не совпала: {row['path']}")
            _progress(progress, "restore", completed=index, total=len(entries), path=row["path"])
        if existing_empty:
            target.rmdir()
        try:
            os.replace(staging, target)
            committed = True
        except Exception:
            if existing_empty and not target.exists():
                target.mkdir(parents=False, exist_ok=True)
            raise
        restored = _verify_restored_tree(target, manifest)
        return restored
    except Exception:
        with contextlib.suppress(OSError):
            if staging.exists():
                shutil.rmtree(staging)
        if committed:
            with contextlib.suppress(OSError):
                shutil.rmtree(target)
            if existing_empty:
                with contextlib.suppress(OSError):
                    target.mkdir(parents=False, exist_ok=True)
        raise


def _verify_restored_tree(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {row["path"]: row for row in manifest["files"]}
    actual = _walk_files(root)
    actual_set = {p.relative_to(root).as_posix() for p in actual}
    if actual_set != set(expected):
        raise BackupError("restore_file_set_mismatch", "Состав восстановленного архива отличается от снимка.")
    for path in actual:
        relative = path.relative_to(root).as_posix()
        row = expected[relative]
        if path.stat().st_size != row["size"] or _sha256(path) != row["sha256"]:
            raise BackupError("restore_hash_mismatch", f"Контрольная сумма восстановленного файла не совпала: {relative}")
    return {"verified": True, "file_count": len(expected), "content_hash": manifest["content_hash"]}


def restore_backup(archive_root: str | Path, backup_root: str | Path, snapshot_id: str,
                   target: str | Path, *, progress: Progress | None = None) -> dict[str, Any]:
    source_archive = _archive_root(archive_root)
    backup = _backup_root(backup_root, source_archive, create=False)
    destination = _target_path(target, source_archive, backup)
    checked = verify_backup(backup, snapshot_id, progress=progress)
    if not checked.get("verified"):
        raise BackupError("backup_verification_failed", "Снимок не прошёл проверку; восстановление остановлено.")
    snapshot, _ = _snapshot_paths(backup, snapshot_id)
    result = _copy_snapshot_to(snapshot, destination, checked["manifest"], progress)
    return {"snapshot_id": snapshot_id, "target": str(destination), **result}


def restore_test(archive_root: str | Path, backup_root: str | Path, snapshot_id: str,
                 *, progress: Progress | None = None) -> dict[str, Any]:
    source_archive = _archive_root(archive_root)
    backup = _backup_root(backup_root, source_archive, create=False)
    checked = verify_backup(backup, snapshot_id, progress=progress)
    if not checked.get("verified"):
        raise BackupError("backup_verification_failed", "Снимок не прошёл проверку; тест восстановления остановлен.")
    snapshot, _ = _snapshot_paths(backup, snapshot_id)
    test_root = backup / ".restore-tests"
    _reject_symlink_path(test_root)
    test_root.mkdir(parents=True, exist_ok=True)
    destination = test_root / f"restore-{uuid.uuid4().hex}"
    try:
        restored = _copy_snapshot_to(snapshot, destination, checked["manifest"], progress)
        receipt = _load_receipt(backup, snapshot_id) or {}
        receipt["restore_test"] = {"passed": True,
                                   "tested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "content_hash": checked["content_hash"],
                                   "file_count": restored["file_count"]}
        _write_receipt(backup, snapshot_id, receipt)
        return {"snapshot_id": snapshot_id, "passed": True, **restored}
    except Exception as exc:
        receipt = _load_receipt(backup, snapshot_id) or {}
        receipt["restore_test"] = {"passed": False,
                                   "tested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "content_hash": checked["content_hash"],
                                   "error": str(exc)}
        _write_receipt(backup, snapshot_id, receipt)
        raise
    finally:
        with contextlib.suppress(OSError):
            if destination.exists():
                shutil.rmtree(destination)
        with contextlib.suppress(OSError):
            test_root.rmdir()


def _policy() -> dict[str, Any]:
    from services.meeting_settings import load_settings
    return load_settings()


def _archive_completeness(archive: MeetingArchive, meeting_id: str) -> tuple[bool, list[str]]:
    folder = archive.meeting_dir(meeting_id)
    manifest = _json_read(folder / "manifest.json", None)
    reasons = []
    if not isinstance(manifest, dict):
        return False, ["Нет актуального манифеста компактного архива."]
    readiness = manifest.get("readiness") or {}
    for key, label in (("transcript", "стенограмма"), ("screen", "экран"), ("audio", "сжатый звук")):
        if readiness.get(key) != "ready":
            reasons.append(f"Этап «{label}» не готов.")
    screen_data = _json_read(folder / "screen-events.json", {})
    if not isinstance(screen_data, dict) or screen_data.get("status") != "ready":
        reasons.append("Статус извлечения экрана в screen-events.json не готов.")
    audio_files = list((folder / "audio").glob("*.opus")) if (folder / "audio").is_dir() else []
    try:
        has_retained_audio = any(path.is_file() and path.stat().st_size > 0 for path in audio_files)
    except OSError:
        has_retained_audio = False
    if not has_retained_audio:
        reasons.append("В архиве нет готового сжатого звука Opus.")
    transcript_data = _json_read(folder / "transcript.json", {})
    try:
        meta = archive.get_meeting(meeting_id)
    except (KeyError, ValueError):
        meta = {}
        reasons.append("Файл встречи отсутствует или повреждён.")
    transcript_data = transcript_data if isinstance(transcript_data, dict) else {}
    transcript_ready = (meta.get("speech_status") in {"ready", "completed"}
                        or bool(transcript_data.get("raw_segments"))
                        or transcript_data.get("raw_text") is not None)
    if not transcript_ready:
        reasons.append("В архиве нет готовой стенограммы или подтверждённого статуса речи.")
    hashes = manifest.get("files")
    if not isinstance(hashes, dict) or not hashes:
        reasons.append("В манифесте нет контрольных сумм архива.")
    else:
        expected = set(hashes)
        try:
            actual = {path.relative_to(folder).as_posix() for path in _walk_files(folder)
                      if path.name != "manifest.json"}
        except BackupError as exc:
            actual = set()
            reasons.append(str(exc))
        if actual != expected:
            reasons.append("Состав файлов компактного архива не совпадает с манифестом.")
        for relative, digest in hashes.items():
            try:
                safe = _safe_relative(relative)
                path = folder / safe
                if not path.is_file() or _sha256(path) != digest:
                    reasons.append(f"Хеш архивного файла не совпадает: {relative}")
            except (BackupError, OSError):
                reasons.append(f"Невозможно проверить архивный файл: {relative}")
    return not reasons, reasons


def _current_entries(archive_root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in _walk_files(archive_root):
        entries.append({"path": path.relative_to(archive_root).as_posix(),
                        "size": path.stat().st_size, "sha256": _sha256(path)})
    return entries


def _backup_matches_current(archive_root: Path, backup_root: Path,
                            snapshot_id: str) -> tuple[bool, str | None, dict[str, Any] | None]:
    checked = verify_backup(backup_root, snapshot_id)
    if not checked.get("verified"):
        return False, "Резервная копия не прошла повторную проверку.", None
    try:
        current = _current_entries(archive_root)
    except BackupError as exc:
        return False, str(exc), None
    current_hash = _entries_fingerprint(current)
    if current_hash != checked["content_hash"]:
        return False, "Архив изменился после создания резервной копии; нужна новая копия.", checked
    return True, None, checked


def _cleanup_preview_unlocked(archive_root: Path, backup_root: Path | None,
                              *, now: datetime, retention_days: int,
                              auto_cleanup_enabled: bool) -> dict[str, Any]:
    archive = MeetingArchive(archive_root, read_only=True)
    cutoff = now - timedelta(days=max(0, retention_days))
    items = []
    snapshots = []
    if backup_root and backup_root.is_dir():
        with contextlib.suppress(OSError):
            snapshots = sorted((p.name for p in (backup_root / "snapshots").iterdir()
                                if p.is_dir() and _is_uuid(p.name)), reverse=True)
    matching_snapshot = None
    backup_block_reason = "Нет папки резервных копий."
    if backup_root:
        backup_block_reason = "Нет созданной резервной копии архива."
    if backup_root and snapshots:
        verified_count = 0
        current_snapshot_count = 0
        restore_receipt_missing = False
        try:
            current_fingerprint = _entries_fingerprint(_current_entries(archive_root))
            for snapshot_id in snapshots:
                checked = verify_backup(backup_root, snapshot_id)
                if not checked.get("verified"):
                    continue
                verified_count += 1
                if checked.get("content_hash") != current_fingerprint:
                    continue
                current_snapshot_count += 1
                receipt = _load_receipt(backup_root, snapshot_id) or {}
                restore = receipt.get("restore_test") or {}
                if (receipt.get("verified") is True and restore.get("passed") is True
                        and restore.get("content_hash") == current_fingerprint):
                    matching_snapshot = snapshot_id
                    break
                restore_receipt_missing = True
            if matching_snapshot is None:
                if current_snapshot_count and restore_receipt_missing:
                    backup_block_reason = "Для текущей копии нет успешного теста восстановления."
                elif verified_count:
                    backup_block_reason = "Архив изменился после создания резервной копии; нужна новая копия."
                else:
                    backup_block_reason = "Резервные копии не прошли повторную проверку."
        except (BackupError, OSError):
            matching_snapshot = None
            backup_block_reason = "Не удалось повторно прочитать архив для сверки с резервной копией."
    for meta in archive.list_meetings(limit=100000, offset=0):
        meeting_id = str(meta.get("meeting_id", ""))
        original_rel = meta.get("original_rel")
        if not original_rel:
            continue
        raw_original = Path(str(original_rel))
        candidate = raw_original if raw_original.is_absolute() else archive_root / raw_original
        expected_originals = (archive_root / "originals").resolve()
        expected_meeting_originals = (archive_root / "originals" / meeting_id).resolve()
        try:
            _reject_symlink_path(candidate)
            resolved_original = candidate.resolve(strict=False)
            if (not resolved_original.is_relative_to(expected_originals)
                    or resolved_original.parent != expected_meeting_originals):
                try:
                    old_enough = candidate.exists() and datetime.fromtimestamp(
                        candidate.stat().st_mtime, timezone.utc) <= cutoff
                except OSError:
                    old_enough = False
                if old_enough:
                    items.append({"meeting_id": meeting_id,
                                  "filename": meta.get("source_filename") or candidate.name,
                                  "eligible": False,
                                  "blocked_reasons": ["Исходник находится вне своей папки originals встречи."]})
                continue
            original = resolved_original
            if not original.is_file():
                continue
            modified = datetime.fromtimestamp(original.stat().st_mtime, timezone.utc)
        except BackupError as exc:
            try:
                old_enough = candidate.exists() and datetime.fromtimestamp(
                    candidate.lstat().st_mtime, timezone.utc) <= cutoff
            except OSError:
                old_enough = False
            if old_enough:
                items.append({"meeting_id": meeting_id,
                              "filename": meta.get("source_filename") or candidate.name,
                              "eligible": False,
                              "blocked_reasons": [str(exc)]})
            continue
        except OSError:
            continue
        if modified > cutoff:
            continue
        blockers = []
        if not auto_cleanup_enabled:
            blockers.append("Автоматическая очистка выключена в настройках.")
        if not backup_root:
            blockers.append("Папка резервных копий не настроена.")
        complete, completeness_reasons = _archive_completeness(archive, meeting_id)
        if not complete:
            blockers.extend(completeness_reasons)
        if not meta.get("sha256") or _sha256(original) != meta.get("sha256"):
            blockers.append("Контрольная сумма оригинала не совпадает с записью встречи.")
        if backup_root:
            if matching_snapshot is None:
                blockers.append(backup_block_reason)
        items.append({"meeting_id": meeting_id, "filename": original.name,
                      "modified_at": modified.isoformat(timespec="seconds"),
                      "age_days": max(0, int((now - modified).total_seconds() // 86400)),
                      "eligible": not blockers, "snapshot_id": matching_snapshot,
                      "blocked_reasons": blockers})
    return {"retention_days": retention_days, "cutoff": cutoff.isoformat(timespec="seconds"),
            "auto_cleanup_enabled": auto_cleanup_enabled, "items": items,
            "eligible_count": sum(1 for item in items if item["eligible"]),
            "blocked_count": sum(1 for item in items if not item["eligible"])}


def _is_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value.lower()
    except (ValueError, TypeError, AttributeError):
        return False


def cleanup_preview(archive_root: str | Path, backup_root: str | Path | None = None, *,
                    progress: Progress | None = None, now: datetime | None = None) -> dict[str, Any]:
    """List eligible old originals and explain every blocked deletion."""
    root = _archive_root(archive_root)
    settings = _policy()
    configured_backup = backup_root or settings.get("backup_root")
    backup = _backup_root(configured_backup, root, create=False) if configured_backup else None
    try:
        retention_days = max(0, int(settings.get("original_retention_days", 30)))
    except (TypeError, ValueError):
        retention_days = 30
    _progress(progress, "cleanup_preview", status="checking")
    with _archive_lock(root):
        return _cleanup_preview_unlocked(root, backup, now=now or datetime.now(timezone.utc),
                                          retention_days=retention_days,
                                          auto_cleanup_enabled=bool(settings.get("auto_cleanup", False)))


def _mark_original_removed(archive_root: Path, meeting_id: str) -> None:
    archive = MeetingArchive(archive_root)
    folder = archive.meeting_dir(meeting_id)
    meta_path = folder / "meeting.json"
    meta = _json_read(meta_path, {})
    meta["source_available"] = False
    meta["source_removed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _json_write(meta_path, meta)
    manifest_path = folder / "manifest.json"
    manifest = _json_read(manifest_path, {})
    if isinstance(manifest, dict):
        manifest.setdefault("source_availability", {})["source"] = False
        manifest.setdefault("readiness", {})["source"] = "unavailable"
        files = manifest.get("files")
        if isinstance(files, dict) and (folder / "meeting.json").is_file():
            files["meeting.json"] = _sha256(folder / "meeting.json")
        _json_write(manifest_path, manifest)


def run_cleanup(archive_root: str | Path, backup_root: str | Path | None = None, *,
                progress: Progress | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Recheck all guards under archive lock, then unlink only eligible originals."""
    root = _archive_root(archive_root)
    settings = _policy()
    if not settings.get("auto_cleanup", False):
        raise BackupError("cleanup_disabled", "Очистка оригиналов не включена в настройках.")
    configured_backup = backup_root or settings.get("backup_root")
    if not configured_backup:
        raise BackupError("backup_unconfigured", "Сначала выберите папку резервных копий.")
    backup = _backup_root(configured_backup, root, create=False)
    try:
        retention_days = max(0, int(settings.get("original_retention_days", 30)))
    except (TypeError, ValueError):
        retention_days = 30
    timestamp = now or datetime.now(timezone.utc)
    removed = []
    with _archive_lock(root):
        preview = _cleanup_preview_unlocked(root, backup, now=timestamp,
                                             retention_days=retention_days,
                                             auto_cleanup_enabled=True)
        eligible = [item for item in preview["items"] if item["eligible"]]
        _progress(progress, "cleanup", completed=0, total=len(eligible))
        archive = MeetingArchive(root)
        deletion_plan = []
        current_backup_ids = {item["snapshot_id"] for item in eligible}
        if len(current_backup_ids) > 1:
            raise BackupError("backup_changed", "Подходящие копии разных версий не позволяют выполнить общую очистку.")
        for item in eligible:
            meeting_id = MeetingArchive._valid_id(item["meeting_id"])
            meta = archive.get_meeting(meeting_id)
            original = archive.original_path(meeting_id)
            originals_root = (root / "originals").resolve()
            expected_meeting_root = (root / "originals" / meeting_id).resolve()
            if original:
                _reject_symlink_path(original)
            if (not original or original.is_symlink() or not original.resolve().is_relative_to(originals_root)
                    or original.resolve().parent != expected_meeting_root):
                raise BackupError("cleanup_path_guard", f"Небезопасный путь оригинала встречи {meeting_id}.")
            if _sha256(original) != meta.get("sha256"):
                raise BackupError("original_hash_changed", f"Оригинал встречи {meeting_id} изменился после проверки.")
            deletion_plan.append((meeting_id, original))
        # One full tree verification immediately before deletion is sufficient;
        # per-file source hashes above then guard each unlink without invalidating
        # the same snapshot after the first metadata update.
        for index, (meeting_id, original) in enumerate(deletion_plan, 1):
            original.unlink()
            with contextlib.suppress(OSError):
                original.parent.rmdir()
            _mark_original_removed(root, meeting_id)
            removed.append(meeting_id)
            _progress(progress, "cleanup", completed=index, total=len(eligible), meeting_id=meeting_id)
    return {"removed_meeting_ids": removed, "removed_count": len(removed),
            "compact_archive_deleted": False}


def backup_status(archive_root: str | Path, backup_root: str | Path | None = None) -> dict[str, Any]:
    root = _archive_root(archive_root)
    settings = _policy()
    configured_backup = backup_root or settings.get("backup_root")
    if not configured_backup:
        return {"configured": False, "backup_root": None, "snapshots": [],
                "error": {"code": "backup_unconfigured", "message": "Папка резервных копий не настроена."}}
    backup = _backup_root(configured_backup, root, create=False)
    rows = []
    snapshot_root = backup / "snapshots"
    if snapshot_root.is_dir():
        for path in sorted((p for p in snapshot_root.iterdir() if p.is_dir() and _is_uuid(p.name)),
                           key=lambda p: p.name, reverse=True):
            manifest = _json_read(path / "manifest.json", {})
            receipt = _load_receipt(backup, path.name) or {}
            rows.append({"snapshot_id": path.name, "created_at": manifest.get("created_at"),
                         "file_count": len(manifest.get("files", [])),
                         "total_bytes": sum(int(row.get("size", 0)) for row in manifest.get("files", []) if isinstance(row, dict)),
                         "verified_receipt": bool(receipt.get("verified")),
                         "restore_test_passed": bool((receipt.get("restore_test") or {}).get("passed"))})
    return {"configured": True, "archive_root": str(root), "backup_root": str(backup),
            "auto_cleanup_enabled": bool(settings.get("auto_cleanup", False)),
            "retention_days": settings.get("original_retention_days", 30), "snapshots": rows}

