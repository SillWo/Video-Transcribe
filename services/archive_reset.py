"""Preview-first one-off cleanup for legacy managed archive data.

This command is intentionally not imported by application startup or API code.
Run ``python -m services.archive_reset`` to inspect the configured archive, or
pass ``--confirm-root`` with its exact resolved path to apply the cleanup.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


class ResetError(RuntimeError):
    """A reset cannot safely proceed with the current configuration."""


_JOB_ID = re.compile(r"[0-9a-fA-F-]{1,80}\Z")
_SIDECARS = ("-wal", "-shm", "-journal")


@dataclass(frozen=True)
class ResetPlan:
    archive_root: Path
    runtime_root: Path
    meeting_ids: tuple[str, ...]
    job_ids: tuple[str, ...]
    job_artifacts: tuple[Path, ...]
    archive_targets: tuple[Path, ...]

    def preview(self) -> dict:
        return {
            "archive_root": str(self.archive_root),
            "runtime_root": str(self.runtime_root),
            "meeting_count": len(self.meeting_ids),
            "job_count": len(self.job_ids),
            "meetings": list(self.meeting_ids),
            "jobs": list(self.job_ids),
            "job_artifact_directories": [str(path) for path in self.job_artifacts],
            "archive_targets": [str(path) for path in self.archive_targets],
        }


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if path.is_symlink():
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _assert_no_reparse_chain(path: Path) -> None:
    """Reject symlinks/junctions in every existing component of a path."""
    absolute = Path(os.path.abspath(path.expanduser()))
    components = [absolute, *absolute.parents]
    for component in reversed(components):
        if component.exists() and _is_reparse(component):
            raise ResetError(f"Refusing a reparse or symbolic-link path: {component}")


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _checked_target(path: Path, allowed_root: Path) -> Path:
    """Return a validated target below one explicit archive/runtime root."""
    _assert_no_reparse_chain(path)
    resolved_root = _resolved(allowed_root)
    resolved_path = _resolved(path)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ResetError(f"Target escapes its managed root: {path}") from exc
    if resolved_path == resolved_root:
        raise ResetError(f"Refusing to remove a managed root itself: {path}")
    return resolved_path


def _assert_tree_no_reparse(path: Path) -> None:
    if not path.is_dir():
        return
    for current, directories, files in os.walk(path, topdown=True, followlinks=False):
        parent = Path(current)
        for name in [*directories, *files]:
            candidate = parent / name
            if _is_reparse(candidate):
                raise ResetError(f"Refusing a reparse or symbolic-link path: {candidate}")


def _valid_id(value: object) -> str | None:
    if not isinstance(value, str) or not _JOB_ID.fullmatch(value):
        return None
    return value


def _meeting_ids(archive_root: Path) -> set[str]:
    directory = archive_root / "meetings"
    if not directory.is_dir():
        return set()
    result = set()
    for entry in directory.iterdir():
        try:
            meeting_id = str(uuid.UUID(entry.name))
        except (ValueError, AttributeError):
            continue
        if meeting_id != entry.name.lower():
            continue
        _assert_no_reparse_chain(entry)
        if entry.is_dir():
            result.add(meeting_id)
    return result


def _root_values_match(record: dict, archive_root: Path) -> bool | None:
    """True/False when explicit roots exist; None when the job is unbound."""
    values = [record.get(key) for key in ("archiveRoot", "meetingRoot")]
    values = [value for value in values if value not in (None, "")]
    if not values:
        return None
    try:
        return all(_resolved(value) == archive_root for value in values if isinstance(value, str)) and all(
            isinstance(value, str) for value in values
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _select_job_ids(runtime_root: Path, archive_root: Path, meeting_ids: set[str]) -> set[str]:
    database = runtime_root / "history.sqlite3"
    if not database.is_file():
        return set()
    _checked_target(database, runtime_root)
    db = sqlite3.connect(database)
    try:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute("SELECT id, metadata FROM jobs").fetchall()
    except sqlite3.Error as exc:
        raise ResetError(f"Could not read simple-job history: {exc}") from exc
    finally:
        db.close()

    selected = set()
    for job_id, raw_metadata in rows:
        safe_id = _valid_id(job_id)
        if safe_id is None:
            continue
        try:
            record = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        meeting_id = str(record.get("meetingId") or "")
        root_match = _root_values_match(record, archive_root)
        if meeting_id:
            if root_match is True or (root_match is None and meeting_id in meeting_ids):
                selected.add(safe_id)
        elif root_match is True or root_match is None:
            # An ordinary legacy row without archiveRoot is exactly the row
            # sync_history would claim for the active archive.
            selected.add(safe_id)
    return selected


def _portable_targets(archive_root: Path, runtime_root: Path, job_ids: set[str]) -> list[Path]:
    directory = archive_root / "transcriptions"
    targets: list[Path] = []
    if not directory.is_dir():
        return targets
    for entry in directory.iterdir():
        if entry.name == "deleted.json" and entry.is_file():
            targets.append(_checked_target(entry, archive_root))
            continue
        if not entry.is_dir() or _valid_id(entry.name) is None:
            continue
        _assert_no_reparse_chain(entry)
        item_path = entry / "item.json"
        summary_path = entry / "catalog.json"
        metadata_path = item_path if item_path.is_file() else summary_path
        metadata = {}
        if metadata_path.is_file():
            _checked_target(metadata_path, archive_root)
            try:
                value = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = value if isinstance(value, dict) else {}
            except (OSError, ValueError):
                # A damaged managed entry still belongs to this archive.
                metadata = {}
        root_match = _root_values_match(metadata, archive_root)
        if root_match is False:
            continue
        job_id = entry.name
        # If archive_root and runtime_root overlap, this path may be a job
        # artifact directory. Remove it only when its job was selected above.
        if (
            _resolved(entry).is_relative_to(runtime_root / "transcriptions")
            and job_id not in job_ids
            and not metadata_path.is_file()
        ):
            continue
        targets.append(_checked_target(entry, archive_root))
    return targets


def _archive_targets(archive_root: Path, runtime_root: Path, job_ids: set[str]) -> tuple[Path, ...]:
    targets: list[Path] = []
    meetings = archive_root / "meetings"
    if meetings.is_dir():
        for entry in meetings.iterdir():
            try:
                parsed = str(uuid.UUID(entry.name))
            except (ValueError, AttributeError):
                continue
            if parsed == entry.name.lower():
                _assert_no_reparse_chain(entry)
            if parsed == entry.name.lower() and entry.is_dir():
                targets.append(_checked_target(entry, archive_root))

    originals = archive_root / "originals"
    if originals.is_dir():
        for entry in originals.iterdir():
            try:
                parsed = str(uuid.UUID(entry.name))
            except (ValueError, AttributeError):
                continue
            if parsed == entry.name.lower():
                _assert_no_reparse_chain(entry)
            if parsed == entry.name.lower() and entry.is_dir():
                targets.append(_checked_target(entry, archive_root))

    targets.extend(_portable_targets(archive_root, runtime_root, job_ids))
    db_path = archive_root / "meeting-index.sqlite3"
    for path in (db_path, *(Path(str(db_path) + suffix) for suffix in _SIDECARS)):
        if path.exists():
            targets.append(_checked_target(path, archive_root))
    # Archive and runtime roots may be the same, so discard duplicate targets.
    return tuple(dict.fromkeys(targets))


def build_plan(archive_root: str | Path, runtime_root: str | Path) -> ResetPlan:
    archive = _resolved(archive_root)
    runtime = _resolved(runtime_root)
    _assert_no_reparse_chain(Path(archive_root))
    _assert_no_reparse_chain(Path(runtime_root))
    meeting_ids = _meeting_ids(archive)
    job_ids = _select_job_ids(runtime, archive, meeting_ids)
    job_artifacts = []
    jobs_dir = runtime / "transcriptions"
    for job_id in sorted(job_ids):
        path = jobs_dir / job_id
        if path.exists():
            target = _checked_target(path, runtime)
            _assert_tree_no_reparse(target)
            job_artifacts.append(target)
    archive_targets = _archive_targets(archive, runtime, job_ids)
    for target in archive_targets:
        _assert_tree_no_reparse(target)
    return ResetPlan(
        archive_root=archive,
        runtime_root=runtime,
        meeting_ids=tuple(sorted(meeting_ids)),
        job_ids=tuple(sorted(job_ids)),
        job_artifacts=tuple(job_artifacts),
        archive_targets=archive_targets,
    )


def _host_lock(runtime_root: Path):
    from services.app_host import _acquire_lock

    lock_path = runtime_root / "desktop-host.lock"
    _checked_target(lock_path, runtime_root)
    try:
        runtime_root.mkdir(parents=True, exist_ok=True)
        handle = _acquire_lock(lock_path)
    except OSError as exc:
        raise ResetError(f"Could not acquire the desktop host lock: {exc}") from exc
    if handle is None:
        raise ResetError("The desktop host is running; close it before resetting the archive.")
    return handle


def _delete_targets(paths: tuple[Path, ...], allowed_root: Path) -> None:
    for path in paths:
        if not path.exists():
            continue
        target = _checked_target(path, allowed_root)
        _assert_tree_no_reparse(target)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()


def _delete_history_rows(runtime_root: Path, job_ids: tuple[str, ...]) -> None:
    if not job_ids:
        return
    database = runtime_root / "history.sqlite3"
    if not database.is_file():
        return
    _checked_target(database, runtime_root)
    try:
        with sqlite3.connect(database, timeout=30) as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.executemany("DELETE FROM queue_entries WHERE job_id=?", ((job_id,) for job_id in job_ids))
            db.executemany("DELETE FROM jobs WHERE id=?", ((job_id,) for job_id in job_ids))
            queued = db.execute("SELECT 1 FROM queue_entries LIMIT 1").fetchone()
            if queued is None:
                db.execute(
                    "UPDATE queue_state SET paused=0, reason_code=NULL, reason_message=NULL WHERE singleton=1"
                )
    except sqlite3.Error as exc:
        raise ResetError(f"Could not remove selected job history: {exc}") from exc


def apply_reset(plan: ResetPlan, confirm_root: str) -> dict:
    if confirm_root != str(plan.archive_root):
        raise ResetError(
            "Confirmation must exactly match the configured resolved archive root: "
            f"{plan.archive_root}"
        )
    _assert_no_reparse_chain(plan.archive_root)
    _assert_no_reparse_chain(plan.runtime_root)
    lock = _host_lock(plan.runtime_root)
    try:
        # Re-read both history and archive manifests while holding the host
        # lock, so jobs created after preview are captured before deletion.
        plan = build_plan(plan.archive_root, plan.runtime_root)
        # Keep the DB rows until all managed job directories have been removed.
        # That makes retries after partial filesystem failures rediscoverable.
        _delete_targets(plan.job_artifacts, plan.runtime_root)
        _delete_history_rows(plan.runtime_root, plan.job_ids)
        _delete_targets(plan.archive_targets, plan.archive_root)
    finally:
        lock.close()
    return {"deleted_jobs": len(plan.job_ids), "deleted_meetings": len(plan.meeting_ids)}


def _configured_roots() -> tuple[Path, Path, Path]:
    from services.meeting_settings import load_settings, settings_path

    project_root = Path(__file__).resolve().parents[1]
    settings = load_settings(project_root)
    raw_archive = os.environ.get("VT_MEETING_ROOT")
    if not raw_archive:
        try:
            raw_settings = json.loads(settings_path().read_text(encoding="utf-8"))
            if isinstance(raw_settings, dict):
                raw_archive = raw_settings.get("archive_root")
        except (OSError, ValueError):
            raw_archive = None
    archive_source = Path(raw_archive).expanduser() if raw_archive else Path(settings["archive_root"])
    _assert_no_reparse_chain(archive_source)
    raw_runtime = os.environ.get("VT_SIMPLE_RUNTIME_DIR")
    runtime_source = Path(raw_runtime).expanduser() if raw_runtime else project_root / ".runtime"
    _assert_no_reparse_chain(runtime_source)
    return _resolved(settings["archive_root"]), _resolved(runtime_source), project_root


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview or reset legacy managed archive data.")
    parser.add_argument(
        "--confirm-root",
        help="apply cleanup only when this exactly matches the configured resolved archive root",
    )
    args = parser.parse_args(argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        archive_root, runtime_root, _project_root = _configured_roots()
        plan = build_plan(archive_root, runtime_root)
        if args.confirm_root is None:
            print(json.dumps(plan.preview(), ensure_ascii=False, indent=2), file=stdout)
            return 0
        result = apply_reset(plan, args.confirm_root)
        print(json.dumps({**plan.preview(), **result}, ensure_ascii=False, indent=2), file=stdout)
        return 0
    except (ResetError, OSError, sqlite3.Error, ValueError) as exc:
        print(str(exc), file=stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
