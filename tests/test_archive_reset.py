import io
import json
import sqlite3
import uuid

import pytest

from services.archive_reset import ResetError, apply_reset, build_plan, main
from services.archive_catalog import sync_history, write_record
from services.app_host import _acquire_lock
from services.meeting_archive import MeetingArchive
from services.transcription_history import TranscriptionHistory


def _seed(tmp_path, archive_root=None, runtime_root=None):
    archive = archive_root or tmp_path / "archive"
    runtime = runtime_root or tmp_path / "runtime"
    archive.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "user-source.mp4"
    source.write_bytes(b"user-owned source")
    backup = tmp_path / "backup-snapshot"
    backup.mkdir()
    (backup / "snapshot.bin").write_bytes(b"backup")
    (runtime / "personal-dictionary.json").write_text('{"items":[]}', encoding="utf-8")

    meeting_id = str(uuid.uuid4())
    meeting_media = tmp_path / "meeting.mp4"
    meeting_media.write_bytes(b"archive media")
    meeting = MeetingArchive(archive).import_file(meeting_media, title="Old meeting")
    meeting_id = meeting["meeting_id"]

    store = TranscriptionHistory(runtime)
    foreign_root = tmp_path / "other-archive"
    foreign_meeting_id = str(uuid.uuid4())
    records = store.create_queued(
        [
            {"id": "1" * 32, "source": str(source), "archiveRoot": str(archive)},
            {"id": "2" * 32, "source": str(source)},
            {
                "id": "3" * 32,
                "source": str(source),
                "meetingId": meeting_id,
                "meetingRoot": str(archive),
                "archiveRoot": str(archive),
            },
            {"id": "4" * 32, "meetingId": meeting_id},
            {"id": "5" * 32, "archiveRoot": str(foreign_root)},
            {"id": "6" * 32, "meetingId": foreign_meeting_id},
            {
                "id": "7" * 32,
                "meetingId": foreign_meeting_id,
                "meetingRoot": str(foreign_root),
            },
        ]
    )
    for record in records:
        store.job_dir(record["id"]).mkdir(parents=True, exist_ok=True)
        (store.job_dir(record["id"]) / "artifact.bin").write_bytes(record["id"].encode())

    portable_id = "8" * 32
    write_record(
        archive,
        {"id": portable_id, "status": "completed", "title": "Old portable item"},
        "old transcript",
    )
    index = archive / "meeting-index.sqlite3"
    index.write_bytes(b"index")
    (archive / "meeting-index.sqlite3-wal").write_bytes(b"wal")
    (archive / "meeting-index.sqlite3-shm").write_bytes(b"shm")
    (archive / "unrelated.txt").write_text("keep", encoding="utf-8")
    return {
        "archive": archive.resolve(),
        "runtime": runtime.resolve(),
        "source": source,
        "backup": backup,
        "meeting_id": meeting_id,
        "meeting_media": meeting_media,
        "store": store,
        "portable_id": portable_id,
        "foreign_ids": {"5" * 32, "6" * 32, "7" * 32},
        "expected_job_ids": {"1" * 32, "2" * 32, "3" * 32, "4" * 32},
    }


def test_preview_lists_only_managed_targets_and_does_not_delete(tmp_path, monkeypatch):
    seeded = _seed(tmp_path)
    monkeypatch.setattr(
        "services.archive_reset._configured_roots",
        lambda: (seeded["archive"], seeded["runtime"], tmp_path),
    )
    output = io.StringIO()
    errors = io.StringIO()

    assert main([], stdout=output, stderr=errors) == 0
    preview = json.loads(output.getvalue())
    assert preview["meeting_count"] == 1
    assert preview["job_count"] == 4
    assert set(preview["jobs"]) == seeded["expected_job_ids"]
    assert (seeded["archive"] / "meetings" / seeded["meeting_id"] / "meeting.json").is_file()
    assert seeded["store"].get("1" * 32) is not None
    assert (seeded["archive"] / "meeting-index.sqlite3-wal").exists()
    assert errors.getvalue() == ""


def test_preview_is_read_only_even_while_desktop_host_lock_is_held(tmp_path, monkeypatch):
    seeded = _seed(tmp_path)
    monkeypatch.setattr(
        "services.archive_reset._configured_roots",
        lambda: (seeded["archive"], seeded["runtime"], tmp_path),
    )
    lock = _acquire_lock(seeded["runtime"] / "desktop-host.lock")
    assert lock is not None
    try:
        output = io.StringIO()
        assert main([], stdout=output, stderr=io.StringIO()) == 0
        assert json.loads(output.getvalue())["job_count"] == 4
    finally:
        lock.close()
    assert seeded["store"].get("1" * 32) is not None


def test_apply_requires_exact_resolved_archive_root_and_host_lock(tmp_path):
    seeded = _seed(tmp_path)
    plan = build_plan(seeded["archive"], seeded["runtime"])

    with pytest.raises(ResetError, match="exactly match"):
        apply_reset(plan, str(seeded["archive"]) + "\\.")

    lock = _acquire_lock(seeded["runtime"] / "desktop-host.lock")
    assert lock is not None
    try:
        with pytest.raises(ResetError, match="desktop host is running"):
            apply_reset(plan, str(seeded["archive"]))
    finally:
        lock.close()

    assert seeded["store"].get("1" * 32) is not None


def test_apply_removes_active_archive_and_linked_jobs_but_preserves_other_data(tmp_path):
    seeded = _seed(tmp_path)
    plan = build_plan(seeded["archive"], seeded["runtime"])
    result = apply_reset(plan, str(seeded["archive"]))

    assert result == {"deleted_jobs": 4, "deleted_meetings": 1}
    assert not (seeded["archive"] / "meetings" / seeded["meeting_id"]).exists()
    assert not (seeded["archive"] / "originals" / seeded["meeting_id"]).exists()
    assert not (seeded["archive"] / "transcriptions" / seeded["portable_id"]).exists()
    assert not (seeded["archive"] / "meeting-index.sqlite3").exists()
    assert not (seeded["archive"] / "meeting-index.sqlite3-wal").exists()
    assert (seeded["archive"] / "unrelated.txt").read_text(encoding="utf-8") == "keep"
    assert seeded["source"].read_bytes() == b"user-owned source"
    assert (seeded["backup"] / "snapshot.bin").read_bytes() == b"backup"
    assert (seeded["runtime"] / "personal-dictionary.json").exists()

    for job_id in seeded["expected_job_ids"]:
        assert seeded["store"].get(job_id) is None
        assert not (seeded["runtime"] / "transcriptions" / job_id).exists()
    for job_id in seeded["foreign_ids"]:
        assert seeded["store"].get(job_id) is not None
        assert (seeded["runtime"] / "transcriptions" / job_id).exists()
    assert seeded["store"].queue_ids() == sorted(seeded["foreign_ids"])

    restarted = TranscriptionHistory(seeded["runtime"])
    summary = sync_history(seeded["archive"], restarted)
    assert summary["synced"] == 0
    assert restarted.get("2" * 32) is None


def test_apply_can_retry_after_partial_job_artifact_io_failure(tmp_path, monkeypatch):
    import services.archive_reset as reset

    seeded = _seed(tmp_path)
    plan = build_plan(seeded["archive"], seeded["runtime"])
    original_rmtree = reset.shutil.rmtree
    calls = {"count": 0}

    def fail_once(path, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated busy artifact")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(reset.shutil, "rmtree", fail_once)
    with pytest.raises(OSError, match="simulated busy"):
        apply_reset(plan, str(seeded["archive"]))
    assert all(seeded["store"].get(job_id) is not None for job_id in seeded["expected_job_ids"])

    monkeypatch.setattr(reset.shutil, "rmtree", original_rmtree)
    result = apply_reset(plan, str(seeded["archive"]))
    assert result["deleted_jobs"] == 4
    assert all(seeded["store"].get(job_id) is None for job_id in seeded["expected_job_ids"])


def test_runtime_archive_overlap_preserves_unselected_job_directory(tmp_path):
    root = tmp_path / "shared"
    store = TranscriptionHistory(root)
    foreign = "9" * 32
    store.create_queued([{"id": foreign, "archiveRoot": str(tmp_path / "other") }])
    store.job_dir(foreign).mkdir(parents=True, exist_ok=True)
    job_artifact = store.job_dir(foreign) / "keep.bin"
    job_artifact.write_bytes(b"foreign runtime artifact")
    portable = "a" * 32
    write_record(root, {"id": portable, "title": "portable"}, "portable transcript")

    plan = build_plan(root, root)
    assert foreign not in plan.job_ids
    assert root / "transcriptions" / foreign not in plan.archive_targets
    assert root / "transcriptions" / portable in plan.archive_targets


def test_apply_clears_stale_queue_pause_when_no_foreign_queue_remains(tmp_path):
    archive = tmp_path / "archive"
    runtime = tmp_path / "runtime"
    archive.mkdir()
    store = TranscriptionHistory(runtime)
    store.create_queued([{"id": "c" * 32, "archiveRoot": str(archive)}])
    store.set_queue_state(True, {"code": "stale", "message": "Old failure"})

    plan = build_plan(archive, runtime)
    apply_reset(plan, str(archive.resolve()))

    assert store.queue_ids() == []
    assert store.queue_state() == {"paused": False, "pauseReason": None}


def test_symlink_target_is_rejected_before_any_deletion(tmp_path, monkeypatch):
    seeded = _seed(tmp_path)
    link = seeded["archive"] / "meetings" / str(uuid.uuid4())
    link.mkdir()
    import services.archive_reset as reset

    real_check = reset._is_reparse
    monkeypatch.setattr(reset, "_is_reparse", lambda path: path == link or real_check(path))
    with pytest.raises(ResetError, match="symbolic-link"):
        build_plan(seeded["archive"], seeded["runtime"])
    assert seeded["store"].get("1" * 32) is not None


def test_missing_archive_and_runtime_preview_does_not_create_directories(tmp_path):
    archive = tmp_path / "absent-archive"
    runtime = tmp_path / "absent-runtime"
    plan = build_plan(archive, runtime)
    assert plan.job_ids == ()
    assert plan.archive_targets == ()
    assert not archive.exists()
    assert not runtime.exists()
