import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fastapi import HTTPException

from services.archive_api import TextEdit, edit_text, get_item, list_archive
from services.archive_catalog import list_items, read_record, sync_history, write_record
from services.meeting_mcp import ArchiveTools
from services.meeting_archive import MeetingArchive
from services.transcription_history import TranscriptionHistory


class _WhitespaceEncoding:
    def __init__(self, text):
        import re

        self.offsets = [
            (match.start(), match.end()) for match in re.finditer(r"\S+", text)
        ]
        self.ids = list(range(len(self.offsets)))


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return _WhitespaceEncoding(text)


class _StubEmbedder:
    available = True
    model_hash = "stub-e5-v1"

    def encode_passages(self, texts):
        return np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (len(texts), 1))

    def encode_queries(self, texts):
        return np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (len(texts), 1))


def _completed(store, job_id, **extra):
    row = store.create(
        {
            "id": job_id,
            "sourceType": "file",
            "source": "C:/original/video.mp4",
            "title": "Portable transcript",
            **extra,
        }
    )
    store.update(job_id, status="completed")
    store.write_text(job_id, "The corrected phrase appears here.", 0)
    return store.get(job_id)


def test_sync_history_migrates_once_excludes_meeting_jobs_and_binds_first_root(
    tmp_path,
):
    store = TranscriptionHistory(tmp_path / "db")
    _completed(store, "a" * 32)
    _completed(store, "b" * 32, meetingId="meeting-1")
    first, second = tmp_path / "archive-one", tmp_path / "archive-two"

    summary = sync_history(first, store)
    migrated = read_record(first, "a" * 32)
    assert summary["synced"] == 1
    assert summary["skipped_meeting"] == 1
    assert migrated["text"] == "The corrected phrase appears here."
    assert migrated["item"]["source"] == "C:/original/video.mp4"
    assert read_record(first, "b" * 32) is None

    before = (migrated["directory"] / "item.json").stat().st_mtime_ns
    sync_history(first, store)
    after = (migrated["directory"] / "item.json").stat().st_mtime_ns
    assert before == after
    sync_history(second, store)
    assert read_record(second, "a" * 32) is None


def test_sync_claim_preserves_legacy_metadata_and_sqlite_timing(tmp_path):
    store = TranscriptionHistory(tmp_path / "history")
    completed_id = "1" * 32
    queued_id = "2" * 32
    _completed(store, completed_id)
    store.create({"id": queued_id, "sourceType": "file", "source": "queued.wav"})
    store.write_text(queued_id, "Queued transcript.", 0)
    historic = {
        completed_id: {
            "createdAt": "2018-04-03T02:01:00+00:00",
            "updatedAt": "2018-04-03T03:02:00+00:00",
            "startedAt": "2018-04-03T02:02:00+00:00",
            "finishedAt": "2018-04-03T02:03:00+00:00",
            "queueWaitSeconds": 60.0,
            "processingDurationSeconds": 60.0,
            "durationSeconds": 120.0,
            "revision": 7,
            "legacyMetadata": {"keep": ["exact", 17]},
        },
        queued_id: {
            "createdAt": "2019-05-06T07:08:00+00:00",
            "updatedAt": "2019-05-06T07:09:00+00:00",
            "startedAt": None,
            "finishedAt": None,
            "queueWaitSeconds": None,
            "processingDurationSeconds": None,
            "durationSeconds": None,
            "revision": 3,
            "legacyMetadata": {"keep": ["queued", 23]},
        },
    }
    sql_before = {}
    with store._connect() as db:
        for job_id, values in historic.items():
            record = store.get(job_id)
            record.update(values)
            db.execute(
                "UPDATE jobs SET created_at=?, updated_at=?, status=?, stage=?, revision=?, metadata=? WHERE id=?",
                (
                    record["createdAt"],
                    record["updatedAt"],
                    record["status"],
                    record["stage"],
                    record["revision"],
                    json.dumps(record, ensure_ascii=False),
                    job_id,
                ),
            )
            sql_before[job_id] = tuple(
                db.execute(
                    "SELECT created_at, updated_at, status, stage, revision FROM jobs WHERE id=?",
                    (job_id,),
                ).fetchone()
            )
    before = {record["id"]: record for record in store.list()}
    archive_root = tmp_path / "archive"

    sync_history(archive_root, store)
    after = {record["id"]: record for record in store.list()}
    with store._connect() as db:
        for job_id in historic:
            sql_after = tuple(
                db.execute(
                    "SELECT created_at, updated_at, status, stage, revision FROM jobs WHERE id=?",
                    (job_id,),
                ).fetchone()
            )
            assert sql_after == sql_before[job_id]
            expected = {**before[job_id], "archiveRoot": str(archive_root.resolve())}
            assert after[job_id] == expected
            portable = json.loads(
                (archive_root / "transcriptions" / job_id / "item.json").read_text(
                    encoding="utf-8"
                )
            )
            assert portable == expected

    sync_history(archive_root, store)
    assert {record["id"]: record for record in store.list()} == after


def test_archive_root_claim_serializes_independent_history_instances(tmp_path):
    database = tmp_path / "history"
    first_store = TranscriptionHistory(database)
    job_id = "3" * 32
    first_store.create({"id": job_id, "source": "legacy.wav", "custom": {"v": 1}})
    second_store = TranscriptionHistory(database)
    before = first_store.get(job_id)
    barrier = threading.Barrier(2)

    def claim(store, root):
        barrier.wait(timeout=5)
        return store.claim_archive_root(job_id, str(root.resolve()))

    first_root = tmp_path / "archive-first"
    second_root = tmp_path / "archive-second"
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [
            pool.submit(claim, first_store, first_root),
            pool.submit(claim, second_store, second_root),
        ]
        records = [future.result(timeout=10) for future in outcomes]

    roots = {record["archiveRoot"] for record in records}
    assert len(roots) == 1
    stored = first_store.get(job_id)
    assert stored["archiveRoot"] in {
        str(first_root.resolve()),
        str(second_root.resolve()),
    }
    assert stored == {**before, "archiveRoot": stored["archiveRoot"]}


def test_catalog_recheck_uses_compact_summary_without_loading_full_records(
    tmp_path, monkeypatch
):
    store = TranscriptionHistory(tmp_path / "db")
    _completed(store, "9" * 32, debugPayload={"trace": "x" * 100_000})
    root = tmp_path / "archive"
    sync_history(root, store)
    calls = {"get": 0, "text": 0}
    original_get = store.get
    original_read_text = store.read_text

    def count_get(job_id):
        calls["get"] += 1
        return original_get(job_id)

    def count_text(job_id):
        calls["text"] += 1
        return original_read_text(job_id)

    monkeypatch.setattr(store, "get", count_get)
    monkeypatch.setattr(store, "read_text", count_text)
    sync_history(root, store)
    assert calls == {"get": 0, "text": 0}


def test_restored_standalone_transcription_can_be_read_edited_and_conflicts(
    tmp_path, monkeypatch
):
    import services.archive_api as api

    root = tmp_path / "portable"
    store = TranscriptionHistory(tmp_path / "empty-db")
    item = {
        "id": "c" * 32,
        "status": "completed",
        "revision": 0,
        "createdAt": "2026-01-01T00:00:00+00:00",
        "title": "Restored",
        "sourceType": "file",
    }
    write_record(root, item, "A typo phrase remains in context.")
    monkeypatch.setattr(api, "_root", lambda: root)
    monkeypatch.setattr(api, "_store", lambda: store)

    detail = get_item("transcription", item["id"])
    assert detail["text"] == "A typo phrase remains in context."
    edited = edit_text(
        item["id"], TextEdit(text="A fixed phrase remains in context.", baseRevision=0)
    )
    assert edited["revision"] == 1
    assert read_record(root, item["id"])["text"] == "A fixed phrase remains in context."
    try:
        edit_text(item["id"], TextEdit(text="Stale edit.", baseRevision=0))
    except HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("stale edit must conflict")


def test_archive_root_binding_prevents_duplicate_id_cross_root_edits(
    tmp_path, monkeypatch
):
    import services.archive_api as api

    root_a = tmp_path / "archive-a"
    root_b = tmp_path / "archive-b"
    job_id = "f" * 32
    store = TranscriptionHistory(tmp_path / "history-db")
    _completed(store, job_id, archiveRoot=str(root_a))
    store.write_text(job_id, "History A transcript.", 0)
    portable_item = {
        "id": job_id,
        "status": "completed",
        "revision": 0,
        "createdAt": "2026-02-01T00:00:00+00:00",
        "title": "Restored copy B",
        "archiveRoot": str(root_a),
    }
    write_record(root_b, portable_item, "Portable B transcript.")
    monkeypatch.setattr(api, "_root", lambda: root_b)
    monkeypatch.setattr(api, "_store", lambda: store)

    detail = get_item("transcription", job_id)
    assert detail["archive_root"] == str(root_b.resolve())
    assert detail["item"]["archive_root"] == str(root_b.resolve())
    assert detail["text"] == "Portable B transcript."
    catalog = list_archive(kind="transcription")
    assert catalog["archive_root"] == str(root_b.resolve())
    assert catalog["items"][0]["archive_root"] == str(root_b.resolve())
    assert catalog["items"][0]["kind"] == "audio"

    try:
        edit_text(
            job_id,
            TextEdit(
                text="Delayed edit from A.",
                baseRevision=0,
                expectedArchiveRoot=str(root_a.resolve()),
            ),
        )
    except HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("a delayed edit from another archive must conflict")

    assert read_record(root_b, job_id)["text"] == "Portable B transcript."
    assert store.read_text(job_id) == "History A transcript."


def test_mcp_reads_and_searches_portable_transcriptions_without_writes(tmp_path):
    root = tmp_path / "archive"
    item = {
        "id": "d" * 32,
        "status": "completed",
        "revision": 0,
        "createdAt": "2026-01-01T00:00:00+00:00",
        "title": "Runbook",
    }
    write_record(root, item, "Quasar fallback handled the outage in January.")
    folder = root / "transcriptions" / item["id"]
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()
    }
    tools = ArchiveTools(root)
    rows = json.loads(tools.find_archive_items("Runbook"))["items"]
    read = json.loads(tools.read_archive_item(item["id"]))
    result = json.loads(tools.search_archive("Quasar fallback"))["items"]
    after = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()
    }
    assert rows[0]["kind"] == "audio"
    assert read["text"].startswith("Quasar fallback")
    assert any(row["job_id"] == item["id"] for row in result)
    assert before == after


def test_universal_search_returns_meeting_and_ordinary_hits_with_stable_pages(tmp_path):
    root = tmp_path / "archive"
    media = tmp_path / "meeting.mp4"
    media.write_bytes(b"synthetic meeting media")
    archive = MeetingArchive(root)
    meeting = archive.import_file(media, title="Deployment review")
    archive.write_transcript(
        meeting["meeting_id"],
        [
            {
                "id": "speech-1",
                "start": 12,
                "end": 14,
                "speaker": "spk-1",
                "text": "Use ARCH-427 for the rollback.",
            }
        ],
        raw_text="Use ARCH-427 for the rollback.",
    )
    archive.finalize(meeting["meeting_id"])
    ordinary_id = "7" * 32
    write_record(
        root,
        {
            "id": ordinary_id,
            "status": "completed",
            "revision": 0,
            "createdAt": "2026-01-02T00:00:00+00:00",
            "title": "Runbook",
        },
        "x" * 1800 + " ARCH-427 is approved near the end.",
    )
    write_record(
        root,
        {
            "id": "8" * 32,
            "status": "completed",
            "revision": 0,
            "createdAt": "2026-01-03T00:00:00+00:00",
            "title": "Near match",
        },
        "ARCH-4270 is a different identifier.",
    )
    tools = ArchiveTools(root)

    first = json.loads(tools.search_archive("ARCH-427", limit=1, mode="text"))
    second = json.loads(
        tools.search_archive("ARCH-427", limit=1, offset=1, mode="text")
    )
    results = first["items"] + second["items"]
    assert first["total"] == 2
    assert first["next_offset"] == 1
    assert second["next_offset"] is None
    assert {row["kind"] for row in results} == {"audio", "video"}
    ordinary = next(row for row in results if row["kind"] == "audio")
    assert ordinary["job_id"] == ordinary_id
    assert "ARCH-427 is approved near the end" in ordinary["text"]
    assert not any(row.get("job_id") == "8" * 32 for row in results)


def test_semantic_index_covers_tail_of_long_transcription_in_chunks(tmp_path):
    from services.meeting_index import MeetingIndex

    root = tmp_path / "archive"
    job_id = "6" * 32
    text = " ".join(["ordinary"] * 1100 + ["TAILKEYWORD"])
    record = {
        "id": job_id,
        "status": "completed",
        "revision": 0,
        "createdAt": "2026-01-01T00:00:00+00:00",
        "title": "Long transcript",
    }
    write_record(root, record, text)
    index = MeetingIndex(root, embedder=_StubEmbedder())
    index._chunk_tokenizer = _WhitespaceTokenizer()
    index._chunk_tokenizer_loaded = True

    indexed = index.index_transcription(record, text)
    rows = index._transcription_rows(job_id)
    assert indexed["indexed_items"] == len(rows)
    assert len(rows) > 2
    assert all(len(_WhitespaceEncoding(row["text"]).ids) <= 384 for row in rows)
    assert rows[-1]["char_end"] == len(text)
    assert "TAILKEYWORD" in rows[-1]["text"]

    reader = MeetingIndex(root, read_only=True, embedder=_StubEmbedder())
    reader._chunk_tokenizer = _WhitespaceTokenizer()
    reader._chunk_tokenizer_loaded = True
    result = reader.search("TAILKEYWORD", mode="semantic")
    assert result["mode"] == "semantic"
    assert result["semantic"]["status"] == "ready"
    assert any(row["text"].endswith("TAILKEYWORD") for row in result["results"])


def test_slow_index_worker_does_not_hold_history_transaction_lock(
    tmp_path, monkeypatch
):
    from services.meeting_index import MeetingIndex

    started = threading.Event()
    release = threading.Event()

    def slow_index(_self, _record, _text=None, **_kwargs):
        started.set()
        release.wait(timeout=10)
        return {"status": "ready", "indexed_items": 1}

    monkeypatch.setattr(MeetingIndex, "index_transcription", slow_index)
    root = tmp_path / "archive"
    store = TranscriptionHistory(tmp_path / "runtime")
    job_id = "5" * 32
    store.create({"id": job_id, "archiveRoot": str(root)})
    store.write_text(job_id, "A completed transcript.", 0)
    store.update(job_id, status="completed")
    assert started.wait(timeout=5)
    try:
        record = store.get(job_id)
        store.replace_text(job_id, "An edited transcript.", record["revision"])
        assert store.get(job_id)["revision"] == record["revision"] + 1
    finally:
        release.set()


def test_active_history_delete_rejection_keeps_portable_record(tmp_path, monkeypatch):
    import services.archive_api as api

    root = tmp_path / "archive"
    store = TranscriptionHistory(tmp_path / "runtime")
    store.create({"id": "e" * 32, "archiveRoot": str(root)})
    store.write_text("e" * 32, "active transcript", 0)
    monkeypatch.setattr(api, "_root", lambda: root)
    monkeypatch.setattr(api, "_store", lambda: store)

    try:
        api.delete_transcription("e" * 32)
    except HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("active history rows cannot be deleted")
    assert read_record(root, "e" * 32)["text"] == "active transcript"


def test_concurrent_restored_text_edits_share_revision_lock(tmp_path, monkeypatch):
    import services.archive_api as api

    root = tmp_path / "portable"
    store = TranscriptionHistory(tmp_path / "empty-db")
    item = {
        "id": "f" * 32,
        "status": "completed",
        "revision": 0,
        "createdAt": "2026-01-01T00:00:00+00:00",
        "title": "Restored",
    }
    write_record(root, item, "Initial text with context.")
    monkeypatch.setattr(api, "_root", lambda: root)
    monkeypatch.setattr(api, "_store", lambda: store)
    original_read = api.read_record
    barrier = threading.Barrier(2)

    def synchronized_read(archive_root, job_id):
        result = original_read(archive_root, job_id)
        if threading.current_thread().name.startswith("edit-"):
            barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(api, "read_record", synchronized_read)
    outcomes = []

    def edit(text):
        try:
            outcomes.append(
                api.edit_text(item["id"], TextEdit(text=text, baseRevision=0))[
                    "revision"
                ]
            )
        except HTTPException as exc:
            outcomes.append(exc.status_code)

    threads = [
        threading.Thread(
            target=edit, args=(f"Edit {i} text with context.",), name=f"edit-{i}"
        )
        for i in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert sorted(outcomes, key=str) == [1, 409]
    assert read_record(root, item["id"])["item"]["revision"] == 1


def test_archive_api_uses_audio_video_kinds_and_accepts_legacy_aliases(
    tmp_path, monkeypatch
):
    import services.archive_api as api

    root = tmp_path / "archive"
    store = TranscriptionHistory(tmp_path / "history")
    audio_id = "a" * 32
    write_record(
        root,
        {
            "id": audio_id,
            "status": "completed",
            "revision": 0,
            "createdAt": "2026-01-02T00:00:00+00:00",
            "title": "Field recording",
        },
        "Transcript text.",
    )
    video = MeetingArchive(root)
    media_file = tmp_path / "meeting.mp4"
    media_file.write_bytes(b"test video")
    video_item = video.import_file(media_file, title="Planning session")
    video.finalize(video_item["meeting_id"])
    monkeypatch.setattr(api, "_root", lambda: root)
    monkeypatch.setattr(api, "_store", lambda: store)

    for kind, expected in (("all", {"audio", "video"}), ("audio", {"audio"}),
                           ("transcription", {"audio"}), ("video", {"video"}),
                           ("meeting", {"video"})):
        rows = list_archive(kind=kind)["items"]
        assert {row["kind"] for row in rows} == expected
    assert get_item("transcription", audio_id)["item"]["kind"] == "audio"
    assert get_item("meeting", video_item["meeting_id"])["item"]["kind"] == "video"


def test_pending_audio_is_hidden_from_catalog_reads_and_stale_search_index(tmp_path):
    from services.meeting_index import MeetingIndex
    from services.archive_catalog import _atomic

    root = tmp_path / "archive"
    job_id = "b" * 32
    record = {
        "id": job_id,
        "status": "completed",
        "revision": 0,
        "createdAt": "2026-01-01T00:00:00+00:00",
        "title": "Pending import",
    }
    write_record(root, record, "A distinctive pending transcript phrase.")
    index = MeetingIndex(root)
    index.index_transcription(record, "A distinctive pending transcript phrase.")

    folder = root / "transcriptions" / job_id
    pending = {**record, "archivePublication": "pending", "archiveKind": "audio"}
    _atomic(folder / "item.json", pending)
    _atomic(folder / "catalog.json", pending)

    assert read_record(root, job_id) is None
    assert not any(row["id"] == job_id for row in list_items(root))
    result = index.search("distinctive pending transcript", mode="text")
    assert result["total"] == 0
