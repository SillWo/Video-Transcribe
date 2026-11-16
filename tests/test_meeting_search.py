import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from services.meeting_archive import MeetingArchive
from services.meeting_index import MeetingIndex
from services.meeting_search import ReadIndexProxy


def _fixture_archive(tmp_path):
    root = tmp_path / "archive"
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"search worker fixture")
    archive = MeetingArchive(root)
    meeting_id = archive.import_file(source, title="Release discussion") ["meeting_id"]
    archive.write_transcript(meeting_id, [
        {"id": "release", "start": 0, "end": 1,
         "text": "Nightly build failed, so launch was postponed"},
        {"id": "other", "start": 2, "end": 3,
         "text": "A person reviewed the office seating plan"},
    ])
    return root, archive, meeting_id


def _snapshot(root):
    return {path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
            for path in root.rglob("*")}


def test_proxy_imports_only_stdlib_and_does_not_load_model_modules():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import sys, json; import services.meeting_search; "
         "print(json.dumps({m: m in sys.modules for m in ('numpy','onnxruntime','tokenizers')}))"],
        cwd=project_root, capture_output=True, text=True, timeout=20, check=True,
    )
    assert json.loads(result.stdout) == {"numpy": False, "onnxruntime": False, "tokenizers": False}


def test_proxy_returns_structured_timeout_error(tmp_path, monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["python", "-m", "services.meeting_index"], 120)

    monkeypatch.setattr("services.meeting_search.subprocess.run", timeout)
    result = ReadIndexProxy(tmp_path).search("query")
    assert result["status"] == "worker_timeout"
    assert result["error"]["type"] == "TimeoutExpired"
    assert result["error"]["timeout_seconds"] == 120


def test_subprocess_text_search_is_read_only_on_unindexed_archive(tmp_path):
    root, _archive, meeting_id = _fixture_archive(tmp_path)
    before = _snapshot(root)
    proxy = ReadIndexProxy(root, timeout=120)
    result = proxy.search("Nightly build", mode="text")
    after = _snapshot(root)
    assert result["status"] == "archive_fallback_index_missing"
    assert result["results"][0]["meeting_id"] == meeting_id
    assert result["results"][0]["evidence_id"] == "release"
    assert before == after
    assert not (root / MeetingIndex.DB_NAME).exists()


def test_read_only_search_during_atomic_database_update_uses_archive_and_writes_no_sidecars(tmp_path):
    root, _archive, meeting_id = _fixture_archive(tmp_path)
    index = MeetingIndex(root, embedding_model_path=root / "missing-model")
    index.rebuild()
    # Keep an uncommitted change under an exclusive DELETE-journal lock. The
    # reader must either see a complete snapshot or fall back to archive files.
    writer = sqlite3.connect(index.db_path, timeout=1)
    writer.execute("BEGIN EXCLUSIVE")
    writer.execute("UPDATE evidence SET text='partial damaged update'")
    result = ReadIndexProxy(root, timeout=20).search("nightly build", mode="text")
    assert any(row["meeting_id"] == meeting_id and "Nightly build" in row["text"]
               for row in result["results"])
    assert not list(root.glob("meeting-index.sqlite3-shm"))
    writer.rollback()
    writer.close()


def test_legacy_wal_reader_falls_back_and_writer_migrates_under_archive_lock(tmp_path):
    root, archive, meeting_id = _fixture_archive(tmp_path)
    index = MeetingIndex(root, embedding_model_path=root / "missing-model")
    index.rebuild()
    with sqlite3.connect(index.db_path) as db:
        assert db.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
    header = index.db_path.read_bytes()[:20]
    assert header[18:20] == b"\x02\x02"

    before_files = {path.name for path in root.iterdir()}
    result = MeetingIndex(root, read_only=True, embedding_model_path=root / "missing-model").search(
        "nightly build", mode="text")
    assert result["index_state"] == "needs_reindex"
    assert result["status"] == "archive_fallback_legacy_wal_needs_reindex"
    assert any(row["meeting_id"] == meeting_id for row in result["results"])
    assert {path.name for path in root.iterdir()} == before_files
    assert not (root / "meeting-index.sqlite3-shm").exists()

    index.index_meeting(meeting_id)
    migrated = index.db_path.read_bytes()[:20]
    assert migrated[18:20] == b"\x01\x01"
