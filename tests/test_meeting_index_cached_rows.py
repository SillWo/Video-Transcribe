import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from services.meeting_archive import MeetingArchive
from services.meeting_index import MeetingIndex


def _archive(tmp_path, name="archive"):
    root = tmp_path / name
    source = tmp_path / f"{name}.wav"
    source.write_bytes(name.encode())
    archive = MeetingArchive(root)
    meeting_id = archive.import_file(source)["meeting_id"]
    return root, archive, meeting_id


def _spy_screen_reads(monkeypatch):
    reads = []
    original = MeetingIndex._read_json

    def spy(path, default):
        if Path(path).name == "screen-events.json":
            reads.append(Path(path))
        return original(path, default)

    monkeypatch.setattr(MeetingIndex, "_read_json", staticmethod(spy))
    return reads


def test_indexed_search_uses_cached_rows_without_parsing_screen_events(tmp_path, monkeypatch):
    root, archive, meeting_id = _archive(tmp_path)
    archive.write_screen_events(meeting_id, [
        {"id": "screen", "start": 10, "end": 11, "text": "cached unique display"}
    ])
    index = MeetingIndex(root, embedding_model_path=root / "missing-model")
    index.index_meeting(meeting_id)
    reads = _spy_screen_reads(monkeypatch)

    results = index.search("cached unique display", mode="text")["results"]

    assert any(row["kind"] == "screen" and row["evidence_id"] == "screen" for row in results)
    assert reads == []


def test_changed_archive_content_is_visible_immediately_and_search_writes_nothing(tmp_path):
    root, archive, meeting_id = _archive(tmp_path)
    archive.write_transcript(meeting_id, [
        {"id": "segment", "speaker": "speaker_1", "start": 0, "end": 1, "text": "before edit"}
    ])
    archive.write_screen_events(meeting_id, [
        {"id": "old-screen", "start": 2, "end": 3, "text": "old display"}
    ])
    index = MeetingIndex(root, embedding_model_path=root / "missing-model")
    index.index_meeting(meeting_id)

    archive.edit_segment(meeting_id, "segment", "immediate transcript phrase")
    archive.confirm_speaker(meeting_id, "speaker_1", "Алиса Новая")
    archive.write_screen_events(meeting_id, [
        {"id": "new-screen", "start": 4, "end": 5, "text": "immediate screen phrase"}
    ])
    before_search = {path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes())
                     for path in root.rglob("*") if path.is_file()}

    transcript = index.search("immediate transcript phrase", mode="text")["results"]
    screen = index.search("immediate screen phrase", mode="text")["results"]
    speaker = index.search("Алиса Новая", mode="text")["results"]

    assert any(row["evidence_id"] == "segment" and row["text"] == "immediate transcript phrase"
               for row in transcript)
    assert any(row["evidence_id"] == "new-screen" for row in screen)
    assert any(row.get("confirmed_name") == "Алиса Новая" for row in speaker)
    after_search = {path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes())
                    for path in root.rglob("*") if path.is_file()}
    assert after_search == before_search


def test_job_metadata_does_not_invalidate_but_title_and_date_do(tmp_path, monkeypatch):
    root, archive, meeting_id = _archive(tmp_path)
    archive.write_screen_events(meeting_id, [
        {"id": "screen", "start": 1, "end": 2, "text": "stable screen content"}
    ])
    index = MeetingIndex(root, embedding_model_path=root / "missing-model")
    index.index_meeting(meeting_id)
    reads = _spy_screen_reads(monkeypatch)

    archive.update_meeting(meeting_id, status="screen_processing", stage_state={"screen": {"status": "active"}},
                           manifest_ready=False)
    cached = index.search("stable screen content", mode="text")["results"]
    assert any(row["evidence_id"] == "screen" for row in cached)
    assert reads == []

    archive.update_meeting(meeting_id, title="Current meeting title", meeting_date="2026-04-03",
                           date_provenance="editorial")
    updated = index.search("Current meeting title", mode="text")["results"]
    meeting_row = next(row for row in updated if row["kind"] == "meeting")
    assert meeting_row["meeting_title"] == "Current meeting title"
    assert meeting_row["meeting_date"] == "2026-04-03"
    assert meeting_row["date_provenance"] == "editorial"
    assert reads == [archive.meeting_dir(meeting_id) / "screen-events.json"]


def test_legacy_index_without_document_snapshots_falls_back_to_archive(tmp_path, monkeypatch):
    root, archive, meeting_id = _archive(tmp_path)
    archive.write_screen_events(meeting_id, [
        {"id": "screen", "start": 1, "end": 2, "text": "legacy visible display"}
    ])
    index = MeetingIndex(root, embedding_model_path=root / "missing-model")
    index.index_meeting(meeting_id)
    with sqlite3.connect(index.db_path) as db:
        db.execute("DROP TABLE document_snapshots")

    reads = _spy_screen_reads(monkeypatch)
    results = MeetingIndex(root, read_only=True, embedding_model_path=root / "missing-model").search(
        "legacy visible display", mode="text")["results"]

    assert any(row["evidence_id"] == "screen" for row in results)
    assert len(reads) == 1


@pytest.mark.parametrize("corruption", ["invalid_start", "missing_meeting_date"])
def test_malformed_valid_json_snapshot_falls_back_without_writing(tmp_path, monkeypatch, corruption):
    root, archive, meeting_id = _archive(tmp_path)
    archive.write_screen_events(meeting_id, [
        {"id": "screen", "start": 10, "end": 11, "text": "safe fallback phrase"}
    ])
    writer = MeetingIndex(root, embedding_model_path=root / "missing-model")
    writer.index_meeting(meeting_id)
    with sqlite3.connect(writer.db_path) as db:
        rows_json = db.execute(
            "SELECT rows_json FROM document_snapshots WHERE meeting_id=?", (meeting_id,)
        ).fetchone()[0]
        rows = json.loads(rows_json)
        screen = next(row for row in rows if row["kind"] == "screen")
        if corruption == "invalid_start":
            screen["start"] = "not-a-number"
        else:
            screen.pop("meeting_date")
        db.execute("UPDATE document_snapshots SET rows_json=? WHERE meeting_id=?",
                   (json.dumps(rows, ensure_ascii=False), meeting_id))

    reads = _spy_screen_reads(monkeypatch)
    index_path = writer.db_path
    before_db = (index_path.stat().st_mtime_ns, index_path.read_bytes())
    before_archive = {path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes())
                      for path in root.rglob("*") if path.is_file()}
    result = MeetingIndex(root, read_only=True, embedding_model_path=root / "missing-model").search(
        "safe fallback phrase", mode="text")

    row = next(row for row in result["results"] if row["evidence_id"] == "screen")
    assert row["start"] == 10
    assert len(reads) == 1
    assert (index_path.stat().st_mtime_ns, index_path.read_bytes()) == before_db
    after_archive = {path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes())
                     for path in root.rglob("*") if path.is_file()}
    assert after_archive == before_archive


def test_removed_meeting_is_not_resurrected_from_snapshot(tmp_path):
    root, archive, meeting_id = _archive(tmp_path)
    archive.write_screen_events(meeting_id, [
        {"id": "screen", "start": 1, "end": 2, "text": "removed meeting marker"}
    ])
    index = MeetingIndex(root, embedding_model_path=root / "missing-model")
    index.index_meeting(meeting_id)
    shutil.rmtree(archive.meeting_dir(meeting_id))

    result = index.search("removed meeting marker", mode="text")

    assert result["results"] == []
    assert result["total"] == 0
