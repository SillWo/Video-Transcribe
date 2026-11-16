import shutil
import sqlite3

import pytest

from services.meeting_archive import MeetingArchive
from services.meeting_index import MeetingIndex
from services import meeting_index_fast


def _archive(tmp_path, name="fast"):
    root = tmp_path / name
    source = tmp_path / f"{name}.wav"
    source.write_bytes(name.encode())
    archive = MeetingArchive(root)
    meeting_id = archive.import_file(source)["meeting_id"]
    return root, archive, meeting_id


def test_fast_identifier_lookup_keeps_boundaries_confusables_and_full_pagination(tmp_path, monkeypatch):
    root, archive, meeting_id = _archive(tmp_path)
    archive.write_transcript(meeting_id, [
        {"id": "exact-one", "speaker": "speaker_1", "start": 1, "end": 2, "text": "Review ARCH-427"},
        {"id": "exact-two", "start": 3, "end": 4, "text": "ARCH-427 approved"},
        {"id": "longer-id", "start": 5, "end": 6, "text": "Review ARCH-4270"},
    ])
    archive.confirm_speaker(meeting_id, "speaker_1", "Alice Speaker")
    archive.update_meeting(meeting_id, title="Quarterly Review Title")
    archive.write_screen_events(meeting_id, [
        {"id": "normalized", "start": 7, "end": 8, "text": "Ticket ARCН-427"},
    ])
    monkeypatch.setattr(MeetingIndex, "FAST_SEARCH_ENABLED", True)
    monkeypatch.setattr(MeetingIndex, "FAST_SEARCH_MIN_DOCUMENTS", 1)
    index = MeetingIndex(root, embedding_model_path=root / "missing-model")
    index.index_meeting(meeting_id)
    monkeypatch.setattr(index, "_all_folder_rows", lambda *_args, **_kwargs: pytest.fail("fast path materialized all rows"))

    first = index.search("ARCH-427", mode="hybrid", limit=2)
    second = index.search("ARCH-427", mode="hybrid", limit=2, offset=2)

    assert first["total"] == 3
    assert [row["evidence_id"] for row in first["results"]] == ["exact-one", "exact-two"]
    assert first["results"][0]["match_kind"] == "identifier_exact"
    assert second["results"][0]["evidence_id"] == "normalized"
    assert second["results"][0]["match_kind"] == "identifier_normalized"
    assert second["next_offset"] is None
    speaker = index.search("Alice Speaker", mode="text")
    title = index.search("Quarterly Review Title", mode="text")
    assert any(row.get("confirmed_name") == "Alice Speaker" for row in speaker["results"])
    assert any(row["kind"] == "meeting" for row in title["results"])


def test_large_fast_text_search_caps_materialized_candidates_and_reports_truncation(tmp_path, monkeypatch):
    root, archive, meeting_id = _archive(tmp_path)
    archive.write_screen_events(meeting_id, [
        {"id": f"screen-{i:05}", "start": i * 31, "end": i * 31 + 1,
         "text": f"common marker row {i}"}
        for i in range(120)
    ])
    monkeypatch.setattr(MeetingIndex, "FAST_SEARCH_ENABLED", True)
    index = MeetingIndex(root, embedding_model_path=root / "missing-model")
    index.index_meeting(meeting_id)
    monkeypatch.setattr(MeetingIndex, "FAST_SEARCH_MIN_DOCUMENTS", 1)
    monkeypatch.setattr(meeting_index_fast, "CANDIDATE_LIMIT", 17)
    monkeypatch.setattr(index, "_all_folder_rows", lambda *_args, **_kwargs: pytest.fail("fast path materialized all rows"))
    original_validate = MeetingIndex._valid_document_row
    validated = []

    def count_validation(mid, row):
        validated.append((mid, row.get("evidence_id")))
        return original_validate(mid, row)

    monkeypatch.setattr(MeetingIndex, "_valid_document_row", staticmethod(count_validation))

    result = index.search("common marker", mode="text", limit=20)

    assert result["candidate_limited"] is True
    assert result["truncated"] is True
    assert result["status"] == "candidate_limited"
    assert result["candidate_limit_reason"]
    assert result["total"] == 17
    assert len(result["results"]) == 17
    assert len(validated) <= 17


def test_fast_search_uses_live_overlays_without_resurrecting_deleted_meetings(tmp_path, monkeypatch):
    root, archive, meeting_id = _archive(tmp_path, "overlay")
    archive.write_transcript(meeting_id, [
        {"id": "segment", "speaker": "speaker_1", "start": 0, "end": 1, "text": "old transcript"}
    ])
    archive.write_screen_events(meeting_id, [
        {"id": "old-screen", "start": 2, "end": 3, "text": "old display"}
    ])
    deleted_source = tmp_path / "deleted.wav"
    deleted_source.write_bytes(b"deleted")
    deleted_id = archive.import_file(deleted_source)["meeting_id"]
    archive.write_screen_events(deleted_id, [
        {"id": "deleted-screen", "start": 0, "end": 1, "text": "deleted display"}
    ])
    monkeypatch.setattr(MeetingIndex, "FAST_SEARCH_ENABLED", True)
    monkeypatch.setattr(MeetingIndex, "FAST_SEARCH_MIN_DOCUMENTS", 1)
    index = MeetingIndex(root, embedding_model_path=root / "missing-model")
    index.index_meeting(meeting_id)
    index.index_meeting(deleted_id)
    shutil.rmtree(archive.meeting_dir(deleted_id))

    archive.edit_segment(meeting_id, "segment", "current transcript phrase")
    archive.confirm_speaker(meeting_id, "speaker_1", "Alice New")
    archive.write_screen_events(meeting_id, [
        {"id": "current-screen", "start": 2, "end": 3, "text": "current display phrase"}
    ])
    new_source = tmp_path / "new.wav"
    new_source.write_bytes(b"new")
    new_id = archive.import_file(new_source)["meeting_id"]
    archive.write_screen_events(new_id, [
        {"id": "new-screen", "start": 0, "end": 1, "text": "fresh meeting marker"}
    ])
    readonly = MeetingIndex(root, read_only=True, embedding_model_path=root / "missing-model")
    monkeypatch.setattr(readonly, "_all_folder_rows", lambda *_args, **_kwargs: pytest.fail("fast path materialized all rows"))
    before_db = readonly.db_path.read_bytes()
    before_archive = {path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes())
                      for path in root.rglob("*") if path.is_file() and path != readonly.db_path}

    display = readonly.search("current display phrase", mode="text")
    speaker = readonly.search("Alice New", mode="text")
    fresh = readonly.search("fresh meeting marker", mode="text")
    removed = readonly.search("deleted display", mode="text")

    assert any(row["evidence_id"] == "current-screen" for row in display["results"])
    assert any(row.get("confirmed_name") == "Alice New" for row in speaker["results"])
    assert any(row["meeting_id"] == new_id for row in fresh["results"])
    assert all(row["meeting_id"] != deleted_id for row in removed["results"])
    assert readonly.db_path.read_bytes() == before_db
    after_archive = {path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes())
                     for path in root.rglob("*") if path.is_file() and path != readonly.db_path}
    assert after_archive == before_archive


def test_corrupt_fast_candidate_falls_back_to_live_meeting_without_writing(tmp_path, monkeypatch):
    root, archive, meeting_id = _archive(tmp_path, "corrupt")
    archive.write_screen_events(meeting_id, [
        {"id": "screen", "start": 10, "end": 11, "text": "safe visible phrase"}
    ])
    monkeypatch.setattr(MeetingIndex, "FAST_SEARCH_ENABLED", True)
    monkeypatch.setattr(MeetingIndex, "FAST_SEARCH_MIN_DOCUMENTS", 1)
    index = MeetingIndex(root, embedding_model_path=root / "missing-model")
    index.index_meeting(meeting_id)
    with sqlite3.connect(index.db_path) as db:
        db.execute("UPDATE search_documents SET payload_json='{}' WHERE meeting_id=? AND evidence_id='screen'",
                   (meeting_id,))
    readonly = MeetingIndex(root, read_only=True, embedding_model_path=root / "missing-model")
    monkeypatch.setattr(readonly, "_all_folder_rows", lambda *_args, **_kwargs: pytest.fail("fast path materialized all rows"))
    before_db = readonly.db_path.read_bytes()

    result = readonly.search("safe visible phrase", mode="text")

    assert any(row["evidence_id"] == "screen" and row["start"] == 10 for row in result["results"])
    assert readonly.db_path.read_bytes() == before_db
