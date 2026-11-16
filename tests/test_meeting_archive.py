import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

from services.meeting_archive import MeetingArchive


def test_unicode_import_dedup_and_stable_id(tmp_path):
    source = tmp_path / "исходный файл 東京.wav"
    source.write_bytes(b"audio bytes")
    archive = MeetingArchive(tmp_path / "archive")
    first = archive.import_file(source, title="Планёрка")
    second = archive.import_file(source, title="Other title")
    assert first["meeting_id"] == second["meeting_id"]
    assert archive.original_path(first["meeting_id"]).read_bytes() == b"audio bytes"
    assert first["sha256"] == hashlib.sha256(b"audio bytes").hexdigest()


def test_raw_rerun_preserves_edits_and_speaker_confirmation(tmp_path):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"recording")
    archive = MeetingArchive(tmp_path / "archive")
    meeting = archive.import_file(source)
    mid = meeting["meeting_id"]
    segments = [{"id": "spk:a/seg.1", "start": 2.0, "end": 3.0, "text": "raw", "speaker": "speaker_1"}]
    archive.write_transcript(mid, segments, raw_text="raw")
    archive.edit_segment(mid, "spk:a/seg.1", "исправлено")
    archive.confirm_speaker(mid, "speaker_1", "Алиса")
    archive.write_transcript(mid, [{**segments[0], "text": "new raw"}], raw_text="new raw")
    restarted = MeetingArchive(tmp_path / "archive")
    assert restarted.timeline(mid)["items"][0]["text"] == "исправлено"
    assert restarted.timeline(mid)["items"][0]["original_text"] == "new raw"
    assert json.loads((restarted.meeting_dir(mid) / "speakers.json").read_text(encoding="utf-8"))["confirmed"]["speaker_1"]["name"] == "Алиса"


def test_orphaned_edit_remains_visible_when_rerun_changes_ids(tmp_path):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"recording")
    archive = MeetingArchive(tmp_path / "archive")
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "stable:segment-1", "start": 2, "end": 3, "text": "before"}])
    archive.edit_segment(mid, "stable:segment-1", "corrected")
    archive.write_transcript(mid, [{"id": "other-seg", "start": 2, "end": 3, "text": "rerun"}])
    orphan = next(row for row in archive.timeline(mid)["items"] if row["kind"] == "orphan_edit")
    assert orphan["evidence_id"] == "stable:segment-1"
    assert orphan["text"] == "corrected"
    assert orphan["reason"] == "segment_missing_or_changed_after_rerun"


def test_chronological_timeline_and_glossary_undo_id(tmp_path):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"x")
    archive = MeetingArchive(tmp_path / "archive")
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "t2", "start": 4, "end": 5, "text": "later"}, {"id": "t1", "start": 1, "end": 2, "text": "first"}])
    archive.write_screen_events(mid, [{"id": "screen", "start": 3, "end": 4, "frame": "frames/1.jpg", "text": "screen", "regions": [], "tables": [], "reason": "change"}])
    assert [i["evidence_id"] for i in archive.timeline(mid)["items"]] == ["t1", "screen", "t2"]
    entry_id = archive.add_glossary_entry(mid, "Кэш", "cache")
    assert archive.remove_glossary_entry(mid, entry_id) == entry_id
    assert archive.add_glossary_entry(mid, "Кэш", "cache", entry_id) == entry_id
    assert archive.glossary(mid)[0]["id"] == entry_id


def test_read_screen_events_returns_screen_file_and_validates_meeting(tmp_path):
    source = tmp_path / "screen-read.wav"
    source.write_bytes(b"screen")
    archive = MeetingArchive(tmp_path / "archive")
    meeting_id = archive.import_file(source)["meeting_id"]
    expected = archive.write_screen_events(
        meeting_id, [{"id": "screen", "start": 3, "end": 4, "frame": "frames/one.jpg"}],
        status="ready", limitations=["partial OCR"])

    assert archive.read_screen_events(meeting_id) == expected
    with pytest.raises(ValueError, match="Invalid meeting_id"):
        archive.read_screen_events("not-a-meeting-id")
    with pytest.raises(KeyError):
        archive.read_screen_events(str(uuid.uuid4()))


def test_read_only_does_not_create_or_change_files(tmp_path):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"x")
    root = tmp_path / "archive"
    mid = MeetingArchive(root).import_file(source)["meeting_id"]
    before = {p.relative_to(root): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir") for p in root.rglob("*")}
    ro = MeetingArchive(root, read_only=True)
    ro.get_meeting(mid)
    after = {p.relative_to(root): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir") for p in root.rglob("*")}
    assert before == after
    missing = tmp_path / "never-created"
    MeetingArchive(missing, read_only=True)
    assert not missing.exists()


def test_literal_evidence_identifier_and_traversal_rejection(tmp_path):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"x")
    archive = MeetingArchive(tmp_path / "archive")
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "Turn:01/A.2", "start": 0, "end": 1, "text": "hello"}])
    with pytest.raises(ValueError):
        archive.meeting_dir("../../etc/passwd")
    assert archive.meeting_dir(mid).is_dir()


def test_timeline_uses_interval_overlap_and_pagination_completion(tmp_path):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"x")
    archive = MeetingArchive(tmp_path / "archive")
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "long", "start": 1, "end": 5, "text": "overlap"},
                                   {"id": "later", "start": 8, "end": 9, "text": "later"}])
    first = archive.timeline(mid, start=3, end=9, limit=1)
    assert first["items"][0]["evidence_id"] == "long"
    assert first["complete"] is False
    last = archive.timeline(mid, start=3, end=9, limit=1, offset=1)
    assert last["complete"] is True
    assert last["next_offset"] is None


def test_finalize_manifest_hashes_retained_frames_and_speaker_name(tmp_path):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"x")
    archive = MeetingArchive(tmp_path / "archive")
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "seg", "start": 0, "end": 1, "text": "hello", "speaker": "speaker_1"}])
    archive.confirm_speaker(mid, "speaker_1", "Alice")
    frame = archive.meeting_dir(mid) / "frames" / "sample.jpg"
    frame.parent.mkdir()
    frame.write_bytes(b"frame bytes")
    manifest = archive.finalize(mid)
    assert manifest["files"]["frames/sample.jpg"] == hashlib.sha256(b"frame bytes").hexdigest()
    assert "Alice: hello" in (archive.meeting_dir(mid) / "transcript.md").read_text(encoding="utf-8")


def test_refresh_editorial_exports_updates_exports_without_reading_media(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"x")
    archive = MeetingArchive(tmp_path / "archive")
    mid = archive.import_file(source, title="Weekly") ["meeting_id"]
    archive.write_transcript(mid, [{"id": "seg", "start": 0, "end": 1,
                                    "text": "raw line", "speaker": "speaker_1"}], raw_text="raw line")
    archive.write_source_artifacts(mid, raw_asr={"text": "untouched source"})
    archive.write_screen_events(mid, [{"id": "event", "start": 0, "text": "screen"}], status="ready")
    audio = archive.meeting_dir(mid) / "audio" / "part.opus"
    audio.parent.mkdir()
    audio.write_bytes(b"audio bytes")
    frame = archive.meeting_dir(mid) / "frames" / "frame.jpg"
    frame.parent.mkdir()
    frame.write_bytes(b"frame bytes")
    baseline = archive.finalize(mid)
    archive.edit_segment(mid, "seg", "corrected line")
    archive.confirm_speaker(mid, "speaker_1", "Алиса")
    archive.add_glossary_entry(mid, "Кэш", "cache")

    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        relative = path.relative_to(archive.meeting_dir(mid)) if path.is_relative_to(archive.meeting_dir(mid)) else None
        if relative and relative.parts[0] in {"audio", "frames"}:
            raise AssertionError(f"editorial refresh read media: {relative}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    manifest = archive.refresh_editorial_exports(mid)
    folder = archive.meeting_dir(mid)
    transcript_md = (folder / "transcript.md").read_text(encoding="utf-8")
    glossary_md = (folder / "glossary.md").read_text(encoding="utf-8")
    assert "Алиса: corrected line" in transcript_md
    assert "**Кэш**: cache" in glossary_md
    assert manifest["files"]["frames/frame.jpg"] == baseline["files"]["frames/frame.jpg"]
    assert manifest["files"]["audio/part.opus"] == baseline["files"]["audio/part.opus"]
    for name in ("transcript.md", "transcript.json", "glossary.md", "meeting.md", "meeting.json"):
        assert manifest["files"][name] == hashlib.sha256((folder / name).read_bytes()).hexdigest()
    assert manifest["status"]["overall"] == "ready"
    assert manifest["status"]["backup"] == "unverified"

    transcript = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
    raw_asr = json.loads((folder / "raw-asr.json").read_text(encoding="utf-8"))
    assert transcript["raw_segments"][0]["text"] == "raw line"
    assert transcript["overrides"]["seg"]["text"] == "corrected line"
    assert transcript["segments"][0]["text"] == "corrected line"
    assert raw_asr == {"text": "untouched source"}
    assert json.loads((folder / "speakers.json").read_text(encoding="utf-8"))["confirmed"]["speaker_1"]["name"] == "Алиса"


def test_finalize_exports_segments_once_then_preserves_transcript_mtime(tmp_path):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"audio")
    archive = MeetingArchive(tmp_path / "archive")
    meeting_id = archive.import_file(source)["meeting_id"]
    archive.write_transcript(meeting_id, [
        {"id": "segment", "start": 1, "end": 2, "text": "first export"}
    ], raw_text="first export")
    path = archive.meeting_dir(meeting_id) / "transcript.json"
    before = json.loads(path.read_text(encoding="utf-8"))
    assert "segments" not in before

    archive.finalize(meeting_id)
    exported = json.loads(path.read_text(encoding="utf-8"))
    assert exported["segments"] == exported["raw_segments"]
    exported_mtime = path.stat().st_mtime_ns

    archive.finalize(meeting_id)

    assert path.stat().st_mtime_ns == exported_mtime


def test_refresh_editorial_exports_without_prior_manifest_stays_pending(tmp_path):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"x")
    archive = MeetingArchive(tmp_path / "archive")
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "seg", "start": 0, "end": 1, "text": "line"}])
    manifest = archive.refresh_editorial_exports(mid)
    assert manifest["status"]["overall"] == "partial"
    assert manifest["readiness"] == {
        "source": "pending", "transcript": "pending", "screen": "pending", "audio": "pending"}


@pytest.mark.parametrize("change", ["screen", "active_stage"])
def test_refresh_editorial_exports_does_not_reuse_ready_after_screen_change_or_active_stage(tmp_path, change):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"x")
    archive = MeetingArchive(tmp_path / "archive")
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "seg", "start": 0, "end": 1, "text": "line"}])
    archive.write_screen_events(mid, [{"id": "one", "start": 0, "text": "ready screen"}], status="ready")
    assert archive.finalize(mid)["status"]["overall"] == "ready"
    if change == "screen":
        archive.write_screen_events(mid, [{"id": "two", "start": 0, "text": "new screen"}], status="ready")
    else:
        archive.update_meeting(mid, stage_state={"screen": {"status": "active"}})
    manifest = archive.refresh_editorial_exports(mid)
    assert manifest["status"]["overall"] == "partial"
    assert manifest["readiness"]["screen"] == "pending"


def test_refresh_editorial_exports_is_read_only_protected(tmp_path):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"x")
    root = tmp_path / "archive"
    mid = MeetingArchive(root).import_file(source)["meeting_id"]
    with pytest.raises(PermissionError):
        MeetingArchive(root, read_only=True).refresh_editorial_exports(mid)


def test_meeting_directory_rejects_symlink_escape(tmp_path):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"x")
    root = tmp_path / "archive"
    archive = MeetingArchive(root)
    mid = archive.import_file(source)["meeting_id"]
    target = tmp_path / "outside"
    target.mkdir()
    path = root / "meetings" / mid
    shutil.rmtree(path)
    try:
        path.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    with pytest.raises(ValueError, match="outside the archive"):
        archive.meeting_dir(mid)
