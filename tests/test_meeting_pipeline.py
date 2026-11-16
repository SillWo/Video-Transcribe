import json
from pathlib import Path

from services.meeting_archive import MeetingArchive
from services.meeting_index import MeetingIndex
from services.meeting_pipeline import MeetingStageFailure, meeting_segments, persist_speech, run_stages


def test_real_diarization_turns_drive_timed_speech_ids_and_raw_asr_is_retained(tmp_path):
    result = {
        "text": "long combined asr result",
        "segments": [{"start": 0, "end": 30, "text": "raw words", "words": [{"start": 1, "end": 2, "text": "raw"}]}],
        "diarization": {"turns": [
            {"start": 1.25, "end": 2.0, "text": "first person", "speakerIds": [1]},
            {"start": 2.1, "end": 3.5, "text": "second person", "speakerIds": [2]},
        ]},
    }
    segments = meeting_segments(result)
    assert [(s["start"], s["end"], s["text"]) for s in segments] == [
        (1.25, 2.0, "first person"), (2.1, 3.5, "second person")]
    assert segments[0]["id"] == meeting_segments({**result, "diarization": {"turns": [
        {"start": 1.25, "end": 2.0, "text": "edited recognition", "speakerIds": [1]}]}})[0]["id"]

    source = tmp_path / "recording.wav"
    source.write_bytes(b"audio")
    archive = MeetingArchive(tmp_path / "archive")
    meeting_id = archive.import_file(source)["meeting_id"]
    persist_speech(tmp_path / "archive", meeting_id, result)
    reopened = MeetingArchive(tmp_path / "archive")
    timeline = reopened.timeline(meeting_id)
    assert [row["text"] for row in timeline["items"]] == ["first person", "second person"]
    raw = json.loads((reopened.meeting_dir(meeting_id) / "raw-asr.json").read_text(encoding="utf-8"))
    assert raw["segments"][0]["words"][0]["text"] == "raw"


def test_speaker_id_reordering_matches_same_audio_intervals_and_keeps_names(tmp_path):
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"audio")
    root = tmp_path / "archive"
    archive = MeetingArchive(root)
    meeting_id = archive.import_file(source)["meeting_id"]
    first = {"segments": [], "diarization": {"turns": [
        {"start": 0, "end": 1, "text": "one", "speakerIds": [1]},
        {"start": 1, "end": 2, "text": "two", "speakerIds": [2]},
    ]}}
    persist_speech(root, meeting_id, first)
    archive.confirm_speaker(meeting_id, "speaker_1", "Alice")
    archive.confirm_speaker(meeting_id, "speaker_2", "Bob")
    rerun = {"segments": [], "diarization": {"turns": [
        {"start": 0, "end": 1, "text": "one corrected", "speakerIds": [2]},
        {"start": 1, "end": 2, "text": "two corrected", "speakerIds": [1]},
    ]}}
    persist_speech(root, meeting_id, rerun)
    rows = MeetingArchive(root).timeline(meeting_id)["items"]
    assert [(r["text"], r["confirmed_speaker"]) for r in rows] == [
        ("one corrected", "Alice"), ("two corrected", "Bob")]


def test_segment_edit_becomes_orphan_when_interval_changes(tmp_path):
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"audio")
    root = tmp_path / "archive"
    archive = MeetingArchive(root)
    meeting_id = archive.import_file(source)["meeting_id"]
    original = {"id": "stable-seg", "start": 1.0, "end": 2.0, "text": "raw"}
    archive.write_transcript(meeting_id, [original])
    archive.edit_segment(meeting_id, "stable-seg", "corrected")
    archive.write_transcript(meeting_id, [{**original, "start": 1.1, "end": 2.1, "text": "new raw"}])
    orphan = next(row for row in archive.timeline(meeting_id)["items"] if row["kind"] == "orphan_edit")
    assert orphan["text"] == "corrected"
    assert orphan["original_text"] == "raw"
    assert orphan["start"] == 1.0


def test_stage_failure_is_persisted_and_propagated_not_marked_complete(tmp_path):
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"video")
    root = tmp_path / "archive"
    archive = MeetingArchive(root)
    meeting_id = archive.import_file(source)["meeting_id"]
    def fail_runner(*_args):
        raise RuntimeError("decoder failed")
    try:
        run_stages(root, meeting_id, source, ["screen"], runner=fail_runner)
    except MeetingStageFailure as exc:
        assert exc.stage == "screen"
    else:
        raise AssertionError("A failed screen stage must fail the queue worker")
    reopened = MeetingArchive(root, read_only=True)
    meta = reopened.get_meeting(meeting_id)
    assert meta["stage_state"]["screen"]["status"] == "failed"
    assert meta["stage_state"]["screen"]["elapsed_seconds"] >= 0
    assert (reopened.meeting_dir(meeting_id) / "manifest.json").is_file()


def test_cancelled_stage_persists_interrupted_state(tmp_path):
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"video")
    root = tmp_path / "archive"
    archive = MeetingArchive(root)
    meeting_id = archive.import_file(source)["meeting_id"]
    try:
        run_stages(root, meeting_id, source, ["screen"], cancel=lambda: True)
    except InterruptedError:
        pass
    else:
        raise AssertionError("Cancellation must propagate to the worker")
    meta = MeetingArchive(root, read_only=True).get_meeting(meeting_id)
    assert meta["stage_state"]["screen"]["status"] == "cancelled"
    assert meta["status"] == "interrupted"


def test_index_stage_exports_transcript_before_snapshot_and_finalize_keeps_cache_hot(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"audio")
    root = tmp_path / "archive"
    archive = MeetingArchive(root)
    meeting_id = archive.import_file(source)["meeting_id"]
    persist_speech(root, meeting_id, {"text": "raw transcript", "segments": [], "diarization": {
        "turns": [{"start": 1, "end": 2, "text": "new transcript words", "speakerIds": []}]
    }})
    archive.write_screen_events(meeting_id, [
        {"id": "screen", "start": 3, "end": 4, "text": "visible screen marker"}
    ], status="ready")
    observed = {}

    def fake_index_runner(stage, _staging, _progress):
        assert stage == "index"
        transcript_path = archive.meeting_dir(meeting_id) / "transcript.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        observed["segments"] = transcript["segments"]
        observed["transcript_mtime"] = transcript_path.stat().st_mtime_ns
        indexed = MeetingIndex(root, embedding_model_path=root / "missing-model").index_meeting(meeting_id)
        return {"ok": True, "indexed_items": indexed, "embeddings": {"status": "unavailable"}}

    run_stages(root, meeting_id, source, ["index"], runner=fake_index_runner)
    transcript_path = archive.meeting_dir(meeting_id) / "transcript.json"
    assert observed["segments"][0]["text"] == "new transcript words"
    assert transcript_path.stat().st_mtime_ns == observed["transcript_mtime"]

    archive.finalize(meeting_id)
    assert transcript_path.stat().st_mtime_ns == observed["transcript_mtime"]

    screen_reads = []
    original_read_json = MeetingIndex._read_json

    def spy_read_json(path, default):
        if Path(path).name == "screen-events.json":
            screen_reads.append(Path(path))
        return original_read_json(path, default)

    monkeypatch.setattr(MeetingIndex, "_read_json", staticmethod(spy_read_json))
    result = MeetingIndex(root, embedding_model_path=root / "missing-model").search(
        "visible screen marker", mode="text")
    assert any(row["evidence_id"] == "screen" for row in result["results"])
    assert screen_reads == []
