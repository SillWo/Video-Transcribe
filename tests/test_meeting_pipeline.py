import json
from pathlib import Path

import pytest

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


def test_persist_speech_disambiguates_repeated_speech_ids_and_indexes_every_segment(tmp_path):
    root = tmp_path / "archive"
    archive = MeetingArchive(root)
    cases = [
        ([{"text": "untimed repeated"}, {"text": "untimed repeated"}, {"text": "untimed other"}],
         [None, None, None]),
        ([{"start": 1, "end": 2, "text": "first"},
          {"start": 1, "end": 2, "text": "second"},
          {"start": 2, "end": 3, "text": "third"}],
         [1.0, 1.0, 2.0]),
    ]

    for case_index, (segments, expected_starts) in enumerate(cases):
        source = tmp_path / f"recording-{case_index}.wav"
        source.write_bytes(f"audio-{case_index}".encode())
        meeting_id = archive.import_file(source)["meeting_id"]
        before = meeting_segments({"segments": segments})
        base_id, third_id = before[0]["id"], before[2]["id"]
        assert before[2]["id"] != base_id
        persist_speech(root, meeting_id, {"segments": segments})

        assert MeetingIndex(root, embedding_model_path=root / "missing-model").index_meeting(meeting_id) == 4
        rows = MeetingArchive(root).timeline(meeting_id)["items"]
        assert [row["id"] for row in rows] == [base_id, f"{base_id}:2", third_id]
        assert [row["text"] for row in rows] == [s["text"] for s in segments]
        assert [row["start"] for row in rows] == expected_starts


def test_legacy_duplicate_speech_ids_keep_corrections_and_index_without_asr(tmp_path):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"audio")
    root = tmp_path / "archive"
    archive = MeetingArchive(root)
    meeting_id = archive.import_file(source)["meeting_id"]
    base_id = meeting_segments({"segments": [
        {"start": 1.0, "end": 2.0, "text": "first original"},
    ]})[0]["id"]
    raw_segments = [
        {"id": base_id, "start": 1.0, "end": 2.0, "text": "first original"},
        {"id": base_id, "start": 1.0, "end": 2.0, "text": "second original"},
    ]
    # Model an existing archive directly so no write path can normalize IDs.
    transcript_path = archive.meeting_dir(meeting_id) / "transcript.json"
    transcript_path.write_text(json.dumps({
        "raw_segments": raw_segments,
        "segments": [dict(segment) for segment in raw_segments],
        "raw_text": None,
        "overrides": {base_id: {
            "text": "corrected legacy text", "original_start": 1.0,
            "original_end": 2.0, "original_text": "first original",
        }},
    }), encoding="utf-8")

    def index_only(_stage, _staging, _progress):
        indexed = MeetingIndex(root, embedding_model_path=root / "missing-model").index_meeting(meeting_id)
        return {"ok": True, "indexed_items": indexed, "embeddings": {"status": "unavailable"}}

    run_stages(root, meeting_id, source, ["index"], runner=index_only)
    exported = json.loads(transcript_path.read_text(encoding="utf-8"))
    expected_ids = [base_id, f"{base_id}:2"]
    assert [row["id"] for row in exported["raw_segments"]] == expected_ids
    assert [row["id"] for row in exported["segments"]] == expected_ids
    first_export = transcript_path.read_bytes()
    archive.refresh_editorial_exports(meeting_id)
    assert transcript_path.read_bytes() == first_export

    rows = MeetingArchive(root, read_only=True).timeline(meeting_id)["items"]
    assert [row["id"] for row in rows] == expected_ids
    assert [row["text"] for row in rows] == ["corrected legacy text", "corrected legacy text"]
    assert [row["original_text"] for row in rows] == ["first original", "second original"]
    assert [row["start"] for row in rows] == [1.0, 1.0]


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


def test_screen_stage_publishes_v2_frames_only_and_timeline_links(tmp_path):
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"video")
    root = tmp_path / "archive"
    archive = MeetingArchive(root)
    meeting_id = archive.import_file(source)["meeting_id"]

    def screen_runner(_stage, staging, _progress):
        frame = staging / "frames" / "screen-new.jpg"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b"new jpeg")
        return {"ok": True, "result": {
            "events": [{"id": "screen-0001", "start": 1, "end": 2,
                        "frame": "frames/screen-new.jpg", "width": 640, "height": 360,
                        "sample_interval_seconds": 1, "time_uncertainty_seconds": 1,
                        "reason": "initial", "text": "should not persist", "regions": [{"text": "OCR"}],
                        "tables": []}],
            "status": "ready", "limitations": [],
            "stats": {"sampled_frames": 2, "saved_frames": 1},
        }}

    run_stages(root, meeting_id, source, ["screen"], runner=screen_runner)
    folder = archive.meeting_dir(meeting_id)
    data = json.loads((folder / "screen-events.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["mode"] == "frames_only"
    assert data["events"][0]["frame"] == "frames/screen-new.jpg"
    assert data["events"][0]["start"] == 1
    assert not {"text", "regions", "tables"} & data["events"][0].keys()
    assert "[screen-0001 captured](frames/screen-new.jpg)" in (folder / "screen-timeline.md").read_text(encoding="utf-8")


def test_cancelled_screen_publication_keeps_previous_json_and_frame(tmp_path):
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"video")
    root = tmp_path / "archive"
    archive = MeetingArchive(root)
    meeting_id = archive.import_file(source)["meeting_id"]
    old_frame = archive.meeting_dir(meeting_id) / "frames" / "old.jpg"
    old_frame.parent.mkdir()
    old_frame.write_bytes(b"old jpeg")
    archive.write_screen_events(meeting_id, [{"id": "old", "start": 0, "end": 2,
                                              "frame": "frames/old.jpg"}], status="ready")
    screen_path = archive.meeting_dir(meeting_id) / "screen-events.json"
    prior_json = screen_path.read_bytes()
    calls = 0

    def cancel_before_publish():
        nonlocal calls
        calls += 1
        return calls >= 4

    def screen_runner(_stage, staging, _progress):
        frame = staging / "frames" / "new.jpg"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b"new jpeg")
        return {"ok": True, "result": {"events": [{"id": "new", "start": 2,
                "frame": "frames/new.jpg"}], "status": "ready"}}

    with pytest.raises(InterruptedError):
        run_stages(root, meeting_id, source, ["screen"], runner=screen_runner, cancel=cancel_before_publish)
    assert screen_path.read_bytes() == prior_json
    assert old_frame.read_bytes() == b"old jpeg"
    assert not (archive.meeting_dir(meeting_id) / "frames" / "new.jpg").exists()


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
