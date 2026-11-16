from __future__ import annotations

import threading
import time

from services.meeting_activity import MeetingActivity


def _wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "activity worker did not reach the expected state"


def test_overlapping_holders_issue_one_request_and_reset_after_both_release():
    calls = []
    changed = threading.Event()

    def backend(active):
        calls.append(active)
        changed.set()

    activity = MeetingActivity(backend)
    activity.acquire("job:1")
    assert changed.wait(1)
    assert calls == [True]
    changed.clear()

    activity.acquire("recorder")
    activity.acquire("recorder")  # Status polling is idempotent for this key.
    activity.release("job:1")
    time.sleep(0.03)
    assert calls == [True]

    activity.release("recorder")
    assert changed.wait(1)
    _wait_for(lambda: activity.active_keys == frozenset())
    assert calls == [True, False]
    _wait_for(lambda: activity._thread is None)


def test_idle_and_read_only_use_do_not_request_activity():
    calls = []
    activity = MeetingActivity(calls.append)

    # Merely constructing the coordinator or inspecting keys represents idle
    # and read-only status work; neither takes a hold.
    assert activity.active_keys == frozenset()
    time.sleep(0.02)
    assert calls == []


def test_power_api_failure_does_not_escape_to_application():
    calls = []

    def broken_backend(active):
        calls.append(active)
        raise OSError("power API unavailable")

    activity = MeetingActivity(broken_backend)
    activity.acquire("job:broken")
    _wait_for(lambda: calls == [True])
    activity.release("job:broken")
    _wait_for(lambda: calls == [True, False])
    _wait_for(lambda: activity._thread is None)


def test_recorder_releases_hold_when_stop_is_confirmed_even_if_import_fails(tmp_path, monkeypatch):
    from services import meeting_recorder
    from services.meeting_recorder import MeetingRecorder

    capture_dir = tmp_path / "meeting-recorder" / "recordings"
    capture_dir.mkdir(parents=True)
    recording = capture_dir / "capture.mkv"
    recording.write_bytes(b"mkv")

    class Client:
        def __init__(self):
            self.statuses = iter([{"outputActive": True}, {"outputActive": False}])

        def get_record_status(self):
            return next(self.statuses)

        def stop_record(self):
            pass

    calls = []
    monkeypatch.setattr(meeting_recorder, "hold_activity", lambda key: calls.append(("hold", key)))
    monkeypatch.setattr(meeting_recorder, "release_activity", lambda key: calls.append(("release", key)))
    recorder = MeetingRecorder(runtime_root=tmp_path, ffprobe=lambda _path: (_ for _ in ()).throw(ValueError("bad media")))
    recorder._session = {"recording_path": str(recording), "title": "Meeting"}
    recorder._is_owned_process = lambda: True
    recorder._connect = lambda **_kwargs: Client()

    import pytest
    with pytest.raises(ValueError, match="bad media"):
        recorder.stop()

    assert calls == [("hold", "meeting-recorder"), ("release", "meeting-recorder")]


def test_recorder_keeps_hold_if_obs_rejects_stop(tmp_path, monkeypatch):
    from services import meeting_recorder
    from services.meeting_recorder import MeetingRecorder, RecorderError

    calls = []

    class Client:
        def get_record_status(self):
            return {"outputActive": True}

        def stop_record(self):
            raise RuntimeError("OBS unavailable")

    monkeypatch.setattr(meeting_recorder, "hold_activity", lambda key: calls.append(("hold", key)))
    monkeypatch.setattr(meeting_recorder, "release_activity", lambda key: calls.append(("release", key)))
    recorder = MeetingRecorder(runtime_root=tmp_path)
    recorder._session = {"recording_path": str(tmp_path / "capture.mkv")}
    recorder._is_owned_process = lambda: True
    recorder._connect = lambda **_kwargs: Client()

    import pytest
    with pytest.raises(RuntimeError, match="OBS unavailable"):
        recorder.stop()

    assert calls == [("hold", "meeting-recorder")]


def test_recorder_status_releases_only_when_owned_obs_process_is_gone(tmp_path, monkeypatch):
    from services import meeting_recorder
    from services.meeting_recorder import MeetingRecorder

    calls = []
    monkeypatch.setattr(meeting_recorder, "release_activity", lambda key: calls.append(key))
    for process_running, expected in ((False, ["meeting-recorder"]), (True, [])):
        calls.clear()
        recorder = MeetingRecorder(runtime_root=tmp_path)
        recorder._session = {"recording_path": str(tmp_path / "recoverable.mkv")}
        recorder._state = "recording"
        recorder._connect = lambda **_kwargs: (_ for _ in ()).throw(ConnectionError("offline"))
        recorder._private_obs_process_running = lambda: process_running

        recorder._status_from_obs()

        assert calls == expected


def _run_simple_meeting_job_for_activity_test(monkeypatch, *, post_stage_error=None):
    import uuid
    from types import SimpleNamespace

    import web_api
    from services import meeting_activity

    job_id = uuid.uuid4().hex
    job = web_api.JobState(job_id, {
        "_simple": True,
        "_meetingId": "meeting-test",
        "_meetingRoot": "unused-archive",
        "_meetingStages": ["speech", "screen"],
        "_executionMode": "auto",
        "_sourceType": "url",
        "_profile": "balanced",
        "_requestedDevice": "cpu",
        "model": "small",
        "device": "cpu",
        "restorePunctuation": False,
    })
    record = {"id": job_id, "sourceType": "url", "profile": "balanced", "device": "cpu"}

    class History:
        def get(self, requested_id):
            return dict(record) if requested_id == job_id else None

        def update(self, requested_id, **changes):
            if requested_id != job_id:
                return None
            record.update(changes)
            return dict(record)

    holds = []
    monkeypatch.setattr(web_api, "_history_store", History())
    monkeypatch.setitem(web_api.jobs, job_id, job)
    monkeypatch.setitem(web_api.simple_worker_states, job_id, "not_started")
    monkeypatch.setattr(web_api, "_mark_simple_job_started", lambda _job_id: setattr(job, "status", "running") or True)
    monkeypatch.setattr(meeting_activity, "hold_activity", lambda key: holds.append(("hold", key)))
    monkeypatch.setattr(meeting_activity, "release_activity", lambda key: holds.append(("release", key)))
    monkeypatch.setattr(web_api, "_resolve_simple_execution", lambda *_args: None)
    monkeypatch.setattr(web_api, "_checkpoint_update", lambda *_args: None)
    monkeypatch.setattr(web_api, "_simple_progress", lambda *_args: None)
    monkeypatch.setattr(web_api, "_simple_stage_event", lambda *_args: None)
    monkeypatch.setattr(web_api, "_finish_simple_checkpoints", lambda *_args: None)
    monkeypatch.setattr(web_api, "append_log", lambda *_args: None)
    monkeypatch.setattr(web_api, "build_command", lambda *_args: ["fake-asr"])
    monkeypatch.setattr(web_api, "_finalize_simple_job", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(web_api, "_pause_simple_queue_for_error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(web_api, "_run_meeting_post_stages",
                        lambda *_args: (_ for _ in ()).throw(post_stage_error) if post_stage_error else {"ok": True})

    class Archive:
        def __init__(self, _root):
            pass

        def get_meeting(self, _meeting_id):
            return {"stage_state": {}}

        def update_meeting(self, _meeting_id, **_changes):
            pass

    monkeypatch.setattr("services.meeting_archive.MeetingArchive", Archive)
    worker_calls = []

    def run_owned_process(_controller, command, **_kwargs):
        worker_calls.append(command)
        if len(worker_calls) == 2:
            job.output_path = "fake-output.txt"
            job.result_text = "transcript"
        return 0

    monkeypatch.setattr(web_api, "_run_owned_process", run_owned_process)
    web_api._run_simple_job(job_id, "https://example.org/meeting")
    return holds, job


def test_simple_job_releases_hold_after_post_asr_success_and_failure(monkeypatch):
    import web_api
    from services import meeting_activity
    from web_api import JobState

    for error in (None, RuntimeError("post stage failed")):
        with monkeypatch.context() as scoped:
            holds, _job = _run_simple_meeting_job_for_activity_test(scoped, post_stage_error=error)
            assert holds == [("hold", f"simple-job:{_job.id}"), ("release", f"simple-job:{_job.id}")]


def test_terminal_simple_job_does_not_take_activity_hold(monkeypatch):
    import uuid
    import web_api
    from services import meeting_activity

    job_id = uuid.uuid4().hex
    job = web_api.JobState(job_id, {"_simple": True}, status="completed")
    calls = []
    monkeypatch.setitem(web_api.jobs, job_id, job)
    monkeypatch.setattr(meeting_activity, "hold_activity", lambda key: calls.append(("hold", key)))
    web_api._run_simple_job(job_id, "unused")
    assert calls == []
