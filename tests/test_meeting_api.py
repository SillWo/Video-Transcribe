import os
import hashlib
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
from unittest.mock import patch

from PIL import Image

from fastapi.testclient import TestClient

import web_api
from services.meeting_archive import MeetingArchive
from services.meeting_recorder import MeetingRecorder
from services.transcription_history import TranscriptionHistory


class MeetingApiTests(unittest.TestCase):
    def setUp(self):
        web_api.reset_simple_queue_for_tests()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env_before = {key: os.environ.get(key) for key in
                           ("VT_SIMPLE_RUNTIME_DIR", "VT_MEETING_ROOT", "VT_MEETING_SETTINGS")}
        os.environ["VT_SIMPLE_RUNTIME_DIR"] = str(self.root / "runtime")
        os.environ["VT_MEETING_ROOT"] = str(self.root / "archive")
        os.environ["VT_MEETING_SETTINGS"] = str(self.root / "settings.json")
        web_api._history_store = None
        with web_api.jobs_lock:
            self.old_jobs = web_api.jobs.copy()
            web_api.jobs.clear()
            web_api.job_process_controllers.clear()
            web_api.simple_worker_states.clear()
        self.client = TestClient(web_api.app)
        web_api.get_simple_queue().set_state(True)

    def tearDown(self):
        web_api.reset_simple_queue_for_tests()
        with web_api.jobs_lock:
            web_api.jobs.clear()
            web_api.jobs.update(self.old_jobs)
            web_api.job_process_controllers.clear()
            web_api.simple_worker_states.clear()
        web_api._history_store = None
        for key, value in self.env_before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp.cleanup()

    def _import(self):
        source = self.root / "запись.wav"
        source.write_bytes(b"audio")
        response = self.client.post("/api/meetings/import", json={"paths": [str(source)]})
        self.assertEqual(response.status_code, 200)
        return response.json()["items"][0]["meeting_id"], source

    def _finalize_meeting_job(self, meeting_id, job_id, *, status, stage, stages=None):
        archive_root = str(self.root / "archive")
        archive = MeetingArchive(archive_root)
        archive.update_meeting(meeting_id, speech_status="ready", status="ready")
        settings = {"_simple": True, "_meetingId": meeting_id, "_meetingRoot": archive_root}
        metadata = {"id": job_id, "meetingId": meeting_id, "meetingRoot": archive_root}
        if stages is not None:
            settings["_meetingStages"] = stages
            metadata["meetingStages"] = stages
        checkpoints = web_api.initial_checkpoints()
        for checkpoint in checkpoints:
            if checkpoint["stage"] == stage:
                checkpoint.update(status="active", startedAt="2026-11-15T10:00:00Z")
        record = web_api.history_store.create({**metadata, "checkpoints": checkpoints})
        job_status = "cancelling" if status == "cancelled" else "running"
        web_api.history_store.update(job_id, status=job_status, stage=stage)
        with web_api.jobs_lock:
            web_api.jobs[job_id] = web_api.JobState(
                id=job_id, settings=settings, status=job_status, stage=stage)
        self.assertTrue(web_api._finalize_simple_job(job_id, status))
        return archive.get_meeting(meeting_id)

    def test_terminal_meeting_job_sync_updates_speech_only_for_asr_outcomes(self):
        meeting_id, _source = self._import()

        for status in ("failed", "cancelled"):
            with self.subTest(stage="screen", status=status):
                result = self._finalize_meeting_job(
                    meeting_id, "a" * 32 if status == "failed" else "b" * 32,
                    status=status, stage="screen",
                    stages=["screen", "audio", "index"])
                self.assertEqual(result["speech_status"], "ready")

        cancelled_speech = self._finalize_meeting_job(
            meeting_id, "c" * 32, status="cancelled", stage="recognition", stages=["speech"])
        self.assertEqual(cancelled_speech["speech_status"], "cancelled")

        failed_screen_after_speech = self._finalize_meeting_job(
            meeting_id, "d" * 32, status="failed", stage="screen", stages=["speech", "screen"])
        self.assertEqual(failed_screen_after_speech["speech_status"], "ready")

        legacy_speech_cancelled = self._finalize_meeting_job(
            meeting_id, "e" * 32, status="cancelled", stage="recognition")
        self.assertEqual(legacy_speech_cancelled["speech_status"], "cancelled")

    def test_static_settings_and_glossary_routes_are_not_captured_by_id_route(self):
        self.assertEqual(self.client.get("/api/meetings/settings").status_code, 200)
        self.assertEqual(self.client.get("/api/meetings/glossary").status_code, 200)

    def test_meeting_list_includes_short_errors_and_timestamps_for_live_and_persisted_jobs(self):
        meeting_id, _source = self._import()
        live = SimpleNamespace(
            id="live-job", settings={"_meetingId": meeting_id}, status="failed", stage="screen",
            error="  Экран   недоступен  ", errorCode="screen_unavailable",
            created_at="2026-11-15T10:00:00+00:00", updated_at="2026-11-15T10:01:00+00:00",
        )
        persisted = web_api.history_store.create({"meetingId": meeting_id, "id": "abc123def456"})
        web_api.history_store.update(persisted["id"], status="failed", stage="speech",
                                     error="Распознавание завершилось с ошибкой",
                                     errorCode="speech_failed")
        persisted = web_api.history_store.get(persisted["id"])
        with web_api.jobs_lock:
            web_api.jobs[live.id] = live

        listed = self.client.get("/api/meetings").json()["items"]
        jobs = {job["id"]: job for job in next(item for item in listed if item["meeting_id"] == meeting_id)["jobs"]}
        self.assertEqual(jobs["live-job"]["error"], "Экран недоступен")
        self.assertEqual(jobs["live-job"]["errorCode"], "screen_unavailable")
        self.assertEqual(jobs["live-job"]["createdAt"], live.created_at)
        self.assertEqual(jobs["live-job"]["updatedAt"], live.updated_at)
        self.assertEqual(jobs["abc123def456"]["error"], "Распознавание завершилось с ошибкой")
        self.assertEqual(jobs["abc123def456"]["errorCode"], "speech_failed")
        self.assertEqual(jobs["abc123def456"]["createdAt"], persisted["createdAt"])
        self.assertEqual(jobs["abc123def456"]["updatedAt"], persisted["updatedAt"])

    def test_meeting_job_status_uses_newer_in_memory_job_than_persisted_failure(self):
        meeting_id, _source = self._import()
        persisted = web_api.history_store.create({"meetingId": meeting_id, "id": "abc123def457"})
        web_api.history_store.update(persisted["id"], status="failed", error="old failure")
        live = SimpleNamespace(
            id="new-queued-job", settings={"_meetingId": meeting_id}, status="queued", stage="source",
            error=None, created_at="2030-11-15T10:00:00Z", updated_at="2030-11-15T10:00:00Z",
        )
        with web_api.jobs_lock:
            web_api.jobs[live.id] = live

        listed = self.client.get("/api/meetings").json()["items"]
        item = next(item for item in listed if item["meeting_id"] == meeting_id)
        self.assertEqual(item["jobs"][-1]["id"], "new-queued-job")
        self.assertEqual(item["jobStatus"], "queued")

    def test_meeting_list_filters_foreign_roots_but_keeps_legacy_jobs(self):
        meeting_id, _source = self._import()
        current_root = str(self.root / "archive")
        foreign_root = str(self.root / "other-archive")
        history = web_api.history_store

        def persisted(job_id, *, root=None, status, created_at, error=None):
            metadata = {"id": job_id, "meetingId": meeting_id}
            if root is not None:
                metadata["meetingRoot"] = root
            history.create(metadata)
            history.update(job_id, status=status, createdAt=created_at, error=error)

        persisted("abc123def457", root=current_root, status="completed",
                  created_at="2026-11-15T10:00:00Z")
        persisted("abc123def458", root=foreign_root, status="failed",
                  created_at="2026-11-16T10:00:00Z", error="foreign newest failure")
        persisted("abc123def459", status="failed", created_at="2026-11-14T10:00:00Z",
                  error="legacy failure")
        # An in-memory job in the active root must not inherit its colliding
        # history ID's error details from a persisted row owned by another root.
        persisted("abc123def460", root=foreign_root, status="failed",
                  created_at="2026-11-16T11:00:00Z", error="foreign collision error")
        live = SimpleNamespace(
            id="abc123def460", settings={"_meetingId": meeting_id, "_meetingRoot": current_root},
            status="completed", stage="result", error=None, errorCode=None,
            created_at="2030-11-15T12:00:00Z", updated_at="2030-11-15T12:01:00Z",
        )
        foreign_live = SimpleNamespace(
            id="abc123def461", settings={"_meetingId": meeting_id, "_meetingRoot": foreign_root},
            status="failed", stage="speech", error="foreign in-memory failure",
            created_at="2026-11-17T10:00:00Z", updated_at="2026-11-17T10:01:00Z",
        )
        with web_api.jobs_lock:
            web_api.jobs[live.id] = live
            web_api.jobs[foreign_live.id] = foreign_live

        response = self.client.get("/api/meetings")
        self.assertEqual(response.status_code, 200, response.text)
        item = next(item for item in response.json()["items"] if item["meeting_id"] == meeting_id)
        jobs = {job["id"]: job for job in item["jobs"]}
        self.assertNotIn("abc123def458", jobs)
        self.assertNotIn("abc123def461", jobs)
        self.assertIn("abc123def459", jobs)
        self.assertEqual(jobs["abc123def460"]["error"], None)
        self.assertEqual(jobs["abc123def460"]["errorCode"], None)
        self.assertEqual(item["jobStatus"], "completed")

    def test_uuid_detail_route_accepts_fastapi_uuid_conversion(self):
        meeting_id, _ = self._import()
        response = self.client.get(f"/api/meetings/{meeting_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["meeting"]["meeting_id"], meeting_id)

    def test_meeting_revision_uses_only_fixed_file_stats(self):
        meeting_id, _ = self._import()
        archive = MeetingArchive(self.root / "archive")
        screen_path = archive.meeting_dir(meeting_id) / "screen-events.json"
        screen_path.write_text("not parsed by revision endpoint", encoding="utf-8")
        with patch.object(MeetingArchive, "timeline", side_effect=AssertionError("revision must not build timeline")):
            first_response = self.client.get(f"/api/meetings/{meeting_id}/revision")
            self.assertEqual(first_response.status_code, 200, first_response.text)
            first_revision = first_response.json()["revision"]

            archive.update_meeting(meeting_id, title="edited title")
            edited_revision = self.client.get(f"/api/meetings/{meeting_id}/revision").json()["revision"]
            self.assertNotEqual(first_revision, edited_revision)

            screen_path.write_text("replacement screen events with changed size", encoding="utf-8")
            replaced_revision = self.client.get(f"/api/meetings/{meeting_id}/revision").json()["revision"]
            self.assertNotEqual(edited_revision, replaced_revision)
        self.assertFalse((archive.meeting_dir(meeting_id) / "timeline-cache.json").exists())

    def test_timeline_geometry_opt_out_preserves_default_full_payload(self):
        meeting_id, _ = self._import()
        archive = MeetingArchive(self.root / "archive")
        archive.write_screen_events(meeting_id, [{
            "id": "screen-1", "start": 2, "end": 3, "text": "slide", "frame": "frames/1.jpg",
            "width": 1920, "height": 1080,
            "regions": [{"text": "header", "bbox": [1, 2, 3, 4]}],
            "tables": [{"bbox": [4, 5, 6, 7]}],
        }])
        full = self.client.get(f"/api/meetings/{meeting_id}/timeline").json()["items"][0]
        light = self.client.get(f"/api/meetings/{meeting_id}/timeline?include_geometry=false").json()["items"][0]
        self.assertEqual(full["regions"][0]["bbox"], [1, 2, 3, 4])
        self.assertIn("tables", full)
        self.assertNotIn("regions", light)
        self.assertNotIn("tables", light)
        for key in ("id", "text", "frame", "start", "end", "width", "height"):
            self.assertEqual(light[key], full[key])

    def test_detail_can_skip_timeline_and_filters_foreign_root_jobs(self):
        meeting_id, _ = self._import()
        current_root = str(self.root / "archive")
        foreign_root = str(self.root / "other-archive")
        history = web_api.history_store
        current = history.create({"id": "abc123def470", "meetingId": meeting_id,
                                  "meetingRoot": current_root})
        history.update(current["id"], status="completed")
        foreign = history.create({"id": "abc123def471", "meetingId": meeting_id,
                                  "meetingRoot": foreign_root})
        history.update(foreign["id"], status="failed", error="foreign detail failure")
        with patch.object(MeetingArchive, "timeline", side_effect=AssertionError("detail requested timeline")):
            response = self.client.get(f"/api/meetings/{meeting_id}?include_timeline=false")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertNotIn("timeline", payload)
        self.assertEqual(payload["meeting"]["meeting_id"], meeting_id)
        self.assertNotIn(foreign["id"], [job["id"] for job in payload["jobs"]])
        self.assertIn(current["id"], [job["id"] for job in payload["jobs"]])

    def test_meeting_summary_routes_use_compact_history_projection(self):
        meeting_id, _ = self._import()
        record = web_api.history_store.create({
            "id": "abc123def472", "meetingId": meeting_id,
            "meetingRoot": str(self.root / "archive"),
            "asrPayload": {"segments": [{"text": "x" * 100_000}]},
        })
        web_api.history_store.update(record["id"], status="failed", stage="screen",
                                     error="Visible failure", errorCode="screen_failed")
        with patch.object(web_api.get_history_store(), "list", side_effect=AssertionError("full history decode")):
            listed = self.client.get("/api/meetings")
            detail = self.client.get(f"/api/meetings/{meeting_id}?include_timeline=false")
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual(detail.status_code, 200, detail.text)
        item = next(item for item in listed.json()["items"] if item["meeting_id"] == meeting_id)
        summary = next(job for job in item["jobs"] if job["id"] == record["id"])
        self.assertEqual(summary["error"], "Visible failure")
        self.assertEqual(summary["errorCode"], "screen_failed")
        self.assertEqual(detail.json()["jobs"][0]["error"], "Visible failure")

    def test_active_recording_holds_dispatch_without_pausing_persistent_queue(self):
        class Queue:
            def state(self): return {"paused": False}
        marker = self.root / "runtime" / "meeting-recorder" / "session.json"
        marker.parent.mkdir(parents=True)
        marker.write_text('{"recording_path":"capture.mkv"}', encoding="utf-8")
        with patch("web_api.get_simple_queue", return_value=Queue()):
            self.assertTrue(web_api._simple_dispatch_paused())
            marker.write_text('{"recording_path":null}', encoding="utf-8")
            self.assertFalse(web_api._simple_dispatch_paused())

    def test_recorder_import_uses_same_nondefault_archive_as_meeting_list(self):
        capture_dir = self.root / "runtime" / "meeting-recorder" / "recordings"
        capture_dir.mkdir(parents=True)
        capture = capture_dir / "QA запись.mkv"
        capture.write_bytes(b"recorded bytes")
        recorder = MeetingRecorder(project_root=web_api.APP_ROOT, runtime_root=self.root / "runtime",
            ffprobe=lambda _path: {"audio": [{}, {}, {}], "video": [{}]})
        recorder._session = {"recording_path": str(capture), "title": "QA meeting"}
        class Client:
            def get_record_status(self): return {"outputActive": False, "outputPath": str(capture)}
        recorder._connect = lambda attempts=1: Client()
        recorder._status_from_obs = lambda: {"state": "idle", "connected": True}
        result = recorder.stop()
        self.assertEqual(result["meeting"]["meeting_id"], recorder.archive.list_meetings()[0]["meeting_id"])
        listed = self.client.get("/api/meetings").json()["items"]
        self.assertIn(result["meeting"]["meeting_id"], [item["meeting_id"] for item in listed])

    def test_import_to_manual_queue_restores_flags_and_persists_real_timed_result(self):
        meeting_id, source = self._import()
        response = self.client.post("/api/meetings/process", json={
            "meeting_ids": [meeting_id], "execution_mode": "manual", "device": "cpu",
            "cpu_threads": 2, "diarization": True, "stages": ["speech", "screen"],
        })
        self.assertEqual(response.status_code, 202, response.text)
        record = response.json()["jobs"][0]
        self.assertEqual(record["meetingId"], meeting_id)
        self.assertEqual(record["meetingStages"], ["speech", "screen"])
        self.assertEqual(record["executionMode"], "manual")
        hydrated = web_api._hydrate_simple_job(record)
        self.assertEqual(hydrated.settings["_meetingId"], meeting_id)
        self.assertEqual(hydrated.settings["_meetingRoot"], str(self.root / "archive"))
        self.assertTrue(hydrated.settings["useTimestamps"])
        with patch.dict(os.environ, {"VT_MEETING_ROOT": ""}):
            settings_change = self.client.put("/api/meetings/settings", json={"archive_root": str(self.root / "other-archive")})
        self.assertEqual(settings_change.status_code, 409)

        result = {"text": "hello there", "segments": [
            {"start": 3.0, "end": 4.0, "text": "raw ASR", "words": [{"start": 3.1, "end": 3.4, "text": "raw"}]},
        ], "diarization": {"turns": [
            {"start": 3.25, "end": 3.8, "text": "hello there", "speakerIds": [1]},
        ]}}
        web_api._persist_meeting_result(record["id"], result)
        archive = MeetingArchive(self.root / "archive", read_only=True)
        timeline = archive.timeline(meeting_id)
        self.assertEqual(timeline["items"][0]["start"], 3.25)
        self.assertEqual(timeline["items"][0]["end"], 3.8)
        self.assertEqual(timeline["items"][0]["speaker_cluster"], "speaker_1")
        reopened = MeetingArchive(self.root / "archive", read_only=True)
        self.assertEqual(reopened.timeline(meeting_id)["items"][0]["text"], "hello there")

    def test_meeting_processing_selects_cpu_diarization_except_manual_gpu(self):
        meeting_id, _source = self._import()
        cases = [
            ({"execution_mode": "auto", "device": "gpu", "diarization": True}, "cpu", True),
            ({"execution_mode": "manual", "device": "cpu", "diarization": True}, "cpu", True),
            ({"execution_mode": "manual", "device": "gpu", "diarization": True}, "auto", True),
            ({"execution_mode": "manual", "device": "gpu", "diarization": False}, "auto", False),
        ]
        for index, (options, expected_device, expected_enabled) in enumerate(cases):
            with self.subTest(options=options), patch.object(
                    web_api, "_enqueue_simple_jobs", return_value=[{"id": f"job-{index}"}]) as enqueue:
                response = self.client.post("/api/meetings/process", json={
                    "meeting_ids": [meeting_id], "stages": ["speech"], **options,
                })
            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(enqueue.call_args.args[5], expected_enabled)
            self.assertEqual(enqueue.call_args.args[6], expected_device)

    def test_meeting_profile_is_persisted_and_used_when_process_profile_is_omitted(self):
        updated = self.client.put("/api/meetings/settings", json={"profile": "quality"})
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(self.client.get("/api/meetings/settings").json()["settings"]["profile"], "quality")
        meeting_id, _ = self._import()
        response = self.client.post("/api/meetings/process", json={"meeting_ids": [meeting_id], "stages": ["speech"]})
        self.assertEqual(response.status_code, 202, response.text)
        record = response.json()["jobs"][0]
        self.assertEqual(record["profile"], "quality")
        self.assertEqual(web_api._hydrate_simple_job(record).settings["_profile"], "quality")

    def test_invalid_meeting_profile_is_rejected(self):
        response = self.client.put("/api/meetings/settings", json={"profile": "unknown"})
        self.assertEqual(response.status_code, 400)

    def test_manual_gpu_worker_is_attempted_without_speed_receipt(self):
        from unittest.mock import patch
        staging = self.root / "worker-stage"
        staging.mkdir()
        command_seen = {}
        def run_owned(_controller, command, **kwargs):
            command_seen["command"] = command
            result_path = Path(command[command.index("--result-path") + 1])
            result_path.write_text(json.dumps({"ok": True, "stage": "screen"}), encoding="utf-8")
            return 0
        job = SimpleNamespace(settings={"_requestedDevice": "gpu", "_executionMode": "manual", "_profile": "balanced"})
        runtime = {"device": "directml", "python": str(Path(__import__("sys").executable)),
                   "env": {}, "verified": False, "reason": "manual attempt", "error": None}
        with patch("web_api.get_job", return_value=job), \
             patch("services.meeting_hardware.plan_stage", return_value=runtime), \
             patch("web_api._run_owned_process", side_effect=run_owned):
            result = web_api._run_meeting_stage_worker("manual-gpu", "screen", str(self.root / "archive"),
                "meeting-id", "", staging, "directml", None)
        self.assertTrue(result["ok"])
        command = command_seen["command"]
        self.assertEqual(command[0], runtime["python"])
        self.assertEqual(command[command.index("--device") + 1], "directml")

    def test_non_speech_retry_reuses_queue_without_calling_asr_resolver(self):
        meeting_id, source = self._import()
        response = self.client.post("/api/meetings/process", json={
            "meeting_ids": [meeting_id], "stages": ["screen"], "device": "cpu", "cpu_threads": 2,
        })
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["jobs"][0]["id"]
        with patch("services.meeting_pipeline.run_stages", return_value={"stages": {"screen": {"status": "ready"}}, "manifest": {}}) as stages, \
             patch.object(web_api, "_resolve_simple_execution", side_effect=AssertionError("ASR must not run")):
            web_api._run_simple_job(job_id, str(source))
        stages.assert_called_once()
        self.assertEqual(stages.call_args.args[3], ["screen"])
        self.assertEqual(web_api.history_store.get(job_id)["status"], "completed")

    def test_manual_gpu_meeting_failure_pauses_shared_fifo_until_resume(self):
        meeting_ids = []
        for number in (1, 2):
            source = self.root / f"meeting-{number}.wav"
            source.write_bytes(f"distinct audio source {number}".encode("ascii"))
            imported = self.client.post("/api/meetings/import", json={"paths": [str(source)]})
            self.assertEqual(imported.status_code, 200, imported.text)
            meeting_ids.append(imported.json()["items"][0]["meeting_id"])
        jobs = []
        for meeting_id in meeting_ids:
            queued = self.client.post("/api/meetings/process", json={
                "meeting_ids": [meeting_id], "execution_mode": "manual", "device": "gpu",
                "stages": ["index"], "cpu_threads": 2,
            })
            self.assertEqual(queued.status_code, 202, queued.text)
            jobs.append(queued.json()["jobs"][0]["id"])

        calls = []
        original_worker = web_api._run_meeting_stage_worker
        def observed_worker(job_id, *args, **kwargs):
            calls.append(job_id)
            return original_worker(job_id, *args, **kwargs)

        def wait_for(predicate, timeout=8):
            deadline = __import__("time").monotonic() + timeout
            while __import__("time").monotonic() < deadline:
                if predicate():
                    return
                __import__("time").sleep(0.02)
            self.fail("The shared meeting queue did not reach the expected state in time.")

        with patch.object(web_api, "_run_meeting_stage_worker", side_effect=observed_worker):
            self.assertTrue(self.client.post("/api/simple/queue/resume").json()["paused"] is False)
            wait_for(lambda: web_api.get_simple_queue().state()["paused"] and
                     web_api.history_store.get(jobs[0])["status"] == "failed")
            self.assertEqual(calls, [jobs[0]])
            self.assertEqual(web_api.history_store.get(jobs[1])["status"], "queued")
            self.assertEqual(web_api.get_simple_queue().state()["pauseReason"]["code"], "gpu_stage_unsupported")
            self.assertIn("GPU", web_api.get_simple_queue().state()["pauseReason"]["message"])
            wait_for(lambda: web_api.simple_dispatcher is not None and
                     web_api.simple_dispatcher.active_job_id is None)

            resumed = self.client.post("/api/simple/queue/resume")
            self.assertEqual(resumed.status_code, 200, resumed.text)
            wait_for(lambda: web_api.get_simple_queue().state()["paused"] and
                     web_api.history_store.get(jobs[1])["status"] == "failed")
            self.assertEqual(calls, jobs)
            self.assertEqual(web_api.jobs[jobs[1]].settings["_requestedDevice"], "gpu")
            self.assertIn("GPU", web_api.get_simple_queue().state()["pauseReason"]["message"])

    def test_recovered_active_job_marks_archive_interrupted(self):
        meeting_id, _source = self._import()
        job = self.client.post("/api/meetings/process", json={
            "meeting_ids": [meeting_id], "stages": ["speech"], "cpu_threads": 2,
        }).json()["jobs"][0]
        web_api.get_simple_queue().mark_started(job["id"])
        reopened_history = TranscriptionHistory(self.root / "runtime")
        web_api._restore_meeting_job_statuses(reopened_history)
        meta = MeetingArchive(self.root / "archive", read_only=True).get_meeting(meeting_id)
        self.assertEqual(meta["job_status"], "interrupted")
        self.assertEqual(meta["status"], "interrupted")

    def test_restore_meeting_status_uses_only_latest_job_per_meeting(self):
        meeting_id = "00000000-0000-0000-0000-000000000001"
        current_root = str(self.root / "archive")

        class FakeArchive:
            def __init__(self, _root):
                self.meta = {"job_id": "new", "job_status": "completed", "status": "ready"}
                self.updates = []

            def get_meeting(self, _meeting_id):
                return dict(self.meta)

            def update_meeting(self, _meeting_id, **changes):
                self.updates.append(changes)
                self.meta.update(changes)

            def finalize(self, _meeting_id):
                pass

        older = {"id": "old", "meetingId": meeting_id, "meetingRoot": current_root,
                 "status": "failed", "createdAt": "2026-11-14T10:00:00Z"}
        newer = {"id": "new", "meetingId": meeting_id, "meetingRoot": current_root,
                 "status": "completed", "createdAt": "2026-11-15T10:00:00Z"}
        for rows in ([newer, older], [older, newer]):
            with self.subTest(order=[row["id"] for row in rows]):
                fake = FakeArchive(current_root)
                with patch.object(web_api, "MeetingArchive", create=True, return_value=fake), \
                     patch("services.meeting_archive.MeetingArchive", return_value=fake), \
                     patch("services.meeting_settings.load_settings", return_value={"archive_root": current_root}), \
                     patch.object(web_api, "APP_ROOT", self.root):
                    web_api._restore_meeting_job_statuses(SimpleNamespace(list=lambda: rows))
                self.assertEqual(fake.meta, {"job_id": "new", "job_status": "completed", "status": "ready"})
                self.assertEqual(fake.updates, [])

    def test_restore_meeting_status_does_not_apply_old_terminal_over_latest_queued(self):
        meeting_id = "00000000-0000-0000-0000-000000000002"
        root = str(self.root / "archive")
        class FakeArchive:
            def __init__(self, _root):
                self.meta = {"job_id": "old", "job_status": "failed", "status": "failed"}
                self.updates = []
            def get_meeting(self, _id): return dict(self.meta)
            def update_meeting(self, _id, **changes): self.updates.append(changes); self.meta.update(changes)
            def finalize(self, _id): pass
        rows = [
            {"id": "new", "meetingId": meeting_id, "meetingRoot": root, "status": "queued", "createdAt": "2026-11-15T10:00:00Z"},
            {"id": "old", "meetingId": meeting_id, "meetingRoot": root, "status": "failed", "createdAt": "2026-11-14T10:00:00Z"},
        ]
        fake = FakeArchive(root)
        with patch("services.meeting_archive.MeetingArchive", return_value=fake), \
             patch("services.meeting_settings.load_settings", return_value={"archive_root": root}):
            web_api._restore_meeting_job_statuses(SimpleNamespace(list=lambda: rows))
        self.assertEqual(fake.updates, [])

    def test_restore_meeting_status_marks_only_latest_interrupted_job(self):
        meeting_id = "00000000-0000-0000-0000-000000000003"
        root = str(self.root / "archive")
        class FakeArchive:
            def __init__(self, _root): self.meta = {"job_id": "old", "job_status": "failed", "status": "failed"}; self.updates = []
            def get_meeting(self, _id): return dict(self.meta)
            def update_meeting(self, _id, **changes): self.updates.append(changes); self.meta.update(changes)
            def finalize(self, _id): pass
        rows = [
            {"id": "new", "meetingId": meeting_id, "meetingRoot": root, "status": "interrupted", "createdAt": "2026-11-15T10:00:00Z"},
            {"id": "old", "meetingId": meeting_id, "meetingRoot": root, "status": "failed", "createdAt": "2026-11-14T10:00:00Z"},
        ]
        fake = FakeArchive(root)
        with patch("services.meeting_archive.MeetingArchive", return_value=fake), \
             patch("services.meeting_settings.load_settings", return_value={"archive_root": root}):
            web_api._restore_meeting_job_statuses(SimpleNamespace(list=lambda: rows))
        self.assertEqual(fake.meta, {"job_id": "new", "job_status": "interrupted", "status": "interrupted"})

    def test_restore_meeting_status_ignores_jobs_for_another_archive_root(self):
        meeting_id = "00000000-0000-0000-0000-000000000004"
        root = str(self.root / "archive")
        class FakeArchive:
            def __init__(self, _root): self.meta = {"job_id": "old", "job_status": "failed", "status": "failed"}; self.updates = []
            def get_meeting(self, _id): return dict(self.meta)
            def update_meeting(self, _id, **changes): self.updates.append(changes); self.meta.update(changes)
            def finalize(self, _id): pass
        rows = [{"id": "new", "meetingId": meeting_id, "meetingRoot": str(self.root / "other"),
                 "status": "completed", "createdAt": "2026-11-15T10:00:00Z"}]
        fake = FakeArchive(root)
        with patch("services.meeting_archive.MeetingArchive", return_value=fake), \
             patch("services.meeting_settings.load_settings", return_value={"archive_root": root}):
            web_api._restore_meeting_job_statuses(SimpleNamespace(list=lambda: rows))
        self.assertEqual(fake.updates, [])

    def test_restore_meeting_status_treats_missing_or_invalid_created_at_as_oldest(self):
        meeting_id = "00000000-0000-0000-0000-000000000005"
        root = str(self.root / "archive")
        class FakeArchive:
            def __init__(self, _root): self.meta = {"job_id": "new", "job_status": "completed", "status": "ready"}; self.updates = []
            def get_meeting(self, _id): return dict(self.meta)
            def update_meeting(self, _id, **changes): self.updates.append(changes); self.meta.update(changes)
            def finalize(self, _id): pass
        for old_time in (None, "bad timestamp"):
            rows = [
                {"id": "new", "meetingId": meeting_id, "meetingRoot": root, "status": "completed", "createdAt": "2026-11-15T10:00:00Z"},
                {"id": "old", "meetingId": meeting_id, "meetingRoot": root, "status": "failed", "createdAt": old_time},
            ]
            fake = FakeArchive(root)
            with self.subTest(createdAt=old_time), \
                 patch("services.meeting_archive.MeetingArchive", return_value=fake), \
                 patch("services.meeting_settings.load_settings", return_value={"archive_root": root}):
                web_api._restore_meeting_job_statuses(SimpleNamespace(list=lambda: rows))
            self.assertEqual(fake.meta["status"], "ready")

    def test_real_history_restore_preserves_completed_archive_on_repeated_startup(self):
        meeting_id, _source = self._import()
        archive = MeetingArchive(self.root / "archive")
        archive.write_transcript(meeting_id, [{"id": "segment-1", "start": 0, "end": 1, "text": "ready"}])
        archive.write_screen_events(meeting_id, [{"id": "screen-1", "start": 0, "end": 1, "text": "slide"}])
        archive.update_meeting(meeting_id, speech_status="ready", job_id="abc123def456",
                               job_status="failed", status="ready")
        manifest = archive.finalize(meeting_id)
        self.assertEqual(manifest["status"]["overall"], "ready")
        archive.update_meeting(meeting_id, status="failed")

        history = TranscriptionHistory(self.root / "runtime")
        history.create({"id": "abc123def456", "meetingId": meeting_id,
                        "meetingRoot": str(self.root / "archive")})
        history.update("abc123def456", status="failed", createdAt="2026-11-14T10:00:00Z")
        history.create({"id": "abc123def457", "meetingId": meeting_id,
                        "meetingRoot": str(self.root / "archive")})
        history.update("abc123def457", status="completed", createdAt="2026-11-15T10:00:00Z")
        history.create({"id": "abc123def458", "meetingId": meeting_id,
                        "meetingRoot": str(self.root / "another-archive")})
        history.update("abc123def458", status="failed", createdAt="2026-11-16T10:00:00Z")

        finalize_calls = []
        original_finalize = MeetingArchive.finalize
        def observed_finalize(instance, target_meeting_id):
            finalize_calls.append(target_meeting_id)
            return original_finalize(instance, target_meeting_id)

        with patch.object(MeetingArchive, "finalize", observed_finalize):
            web_api._restore_meeting_job_statuses(history)
            restored = MeetingArchive(self.root / "archive", read_only=True).get_meeting(meeting_id)
            self.assertEqual(restored["status"], "ready")
            self.assertEqual(restored["job_id"], "abc123def457")
            self.assertEqual(restored["job_status"], "completed")
            manifest_path = archive.meeting_dir(meeting_id) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            meeting_path = archive.meeting_dir(meeting_id) / "meeting.json"
            expected_hash = hashlib.sha256(meeting_path.read_bytes()).hexdigest()
            self.assertEqual(manifest["files"]["meeting.json"], expected_hash)
            self.assertEqual(finalize_calls, [meeting_id])

            unchanged_metadata = meeting_path.read_bytes()
            web_api._restore_meeting_job_statuses(history)
            self.assertEqual(finalize_calls, [meeting_id])
            self.assertEqual(meeting_path.read_bytes(), unchanged_metadata)

    def test_edit_remember_stores_short_term_and_explicit_undo_id(self):
        meeting_id, source = self._import()
        archive = MeetingArchive(self.root / "archive")
        segment_id = "speech:5955a32f8bb19d868a6f3880"
        archive.write_transcript(meeting_id, [{"id": segment_id, "start": 0, "end": 1,
                                               "text": "open old app now"}])
        response = self.client.put(f"/api/meetings/{meeting_id}/segments/{quote(segment_id, safe='')}",
                                   json={"text": "open DBeaver app now", "remember": True})
        self.assertEqual(response.status_code, 200, response.text)
        undo_id = response.json()["remembered_term_undo_id"]
        item = archive.glossary(meeting_id)[0]
        self.assertLessEqual(len(item["term"].split()), 3)
        self.assertIn("DBeaver", item["term"])
        self.assertEqual(item["provenance"]["segment_id"], segment_id)
        removed = self.client.delete(f"/api/meetings/glossary/{meeting_id}/{undo_id}")
        self.assertEqual(removed.json()["undo_id"], undo_id)
        restored = self.client.post(f"/api/meetings/glossary/{meeting_id}/{undo_id}/undo")
        self.assertEqual(restored.status_code, 200, restored.text)
        self.assertEqual(restored.json()["item"]["id"], undo_id)
        speaker = self.client.put(f"/api/meetings/{meeting_id}/speakers/speaker_1", json={"name": "Alice"})
        self.assertEqual(speaker.status_code, 200, speaker.text)
        meeting_dir = archive.meeting_dir(meeting_id)
        self.assertIn("DBeaver", (meeting_dir / "transcript.md").read_text(encoding="utf-8"))
        self.assertIn("DBeaver", (meeting_dir / "glossary.md").read_text(encoding="utf-8"))
        manifest = json.loads((meeting_dir / "manifest.json").read_text(encoding="utf-8"))
        for filename in ("meeting.json", "transcript.md", "speakers.json", "glossary.md"):
            expected = hashlib.sha256((meeting_dir / filename).read_bytes()).hexdigest()
            self.assertEqual(manifest["files"][filename], expected)

    def test_retained_frame_api_returns_jpeg_bytes(self):
        meeting_id, _source = self._import()
        archive = MeetingArchive(self.root / "archive")
        frame = archive.meeting_dir(meeting_id) / "frames" / "frame.jpg"
        frame.parent.mkdir()
        Image.new("RGB", (12, 8), color="red").save(frame, format="JPEG")
        archive.write_screen_events(meeting_id, [{"id": "screen-1", "start": 0, "end": 1,
                                                  "frame": "frames/frame.jpg", "text": "slide"}])
        response = self.client.get(f"/api/meetings/{meeting_id}/frames/screen-1")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertTrue(response.content.startswith(b"\xff\xd8"))

    def test_late_screen_frame_lookup_uses_screen_events_without_building_timeline(self):
        meeting_id, _source = self._import()
        archive = MeetingArchive(self.root / "archive")
        meeting_dir = archive.meeting_dir(meeting_id)
        frame = meeting_dir / "frames" / "late.jpg"
        frame.parent.mkdir()
        Image.new("RGB", (20, 12), color="blue").save(frame, format="JPEG")
        screen_events = [
            {"id": "screen-1", "frame": "frames/late.jpg"},
            {"id": "screen-2", "frame": "frames/late.jpg"},
            {"id": "screen-late", "frame": "frames/late.jpg"},
        ]

        class ScreenEventArchive:
            def get_meeting(self, target_id):
                return archive.get_meeting(target_id)

            def read_screen_events(self, target_id):
                return {"events": screen_events}

            def timeline(self, *_args, **_kwargs):
                raise AssertionError("frame lookup must not build the combined timeline")

            def meeting_dir(self, target_id):
                return archive.meeting_dir(target_id)

        with patch("services.meeting_api._archive", return_value=ScreenEventArchive()):
            response = self.client.get(f"/api/meetings/{meeting_id}/frames/screen-late")
            missing = self.client.get(f"/api/meetings/{meeting_id}/frames/not-found")
            invalid_crop = self.client.get(
                f"/api/meetings/{meeting_id}/frames/screen-late?crop=not,a,crop")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertTrue(response.content.startswith(b"\xff\xd8"))
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(invalid_crop.status_code, 400)

    def test_reindex_uses_the_shared_queue_and_index_worker_stage(self):
        meeting_id, _source = self._import()
        response = self.client.post("/api/meetings/reindex")
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()["jobs"][0]
        self.assertEqual(job["meetingId"], meeting_id)
        self.assertEqual(job["meetingStages"], ["index"])
        self.assertEqual(job["source"], "")


if __name__ == "__main__":
    unittest.main()
