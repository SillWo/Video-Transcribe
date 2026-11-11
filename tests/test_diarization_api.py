import os
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_api
from services.task_errors import TaskFailure
from web_api import JobState


class DiarizationApiTests(unittest.TestCase):
    def setUp(self):
        web_api.reset_simple_queue_for_tests()
        self.temp = tempfile.TemporaryDirectory()
        self.previous_store = web_api.history_store
        self.previous_cached_store = web_api._history_store
        self.previous_runtime_dir = os.environ.get("VT_SIMPLE_RUNTIME_DIR")
        os.environ["VT_SIMPLE_RUNTIME_DIR"] = str(Path(self.temp.name) / "runtime")
        web_api._history_store = None
        with web_api.jobs_lock:
            self.previous_jobs = web_api.jobs.copy()
            web_api.jobs.clear()
            web_api.job_process_controllers.clear()
            web_api.simple_worker_states.clear()
        self.client = TestClient(web_api.app)
        web_api.get_simple_queue().set_state(True)

    def tearDown(self):
        web_api.reset_simple_queue_for_tests()
        web_api.history_store = self.previous_store
        web_api._history_store = self.previous_cached_store
        if self.previous_runtime_dir is None:
            os.environ.pop("VT_SIMPLE_RUNTIME_DIR", None)
        else:
            os.environ["VT_SIMPLE_RUNTIME_DIR"] = self.previous_runtime_dir
        with web_api.jobs_lock:
            web_api.jobs.clear()
            web_api.jobs.update(self.previous_jobs)
            web_api.job_process_controllers.clear()
            web_api.simple_worker_states.clear()
        self.temp.cleanup()

    def payload(self, **changes):
        data = {"sourceType": "url", "url": "https://example.org/recording", "profile": "balanced",
                "device": "cpu", "cpuThreads": 2, "restorePunctuation": False}
        data.update(changes)
        return data

    def test_old_request_defaults_and_options_are_separate_from_whisper_devices(self):
        with patch.object(web_api.simple_profiles, "options", return_value={"devices": [{"id": "whisper"}]}) as whisper_options, \
             patch("services.diarization_runtime.diarization_options", return_value={
                 "enabledByDefault": False, "defaultDevice": "auto", "devices": [], "ready": False
             }):
            options = self.client.get("/api/simple/options").json()
            self.assertEqual(options["devices"], [{"id": "whisper"}])
            self.assertFalse(options["diarization"]["ready"])
            whisper_options.assert_called_once()
        response = self.client.post("/api/simple/jobs", json=self.payload())
        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertFalse(body["diarizationEnabled"])
        self.assertEqual(body["diarizationDevice"], "auto")
        self.assertEqual(body["diarization"]["status"], "disabled")
        record = web_api.history_store.get(body["id"])
        self.assertFalse(record["settings"]["diarizationEnabled"])
        self.assertEqual(record["settings"]["diarizationDevice"], "auto")
        self.assertEqual(next(item for item in record["checkpoints"] if item["stage"] == "diarization")["status"], "skipped")

    def test_batch_settings_survive_queue_restore_and_hydration(self):
        request = {"paths": [str(Path(self.temp.name) / "one.wav"), str(Path(self.temp.name) / "two.wav")],
                   "profile": "balanced", "device": "cpu", "cpuThreads": 2,
                   "restorePunctuation": False, "diarizationEnabled": True,
                   "diarizationDevice": "opaque-intel-id-123"}
        response = self.client.post("/api/simple/jobs/batch", json=request)
        self.assertEqual(response.status_code, 202, response.text)
        for item in response.json()["jobs"]:
            record = web_api.history_store.get(item["id"])
            self.assertTrue(record["diarizationEnabled"])
            self.assertEqual(record["diarizationDevice"], "opaque-intel-id-123")
            self.assertEqual(record["settings"]["diarizationDevice"], "opaque-intel-id-123")
            web_api.jobs.pop(item["id"])
            restored = web_api._hydrate_simple_job(record)
            self.assertTrue(restored.settings["diarizationEnabled"])
            self.assertEqual(restored.settings["diarizationDevice"], "opaque-intel-id-123")

    def test_build_command_adds_diarization_flags_only_when_enabled(self):
        base = {"_simple": True, "language": "auto", "model": "small", "device": "cpu", "nproc": 2,
                "outputFormat": "txt", "saveAudio": False, "useTimestamps": False,
                "restorePunctuation": False, "_outputDir": str(Path(self.temp.name) / "out")}
        disabled = web_api.build_command(JobState("1" * 32, dict(base)), "input.wav")
        self.assertNotIn("--diarization", disabled)
        enabled = web_api.build_command(JobState("2" * 32, {**base, "diarizationEnabled": True,
                                                             "diarizationDevice": "opaque-gpu-id"}), "input.wav")
        self.assertIn("--diarization", enabled)
        self.assertEqual(enabled[enabled.index("--diarization-device") + 1], "opaque-gpu-id")

    def test_missing_device_fails_before_preparation_or_asr_without_fallback(self):
        response = self.client.post("/api/simple/jobs", json=self.payload(
            diarizationEnabled=True, diarizationDevice="vanished-gpu"))
        job_id = response.json()["id"]
        with patch("services.diarization_runtime.resolve_diarization_device",
                   side_effect=TaskFailure("missing", "device_unavailable", stage="diarization")), \
             patch.object(web_api, "_run_owned_process") as owned_process:
            web_api._run_simple_job(job_id, "https://example.org/recording")
        owned_process.assert_not_called()
        record = web_api.history_store.get(job_id)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["stage"], "diarization")
        self.assertEqual(record["diarization"]["status"], "failed")
        self.assertEqual(next(item for item in record["checkpoints"] if item["stage"] == "diarization")["status"], "failed")

    def test_completed_turns_are_sidecar_and_text_edits_only_change_txt_md(self):
        response = self.client.post("/api/simple/jobs", json=self.payload(
            diarizationEnabled=True, diarizationDevice="cpu"))
        job_id = response.json()["id"]
        self.assertTrue(web_api._mark_simple_job_started(job_id))
        web_api.jobs[job_id].status = "running"
        turns = [{"start": 0.0, "end": 1.0, "text": "Привет", "speakerIds": [0], "label": "Говорящий 1"}]
        web_api._simple_stage_event(job_id, "diarization", "active")
        web_api._simple_stage_event(job_id, "diarization", "completed")
        web_api.apply_result(job_id, {"text": "[00:00:00] Говорящий 1: Привет", "rendered_result": "[00:00:00] Говорящий 1: Привет",
                                      "diarization": {"status": "completed", "requestedDevice": "cpu",
                                                      "effectiveDevice": {"id": "cpu", "name": "CPU", "backend": "cpu"},
                                                      "turns": turns}})
        self.assertTrue(web_api._finalize_simple_job(job_id, "completed",
                                                     transcript=web_api.jobs[job_id].result_text))
        sidecar = web_api.history_store.job_dir(job_id) / "diarization.json"
        before = sidecar.read_bytes()
        edited = self.client.put(f"/api/simple/jobs/{job_id}/text", json={"text": "Мой вариант", "baseRevision": 0})
        self.assertEqual(edited.status_code, 200)
        self.assertEqual(sidecar.read_bytes(), before)
        shaped = self.client.get(f"/api/simple/jobs/{job_id}").json()
        self.assertEqual(shaped["diarization"]["turns"], turns)
        self.assertEqual(shaped["text"], "Мой вариант")
        self.assertEqual(self.client.get(f"/api/simple/jobs/{job_id}/download?format=txt").text, "Мой вариант")
        self.assertEqual(self.client.get(f"/api/simple/jobs/{job_id}/download?format=md").text, "Мой вариант")

    def test_terminal_status_normalizes_old_diarization_summary(self):
        base = {"diarizationEnabled": True, "diarizationDevice": "old-id",
                "status": "interrupted", "diarization": {"status": "pending", "requestedDevice": "old-id"}}
        for terminal in ("cancelled", "interrupted", "failed"):
            self.assertEqual(web_api._diarization_record(base, terminal_status=terminal)["status"], terminal)
            self.assertEqual(web_api._diarization_record({**base, "status": terminal})["status"], terminal)
        completed = {**base, "diarization": {"status": "completed", "requestedDevice": "old-id"}}
        self.assertEqual(web_api._diarization_record(completed, terminal_status="cancelled")["status"], "completed")

    def test_off_stage_stays_skipped_while_advancing_recognition_checkpoint(self):
        response = self.client.post("/api/simple/jobs", json=self.payload())
        job_id = response.json()["id"]
        record = web_api.history_store.get(job_id)
        checkpoints = [dict(item) for item in record["checkpoints"]]
        next(item for item in checkpoints if item["stage"] == "recognition")["status"] = "active"
        web_api.history_store.update(job_id, checkpoints=checkpoints)
        web_api._simple_stage_event(job_id, "diarization", "completed")
        checkpoints = {item["stage"]: item["status"] for item in web_api.history_store.get(job_id)["checkpoints"]}
        self.assertEqual(checkpoints["recognition"], "completed")
        self.assertEqual(checkpoints["diarization"], "skipped")

    def test_sidecar_write_failure_rolls_back_artifacts_before_terminal_commit(self):
        response = self.client.post("/api/simple/jobs", json=self.payload(
            diarizationEnabled=True, diarizationDevice="cpu"))
        job_id = response.json()["id"]
        self.assertTrue(web_api._mark_simple_job_started(job_id))
        web_api.jobs[job_id].status = "running"
        web_api.apply_result(job_id, {"text": "new text", "rendered_result": "new text",
                                      "diarization": {"status": "completed", "requestedDevice": "cpu",
                                                      "effectiveDevice": {"id": "cpu", "name": "CPU", "backend": "cpu"},
                                                      "turns": [{"text": "turn"}]}})
        txt = web_api.history_store.artifact(job_id, "txt")
        md = web_api.history_store.artifact(job_id, "md")
        txt.write_text("old txt", encoding="utf-8")
        md.write_text("old md", encoding="utf-8")
        sidecar = web_api.history_store.job_dir(job_id) / "diarization.json"
        old_sidecar = sidecar.read_bytes()
        with patch.object(web_api.history_store, "write_diarization", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                web_api._finalize_simple_job(job_id, "completed", transcript="new text")
        self.assertEqual(web_api.history_store.get(job_id)["status"], "running")
        self.assertEqual(web_api.jobs[job_id].status, "running")
        self.assertEqual(txt.read_text(encoding="utf-8"), "old txt")
        self.assertEqual(md.read_text(encoding="utf-8"), "old md")
        self.assertEqual(sidecar.read_bytes(), old_sidecar)

    def test_turns_stay_out_of_sqlite_and_legacy_apply_result_does_not_create_history(self):
        response = self.client.post("/api/simple/jobs", json=self.payload(
            diarizationEnabled=True, diarizationDevice="cpu"))
        job_id = response.json()["id"]
        self.assertTrue(web_api._mark_simple_job_started(job_id))
        web_api.jobs[job_id].status = "running"
        turns = [{"start": 1, "end": 2, "text": "hello", "speakerIds": [1], "label": "Speaker 1"}]
        web_api.apply_result(job_id, {"text": "hello", "diarization": {"status": "completed",
                                  "requestedDevice": "cpu", "effectiveDevice": {"id": "cpu"}, "turns": turns}})
        record = web_api.history_store.get(job_id)
        self.assertNotIn("turns", record["diarization"])
        self.assertEqual(web_api._simple_shape(record, with_text=False)["diarization"]["turns"], [])
        self.assertEqual(web_api._simple_shape(record, with_text=True)["diarization"]["turns"], turns)

        legacy_id = "legacy-non-simple"
        web_api.jobs[legacy_id] = JobState(legacy_id, {"_simple": False, "language": "auto", "model": "small",
                                                        "device": "cpu", "nproc": 1, "outputFormat": "txt",
                                                        "saveAudio": False, "useTimestamps": True,
                                                        "restorePunctuation": False})
        web_api.apply_result(legacy_id, {"text": "legacy", "rendered_result": "legacy"})
        self.assertFalse((web_api.history_store.transcriptions_dir / legacy_id).exists())

    def test_download_names_match_unicode_source_and_sanitize_url_result_basename(self):
        media = Path(self.temp.name) / "Встреча.mp4"
        media.write_bytes(b"media")
        response = self.client.post("/api/simple/jobs", json=self.payload(
            sourceType="file", path=str(media)))
        job_id = response.json()["id"]
        self.assertTrue(web_api._mark_simple_job_started(job_id))
        web_api.jobs[job_id].status = "running"
        web_api.apply_result(job_id, {"text": "original", "rendered_result": "original"})
        self.assertTrue(web_api._finalize_simple_job(job_id, "completed", transcript="original"))
        single = self.client.get(f"/api/simple/jobs/{job_id}").json()
        history = self.client.get("/api/simple/history").json()["items"][0]
        self.assertEqual([item["name"] for item in single["files"]], ["Встреча.txt", "Встреча.md"])
        self.assertEqual([item["name"] for item in history["files"]], ["Встреча.txt", "Встреча.md"])
        for ext in ("txt", "md"):
            downloaded = self.client.get(f"/api/simple/jobs/{job_id}/download?format={ext}")
            self.assertIn("filename*=utf-8''", downloaded.headers["content-disposition"].lower())
            self.assertIn(quote(f"Встреча.{ext}").lower(),
                          downloaded.headers["content-disposition"].lower())

        edited = self.client.put(f"/api/simple/jobs/{job_id}/text", json={"text": "edited", "baseRevision": 0})
        self.assertEqual(edited.status_code, 200)
        self.assertEqual(self.client.get(f"/api/simple/jobs/{job_id}/download?format=txt").text, "edited")
        self.assertEqual(self.client.get(f"/api/simple/jobs/{job_id}/download?format=md").text, "edited")

        url_job = self.client.post("/api/simple/jobs", json=self.payload()).json()["id"]
        self.assertTrue(web_api._mark_simple_job_started(url_job))
        web_api.jobs[url_job].status = "running"
        web_api.apply_result(url_job, {"text": "url", "rendered_result": "url",
                                       "input": r"C:\temp\..\NUL.mp4"})
        self.assertEqual(web_api.history_store.get(url_job)["sourceFilename"], "_NUL.mp4")
        self.assertTrue(web_api._finalize_simple_job(url_job, "completed", transcript="url"))
        names = [item["name"] for item in self.client.get(f"/api/simple/jobs/{url_job}").json()["files"]]
        self.assertEqual(names, ["_NUL.txt", "_NUL.md"])
        content_disposition = self.client.get(f"/api/simple/jobs/{url_job}/download?format=txt").headers[
            "content-disposition"]
        self.assertIn('filename="_NUL.txt"', content_disposition)


if __name__ == "__main__":
    unittest.main()
