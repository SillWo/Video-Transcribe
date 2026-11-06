import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_api


class BatchApiTests(unittest.TestCase):
    def setUp(self):
        web_api.reset_simple_queue_for_tests()
        self.temp = tempfile.TemporaryDirectory()
        self.old_runtime = os.environ.get("VT_SIMPLE_RUNTIME_DIR")
        self.old_store = web_api._history_store
        os.environ["VT_SIMPLE_RUNTIME_DIR"] = str(Path(self.temp.name) / "runtime")
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
        web_api._history_store = self.old_store
        if self.old_runtime is None:
            os.environ.pop("VT_SIMPLE_RUNTIME_DIR", None)
        else:
            os.environ["VT_SIMPLE_RUNTIME_DIR"] = self.old_runtime
        with web_api.jobs_lock:
            web_api.jobs.clear()
            web_api.jobs.update(self.old_jobs)
            web_api.job_process_controllers.clear()
            web_api.simple_worker_states.clear()
        self.temp.cleanup()

    def payload(self, **changes):
        result = {"profile": "balanced", "device": "cpu", "cpuThreads": 2,
                  "restorePunctuation": False}
        result.update(changes)
        return result

    def test_batch_accepts_absolute_missing_and_duplicate_paths_and_rejects_bad_entries(self):
        root = Path(self.temp.name)
        source = root / "audio.wav"
        source.touch()
        folder = root / "folder"
        folder.mkdir()
        missing = root / "gone.wav"
        response = self.client.post("/api/simple/jobs/batch", json=self.payload(paths=[
            str(source), str(source), str(missing), "relative.wav", str(folder)]))
        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(len(body["jobs"]), 3)
        self.assertEqual([item["source"] for item in body["jobs"][:2]], [str(source)] * 2)
        self.assertEqual([item["queuePosition"] for item in body["queue"]["waitingJobs"]], [1, 2, 3])
        self.assertEqual(body["queue"]["waitingCount"], 3)
        self.assertEqual(len(body["rejected"]), 2)
        self.assertEqual(len({item["id"] for item in body["jobs"]}), 3)
        self.assertTrue(all("text" not in item for item in body["queue"]["waitingJobs"]))

    def test_common_validation_empty_batch_and_picker_multiple_contract(self):
        self.assertEqual(self.client.post("/api/simple/jobs/batch", json=self.payload(paths=[])).status_code, 400)
        for changes in ({"profile": "unknown"}, {"device": "cuda"}, {"cpuThreads": 0}):
            self.assertEqual(self.client.post("/api/simple/jobs/batch", json=self.payload(
                paths=[str(Path(self.temp.name) / "x.wav")], **changes)).status_code, 400)
        source = Path(self.temp.name) / "picked.wav"
        source.touch()
        with patch.object(web_api.native_file_picker, "pick_files", return_value=[str(source), "gone"]):
            result = self.client.post("/api/simple/pick-files").json()
        self.assertEqual(result["files"], [{"path": str(source.resolve()), "name": "picked.wav"}])
        self.assertEqual(result["rejected"][0]["path"], "gone")

    def test_queued_cancel_and_cancel_all_remove_fifo_entries(self):
        paths = [str(Path(self.temp.name) / "one.wav"), str(Path(self.temp.name) / "two.wav")]
        batch = self.client.post("/api/simple/jobs/batch", json=self.payload(paths=paths)).json()
        first, second = [item["id"] for item in batch["jobs"]]
        cancelled = self.client.post(f"/api/simple/jobs/{first}/cancel")
        self.assertEqual(cancelled.status_code, 202)
        self.assertEqual(cancelled.json()["status"], "cancelled")
        self.assertEqual(self.client.get("/api/simple/queue").json()["waitingCount"], 1)
        state = self.client.post("/api/simple/queue/cancel")
        self.assertEqual(state.status_code, 202)
        self.assertEqual(state.json()["waitingCount"], 0)
        self.assertEqual(web_api.history_store.get(second)["status"], "cancelled")

    def test_resume_clears_persisted_pause_without_repeating_failed_job(self):
        self.client.post("/api/simple/jobs/batch", json=self.payload(
            paths=[str(Path(self.temp.name) / "one.wav")]))
        web_api.get_simple_queue().set_state(True, {"code": "device_error", "message": "GPU недоступен."})
        dispatcher = web_api._get_simple_dispatcher()
        with patch.object(dispatcher, "start"):
            response = self.client.post("/api/simple/queue/resume")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["paused"])
        self.assertIsNone(response.json()["pauseReason"])
        self.assertEqual(response.json()["waitingCount"], 1)

    def test_local_source_failure_continues_but_systemic_device_failure_pauses_fifo(self):
        missing = Path(self.temp.name) / "missing.wav"
        batch = self.client.post("/api/simple/jobs/batch", json=self.payload(paths=[str(missing)])).json()
        local_id = batch["jobs"][0]["id"]
        web_api.get_simple_queue().set_state(False)
        web_api.run_job(local_id, str(missing))
        local = web_api.history_store.get(local_id)
        self.assertEqual(local["status"], "failed")
        self.assertEqual(local["errorCode"], "source_unavailable")
        self.assertFalse(web_api.get_simple_queue().state()["paused"])

        source = Path(self.temp.name) / "present.wav"
        source.touch()
        gpu = self.client.post("/api/simple/jobs", json={**self.payload(device="gpu"),
                                                          "sourceType": "file", "path": str(source)}).json()
        with patch.object(web_api.simple_profiles, "resolve", side_effect=ValueError("runtime unavailable")):
            web_api.run_job(gpu["id"], str(source))
        failed = web_api.history_store.get(gpu["id"])
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["errorCode"], "device_unavailable")
        self.assertTrue(web_api.get_simple_queue().state()["paused"])
        self.assertEqual(web_api.get_simple_queue().state()["pauseReason"]["code"], "device_unavailable")


if __name__ == "__main__":
    unittest.main()
