import tempfile
import unittest
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_api
from services import native_file_picker
from web_api import JobState


class _FakeStream(list):
    def close(self):
        return None


class _FakeProcess:
    def __init__(self, lines=(), returncode=0):
        self.stdout = _FakeStream(lines)
        self.returncode = returncode

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class _BlockingStream:
    def __init__(self):
        self.items = queue.Queue()

    def __iter__(self):
        while True:
            line = self.items.get()
            if line is None:
                return
            yield line

    def close(self):
        self.items.put(None)


class _BlockingProcess:
    def __init__(self):
        self.stdout = _BlockingStream()
        self.returncode = None
        self._stopped = threading.Event()
        self.reading = threading.Event()

    def wait(self):
        self._stopped.wait(3)
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self._stopped.set()
        self.stdout.close()

    def kill(self):
        self.returncode = -9
        self._stopped.set()
        self.stdout.close()


class _FakeController:
    def __init__(self):
        self.cancelled = False
        self.process = None

    def check_cancelled(self):
        if self.cancelled:
            raise web_api.JobCancelled("cancelled")

    def register(self, process):
        self.process = process
        self.check_cancelled()

    def unregister(self, process):
        if self.process is process:
            self.process = None

    def stop_owned(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def cancel(self):
        self.cancelled = True
        if self.process is not None:
            self.process.terminate()


class SimpleApiTests(unittest.TestCase):
    def setUp(self):
        web_api.reset_simple_queue_for_tests()
        self.temp = tempfile.TemporaryDirectory()
        self.previous_store = web_api.history_store
        self.previous_cached_store = web_api._history_store
        self.previous_runtime_dir = os.environ.get("VT_SIMPLE_RUNTIME_DIR")
        os.environ["VT_SIMPLE_RUNTIME_DIR"] = str(Path(self.temp.name) / "runtime")
        web_api._history_store = None
        self.previous_jobs = web_api.jobs.copy()
        with web_api.jobs_lock:
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
        data = {"sourceType": "url", "url": "https://example.org/watch/123", "profile": "balanced",
                "device": "cpu", "cpuThreads": 2, "restorePunctuation": False}
        data.update(changes)
        return data

    def test_picker_cancellation_and_canonical_path(self):
        with patch.object(web_api.native_file_picker, "pick_file", return_value=None):
            self.assertEqual(self.client.post("/api/simple/pick-file").json(), {"path": None, "name": None})
        selected = Path(self.temp.name) / "файл с пробелом.wav"
        selected.touch()
        with patch.object(web_api.native_file_picker, "pick_file", return_value=str(selected)):
            response = self.client.post("/api/simple/pick-file")
        self.assertEqual(response.json(), {"path": str(selected.resolve()), "name": "файл с пробелом.wav"})

    def test_native_picker_starts_owned_tk_dialog_without_console(self):
        result = type("Result", (), {"returncode": 0, "stdout": "C:/файл.wav"})()
        with (patch.object(native_file_picker.os, "name", "nt"),
              patch.object(native_file_picker.subprocess, "run", return_value=result) as run):
            self.assertEqual(native_file_picker._show_windows_dialog(), "C:/файл.wav")
        command = run.call_args.args[0]
        self.assertEqual(command, [native_file_picker.sys.executable, "-m", "services.native_file_picker", "--dialog"])
        self.assertEqual(run.call_args.kwargs["creationflags"], getattr(native_file_picker.subprocess, "CREATE_NO_WINDOW", 0))
        self.assertIs(run.call_args.kwargs["stdin"], native_file_picker.subprocess.DEVNULL)
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(run.call_args.kwargs["timeout"], 300)
        self.assertEqual(run.call_args.kwargs["cwd"], Path(native_file_picker.__file__).resolve().parents[1])

    def test_native_picker_multiple_mode_uses_json_protocol(self):
        selected = ["C:/первый.wav", "D:/второй.mp4"]
        result = type("Result", (), {"returncode": 0, "stdout": json.dumps(selected, ensure_ascii=False)})()
        with (patch.object(native_file_picker.os, "name", "nt"),
              patch.object(native_file_picker.subprocess, "run", return_value=result) as run):
            self.assertEqual(native_file_picker._show_windows_dialog(multiple=True), selected)
        self.assertEqual(run.call_args.args[0], [native_file_picker.sys.executable, "-m",
                                                 "services.native_file_picker", "--dialog-multiple"])
        self.assertIs(run.call_args.kwargs["stdin"], native_file_picker.subprocess.DEVNULL)

    def test_native_picker_multiple_rejects_malformed_json_and_non_string_items(self):
        for output in ("not-json", "{}", '["valid.wav", 12]'):
            result = type("Result", (), {"returncode": 0, "stdout": output})()
            with (patch.object(native_file_picker.os, "name", "nt"),
                  patch.object(native_file_picker.subprocess, "run", return_value=result)):
                with self.assertRaisesRegex(RuntimeError, "список выбранных файлов"):
                    native_file_picker._show_windows_dialog(multiple=True)

    def test_native_picker_timeout_becomes_readable_error(self):
        with (patch.object(native_file_picker.os, "name", "nt"),
              patch.object(native_file_picker.subprocess, "run", side_effect=native_file_picker.subprocess.TimeoutExpired("python.exe", 300))):
            with self.assertRaisesRegex(RuntimeError, "Время ожидания выбора файла"):
                native_file_picker._show_windows_dialog()

    def test_tk_dialog_uses_owned_native_picker_and_always_destroys_owner(self):
        calls = {}

        class FakeOwner:
            def title(self, value): calls["title"] = value
            def resizable(self, *value): calls["resizable"] = value
            def attributes(self, *value): calls["attributes"] = value
            def winfo_screenwidth(self): return 1920
            def winfo_screenheight(self): return 1080
            def geometry(self, value): calls["geometry"] = value
            def update(self): calls["updated"] = True
            def destroy(self): calls["destroyed"] = True

        class FakeLabel:
            def __init__(self, owner, **kwargs): calls["label"] = (owner, kwargs)
            def pack(self, **kwargs): calls["packed"] = kwargs

        chosen_path = "C:/данные/файл.wav"
        tk_module = ModuleType("tkinter")
        tk_module.Tk = FakeOwner
        tk_module.Label = FakeLabel
        filedialog = ModuleType("tkinter.filedialog")
        filedialog.askopenfilename = lambda **kwargs: calls.update(dialog=kwargs) or chosen_path
        tk_module.filedialog = filedialog
        with patch.dict("sys.modules", {"tkinter": tk_module, "tkinter.filedialog": filedialog}):
            result = native_file_picker._run_tk_dialog()
        self.assertEqual(result, chosen_path)
        self.assertEqual(calls["title"], "Video Transcribe — выбор файла")
        self.assertEqual(calls["attributes"], ("-topmost", True))
        self.assertEqual(calls["geometry"], "420x120+750+480")
        self.assertTrue(calls["updated"])
        self.assertTrue(calls["destroyed"])
        self.assertIs(calls["dialog"]["parent"], calls["label"][0])

    def test_concurrent_first_requests_construct_one_history_store(self):
        constructor = web_api.TranscriptionHistory
        web_api._history_store = None
        count = 0
        count_lock = threading.Lock()

        def tracked_constructor(root):
            nonlocal count
            with count_lock:
                count += 1
            time.sleep(0.02)
            return constructor(root)

        with patch.object(web_api, "TranscriptionHistory", side_effect=tracked_constructor):
            with ThreadPoolExecutor(max_workers=8) as pool:
                stores = list(pool.map(lambda _: web_api.get_history_store(), range(8)))
        self.assertEqual(count, 1)
        self.assertTrue(all(store is stores[0] for store in stores))

    def test_submit_records_only_metadata_and_adds_second_job_to_fifo(self):
        with patch.object(web_api.simple_profiles, "resolve", return_value=("large-v3-turbo", "cpu")), \
             patch.object(web_api, "run_job", return_value=None):
            response = self.client.post("/api/simple/jobs", json=self.payload())
            self.assertEqual(response.status_code, 202)
            second = self.client.post("/api/simple/jobs", json=self.payload())
        self.assertEqual(second.status_code, 202)
        body = response.json()
        self.assertIsNone(body["text"])
        self.assertNotIn(b"transcript text", web_api.history_store.db_path.read_bytes())
        self.assertEqual(self.client.delete(f"/api/simple/history/{body['id']}").status_code, 409)

    def test_source_validation_and_history_shapes(self):
        with patch.object(web_api, "run_job", return_value=None), \
             patch.object(web_api.simple_profiles, "resolve", return_value=("small", "cpu")):
            self.assertEqual(self.client.post("/api/simple/jobs", json=self.payload(url="file:///secret")).status_code, 400)
            created = self.client.post("/api/simple/jobs", json=self.payload()).json()
        session = self.client.get("/api/simple/session").json()["job"]
        history = self.client.get("/api/simple/history").json()["items"][0]
        self.assertEqual(session["id"], created["id"])
        self.assertNotIn("text", history)
        self.assertEqual(history["source"], "https://example.org/watch/123")
        self.assertEqual([item["stage"] for item in created["checkpoints"]],
                         ["preparation", "source", "audio", "recognition", "diarization", "postprocess", "result"])
        self.assertIsNone(created["progress"])
        self.assertIsNone(created["finishedAt"])
        self.assertIsNone(created["durationSeconds"])
        self.assertEqual(self.client.get(f"/api/simple/jobs/{created['id']}").json()["id"], created["id"])

    def test_progress_is_bounded_persisted_and_does_not_expose_pipeline_messages(self):
        job_id = "8" * 32
        web_api.history_store.create({"id": job_id, "sourceType": "file", "source": "C:/source.wav",
                                      "profile": "quality", "device": "cpu", "cpuThreads": 2,
                                      "restorePunctuation": False})
        web_api._simple_stage_event(job_id, "recognition", "active")
        web_api._simple_progress(job_id, "recognition", {"percent": 140, "completedBytes": 999,
                                                            "totalBytes": 100, "message": "C:/private.wav"})
        public = self.client.get(f"/api/simple/jobs/{job_id}").json()
        self.assertEqual(public["progress"]["percent"], 100)
        self.assertEqual(public["progress"]["completedBytes"], 100)
        self.assertEqual(public["progress"]["message"], "Распознавание речи")
        active = next(item for item in public["checkpoints"] if item["stage"] == "recognition")
        self.assertEqual(active["status"], "active")
        self.assertIsNotNone(active["startedAt"])

    def test_progress_allows_only_exact_cli_warmup_and_fragment_count_messages(self):
        job_id = "7" * 32
        web_api.history_store.create({"id": job_id, "sourceType": "file", "source": "C:/source.wav",
                                      "profile": "quality", "device": "cpu", "cpuThreads": 2,
                                      "restorePunctuation": True})
        web_api._simple_progress(job_id, "recognition", {"message": "Загружаем модель в память."})
        warmup = self.client.get(f"/api/simple/jobs/{job_id}").json()["progress"]
        self.assertEqual(warmup["message"], "Загружаем модель в память")
        web_api._simple_progress(job_id, "postprocess", {"message": "Обработано фрагментов 12 из 37."})
        fragments = self.client.get(f"/api/simple/jobs/{job_id}").json()["progress"]
        self.assertEqual(fragments["message"], "Обработано фрагментов 12 из 37.")
        web_api._simple_progress(job_id, "postprocess", {"message": "Обработано фрагментов C:/secret из 37."})
        private = self.client.get(f"/api/simple/jobs/{job_id}").json()["progress"]
        self.assertEqual(private["message"], "Обработка текста")

    def test_worker_events_persist_success_artifacts_and_failure_without_artifacts(self):
        def make_job(job_id, status_lines, exit_code=0):
            output_dir = web_api.history_store.job_dir(job_id) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            generated = output_dir / "captions.txt"
            generated.write_text("backend output", encoding="utf-8")
            metadata = {"id": job_id, "sourceType": "file", "source": "C:/source.wav",
                        "profile": "economy", "device": "cpu", "cpuThreads": 2,
                        "restorePunctuation": False}
            web_api.history_store.create(metadata)
            settings = {"_simple": True, "_outputDir": str(output_dir), "language": "auto",
                        "model": "small", "device": "cpu", "nproc": 2, "outputFormat": "txt",
                        "saveAudio": False, "useTimestamps": False, "restorePunctuation": False}
            job = JobState(id=job_id, settings=settings)
            with web_api.jobs_lock:
                web_api.jobs[job_id] = job
            encoded = [web_api.JSON_EVENT_PREFIX + json.dumps(item) + "\n" for item in status_lines]

            def factory(command, **kwargs):
                if "services.resource_preparation" in command:
                    return _FakeProcess()
                return _FakeProcess(encoded, exit_code)

            return factory, generated

        success_id = "c" * 32
        result_event = {"type": "result", "stage": "result", "result": {
            "text": "recognized words", "output_path": str(web_api.history_store.job_dir(success_id) / "output" / "captions.txt"),
            "output_format": "txt", "detected_language": "en"}}
        factory, _ = make_job(success_id, [result_event])
        with (patch.object(web_api, "JobProcessController", _FakeController),
              patch.object(web_api.subprocess, "Popen", side_effect=factory)):
            web_api.run_job(success_id, "C:/source.wav")
        completed = web_api.history_store.get(success_id)
        self.assertEqual(completed["status"], "completed")
        completed_shape = self.client.get(f"/api/simple/jobs/{success_id}").json()
        self.assertIsInstance(completed_shape["finishedAt"], str)
        self.assertGreaterEqual(completed_shape["durationSeconds"], 0)
        self.assertEqual(web_api.history_store.read_text(success_id), "recognized words")
        self.assertEqual(web_api.history_store.artifact(success_id, "md").read_text(encoding="utf-8"), "recognized words")

        failed_id = "d" * 32
        error_event = {"type": "error", "stage": "recognition", "message": "decode failed"}
        factory, _ = make_job(failed_id, [error_event], exit_code=1)
        with (patch.object(web_api, "JobProcessController", _FakeController),
              patch.object(web_api.subprocess, "Popen", side_effect=factory)):
            web_api.run_job(failed_id, "C:/source.wav")
        failed = web_api.history_store.get(failed_id)
        self.assertEqual(failed["status"], "failed")
        failed_shape = self.client.get(f"/api/simple/jobs/{failed_id}").json()
        self.assertIsInstance(failed_shape["finishedAt"], str)
        self.assertGreaterEqual(failed_shape["durationSeconds"], 0)
        self.assertEqual(failed["stage"], "recognition")
        self.assertIsNone(web_api.history_store.read_text(failed_id))
        self.assertEqual(failed_shape["error"], "Не удалось обработать эту запись. Проверьте источник и повторите попытку.")
        self.assertIn("[error] decode failed", web_api.jobs[failed_id].logs)

    def test_missing_simple_model_is_prepared_before_process_start(self):
        job_id = "b" * 32
        output_dir = web_api.history_store.job_dir(job_id) / "output"
        output_dir.mkdir(parents=True)
        generated = output_dir / "backend.txt"
        generated.write_text("backend output", encoding="utf-8")
        web_api.history_store.create({"id": job_id, "sourceType": "file", "source": "C:/source.wav",
                                      "profile": "quality", "device": "cpu", "cpuThreads": 2,
                                      "restorePunctuation": False})
        settings = {"_simple": True, "_outputDir": str(output_dir), "language": "auto",
                    "model": "large-v3", "device": "cpu", "nproc": 2, "outputFormat": "txt",
                    "saveAudio": False, "useTimestamps": False, "restorePunctuation": False}
        with web_api.jobs_lock:
            web_api.jobs[job_id] = JobState(id=job_id, settings=settings)
        result_line = web_api.JSON_EVENT_PREFIX + json.dumps({"type": "result", "stage": "result", "result": {
            "text": "recognized words", "output_path": str(generated), "output_format": "txt"}}) + "\n"

        def process_factory(command, **kwargs):
            return _FakeProcess() if "services.resource_preparation" in command else _FakeProcess([result_line])

        with (patch.object(web_api, "JobProcessController", _FakeController),
              patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
              patch.object(web_api.subprocess, "Popen", side_effect=process_factory) as popen):
            web_api.run_job(job_id, "C:/source.wav")
        self.assertEqual(popen.call_count, 2)
        prepare_command = popen.call_args_list[0].args[0]
        self.assertEqual(prepare_command, [web_api.sys.executable, "-m", "services.resource_preparation",
                                           "--model", "large-v3", "--device", "cpu"])
        prepare_env = popen.call_args_list[0].kwargs["env"]
        asr_env = popen.call_args_list[1].kwargs["env"]
        self.assertIs(popen.call_args_list[0].kwargs["stdin"], web_api.subprocess.DEVNULL)
        self.assertIs(popen.call_args_list[1].kwargs["stdin"], web_api.subprocess.DEVNULL)
        self.assertNotIn("HF_HUB_OFFLINE", prepare_env)
        self.assertNotIn("TRANSFORMERS_OFFLINE", prepare_env)
        self.assertEqual(asr_env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(asr_env["TRANSFORMERS_OFFLINE"], "1")
        record = web_api.history_store.get(job_id)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(web_api.history_store.read_text(job_id), "recognized words")

    def test_failed_preparation_keeps_checkpoint_and_resubmit_creates_retry_job(self):
        with (patch.object(web_api.simple_profiles, "resolve", return_value=("small", "cpu")),
              patch.object(web_api, "run_job", return_value=None)):
            first = self.client.post("/api/simple/jobs", json=self.payload()).json()
        error_line = "__VT_PREP__ " + json.dumps({"type": "error", "stage": "preparation",
                                                  "message": "Не удалось загрузить модель. Проверьте подключение к интернету и повторите попытку."})
        with (patch.object(web_api, "JobProcessController", _FakeController),
              patch.object(web_api.subprocess, "Popen", return_value=_FakeProcess([error_line], 1)) as popen):
            web_api.run_job(first["id"], "https://example.org/watch/123")
        popen.assert_called_once()
        self.assertIn("services.resource_preparation", popen.call_args.args[0])
        failed = self.client.get(f"/api/simple/jobs/{first['id']}").json()
        self.assertEqual(failed["status"], "failed")
        prep = next(item for item in failed["checkpoints"] if item["stage"] == "preparation")
        self.assertEqual(prep["status"], "failed")
        with (patch.object(web_api.simple_profiles, "resolve", return_value=("small", "cpu")),
              patch.object(web_api, "run_job", return_value=None)):
            second = self.client.post("/api/simple/jobs", json=self.payload()).json()
        self.assertNotEqual(first["id"], second["id"])

    def _create_controlled_job(self):
        with (patch.object(web_api.simple_profiles, "resolve", return_value=("small", "cpu")),
              patch.object(web_api, "run_job", return_value=None),
              patch.object(web_api, "JobProcessController", _FakeController)):
            response = self.client.post("/api/simple/jobs", json=self.payload())
        self.assertEqual(response.status_code, 202)
        return response.json()

    def test_cancel_before_worker_starts_is_idempotent_and_cancels_queued_job(self):
        body = self._create_controlled_job()
        first = self.client.post(f"/api/simple/jobs/{body['id']}/cancel")
        second = self.client.post(f"/api/simple/jobs/{body['id']}/cancel")
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(first.json()["status"], "cancelled")
        self.assertEqual(second.json()["status"], "cancelled")
        self.assertIsInstance(first.json()["finishedAt"], str)
        self.assertGreaterEqual(first.json()["durationSeconds"], 0)
        with patch.object(web_api.simple_profiles, "resolve", return_value=("small", "cpu")):
            self.assertEqual(self.client.post("/api/simple/jobs", json=self.payload()).status_code, 202)
        web_api.run_job(body["id"], "https://example.org/watch/123")
        cancelled = self.client.get(f"/api/simple/jobs/{body['id']}").json()
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIsNone(cancelled["error"])
        self.assertIsInstance(cancelled["finishedAt"], str)
        self.assertGreaterEqual(cancelled["durationSeconds"], 0)
        prep = next(item for item in cancelled["checkpoints"] if item["stage"] == "preparation")
        self.assertEqual(prep["status"], "cancelled")
        self.assertEqual(self.client.post(f"/api/simple/jobs/{body['id']}/cancel").json()["status"], "cancelled")
        self.assertEqual(self.client.delete(f"/api/simple/history/{body['id']}").status_code, 204)

    def test_cancel_while_preparation_child_runs_stops_child_before_terminal_state(self):
        body = self._create_controlled_job()
        controller = web_api.job_process_controllers[body["id"]]
        child = _BlockingProcess()
        thread = threading.Thread(target=web_api.run_job, args=(body["id"], "https://example.org/watch/123"))
        with patch.object(web_api.subprocess, "Popen", return_value=child) as popen:
            thread.start()
            deadline = time.time() + 2
            while controller.process is None and time.time() < deadline:
                time.sleep(0.01)
            self.assertIs(controller.process, child)
            response = self.client.post(f"/api/simple/jobs/{body['id']}/cancel")
            self.assertEqual(response.status_code, 202)
            thread.join(3)
        self.assertFalse(thread.is_alive())
        record = web_api.history_store.get(body["id"])
        self.assertEqual(record["status"], "cancelled")
        self.assertIsNone(web_api.history_store.read_text(body["id"]))
        self.assertEqual(popen.call_count, 1)

    def test_shutdown_keeps_claimed_job_queued_when_it_has_not_acquired_runtime_slot(self):
        self.assertTrue(web_api.transcription_semaphore.acquire(blocking=False))
        try:
            body = self.client.post("/api/simple/jobs", json=self.payload()).json()
            web_api.get_simple_queue().set_state(False)
            dispatcher = web_api._get_simple_dispatcher()
            dispatcher.start()
            deadline = time.time() + 2
            while (dispatcher.active_job_id != body["id"] or
                   web_api.simple_worker_states.get(body["id"]) != "running") and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(dispatcher.active_job_id, body["id"])
            record = web_api.history_store.get(body["id"])
            self.assertEqual(record["status"], "queued")
            self.assertIsNone(record["startedAt"])
            snapshot = web_api.get_simple_queue_snapshot()
            self.assertIsNone(snapshot["activeJob"])
            self.assertEqual(snapshot["waitingCount"], 1)
            self.assertEqual(snapshot["waitingJobs"][0]["id"], body["id"])
            self.assertEqual(snapshot["waitingJobs"][0]["queuePosition"], 1)
            web_api.prepare_simple_shutdown(timeout=2)
            record = web_api.history_store.get(body["id"])
            self.assertEqual(record["status"], "queued")
            self.assertIn(body["id"], web_api.get_simple_queue().ids())
            self.assertTrue(web_api.get_simple_queue().state()["paused"])
        finally:
            web_api.transcription_semaphore.release()

    def test_shutdown_stops_active_owned_process_and_records_interrupted_not_cancelled(self):
        body = self._create_controlled_job()
        controller = web_api.job_process_controllers[body["id"]]
        child = _BlockingProcess()
        thread = threading.Thread(target=web_api.run_job,
                                  args=(body["id"], "https://example.org/watch/123"))
        with patch.object(web_api.subprocess, "Popen", return_value=child):
            thread.start()
            deadline = time.time() + 2
            while controller.process is None and time.time() < deadline:
                time.sleep(0.01)
            self.assertIs(controller.process, child)
            web_api.prepare_simple_shutdown(timeout=2)
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(web_api.history_store.get(body["id"])["status"], "interrupted")
        self.assertEqual(web_api.history_store.get(body["id"])["errorCode"], None)

    def test_cancel_while_asr_child_runs_does_not_publish_final_artifacts(self):
        body = self._create_controlled_job()
        controller = web_api.job_process_controllers[body["id"]]
        child = _BlockingProcess()
        asr_started = threading.Event()
        prep_line = "__VT_PREP__ " + json.dumps({"type": "progress", "stage": "preparation",
                                                  "status": "completed", "percent": 100}) + "\n"
        calls = []

        def factory(command, **kwargs):
            calls.append(command)
            if "services.resource_preparation" in command:
                return _FakeProcess([prep_line])
            asr_started.set()
            return child

        thread = threading.Thread(target=web_api.run_job, args=(body["id"], "https://example.org/watch/123"))
        with patch.object(web_api.subprocess, "Popen", side_effect=factory):
            thread.start()
            self.assertTrue(asr_started.wait(2))
            deadline = time.time() + 2
            while controller.process is not child and time.time() < deadline:
                time.sleep(0.01)
            self.assertIs(controller.process, child)
            response = self.client.post(f"/api/simple/jobs/{body['id']}/cancel")
            self.assertEqual(response.status_code, 202)
            thread.join(3)
        self.assertFalse(thread.is_alive())
        record = web_api.history_store.get(body["id"])
        self.assertEqual(record["status"], "cancelled")
        self.assertIsNone(web_api.history_store.read_text(body["id"]))
        self.assertFalse(web_api.history_store.artifact(body["id"], "md").exists())
        self.assertEqual(len(calls), 2)

    def test_cancel_after_completion_conflicts_and_unknown_job_is_not_found(self):
        job_id = "6" * 32
        web_api.history_store.create({"id": job_id, "sourceType": "file", "source": "C:/source.wav",
                                      "profile": "quality", "device": "cpu", "cpuThreads": 2,
                                      "restorePunctuation": False})
        web_api.history_store.update(job_id, status="completed")
        self.assertEqual(self.client.post(f"/api/simple/jobs/{job_id}/cancel").status_code, 409)
        self.assertEqual(self.client.post(f"/api/simple/jobs/{'5' * 32}/cancel").status_code, 404)

    def test_artifact_write_failure_does_not_commit_completed_status(self):
        job_id = "4" * 32
        output_dir = web_api.history_store.job_dir(job_id) / "output"
        web_api.history_store.create({"id": job_id, "sourceType": "file", "source": "C:/source.wav",
                                      "profile": "quality", "device": "cpu", "cpuThreads": 2,
                                      "restorePunctuation": False})
        settings = {"_simple": True, "_outputDir": str(output_dir), "language": "auto", "model": "small",
                    "device": "cpu", "nproc": 2, "outputFormat": "txt", "saveAudio": False,
                    "useTimestamps": False, "restorePunctuation": False}
        with web_api.jobs_lock:
            web_api.jobs[job_id] = JobState(id=job_id, settings=settings)
            web_api.job_process_controllers[job_id] = _FakeController()
            web_api.simple_worker_states[job_id] = "not_started"
        result = {"type": "result", "stage": "result", "result": {
            "text": "temporary result", "output_path": str(output_dir / "captions.txt"), "output_format": "txt"}}
        processes = iter([_FakeProcess(), _FakeProcess([web_api.JSON_EVENT_PREFIX + json.dumps(result) + "\n"])] )
        with (patch.object(web_api.subprocess, "Popen", side_effect=lambda *_args, **_kwargs: next(processes)),
              patch.object(web_api.history_store, "write_text", side_effect=OSError(28, "disk full"))):
            web_api.get_simple_queue().set_state(False)
            web_api.run_job(job_id, "C:/source.wav")
        self.assertEqual(web_api.history_store.get(job_id)["status"], "failed")
        self.assertEqual(web_api.history_store.get(job_id)["errorCode"], "storage_unavailable")
        self.assertEqual(web_api.get_simple_queue().state()["pauseReason"]["code"], "storage_unavailable")
        self.assertFalse(web_api.history_store.artifact(job_id, "txt").exists())
        self.assertFalse(web_api.history_store.artifact(job_id, "md").exists())

    def test_cancel_racing_result_publication_wins_without_writing_final_files(self):
        body = self._create_controlled_job()
        output_path = str(web_api.history_store.job_dir(body["id"]) / "output" / "captions.txt")
        prep = _FakeProcess()
        result = {"type": "result", "stage": "result", "result": {
            "text": "must not publish", "output_path": output_path, "output_format": "txt"}}
        asr = _FakeProcess([web_api.JSON_EVENT_PREFIX + json.dumps(result) + "\n"])
        processes = iter([prep, asr])
        original_finalize = web_api._finalize_simple_job

        def cancel_before_finalize(job_id, status, **kwargs):
            if status == "completed":
                response = self.client.post(f"/api/simple/jobs/{job_id}/cancel")
                self.assertEqual(response.status_code, 202)
            return original_finalize(job_id, status, **kwargs)

        with (patch.object(web_api.subprocess, "Popen", side_effect=lambda *_args, **_kwargs: next(processes)),
              patch.object(web_api, "_finalize_simple_job", side_effect=cancel_before_finalize)):
            web_api.run_job(body["id"], "https://example.org/watch/123")
        record = web_api.history_store.get(body["id"])
        self.assertEqual(record["status"], "cancelled")
        self.assertIsNone(web_api.history_store.read_text(body["id"]))
        self.assertFalse(web_api.history_store.artifact(body["id"], "md").exists())

    def test_cancel_stop_failure_remains_cancelling_and_retryable(self):
        body = self._create_controlled_job()
        controller = web_api.job_process_controllers[body["id"]]
        controller.cancel = lambda: (_ for _ in ()).throw(RuntimeError("stop failed"))
        response = self.client.post(f"/api/simple/jobs/{body['id']}/cancel")
        self.assertEqual(response.status_code, 503)
        self.assertIn("Повторите отмену", response.json()["detail"])
        self.assertEqual(web_api.history_store.get(body["id"])["status"], "cancelling")
        self.assertEqual(self.client.delete(f"/api/simple/history/{body['id']}").status_code, 409)
        with patch.object(web_api.simple_profiles, "resolve", return_value=("small", "cpu")):
            self.assertEqual(self.client.post("/api/simple/jobs", json=self.payload()).status_code, 409)
        controller.cancel = lambda: setattr(controller, "cancelled", True)
        self.assertEqual(self.client.post(f"/api/simple/jobs/{body['id']}/cancel").status_code, 202)
        web_api.run_job(body["id"], "https://example.org/watch/123")
        self.assertEqual(web_api.history_store.get(body["id"])["status"], "cancelled")

    def test_retry_after_worker_exits_on_stop_failure_finalizes_cancelled(self):
        body = self._create_controlled_job()
        controller = web_api.job_process_controllers[body["id"]]
        original_cancel = controller.cancel
        controller.cancel = lambda: (_ for _ in ()).throw(RuntimeError("stop failed"))
        controller.stop_owned = lambda: (_ for _ in ()).throw(RuntimeError("stop failed"))
        self.assertEqual(self.client.post(f"/api/simple/jobs/{body['id']}/cancel").status_code, 503)
        web_api.run_job(body["id"], "https://example.org/watch/123")
        self.assertEqual(web_api.history_store.get(body["id"])["status"], "cancelling")
        self.assertEqual(web_api.simple_worker_states[body["id"]], "finished")
        with patch.object(web_api.simple_profiles, "resolve", return_value=("small", "cpu")):
            self.assertEqual(self.client.post("/api/simple/jobs", json=self.payload()).status_code, 409)
        controller.cancel = original_cancel
        controller.stop_owned = lambda: None
        response = self.client.post(f"/api/simple/jobs/{body['id']}/cancel")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "cancelled")
        self.assertIsInstance(response.json()["finishedAt"], str)
        self.assertGreaterEqual(response.json()["durationSeconds"], 0)

    def test_simple_public_payload_hides_old_script_directions(self):
        job_id = "9" * 32
        web_api.history_store.create({"id": job_id, "sourceType": "file", "source": "C:/source.wav",
                                      "profile": "quality", "device": "cpu", "cpuThreads": 2,
                                      "restorePunctuation": False})
        web_api.history_store.update(job_id, status="failed", error="RunStart.ps1 failed")
        self.assertNotIn("start.ps1", self.client.get(f"/api/simple/jobs/{job_id}").text.lower())
        options = self.client.get("/api/simple/options").text.lower()
        self.assertNotIn("start.ps1", options)
        self.assertNotIn("runstart.ps1", options)

    def test_cache_preflight_checks_selected_backend_and_optional_punctuation(self):
        settings = {"model": "large-v3", "device": "cpu", "restorePunctuation": False}
        missing_model = "Не удалось загрузить модель. Проверьте подключение к интернету и повторите попытку."
        with patch.object(web_api, "get_model_status", return_value={"status": "not_downloaded"}):
            self.assertEqual(web_api.verify_simple_model_cache(settings), missing_model)
        with patch.object(web_api, "get_model_status", return_value={"status": "downloaded"}):
            self.assertIsNone(web_api.verify_simple_model_cache(settings))
        settings["restorePunctuation"] = True
        with (patch.object(web_api, "get_model_status", return_value={"status": "downloaded"}),
              patch("services.prepare_models._punctuation_cached", return_value=False)):
            self.assertEqual(web_api.verify_simple_model_cache(settings), "Отключите пунктуацию и повторите попытку.")

    def test_simple_failure_errors_are_friendly_while_legacy_error_stays_technical(self):
        ffmpeg_error = "FFmpeg audio extraction failed: --enable-libaom --enable-libx264 --enable-cuda"
        oom_error = "RuntimeError: CUDA out of memory. CUDNN_STATUS_ALLOC_FAILED"
        self.assertEqual(
            web_api.concise_simple_error(ffmpeg_error),
            "Не удалось прочитать звуковую дорожку. Выберите доступный аудио- или видеофайл.",
        )
        self.assertEqual(
            web_api.concise_simple_error(oom_error),
            "Недостаточно памяти видеокарты. Выберите «Экономный» или процессор и повторите запуск.",
        )
        self.assertNotIn("--enable", web_api.concise_simple_error("Traceback: backend error --enable-libaom"))

        job_id = "f" * 32
        web_api.history_store.create({"id": job_id, "sourceType": "file", "source": "C:/bad.wav",
                                      "profile": "quality", "device": "gpu", "cpuThreads": 2,
                                      "restorePunctuation": False})
        web_api.history_store.update(job_id, status="failed", stage="result", error=ffmpeg_error)
        simple = self.client.get(f"/api/simple/jobs/{job_id}").json()
        self.assertEqual(simple["error"], "Не удалось прочитать звуковую дорожку. Выберите доступный аудио- или видеофайл.")
        self.assertEqual(simple["files"], [])

        legacy = JobState(id="legacy", settings={}, error=oom_error)
        self.assertEqual(web_api.serialize_job(legacy)["error"], oom_error)

    def test_edit_endpoint_updates_both_files_and_rejects_stale_revision(self):
        job_id = "e" * 32
        web_api.history_store.create({"id": job_id, "sourceType": "url", "source": "https://example.org/a",
                                      "profile": "quality", "device": "cpu", "cpuThreads": 1,
                                      "restorePunctuation": False})
        web_api.history_store.update(job_id, status="completed", stage="result")
        web_api.history_store.write_text(job_id, "before", 0)
        response = self.client.put(f"/api/simple/jobs/{job_id}/text", json={"text": "after", "baseRevision": 0})
        self.assertEqual(response.json(), {"text": "after", "revision": 1})
        for ext in ("txt", "md"):
            self.assertEqual(web_api.history_store.artifact(job_id, ext).read_text(encoding="utf-8"), "after")
        stale = self.client.put(f"/api/simple/jobs/{job_id}/text", json={"text": "stale", "baseRevision": 0})
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(self.client.get(f"/api/simple/jobs/{job_id}").json()["text"], "after")


if __name__ == "__main__":
    unittest.main()
