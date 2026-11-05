import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import web_api
from services import runtime_devices, whisper_cpp
from services.model_paths import HF_HUB_CACHE, MODEL_STORAGE_ROOT, MODEL_TEMP_DIR
from services.model_management import get_model_status


class WhisperCppParsingTests(unittest.TestCase):
    def test_parses_real_whisper_cpp_offsets_schema(self):
        segments, stats = whisper_cpp.parse_whisper_cpp_json({
            "transcription": [
                {"offsets": {"from": 250, "to": 1475}, "timestamps": {"from": "00:00:00,250", "to": "00:00:01,475"}, "text": " hello"},
            ],
            "result": {"language": "ru"},
        })
        self.assertEqual((segments[0].start, segments[0].end, segments[0].text), (0.25, 1.475, "hello"))
        self.assertEqual(stats.language, "ru")
        self.assertIsNone(stats.language_probability)

    def test_parses_captured_whisper_cpp_fixture(self):
        fixture = Path(__file__).resolve().parents[1] / ".runtime" / "test-fixtures" / "direct-vulkan.json"
        if not fixture.exists():
            self.skipTest("Vulkan runtime fixture is not present")
        segments, _stats = whisper_cpp.parse_whisper_cpp_json(json.loads(fixture.read_text(encoding="utf-8")))
        self.assertTrue(segments)
        self.assertGreaterEqual(segments[0].start, 0)
        self.assertGreater(segments[0].end, segments[0].start)

    def test_parses_timestamp_strings_when_offsets_are_missing(self):
        segments, _stats = whisper_cpp.parse_whisper_cpp_json({
            "transcription": [{"timestamps": {"from": "00:01:02,500", "to": "00:01:03,000"}, "text": "text"}],
        })
        self.assertEqual((segments[0].start, segments[0].end), (62.5, 63.0))

    def test_rejects_malformed_whisper_cpp_json(self):
        with self.assertRaisesRegex(RuntimeError, "missing transcription array"):
            whisper_cpp.parse_whisper_cpp_json({})
        with self.assertRaisesRegex(RuntimeError, "malformed segment"):
            whisper_cpp.parse_whisper_cpp_json({"transcription": [{"text": "bad"}]})

    def test_whisper_cpp_adapter_converts_any_input_and_reads_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "model.bin"
            model_file.touch()
            model = whisper_cpp.WhisperCppModel.__new__(whisper_cpp.WhisperCppModel)
            model.executable = "whisper-cli.exe"
            model.model_path = model_file
            model.threads = 6
            model.model_name = "tiny"

            def fake_run(command, **_kwargs):
                if command[0] == "ffmpeg.exe":
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                output_json = Path(command[command.index("-of") + 1] + ".json")
                output_json.write_text(json.dumps({
                    "transcription": [{"offsets": {"from": 0, "to": 900}, "text": " hello"}],
                    "result": {"language": "en"},
                }), encoding="utf-8")
                self.assertIn("-m", command)
                self.assertIn("-f", command)
                return SimpleNamespace(returncode=0, stdout="", stderr="whisper_backend_init_gpu: using Vulkan0 backend")

            with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                    patch.object(whisper_cpp.subprocess, "run", side_effect=fake_run):
                segments, stats = model.transcribe("movie.webm", language=None)
            self.assertEqual(segments[0].text, "hello")
            self.assertEqual(stats.language, "en")

    def test_ffmpeg_failures_are_reported(self):
        model = whisper_cpp.WhisperCppModel.__new__(whisper_cpp.WhisperCppModel)
        model.executable, model.model_path, model.threads = "whisper-cli", Path("model.bin"), 1
        with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                patch.object(whisper_cpp.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="bad input")):
            with self.assertRaisesRegex(RuntimeError, "FFmpeg could not prepare audio: bad input"):
                model.transcribe("bad.webm")


class RuntimeDeviceTests(unittest.TestCase):
    def test_runtime_discovery_requires_executable_and_non_cpu_adapter(self):
        with patch.object(runtime_devices, "find_whisper_cpp_executable", return_value="cli"), \
                patch.object(runtime_devices, "enumerate_vulkan_gpus", return_value=[{"name": "Radeon 780M"}]), \
                patch.object(runtime_devices, "cuda_device_count", return_value=0):
            details = {entry["id"]: entry for entry in runtime_devices.get_device_details()}
        self.assertTrue(details["vulkan"]["available"])
        self.assertEqual(details["vulkan"]["name"], "Radeon 780M")
        self.assertFalse(details["cuda"]["available"])

    def test_cpu_software_adapter_does_not_count_as_vulkan_gpu(self):
        with patch.object(runtime_devices, "find_whisper_cpp_executable", return_value="cli"), \
                patch.object(runtime_devices, "enumerate_vulkan_gpus", return_value=[]), \
                patch.object(runtime_devices, "cuda_device_count", return_value=0):
            details = {entry["id"]: entry for entry in runtime_devices.get_device_details()}
        self.assertFalse(details["vulkan"]["available"])
        self.assertIn("No non-CPU Vulkan GPU", details["vulkan"]["reason"])

    def test_windows_cuda_is_unavailable_when_required_runtime_dlls_are_missing(self):
        with patch.object(runtime_devices, "find_whisper_cpp_executable", return_value="cli"), \
                patch.object(runtime_devices, "enumerate_vulkan_gpus", return_value=[{"name": "Radeon 780M"}]), \
                patch.object(runtime_devices, "cuda_device_count", return_value=1), \
                patch.object(runtime_devices, "_is_windows", return_value=True), \
                patch.object(runtime_devices.ctypes, "WinDLL", create=True, side_effect=OSError("not found")):
            details = {entry["id"]: entry for entry in runtime_devices.get_device_details()}
        self.assertFalse(details["cuda"]["available"])
        self.assertIn("CUDA libraries are missing", details["cuda"]["reason"])
        self.assertIn("Use Vulkan or install CUDA 12 and cuDNN 9", details["cuda"]["reason"])
        self.assertTrue(details["vulkan"]["available"])

    def test_cpp_model_rejects_distil(self):
        with self.assertRaisesRegex(ValueError, "not supported by the Vulkan backend"):
            whisper_cpp.model_filename("distil-large-v2")


class ModelStorageTests(unittest.TestCase):
    def test_model_caches_default_inside_project_and_override_external_hf_home(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop("VT_MODEL_ROOT", None)
        env["HF_HOME"] = r"C:\Users\Someone\.cache\huggingface"
        result = subprocess.run(
            [sys.executable, "-c", "import json,os; from services.model_paths import MODEL_STORAGE_ROOT,HF_HUB_CACHE; print(json.dumps([str(MODEL_STORAGE_ROOT),str(HF_HUB_CACHE),os.environ['HF_HOME']]))"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        storage_root, hub_cache, hf_home = json.loads(result.stdout.strip())
        expected_root = root / ".runtime" / "models"
        self.assertEqual(Path(storage_root), expected_root.resolve())
        self.assertEqual(Path(hub_cache), expected_root.resolve() / "huggingface" / "hub")
        self.assertEqual(Path(hf_home), expected_root.resolve() / "huggingface")

    def test_default_model_status_scan_uses_project_cache(self):
        with patch("services.model_management.scan_cache_dir", return_value=None) as scan:
            status = get_model_status("small")
        self.assertEqual(status["status"], "not_downloaded")
        scan.assert_called_once_with(cache_dir=HF_HUB_CACHE)

    def test_legacy_whisper_cli_loads_weights_under_project_model_root(self):
        root = Path(__file__).resolve().parents[1]
        module_path = root / "whisper-og.py"
        whisper_stub = types.ModuleType("whisper")
        whisper_stub.load_model = Mock(return_value=object())
        whisper_stub.available_models = Mock(return_value=["tiny"])
        download_stub = types.ModuleType("utils.download_best")
        download_stub.Download = object
        audio_stub = types.ModuleType("utils.get_audio")
        audio_stub.AudioProcess = object
        spec = importlib.util.spec_from_file_location("legacy_whisper_test_module", module_path)
        module = importlib.util.module_from_spec(spec)
        old_omp_threads = os.environ.get("OMP_NUM_THREADS")
        try:
            with patch.dict(sys.modules, {
                "whisper": whisper_stub,
                "utils.download_best": download_stub,
                "utils.get_audio": audio_stub,
            }):
                spec.loader.exec_module(module)
                with tempfile.TemporaryDirectory(dir=MODEL_TEMP_DIR) as temp_dir:
                    module.temp_audio_filepath = str(Path(temp_dir) / "intermediate.mp3")
                    result = module.initialize(SimpleNamespace(model_size="tiny", nproc=3))
        finally:
            if old_omp_threads is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = old_omp_threads

        self.assertIsNotNone(result)
        whisper_stub.load_model.assert_called_once_with(
            "tiny", device="cpu", download_root=str(MODEL_STORAGE_ROOT / "openai-whisper")
        )


class RunJobResourceTests(unittest.TestCase):
    def test_run_job_releases_runtime_slot_when_command_build_fails(self):
        job_id = "command-error"
        web_api.jobs[job_id] = web_api.JobState(id=job_id, settings={})
        with tempfile.TemporaryDirectory() as temp_dir, \
                patch.object(web_api, "RUNS_DIR", Path(temp_dir)), \
                patch.object(web_api, "build_command", side_effect=RuntimeError("command failure")):
            web_api.run_job(job_id, "input.wav")
        self.assertEqual(web_api.get_job(job_id).status, "failed")
        self.assertTrue(web_api.transcription_semaphore.acquire(blocking=False))
        web_api.transcription_semaphore.release()
        web_api.jobs.pop(job_id, None)


class TranscriptionApiValidationTests(unittest.TestCase):
    def setUp(self):
        web_api.jobs.clear()

    def test_options_expose_available_device_contract(self):
        with patch("web_api.get_device_details", return_value=[{"id": "auto", "available": True}]), \
                patch("web_api.importlib.util.find_spec", return_value=object()):
            payload = TestClient(web_api.app).get("/api/options").json()
        self.assertEqual(payload["defaultDevice"], "auto")
        self.assertEqual(payload["devices"], ["auto", "cpu", "vulkan", "cuda"])
        self.assertIn("large-v3-turbo", payload["vulkanModels"])
        self.assertIn("punctuationAvailable", payload)
        self.assertEqual(payload["modelsDirectory"], str(MODEL_STORAGE_ROOT))

    def test_rejects_non_http_url_and_invalid_device_before_queuing(self):
        client = TestClient(web_api.app)
        base = {"sourceType": "url", "url": "C:/private/video.mp4", "model": "small", "language": "ru"}
        self.assertEqual(client.post("/api/transcriptions", data=base).status_code, 400)
        base["url"] = "https://example.test/video"
        base["device"] = "bogus"
        self.assertEqual(client.post("/api/transcriptions", data=base).status_code, 400)
        self.assertEqual(web_api.jobs, {})

    def test_unknown_api_route_is_not_served_as_spa_html(self):
        response = TestClient(web_api.app).get("/api/no-such-route")
        self.assertEqual(response.status_code, 404)
        self.assertIsInstance(response.json(), dict)

    def test_health_has_stable_app_id(self):
        payload = TestClient(web_api.app).get("/api/health").json()
        self.assertEqual(payload, {"status": "ok", "appId": "video-transcribe"})


if __name__ == "__main__":
    unittest.main()
