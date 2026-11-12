import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import asr_resources as resources
from services.task_errors import TaskFailure


class AsrResourceTests(unittest.TestCase):
    def test_new_model_contract_and_canonical_paths(self):
        self.assertEqual(resources.NEW_MODELS, {"parakeet-tdt-0.6b-v3", "qwen3-asr-1.7b"})
        self.assertTrue(resources.is_new_asr("qwen3-asr-1.7b"))
        self.assertFalse(resources.is_new_asr("small"))
        self.assertEqual(resources.model_paths("qwen3-asr-1.7b")["model"],
                         resources.MODEL_ROOT / "Qwen3-ASR-1.7B-F16.gguf")
        self.assertEqual(resources.validation_audio_path(), resources.MODEL_ROOT / "validation" / "fleurs-en.wav")

    def test_hash_verification_rejects_same_size_corrupt_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            artifact.write_bytes(b"bad!")
            self.assertFalse(resources._valid(artifact, (4, hashlib.sha256(b"good").hexdigest())))

    def test_resumable_download_uses_range_and_atomic_replace(self):
        payload = b"model bytes"
        expected = (len(payload), hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.gguf"
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(payload[:5])
            response = io.BytesIO(payload[5:])
            response.status = 206
            response.headers = {"Content-Range": f"bytes 5-{len(payload) - 1}/{len(payload)}"}
            with patch.object(resources.urllib.request, "urlopen", return_value=response) as open_url:
                resources._download("repo", "rev", target.name, target, expected, None, 5, len(payload))
            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(partial.exists())
            self.assertEqual(open_url.call_args.args[0].get_header("Range"), "bytes=5-")

    def test_download_hash_mismatch_never_promotes_partial(self):
        payload = b"bad"
        response = io.BytesIO(payload)
        response.status = 200
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "file"
            with patch.object(resources.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(TaskFailure):
                    resources._download("repo", "rev", "file", target,
                                        (len(payload), hashlib.sha256(b"good").hexdigest()), None, 0, 3)
            self.assertFalse(target.exists())

    def test_runtime_reuses_only_exact_project_environment(self):
        candidate = resources._CANDIDATES[("native", "cpu")][0]
        with patch.object(resources, "_installed_exact", side_effect=lambda path, *_: Path(path) == resources._python_at(candidate)):
            self.assertEqual(resources.runtime_python("native", "cpu"), resources._python_at(candidate))

    def test_runtime_cache_invalidation_rechecks_alias_after_external_preparation(self):
        resources.invalidate_runtime_cache()
        candidate = resources._CANDIDATES[("parakeet", "directml")][0]
        canonical = resources._python_at(resources._canonical_env("parakeet", "directml"))
        alias = resources._python_at(candidate)
        prepared = False

        def run_probe(command, **_kwargs):
            valid = prepared and Path(command[0]) == alias
            state = ({"versions": resources.PACKAGE_VERSIONS[("parakeet", "directml")],
                      "providers": ["CPUExecutionProvider", "DmlExecutionProvider"]}
                     if valid else {})
            return type("ProbeResult", (), {"returncode": 0, "stdout": json.dumps(state)})()

        with patch.object(resources.subprocess, "run", side_effect=run_probe):
            self.assertEqual(resources.runtime_python("parakeet", "directml"), canonical)
            self.assertTrue(any(not valid for valid in resources._ENV_CACHE.values()))
            self.assertTrue(all(call.kwargs["stdin"] is subprocess.DEVNULL
                                for call in resources.subprocess.run.call_args_list))

            # Preparation ran in a child process; it changed no file fingerprint
            # used by _ENV_CACHE, but made the existing alias usable.
            prepared = True
            resources.invalidate_runtime_cache()
            self.assertEqual(resources.runtime_python("parakeet", "directml"), alias)

    def test_runtime_environment_creation_and_pip_install_use_noninteractive_stdin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "runtime" / "native"
            python = canonical / "Scripts" / "python.exe"
            requirements = root / "requirements.txt"
            requirements.write_text("package==1.0", encoding="utf-8")
            result = type("RunResult", (), {"returncode": 0})()

            with (patch.object(resources, "runtime_python", return_value=python),
                  patch.object(resources, "_canonical_env", return_value=canonical),
                  patch.object(resources, "_inside_project", return_value=True),
                  patch.object(resources, "_requirements_file", return_value=requirements),
                  patch.object(resources, "RUNTIME", root / "runtime"),
                  patch.object(resources, "_installed_exact", side_effect=[False, True]),
                  patch.object(resources.subprocess, "run", return_value=result) as run):
                resources._ensure_runtime("native", "cpu", None)

            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(call.kwargs["stdin"] is subprocess.DEVNULL for call in run.call_args_list))
            self.assertIn("venv", run.call_args_list[0].args[0])
            self.assertIn("pip", run.call_args_list[1].args[0])

    def test_cpu_aligner_accepts_normalized_distribution_version_only_for_cpu_torch(self):
        env = resources._CANDIDATES[("aligner", "cpu")][0]
        resources._ENV_CACHE.clear()
        self.assertTrue(resources._installed_exact(resources._python_at(env), "aligner", "cpu"))

    def test_parakeet_cpu_requires_onnxruntime_cpu_provider_but_accepts_directml_distribution(self):
        versions = {"onnx-asr": "0.12.0", "numpy": "2.4.4", "onnxruntime-directml": "1.24.4"}
        self.assertFalse(resources._runtime_versions_match("parakeet", "cpu", {"versions": versions, "providers": []}))
        self.assertTrue(resources._runtime_versions_match(
            "parakeet", "cpu", {"versions": versions, "providers": ["DmlExecutionProvider", "CPUExecutionProvider"]}))

    def test_cuda_aligner_requires_cuda_torch_and_available_device(self):
        versions = {"torch": "2.15.0", "transformers": "5.22.1", "safetensors": "0.8.0"}
        self.assertFalse(resources._runtime_versions_match(
            "aligner", "cuda", {"versions": versions, "torch": {"version": "2.15.0+cpu", "cuda": None, "available": False}}))
        self.assertTrue(resources._runtime_versions_match(
            "aligner", "cuda", {"versions": versions, "torch": {"version": "2.15.0+cu128", "cuda": "12.8", "available": True}}))

    def test_native_cuda_has_dedicated_runtime_and_requirements(self):
        with patch.object(resources, "_installed_exact", return_value=False):
            python = resources.runtime_python("native", "cuda")
        self.assertEqual(python, resources.ENV_ROOT / "native-cuda" / "Scripts" / "python.exe")
        self.assertIn("transcribe-cpp-native-cu12", (resources.APP_ROOT / "requirements-asr-native-cuda.txt").read_text())
        requirements = (resources.APP_ROOT / "requirements-asr-native-cuda.txt").read_text()
        self.assertIn("transcribe-cpp[cu12]==0.4.1", requirements)
        self.assertIn("transcribe-cpp-native==0.4.1", requirements)
        self.assertIn("transcribe_cpp_native_cu12-0.4.1-py3-none-win_amd64.whl#sha256=7af39bbbddd79039d9b2d400ab96c7248b1ad5073be2d4161dafddb649bfe7d1 ; sys_platform == 'win32'", requirements)
        self.assertIn("transcribe_cpp_native_cu12-0.4.1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=cfbb65ec833effdc18e424d1e3ca7312d33dcd0e92f42866de0d6b8ef52492f0 ; sys_platform == 'linux' and platform_machine == 'x86_64'", requirements)
        self.assertIn("nvidia-cuda-runtime-cu12==12.9.79", requirements)
        self.assertIn("nvidia-cuda-nvrtc-cu12==12.9.86", requirements)
        self.assertIn("nvidia-cublas-cu12==12.9.2.10", requirements)

    def test_native_cuda_requires_exact_nvidia_runtime_dependencies(self):
        versions = {"transcribe-cpp": "0.4.1", "transcribe-cpp-native-cu12": "0.4.1",
                    "transcribe-cpp-native": "0.4.1", "nvidia-cuda-runtime-cu12": "12.9.79",
                    "nvidia-cuda-nvrtc-cu12": "12.9.86", "nvidia-cublas-cu12": "12.9.2.10",
                    "numpy": "2.4.4"}
        self.assertTrue(resources._runtime_versions_match("native", "cuda", {"versions": versions}))
        missing_runtime = dict(versions)
        del missing_runtime["nvidia-cuda-runtime-cu12"]
        self.assertFalse(resources._runtime_versions_match("native", "cuda", {"versions": missing_runtime}))
        missing_nvrtc = dict(versions)
        del missing_nvrtc["nvidia-cuda-nvrtc-cu12"]
        self.assertFalse(resources._runtime_versions_match("native", "cuda", {"versions": missing_nvrtc}))
        wrong_cublas = {**versions, "nvidia-cublas-cu12": "12.9.2.9"}
        self.assertFalse(resources._runtime_versions_match("native", "cuda", {"versions": wrong_cublas}))

    def test_status_reports_missing_runtime_and_optional_alignment(self):
        with patch.object(resources, "_manifest_for", return_value=(0, [])), \
             patch.object(resources, "runtime_python", return_value=Path("/missing/python")), \
             patch.object(resources, "_installed_exact", return_value=False):
            status = resources.asr_resource_status("qwen3-asr-1.7b", "cpu")
        self.assertFalse(status["ready"])
        self.assertIn("native environment (cpu)", status["missing"])
        with patch.object(resources, "alignment_model_dir", return_value=Path("aligner")):
            total, groups = resources._manifest_for("qwen3-asr-1.7b", True)
        self.assertTrue(any(group[0] == "Qwen/Qwen3-ForcedAligner-0.6B-hf" for group in groups))
        self.assertGreater(total, 0)

    def test_diarization_status_requires_cpu_aligner_environment(self):
        with patch.object(resources, "_manifest_for", return_value=(0, [])), \
             patch.object(resources, "runtime_python", side_effect=lambda kind, _backend: Path(kind)), \
             patch.object(resources, "_installed_exact", side_effect=lambda _python, kind, _backend: kind != "aligner"):
            status = resources.asr_resource_status("qwen3-asr-1.7b", "cpu", diarization=True)
        self.assertFalse(status["ready"])
        self.assertIn("aligner environment (cpu)", status["missing"])

    def test_parakeet_diarization_manifest_does_not_include_aligner(self):
        _, groups = resources._manifest_for("parakeet-tdt-0.6b-v3", True)
        self.assertFalse(any(repo == "Qwen/Qwen3-ForcedAligner-0.6B-hf"
                             for repo, *_ in groups))

    def test_parakeet_diarization_preparation_never_prepares_aligner(self):
        with patch.object(resources, "_ensure_files", side_effect=lambda _repo, _rev, _manifest, _dir, _source, _progress, _done, _total: 1), \
             patch.object(resources, "_ensure_runtime") as ensure_runtime, \
             patch.object(resources, "_resource_status", return_value={"ready": True, "missing": []}):
            resources.ensure_asr_resources("parakeet-tdt-0.6b-v3", "cuda", diarization=True)
        self.assertNotIn(("aligner", "cpu"), [call.args[:2] for call in ensure_runtime.call_args_list])
        self.assertNotIn(("aligner", "cuda"), [call.args[:2] for call in ensure_runtime.call_args_list])

    def test_parakeet_diarization_status_is_ready_without_aligner(self):
        with patch.object(resources, "_valid", return_value=True), \
             patch.object(resources, "runtime_python", side_effect=lambda kind, backend: Path(f"{kind}-{backend}")), \
             patch.object(resources, "_installed_exact", return_value=True) as installed:
            status = resources.asr_resource_status("parakeet-tdt-0.6b-v3", "cpu", diarization=True)
        self.assertTrue(status["ready"], status["missing"])
        self.assertNotIn("aligner", [call.args[1] for call in installed.call_args_list])

    def test_parakeet_diarization_fingerprint_ignores_aligner_state(self):
        aligner_ready = False
        with patch.object(resources, "runtime_python", side_effect=lambda kind, backend: Path(f"{kind}-{backend}")), \
             patch.object(resources, "_environment_fingerprint", side_effect=lambda python, kind, backend: (str(python), kind, backend)), \
             patch.object(resources, "_installed_exact", side_effect=lambda _python, kind, _backend: True if kind != "aligner" else aligner_ready):
            before = resources._fingerprint("parakeet-tdt-0.6b-v3", "cpu", True)
            aligner_ready = True
            after = resources._fingerprint("parakeet-tdt-0.6b-v3", "cpu", True)
        self.assertEqual(before, after)

    def test_parakeet_prepares_native_language_id_runtime_with_matching_backend(self):
        with patch.object(resources, "_ensure_files", side_effect=lambda _repo, _rev, _manifest, _dir, _source, _progress, _done, _total: 1), \
             patch.object(resources, "_ensure_runtime") as ensure_runtime, \
             patch.object(resources, "_resource_status", return_value={"ready": True, "missing": []}):
            resources.ensure_asr_resources("parakeet-tdt-0.6b-v3", "directml")
        ensure_runtime.assert_any_call("parakeet", "directml", None)
        ensure_runtime.assert_any_call("native", "vulkan", None)

    def test_cuda_aligner_failure_is_logged_and_cpu_fallback_is_required(self):
        progress = []
        with patch.object(resources, "_ensure_files", side_effect=lambda _repo, _rev, _manifest, _dir, _source, _progress, _done, _total: 1), \
             patch.object(resources, "_ensure_runtime", side_effect=[None, RuntimeError("CUDA unavailable"), None]) as ensure_runtime, \
             patch.object(resources, "_resource_status", return_value={"ready": True, "missing": []}):
            resources.ensure_asr_resources("qwen3-asr-1.7b", "cuda", diarization=True, on_progress=progress.append)
        self.assertEqual(ensure_runtime.call_args_list[0].args[:2], ("native", "cuda"))
        self.assertEqual(ensure_runtime.call_args_list[1].args[:2], ("aligner", "cuda"))
        self.assertEqual(ensure_runtime.call_args_list[2].args[:2], ("aligner", "cpu"))
        self.assertTrue(any("CUDA unavailable" in event["message"] for event in progress))

    def test_directml_is_accepted_by_resource_preparation_cli(self):
        from services import resource_preparation
        called = []
        with patch.object(resource_preparation, "ensure_selected_resources",
                           side_effect=lambda settings, _progress: called.append(settings)):
            result = resource_preparation.main(["--model", "parakeet-tdt-0.6b-v3", "--device", "directml"])
        self.assertEqual(result, 0)
        self.assertEqual(called[0]["device"], "directml")

    def test_new_profile_routing_keeps_legacy_path_untouched(self):
        with patch.object(resources, "ensure_asr_resources") as ensure:
            from services import resource_preparation
            resource_preparation.ensure_selected_resources({"model": "parakeet-tdt-0.6b-v3", "device": "vulkan",
                                                             "restorePunctuation": False})
        ensure.assert_called_once()
        self.assertEqual(ensure.call_args.args[:2], ("parakeet-tdt-0.6b-v3", "directml"))
        with patch.object(resources, "ensure_asr_resources") as ensure:
            with patch("services.resource_preparation.get_model_status", return_value={"status": "downloaded"}):
                from services.resource_preparation import ensure_selected_resources
                ensure_selected_resources({"model": "small", "device": "cpu", "restorePunctuation": False})
        ensure.assert_not_called()

    def test_legacy_gpu_preparation_downloads_only_validation_fixture(self):
        from services import resource_preparation
        from services import prepare_models
        with patch.object(resources, "ensure_validation_audio") as ensure_fixture, \
             patch.object(resources, "ensure_asr_resources") as ensure_asr, \
             patch.object(prepare_models, "_vulkan_cached", return_value=True):
            resource_preparation.ensure_selected_resources(
                {"model": "small", "device": "vulkan", "restorePunctuation": False})
        ensure_fixture.assert_called_once()
        ensure_asr.assert_not_called()

    def test_setup_only_check_is_offline_and_skips_large_legacy_models(self):
        from services import prepare_models
        with patch.object(prepare_models, "get_simple_gpu_details", return_value={"backend": None}), \
             patch.object(prepare_models, "enumerate_vulkan_gpus", return_value=[]), \
             patch.object(prepare_models, "_windows_gpu_inventory", return_value=[]), \
             patch.object(prepare_models, "asr_gpu_candidate", return_value={"canPrepare": False, "backend": None}), \
             patch.object(prepare_models, "get_model_status", return_value={"status": "not_downloaded"}), \
             patch("services.asr_resources.asr_resource_status", return_value={"ready": False, "missing": ["model"]}) as status, \
             patch("services.asr_resources.ensure_asr_resources") as ensure:
            output = []
            result = prepare_models.prepare(check_only=True, skip_postprocess=True, setup_only=True, progress=output.append)
        self.assertEqual(result, 1)
        self.assertEqual(status.call_count, 2)
        ensure.assert_not_called()
        self.assertTrue(any("faster-whisper small" in line for line in output))
        self.assertFalse(any("large-v3" in line for line in output))


if __name__ == "__main__":
    unittest.main()
