import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from services import cuda_runtime, prepare_models


class ModelPreparationTests(unittest.TestCase):
    def test_setup_only_prepares_aligner_weights_and_cpu_runtime_for_each_asr_profile(self):
        with patch.object(prepare_models, "get_simple_gpu_details", return_value={"backend": None}), \
                patch.object(prepare_models, "enumerate_vulkan_gpus", return_value=[]), \
                patch.object(prepare_models, "_windows_gpu_inventory", return_value=[]), \
                patch.object(prepare_models, "get_model_status", return_value={"status": "downloaded"}), \
                patch.object(prepare_models, "_punctuation_cached", return_value=True), \
                patch.object(prepare_models, "asr_gpu_candidate", side_effect=lambda model: {
                    "backend": "directml" if model == "parakeet-tdt-0.6b-v3" else "cuda",
                    "canPrepare": True,
                }), \
                patch("services.asr_resources.ensure_asr_resources") as ensure:
            result = prepare_models.prepare(setup_only=True, skip_postprocess=True, progress=lambda _message: None)

        self.assertEqual(result, 0)
        selected = [(call.args[0], call.args[1]) for call in ensure.call_args_list]
        self.assertEqual(selected, [
            ("parakeet-tdt-0.6b-v3", "cpu"), ("parakeet-tdt-0.6b-v3", "directml"),
            ("qwen3-asr-1.7b", "cpu"), ("qwen3-asr-1.7b", "cuda"),
        ])
        self.assertTrue(all(call.kwargs["diarization"] is True for call in ensure.call_args_list))

    def test_setup_only_check_validates_optional_alignment_without_downloading(self):
        with patch.object(prepare_models, "get_simple_gpu_details", return_value={"backend": None}), \
                patch.object(prepare_models, "enumerate_vulkan_gpus", return_value=[]), \
                patch.object(prepare_models, "_windows_gpu_inventory", return_value=[]), \
                patch.object(prepare_models, "get_model_status", return_value={"status": "downloaded"}), \
                patch.object(prepare_models, "_punctuation_cached", return_value=True), \
                patch.object(prepare_models, "asr_gpu_candidate", return_value={"backend": None, "canPrepare": False}), \
                patch("services.asr_resources.asr_resource_status", return_value={"ready": True, "missing": []}) as status, \
                patch("services.asr_resources.ensure_asr_resources") as ensure:
            result = prepare_models.prepare(check_only=True, setup_only=True, skip_postprocess=True,
                                           progress=lambda _message: None)

        self.assertEqual(result, 0)
        self.assertEqual(status.call_count, 2)
        self.assertTrue(all(call.kwargs["diarization"] is True for call in status.call_args_list))
        ensure.assert_not_called()

    def test_regular_lazy_check_does_not_probe_optional_asr_or_alignment_resources(self):
        with patch.object(prepare_models, "get_simple_gpu_details", return_value={"backend": None}), \
                patch.object(prepare_models, "enumerate_vulkan_gpus", return_value=[]), \
                patch.object(prepare_models, "_windows_gpu_inventory", return_value=[]), \
                patch.object(prepare_models, "get_model_status", return_value={"status": "downloaded"}) as model_status, \
                patch.object(prepare_models, "_punctuation_cached", return_value=True), \
                patch("services.asr_resources.asr_resource_status") as status, \
                patch("services.asr_resources.ensure_asr_resources") as ensure:
            result = prepare_models.prepare(check_only=True, skip_postprocess=True,
                                           progress=lambda _message: None)

        self.assertEqual(result, 0)
        status.assert_not_called()
        ensure.assert_not_called()
        self.assertEqual([call.args[0] for call in model_status.call_args_list], ["small"])

    def test_regular_prepare_downloads_only_small_and_leaves_new_profiles_lazy(self):
        downloaded = []
        with patch.object(prepare_models, "get_simple_gpu_details", return_value={"backend": None}), \
                patch.object(prepare_models, "get_model_status", return_value={"status": "not_downloaded"}) as model_status, \
                patch.object(prepare_models, "_report_size", return_value="offline"), \
                patch.object(prepare_models, "download_registered_model", side_effect=lambda model, _progress: downloaded.append(model)), \
                patch("services.asr_resources.asr_resource_status") as status, \
                patch("services.asr_resources.ensure_asr_resources") as ensure:
            result = prepare_models.prepare(skip_postprocess=True, progress=lambda _message: None)

        self.assertEqual(result, 0)
        self.assertEqual(downloaded, ["small"])
        self.assertEqual([call.args[0] for call in model_status.call_args_list], ["small"])
        status.assert_not_called()
        ensure.assert_not_called()

    def test_check_only_reports_missing_models_without_downloading(self):
        with patch.object(prepare_models, "get_simple_gpu_details", return_value={"backend": None}), \
                patch.object(prepare_models, "get_model_status", return_value={"status": "not_downloaded"}), \
                patch.object(prepare_models, "download_registered_model") as download, \
                patch.object(prepare_models, "resolve_model_path") as cpp_download:
            output = []
            result = prepare_models.prepare(check_only=True, skip_postprocess=True, progress=output.append)
        self.assertEqual(result, 1)
        self.assertTrue(any("faster-whisper small" in line and "offline mode" in line for line in output))
        download.assert_not_called()
        cpp_download.assert_not_called()

    def test_fully_cached_cpu_models_pass_without_downloading(self):
        with patch.object(prepare_models, "get_simple_gpu_details", return_value={"backend": None}), \
                patch.object(prepare_models, "enumerate_vulkan_gpus", return_value=[]), \
                patch.object(prepare_models, "_windows_gpu_inventory", return_value=[]), \
                patch.object(prepare_models, "get_model_status", return_value={"status": "downloaded"}), \
                patch.object(prepare_models, "download_registered_model") as download:
            result = prepare_models.prepare(check_only=True, skip_postprocess=True, progress=lambda _message: None)
        self.assertEqual(result, 0)
        download.assert_not_called()

    def test_amd_models_are_included_even_when_nvidia_is_the_selected_gpu(self):
        with patch.object(prepare_models, "get_simple_gpu_details", return_value={"backend": "cuda", "name": "NVIDIA RTX"}), \
                patch.object(prepare_models, "enumerate_vulkan_gpus", return_value=[{"vendorId": 0x1002, "name": "AMD Radeon"}]), \
                patch.object(prepare_models, "_windows_gpu_inventory", return_value=[]), \
                patch.object(prepare_models, "get_model_status", return_value={"status": "downloaded"}), \
                patch.object(prepare_models, "_vulkan_cached", return_value=False), \
                patch.object(prepare_models, "_report_size", return_value="offline"), \
                patch.object(prepare_models, "resolve_model_path") as cpp_download:
            output = []
            result = prepare_models.prepare(check_only=True, skip_postprocess=True, progress=output.append)
        self.assertEqual(result, 1)
        self.assertTrue(any("whisper.cpp small" in line for line in output))
        cpp_download.assert_not_called()

    def test_empty_model_named_folder_does_not_count_as_punctuation_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "models--kontur-ai--sbert_punc_case_ru"
            empty.mkdir()
            with patch.object(prepare_models, "HF_HOME", Path(directory)), \
                    patch("huggingface_hub.try_to_load_from_cache", return_value=str(empty)), \
                    patch.object(prepare_models.importlib.util, "find_spec", return_value=object()):
                self.assertFalse(prepare_models._punctuation_cached())

    def test_cuda_archive_download_checks_exact_size_and_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = b"small archive fixture"
            spec = {"name": "fake cuda", "url": "https://example.test/runtime.zip",
                    "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}

            class Response(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    self.close()

            with patch.object(cuda_runtime.urllib.request, "urlopen", return_value=Response(payload)):
                archive = cuda_runtime._download_archive(spec, Path(directory))
            self.assertEqual(archive.read_bytes(), payload)

    def test_cuda_archive_rejects_hash_mismatch_and_removes_partial_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = b"wrong content"
            spec = {"name": "fake cuda", "url": "https://example.test/runtime.zip",
                    "sha256": "0" * 64, "size": len(payload)}

            class Response(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    self.close()

            with patch.object(cuda_runtime.urllib.request, "urlopen", return_value=Response(payload)):
                with self.assertRaisesRegex(RuntimeError, "Checksum or size validation failed"):
                    cuda_runtime._download_archive(spec, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cuda_archive_extraction_rejects_paths_outside_target(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.txt", "unsafe")
            with self.assertRaisesRegex(RuntimeError, "Unsafe path"):
                cuda_runtime._safe_extract(archive, Path(directory) / "extract")

    def test_invalid_new_cuda_runtime_keeps_existing_working_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cuda"
            bin_dir = root / "bin"
            bin_dir.mkdir(parents=True)
            for name in cuda_runtime.REQUIRED_DLLS:
                (bin_dir / name).write_bytes(b"working runtime")
            archive = Path(directory) / "cached.zip"
            archive.write_bytes(b"verified archive placeholder")
            with patch.object(cuda_runtime, "_windows_gpu_inventory", return_value=[{"vendorId": 0x10DE}]), \
                    patch.object(cuda_runtime, "ARCHIVES", ({"name": "fixture", "url": "fixture", "sha256": "x", "size": 1},)), \
                    patch.object(cuda_runtime, "_download_archive", return_value=archive), \
                    patch.object(cuda_runtime, "_safe_extract"):
                with self.assertRaisesRegex(RuntimeError, "did not contain the required"):
                    cuda_runtime.ensure_local_cuda_runtime(force_download=True, root=root)
            self.assertTrue(cuda_runtime.local_cuda_ready(root))
            self.assertFalse((Path(directory) / "cuda.previous").exists())


if __name__ == "__main__":
    unittest.main()
