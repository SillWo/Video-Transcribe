import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from services import whisper_cpp
from services.task_errors import TaskFailure


class WhisperCppTypedFailureTests(unittest.TestCase):
    def make_model(self):
        model = whisper_cpp.WhisperCppModel.__new__(whisper_cpp.WhisperCppModel)
        model.executable = "whisper-cli"
        model.model_path = Path("model.bin")
        model.threads = 1
        return model

    def run_failure(self, runtime_detail):
        model = self.make_model()

        def fake_run(command, **_kwargs):
            if command[0] == "ffmpeg.exe":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr=runtime_detail)

        with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                patch.object(whisper_cpp.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(TaskFailure) as raised:
                model.transcribe("damaged.wav")
        return raised.exception

    def test_corrupt_input_decode_failure_is_source_invalid_even_with_vulkan_logs(self):
        failure = self.run_failure("ggml_vk_buffer initialized successfully; input decode failed")
        self.assertEqual(failure.code, "source_invalid")
        self.assertEqual(failure.stage, "source")

    def test_explicit_oom_code_takes_priority_over_decode_failure_text(self):
        failure = self.run_failure("input decode failed: CUDA_ERROR_OUT_OF_MEMORY")
        self.assertEqual(failure.code, "out_of_memory")
        self.assertEqual(failure.stage, "recognition")

    def test_native_cuda_oom_signature_is_typed_as_systemic_oom(self):
        failure = self.run_failure("CUDA_ERROR_OUT_OF_MEMORY while allocating decoder buffers")
        self.assertEqual(failure.code, "out_of_memory")
        self.assertEqual(failure.stage, "recognition")

    def test_ffmpeg_input_conversion_failure_is_source_invalid(self):
        model = self.make_model()
        with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                patch.object(whisper_cpp.subprocess, "run",
                             return_value=SimpleNamespace(returncode=1, stdout="", stderr="Invalid data found when processing input")):
            with self.assertRaises(TaskFailure) as raised:
                model.transcribe("damaged.wav")
        self.assertEqual(raised.exception.code, "source_invalid")

    def test_missing_vulkan_runtime_is_typed_device_unavailable(self):
        with patch.object(whisper_cpp, "find_whisper_cpp_executable", return_value=None):
            with self.assertRaises(TaskFailure) as raised:
                whisper_cpp.WhisperCppModel("small")
        self.assertEqual(raised.exception.code, "device_unavailable")


if __name__ == "__main__":
    unittest.main()
