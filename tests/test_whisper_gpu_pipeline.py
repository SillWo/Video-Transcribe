import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_whisper_gpu_module():
    module_path = Path(__file__).resolve().parents[1] / "whisper-gpu.py"
    spec = importlib.util.spec_from_file_location("whisper_gpu_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


whisper_gpu = load_whisper_gpu_module()


class WhisperGpuOutputTests(unittest.TestCase):
    def test_build_plain_text_without_timestamps_uses_combined_text(self):
        result = whisper_gpu.build_plain_text(
            [{"start": 0.0, "end": 1.0, "text": "segment one"}],
            "combined text",
            use_timestamps=False,
        )

        self.assertEqual(result, "combined text\n")

    def test_build_plain_text_with_timestamps_keeps_segment_structure(self):
        result = whisper_gpu.build_plain_text(
            [
                {"start": 1.0, "end": 2.0, "text": "first"},
                {"start": 3.0, "end": 4.0, "text": "second"},
            ],
            "ignored combined text",
            use_timestamps=True,
        )

        self.assertEqual(result, "[00:00:01,000] first\n[00:00:03,000] second\n")

    def test_build_output_payload_without_timestamps_keeps_text_only_segments(self):
        stats = SimpleNamespace(language="ru", language_probability=0.99)
        payload = whisper_gpu.build_output_payload(
            [
                {"start": 0.0, "end": 1.0, "text": "first"},
                {"start": 1.0, "end": 2.0, "text": "second"},
            ],
            stats,
            "combined text",
            use_timestamps=False,
        )

        self.assertEqual(payload["text"], "combined text")
        self.assertEqual(payload["segments"], [{"text": "first"}, {"text": "second"}])

    def test_build_output_payload_preserves_unavailable_language_probability_as_null(self):
        stats = SimpleNamespace(language="ru", language_probability=None)
        payload = whisper_gpu.build_output_payload([], stats, "recognized", use_timestamps=True)
        self.assertIsNone(payload["language_probability"])

    def test_write_output_txt_without_timestamps_writes_combined_text(self):
        args = SimpleNamespace(
            output_format="txt",
            output_dir="",
            language="ru",
            use_timestamps=False,
        )
        stats = SimpleNamespace(language="ru", language_probability=0.99)
        segments = [{"start": 0.0, "end": 1.0, "text": "segment text"}]

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(whisper_gpu.time, "strftime", return_value="20260417-120000"):
            args.output_dir = temp_dir
            output_path, rendered_result = whisper_gpu.write_output(
                args,
                segments,
                "combined polished text",
                stats,
                "input.wav",
            )

            self.assertEqual(rendered_result, "combined polished text\n")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "combined polished text\n")

    def test_write_output_json_with_timestamps_keeps_segments_and_combined_text(self):
        args = SimpleNamespace(
            output_format="json",
            output_dir="",
            language="ru",
            use_timestamps=True,
        )
        stats = SimpleNamespace(language="ru", language_probability=0.99)
        segments = [{"start": 0.0, "end": 1.0, "text": "segment text"}]

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(whisper_gpu.time, "strftime", return_value="20260417-120000"):
            args.output_dir = temp_dir
            output_path, rendered_result = whisper_gpu.write_output(
                args,
                segments,
                "combined polished text",
                stats,
                "input.wav",
            )

            self.assertIn('"text": "combined polished text"', rendered_result)
            self.assertIn('"start": 0.0', rendered_result)
            self.assertIn('"end": 1.0', rendered_result)
            self.assertEqual(output_path.read_text(encoding="utf-8"), rendered_result)

    def test_write_output_srt_keeps_segment_order_and_timestamps(self):
        args = SimpleNamespace(
            output_format="srt",
            output_dir="",
            language="ru",
            use_timestamps=True,
        )
        stats = SimpleNamespace(language="ru", language_probability=0.99)
        segments = [
            {"start": 0.0, "end": 1.0, "text": "first line"},
            {"start": 1.5, "end": 3.0, "text": "second line"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(whisper_gpu.time, "strftime", return_value="20260417-120000"):
            args.output_dir = temp_dir
            output_path, rendered_result = whisper_gpu.write_output(
                args,
                segments,
                "ignored",
                stats,
                "input.wav",
            )

            self.assertIn("1\n00:00:00,000 --> 00:00:01,000\nfirst line", rendered_result)
            self.assertIn("2\n00:00:01,500 --> 00:00:03,000\nsecond line", rendered_result)
            self.assertEqual(output_path.read_text(encoding="utf-8"), rendered_result)


class WhisperGpuTranscribeTests(unittest.TestCase):
    def build_args(self, *, language: str, restore_punctuation: bool, output_format: str = "json", use_timestamps: bool = True):
        return SimpleNamespace(
            web_json=False,
            language=language,
            beam_size=5,
            output_format=output_format,
            output_dir="",
            use_timestamps=use_timestamps,
            restore_punctuation=restore_punctuation,
        )

    def build_model(self, segments):
        stats = SimpleNamespace(language="ru", language_probability=0.97)

        class FakeModel:
            def transcribe(self, full_filepath, beam_size, language):
                return segments, stats

        return FakeModel(), stats

    def test_transcribe_applies_russian_postprocessing_and_preserves_segment_timestamps(self):
        segments = [
            SimpleNamespace(start=0.0, end=1.0, text="privet"),
            SimpleNamespace(start=1.0, end=2.0, text="mir"),
        ]
        model, _stats = self.build_model(segments)
        args = self.build_args(language="ru", restore_punctuation=True, output_format="json", use_timestamps=True)

        with tempfile.TemporaryDirectory() as temp_dir, \
            patch.object(whisper_gpu.time, "strftime", return_value="20260417-120000"), \
            patch.object(whisper_gpu, "restore_russian_segment_text", side_effect=["Privet.", "Mir."]) as restore_mock, \
            patch.object(whisper_gpu, "build_polished_russian_text_from_segments", return_value="Privet. Mir.") as polish_mock:
            args.output_dir = temp_dir
            result = whisper_gpu.transcribe(args, model, "input.wav", "media.wav")

        self.assertEqual(restore_mock.call_count, 2)
        polish_mock.assert_called_once_with(["Privet.", "Mir."])
        self.assertEqual(result["text"], "Privet. Mir.")
        self.assertEqual(result["segments"], [
            {"start": 0.0, "end": 1.0, "text": "Privet."},
            {"start": 1.0, "end": 2.0, "text": "Mir."},
        ])

    def test_transcribe_ignores_restore_flag_for_non_russian_language(self):
        segments = [SimpleNamespace(start=0.0, end=1.0, text="hello world")]
        model, _stats = self.build_model(segments)
        args = self.build_args(language="en", restore_punctuation=True, output_format="json", use_timestamps=False)

        with tempfile.TemporaryDirectory() as temp_dir, \
            patch.object(whisper_gpu.time, "strftime", return_value="20260417-120000"), \
            patch.object(whisper_gpu, "restore_russian_segment_text") as restore_mock, \
            patch.object(whisper_gpu, "build_polished_russian_text_from_segments") as polish_mock:
            args.output_dir = temp_dir
            result = whisper_gpu.transcribe(args, model, "input.wav", "media.wav")

        restore_mock.assert_not_called()
        polish_mock.assert_not_called()
        self.assertEqual(result["text"], "hello world")
        self.assertEqual(result["segments"], [{"text": "hello world"}])

    def test_transcribe_ignores_restore_flag_for_auto_language(self):
        segments = [SimpleNamespace(start=0.0, end=1.0, text="hello world")]
        model, _stats = self.build_model(segments)
        args = self.build_args(language="auto", restore_punctuation=True, output_format="txt", use_timestamps=False)

        with tempfile.TemporaryDirectory() as temp_dir, \
            patch.object(whisper_gpu.time, "strftime", return_value="20260417-120000"), \
            patch.object(whisper_gpu, "restore_russian_segment_text") as restore_mock:
            args.output_dir = temp_dir
            result = whisper_gpu.transcribe(args, model, "input.wav", "media.wav")

        restore_mock.assert_not_called()
        self.assertEqual(result["text"], "hello world")
        self.assertEqual(result["rendered_result"], "hello world\n")

    def test_auto_gpu_fallback_catches_deferred_iteration_failure_without_partial_results(self):
        class FailingGpu:
            def transcribe(self, *_args, **_kwargs):
                def partial_then_fail():
                    yield SimpleNamespace(start=0, end=1, text="partial")
                    raise RuntimeError("GPU allocation failed during lazy inference")
                return partial_then_fail(), SimpleNamespace(language="ru", language_probability=0.9)

        cpu_stats = SimpleNamespace(language="ru", language_probability=0.9)

        class CpuModel:
            def transcribe(self, *_args, **_kwargs):
                return [SimpleNamespace(start=0, end=1, text="cpu result")], cpu_stats

        args = SimpleNamespace(nproc=2)
        model = whisper_gpu.AutoFallbackModel(FailingGpu(), args, "faster-whisper CUDA")
        with patch.object(whisper_gpu, "_faster_model", return_value=CpuModel()) as cpu_factory:
            segments, stats = model.transcribe("input.wav", language="ru")

        cpu_factory.assert_called_once_with(args, "cpu")
        self.assertEqual([segment.text for segment in segments], ["cpu result"])
        self.assertEqual(stats.language, "ru")
        self.assertEqual(model.effective_device, "cpu")
        self.assertIn("CPU", model.effective_backend)

    def test_explicit_unavailable_vulkan_reports_actionable_error_without_cpu_fallback(self):
        args = SimpleNamespace(device="vulkan", model_size="small", precision="auto", nproc=2)
        devices = [
            {"id": "auto", "available": True},
            {"id": "cpu", "available": True},
            {"id": "vulkan", "available": False, "reason": "No non-CPU Vulkan GPU was detected."},
            {"id": "cuda", "available": False, "reason": "No CUDA device is available."},
        ]
        with patch.object(whisper_gpu, "get_device_details", return_value=devices), \
                patch.object(whisper_gpu, "_faster_model") as cpu_factory:
            with self.assertRaisesRegex(RuntimeError, "VULKAN is unavailable: No non-CPU Vulkan GPU"):
                whisper_gpu.initialize(args)
        cpu_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
