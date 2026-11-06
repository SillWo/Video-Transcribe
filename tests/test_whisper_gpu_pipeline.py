import importlib.util
import argparse
import json
import io
import contextlib
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

    def test_transcribe_restores_russian_punctuation_for_auto_detected_russian(self):
        segments = [SimpleNamespace(start=0.0, end=1.0, text="privet mir")]
        model, _stats = self.build_model(segments)
        args = self.build_args(language="auto", restore_punctuation=True, output_format="txt", use_timestamps=False)

        with tempfile.TemporaryDirectory() as temp_dir, \
            patch.object(whisper_gpu.time, "strftime", return_value="20260417-120000"), \
            patch.object(whisper_gpu, "restore_russian_segment_text", return_value="Привет, мир.") as restore_mock:
            args.output_dir = temp_dir
            result = whisper_gpu.transcribe(args, model, "input.wav", "media.wav")

        restore_mock.assert_called_once_with("privet mir")
        self.assertEqual(result["text"], "Привет, мир.")
        self.assertEqual(result["rendered_result"], "Привет, мир.\n")

    def test_single_source_emits_progress_from_real_segment_iteration_and_completes_before_postprocess(self):
        segments = [SimpleNamespace(start=0.0, end=1.0, text="first"),
                    SimpleNamespace(start=1.0, end=3.0, text="second")]
        model, stats = self.build_model(segments)
        stats.duration = 4.0
        args = self.build_args(language="en", restore_punctuation=False)
        args.single_source = True
        args.web_json = True
        events = []
        with tempfile.TemporaryDirectory() as temp_dir, \
                patch.object(whisper_gpu.time, "strftime", return_value="20260417-120000"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            args.output_dir = temp_dir
            result = whisper_gpu.transcribe(args, model, "input.wav", "media.wav")
        for line in output.getvalue().splitlines():
            if line.startswith(whisper_gpu.JSON_EVENT_PREFIX):
                events.append(json.loads(line[len(whisper_gpu.JSON_EVENT_PREFIX):]))

        recognition = [event for event in events if event.get("type") == "progress" and event.get("stage") == "recognition"]
        self.assertEqual([event["percent"] for event in recognition], [25.0, 75.0, 100.0])
        self.assertTrue(all(event["processedSeconds"] is not None and event["totalSeconds"] == 4.0 for event in recognition))
        self.assertEqual(result["text"], "first\nsecond")
        self.assertLess(next(i for i, event in enumerate(events) if event.get("stage") == "recognition" and event.get("status") == "completed"),
                        next(i for i, event in enumerate(events) if event.get("stage") == "postprocess" and event.get("status") == "skipped"))

    def test_single_source_recognition_uses_indeterminate_progress_without_duration(self):
        segments = [SimpleNamespace(start=0.0, end=1.0, text="hello")]
        model, _stats = self.build_model(segments)
        args = self.build_args(language="en", restore_punctuation=False)
        args.single_source = True
        args.web_json = True
        with tempfile.TemporaryDirectory() as temp_dir, \
                patch.object(whisper_gpu.time, "strftime", return_value="20260417-120000"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            args.output_dir = temp_dir
            whisper_gpu.transcribe(args, model, "input.wav", "media.wav")
        progress_events = [json.loads(line[len(whisper_gpu.JSON_EVENT_PREFIX):])
                           for line in output.getvalue().splitlines()
                           if line.startswith(whisper_gpu.JSON_EVENT_PREFIX)
                           and '"type": "progress"' in line]
        recognition = [event for event in progress_events if event["stage"] == "recognition"]
        self.assertEqual([event["percent"] for event in recognition], [None])
        self.assertIsNone(recognition[0]["totalSeconds"])

    def test_single_source_prepares_audio_before_model_warmup_and_preserves_it_for_transcription(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.mp4"
            source.touch()
            class MainArgs(argparse.Namespace):
                def __contains__(self, key):
                    return hasattr(self, key)

            args = MainArgs(
                filename=str(source), input_dir=None, output_name=None, output_dir=temp_dir,
                language="ru", beam_size=5, precision="auto", device="cpu", model_size="tiny",
                start=None, end=None, nproc=1, keep=False, verbose=False, quiet=True,
                playlist_start=None, playlist_end=None, codec=None, bitrate=None,
                output_format="json", no_timestamps=False, restore_punctuation=False,
                single_source=True, web_json=True,
            )
            audio_path = Path(temp_dir) / "intermediate_audio_file.mp3"
            timeline = []
            original_emit_event = whisper_gpu.emit_event

            def record_event(event_args, event_type, **payload):
                if event_type == "stage":
                    timeline.append(("stage", payload.get("stage"), payload.get("status"), payload.get("message")))
                return original_emit_event(event_args, event_type, **payload)

            def add_source(_args, media_files, **_kwargs):
                media_files.append(str(source))
                record_event(args, "stage", stage="source", status="completed", message="Source ready")

            class FakeAudioProcess:
                valid_time = staticmethod(lambda value: value)

                def __init__(self, _args):
                    pass

                def audio_only(self, _path):
                    return False

                def extract_audio(self, *, output_filepath, progress_callback=None, **_kwargs):
                    timeline.append(("audio_written",))
                    Path(output_filepath).write_bytes(b"prepared")

            def load_model(_args, _device):
                self.assertTrue(audio_path.is_file(), "initialize must not remove the prepared simple-source audio")
                timeline.append(("model_loaded",))
                return SimpleNamespace()

            def transcribe(_args, _model, full_filepath, _filename):
                self.assertEqual(Path(full_filepath), audio_path)
                self.assertTrue(audio_path.is_file(), "prepared audio must remain available until transcription starts")
                timeline.append(("transcribe",))
                return {"text": "done"}

            with patch.object(whisper_gpu.argparse.ArgumentParser, "parse_args", return_value=args), \
                    patch.object(whisper_gpu, "add_media_files", side_effect=add_source), \
                    patch.object(whisper_gpu, "AudioProcess", FakeAudioProcess), \
                    patch.object(whisper_gpu, "get_device_details", return_value=[{"id": "cpu", "available": True}]), \
                    patch.object(whisper_gpu, "_faster_model", side_effect=load_model), \
                    patch.object(whisper_gpu, "transcribe", side_effect=transcribe), \
                    patch.object(whisper_gpu, "emit_event", side_effect=record_event), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(whisper_gpu.main(), 0)

            audio_ready = next(index for index, item in enumerate(timeline)
                               if item[:3] == ("stage", "audio", "completed"))
            warmup_started = next(index for index, item in enumerate(timeline)
                                   if item[:3] == ("stage", "recognition", "in_progress")
                                   and "Загружаем модель" in item[3])
            model_loaded = next(index for index, item in enumerate(timeline) if item[0] == "model_loaded")
            transcribe_started = next(index for index, item in enumerate(timeline) if item[0] == "transcribe")
            self.assertLess(audio_ready, warmup_started)
            self.assertLess(warmup_started, model_loaded)
            self.assertLess(model_loaded, transcribe_started)
            self.assertFalse(audio_path.exists(), "audio is cleaned up after successful transcription")

    def test_transcribe_skips_punctuation_for_auto_detected_english(self):
        segments = [SimpleNamespace(start=0.0, end=1.0, text="hello world")]
        model, stats = self.build_model(segments)
        stats.language = "en"
        args = self.build_args(language="auto", restore_punctuation=True, output_format="txt", use_timestamps=False)

        with tempfile.TemporaryDirectory() as temp_dir, \
            patch.object(whisper_gpu.time, "strftime", return_value="20260417-120000"), \
            patch.object(whisper_gpu, "restore_russian_segment_text") as restore_mock:
            args.output_dir = temp_dir
            result = whisper_gpu.transcribe(args, model, "input.wav", "media.wav")

        restore_mock.assert_not_called()
        self.assertEqual(result["text"], "hello world")
        self.assertIn("не русский язык", result["notice"])

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
