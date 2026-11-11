import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from services import diarization, whisper_cpp
from services.task_errors import TaskFailure


def load_whisper_gpu():
    path = Path(__file__).resolve().parents[1] / "whisper-gpu.py"
    spec = importlib.util.spec_from_file_location("whisper_gpu_diarization", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


whisper_gpu = load_whisper_gpu()


class DiarizationRuntimeTests(unittest.TestCase):
    def test_cli_defaults_diarization_off_and_device_auto(self):
        args = whisper_gpu.build_argument_parser().parse_args(["--filename", "input.wav"])
        self.assertFalse(args.diarization)
        self.assertEqual(args.diarization_device, "auto")

    def test_run_uses_single_explicit_runtime_pass_and_preset(self):
        device = {"id": "gpu-hash", "name": "Radeon", "backend": "vulkan", "deviceIndex": 0,
                  "executable": "nemo.exe", "modelPath": Path("model.gguf")}
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "nemo.exe"
            model_path = Path(directory) / "model.gguf"
            wav_path = Path(directory) / "in.wav"
            executable.touch()
            model_path.touch()
            with wave.open(str(wav_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0\0" * 16000)
            device["executable"] = str(executable)

            def run(command, **kwargs):
                self.assertEqual(command[1:3], ["diarize", str(wav_path.resolve())])
                self.assertEqual(command[command.index("--device") + 1], "vulkan:0")
                self.assertEqual(command[command.index("--preset") + 1], "v3-offline")
                self.assertIn("--force", command)
                self.assertTrue(Path(command[-1]).exists())
                self.assertEqual(kwargs["cwd"], str(whisper_gpu.Path(__file__).resolve().parents[1]))
                if "--force" not in command:
                    return SimpleNamespace(returncode=1, stdout="", stderr="output already exists")
                Path(command[-1]).write_text(json.dumps({"file": "in.wav", "segments": [
                    {"start": 0.1, "end": 0.8, "speaker": 4}]}), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.object(diarization, "MODEL_TEMP_DIR", Path(directory)), \
                 patch.object(diarization.subprocess, "run", side_effect=run), \
                 patch("services.diarization_runtime.resolve_diarization_device", return_value=device), \
                 patch("services.diarization_runtime.resolve_diarization_model", return_value=model_path):
                result = diarization.run_diarization(wav_path, "gpu-hash")
        self.assertEqual(result["effectiveDevice"], {"id": "gpu-hash", "name": "Radeon", "backend": "vulkan"})
        self.assertEqual(result["intervals"][0]["speaker"], 4)

    def test_gpu_failure_is_not_retried_on_cpu(self):
        device = {"id": "gpu-hash", "name": "Radeon", "backend": "vulkan", "deviceIndex": 0,
                  "executable": "nemo.exe", "modelPath": Path("model.gguf")}
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "nemo.exe"
            model_path = Path(directory) / "model.gguf"
            executable.touch()
            model_path.touch()
            device["executable"] = str(executable)
            with patch.object(diarization, "MODEL_TEMP_DIR", Path(directory)), \
             patch("services.diarization_runtime.resolve_diarization_device", return_value=device), \
             patch("services.diarization_runtime.resolve_diarization_model", return_value=model_path), \
             patch.object(diarization.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="Vulkan device lost")) as run:
                with self.assertRaises(TaskFailure) as raised:
                    diarization.run_diarization("in.wav", "gpu-hash")
        self.assertEqual(raised.exception.code, "device_error")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][run.call_args.args[0].index("--device") + 1], "vulkan:0")


class WhisperCppWordTimestampTests(unittest.TestCase):
    def test_v184_probe_reconstructs_all_words_and_keeps_zero_duration_word_invalid(self):
        fixture = Path(__file__).resolve().parents[1] / ".runtime" / "diarization-probe" / "asr-word-probe.json"
        if not fixture.exists():
            self.skipTest("Pinned whisper.cpp v1.8.4 word probe is unavailable")
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        segments, _stats = whisper_cpp.parse_whisper_cpp_json(payload, word_timestamps=True)
        self.assertIsNotNone(segments[0].words)
        self.assertEqual("".join(word["text"] for word in segments[0].words).strip(), segments[0].text)
        self.assertEqual(segments[0].words[0]["start"], segments[0].words[0]["end"])

    def test_word_timestamps_adds_full_json_and_profile_dtw_flags(self):
        model = whisper_cpp.WhisperCppModel.__new__(whisper_cpp.WhisperCppModel)
        model.executable, model.model_path, model.threads, model.model_name = "whisper.exe", Path("m.bin"), 1, "large-v3-turbo"
        model.gpu_index = None
        model.transcription_progress = None
        commands = []

        def run(command, **_kwargs):
            commands.append(command)
            if command[0] == "ffmpeg.exe":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            output = Path(command[command.index("-of") + 1] + ".json")
            output.write_text(json.dumps({"transcription": [{"offsets": {"from": 0, "to": 500},
                "text": " hi", "tokens": [{"text": " hi", "offsets": {"from": 100, "to": 400}}]}]}), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="using Vulkan0 backend")

        with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
             patch.object(whisper_cpp.subprocess, "run", side_effect=run):
            segments, _stats = model.transcribe("input.wav", word_timestamps=True)
        command = commands[1]
        self.assertIn("-ojf", command)
        self.assertEqual(command[command.index("-dtw") + 1], "large.v3.turbo")
        self.assertEqual(segments[0].words[0]["text"], " hi")


class DiarizationPipelineTests(unittest.TestCase):
    def test_disabled_branch_keeps_asr_text_and_does_not_request_word_times(self):
        segment = SimpleNamespace(start=0, end=1, text="plain text")
        stats = SimpleNamespace(language="en", language_probability=0.8)

        class Model:
            def transcribe(self, filename, beam_size, language):
                self.calls = (filename, beam_size, language)
                return [segment], stats

        model = Model()
        args = SimpleNamespace(web_json=False, language="en", beam_size=5, output_format="txt", output_dir="",
                               use_timestamps=True, restore_punctuation=False, diarization=False,
                               diarization_device="auto", device="cpu", single_source=False)
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            result = whisper_gpu.transcribe(args, model, "source.wav", "source.wav")
            written = Path(result["output_path"]).read_text(encoding="utf-8")
        self.assertEqual(model.calls, ("source.wav", 5, "en"))
        self.assertEqual(result["text"], "plain text")
        self.assertEqual(written, "[00:00:00,000] plain text\n")
        self.assertEqual(result["diarization"]["status"], "disabled")

    def test_transcribe_prepares_pcm_words_unloads_asr_and_keeps_asr_metadata(self):
        word = SimpleNamespace(word=" hello", start=0.1, end=0.5)
        segment = SimpleNamespace(start=0.0, end=1.0, text="hello", words=[word])
        stats = SimpleNamespace(language="en", language_probability=0.9)

        class Model:
            effective_device = "cuda"
            effective_backend = "faster-whisper CUDA"

            def __init__(self):
                self.unloaded = False

            def transcribe(self, filename, **kwargs):
                self.filename, self.kwargs = filename, kwargs
                return [segment], stats

            def unload_model(self):
                self.unloaded = True

        model = Model()
        args = SimpleNamespace(web_json=False, language="en", beam_size=5, output_format="json", output_dir="",
                               use_timestamps=True, restore_punctuation=False, diarization=True,
                               diarization_device="auto", device="cuda", single_source=False)
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            def ffmpeg_run(command, **kwargs):
                if command[0] != "ffmpeg.exe":
                    raise AssertionError("only the shared PCM conversion is expected")
                self.assertIn("pcm_s16le", command)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def nemo(*_args):
                self.assertTrue(model.unloaded)
                return {"effectiveDevice": {"id": "cpu", "name": "CPU", "backend": "cpu"},
                        "intervals": [{"start": 0, "end": 1, "speaker": 1}]}

            with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                 patch.object(whisper_gpu.subprocess, "run", side_effect=ffmpeg_run), \
                 patch.object(diarization, "run_diarization", side_effect=nemo):
                result = whisper_gpu.transcribe(args, model, "source.mp4", "source.mp4")
        self.assertTrue(model.unloaded)
        self.assertTrue(model.kwargs["word_timestamps"])
        self.assertEqual(result["effective_device"], "cuda")
        self.assertEqual(result["diarization"]["turns"][0]["speakerIds"], [1])

    def test_cpp_dict_words_are_serialized_and_txt_contains_speakers(self):
        asr_segment = whisper_cpp.WhisperCppSegment(0, 1, "hello", [
            {"text": " hel", "start": 0.1, "end": 0.4}, {"text": "lo", "start": 0.4, "end": 0.8},
        ])
        stats = SimpleNamespace(language="en", language_probability=0.9)

        class FakeCpp(whisper_cpp.WhisperCppModel):
            effective_device = "vulkan"
            effective_backend = "whisper.cpp Vulkan"

            def transcribe(self, _filename, **_kwargs):
                return [asr_segment], stats

        cpp_model = FakeCpp.__new__(FakeCpp)

        args = SimpleNamespace(web_json=False, language="en", beam_size=5, output_format="txt", output_dir="",
                               use_timestamps=True, restore_punctuation=False, diarization=True,
                               diarization_device="cpu", device="vulkan", single_source=False)
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                 patch.object(whisper_gpu.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), \
                 patch.object(whisper_gpu.time, "strftime", return_value="20261111-120000"), \
                 patch("services.diarization.run_diarization", return_value={
                     "effectiveDevice": {"id": "cpu", "name": "CPU", "backend": "cpu"},
                     "intervals": [{"start": 0, "end": 1, "speaker": 1}]}):
                result = whisper_gpu.transcribe(args, cpp_model, "source.wav", "source.wav")
            written = Path(result["output_path"]).read_text(encoding="utf-8")
        self.assertIn("Говорящий 1: hello", written)
        self.assertIn("Говорящий 1: hello", result["text"])

    def test_faster_word_object_normalizes_text_and_unloads_nested_ctranslate_before_diar(self):
        class Word:
            word, start, end = " one", 0.1, 0.7

        segment = SimpleNamespace(start=0, end=1, text="one", words=[Word()])
        stats = SimpleNamespace(language="en", language_probability=0.8)

        class CTranslate:
            def __init__(self):
                self.unloaded = False

            def unload_model(self):
                self.unloaded = True

        class FasterModel:
            effective_device = "cuda"
            effective_backend = "faster-whisper CUDA"

            def __init__(self):
                self.model = CTranslate()

            def transcribe(self, _filename, **_kwargs):
                return [segment], stats

        model = FasterModel()
        args = SimpleNamespace(web_json=False, language="en", beam_size=5, output_format="json", output_dir="",
                               use_timestamps=True, restore_punctuation=False, diarization=True,
                               diarization_device="auto", device="cuda", single_source=False)
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            def nemo(*_args):
                self.assertTrue(model.model.unloaded)
                return {"effectiveDevice": {"id": "cpu", "name": "CPU", "backend": "cpu"},
                        "intervals": [{"start": 0, "end": 1, "speaker": 1}]}

            with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                 patch.object(whisper_gpu.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), \
                 patch("services.diarization.run_diarization", side_effect=nemo):
                result = whisper_gpu.transcribe(args, model, "source.wav", "source.wav")
        self.assertEqual(result["diarization"]["turns"][0]["text"], "one")

    def test_diarization_precedes_one_turn_level_punctuation_restore(self):
        segment = SimpleNamespace(start=0, end=1, text="hello", words=[SimpleNamespace(word=" hello", start=0.1, end=0.8)])
        stats = SimpleNamespace(language="ru", language_probability=0.9)

        class Model:
            def transcribe(self, _filename, **_kwargs):
                return [segment], stats

        args = SimpleNamespace(web_json=True, language="ru", beam_size=5, output_format="txt", output_dir="",
                               use_timestamps=True, restore_punctuation=True, diarization=True,
                               diarization_device="auto", device="cpu", single_source=False)
        events = []
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            def diarize(*_args):
                events.append("diarize")
                return {"effectiveDevice": {"id": "cpu", "name": "CPU", "backend": "cpu"},
                        "intervals": [{"start": 0, "end": 1, "speaker": 1}]}

            def punctuate(text):
                events.append("punctuate")
                return text.title() + "."

            with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                 patch.object(whisper_gpu.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), \
                 patch.object(whisper_gpu, "restore_russian_segment_text", side_effect=punctuate), \
                 patch("services.diarization.run_diarization", side_effect=diarize), \
                 patch.object(whisper_gpu, "emit_event", side_effect=lambda _args, event, **payload: events.append((event, payload))):
                whisper_gpu.transcribe(args, Model(), "source.wav", "source.wav")
        self.assertEqual(events.count("punctuate"), 1)
        self.assertLess(events.index("diarize"), events.index("punctuate"))
        diar_done = next(i for i, value in enumerate(events) if isinstance(value, tuple) and value[0] == "stage"
                         and value[1].get("stage") == "diarization" and value[1].get("status") == "completed")
        post_start = next(i for i, value in enumerate(events) if isinstance(value, tuple) and value[0] == "stage"
                          and value[1].get("stage") == "postprocess" and value[1].get("status") == "in_progress")
        self.assertLess(diar_done, post_start)


if __name__ == "__main__":
    unittest.main()
