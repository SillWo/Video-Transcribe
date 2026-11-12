import importlib.util
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from services import asr_engine
from services.task_errors import TaskFailure


def write_pcm_wav(path: Path, seconds=2):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x01\x00" * 16000 * seconds)


def fake_worker_result(mode, device="cpu"):
    if mode == "transcribe":
        return {"ok": True, "text": "hello", "segments": [{"start": 0.25, "end": 1.25, "text": "hello"}],
                "language": "en", "language_probability": None, "duration": 2.0,
                "effective_device": device, "effective_backend": f"backend {device}", "precision": "F16",
                "diagnostics": {"gpu_verified": False}}
    return {"ok": True, "segments": [{"start": 0.25, "end": 1.25, "text": "hello",
                                          "words": [{"text": "hello", "start": 0.3, "end": 0.9}]}],
            "effective_device": "cuda:0" if device == "cuda" else device,
            "effective_backend": f"Forced Aligner {device} (FP32)",
            "precision": "FP32", "diagnostics": {"runtime": "transformers"}}


class AsrEngineTests(unittest.TestCase):
    def test_model_transcribe_runs_asr_then_aligner_after_worker_exit_and_preserves_times(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "audio.wav"
            write_pcm_wav(source)
            callback_events = []
            alignment_started_before_worker = []
            model = asr_engine.NewAsrModel("qwen3-asr-1.7b", "cuda", progress=callback_events.append)
            completed = []

            class FakeProcess:
                def __init__(self, command, **_kwargs):
                    self.command = command
                    self.stdout = io.StringIO('__VT_ASR__ {"type":"progress","stage":"alignment","message":"Aligned"}\n')
                    self.returncode = None
                    self.mode = command[command.index("--mode") + 1]
                    if self.mode == "align":
                        alignment_started_before_worker.append(
                            bool(callback_events) and callback_events[-1]["phase"] == "alignment"
                            and callback_events[-1]["message"] == "Выравнивание слов")
                    self.output = Path(command[command.index("--output") + 1])
                    if completed:
                        self.assert_previous_completed = completed[-1]
                    result = fake_worker_result(self.mode, "cuda")
                    self.output.write_text(json.dumps(result), encoding="utf-8")

                def wait(self):
                    self.returncode = 0
                    completed.append(self.mode)
                    return self.returncode

                def poll(self):
                    return self.returncode

            with patch.object(asr_engine, "runtime_python", return_value=Path("python.exe")), \
                 patch.object(asr_engine.subprocess, "Popen", side_effect=FakeProcess):
                segments, stats = model.transcribe(source, language="en", word_timestamps=True)

        self.assertEqual(completed, ["transcribe", "align"])
        self.assertEqual((segments[0].start, segments[0].end, segments[0].text), (0.25, 1.25, "hello"))
        self.assertEqual(segments[0].words[0], {"text": "hello", "start": 0.3, "end": 0.9})
        self.assertEqual(stats.language, "en")
        self.assertIsNone(stats.language_probability)
        self.assertEqual(model.precision, "F16")
        self.assertEqual(alignment_started_before_worker, [True])
        alignment_start = next(event for event in callback_events if event["message"] == "Выравнивание слов")
        self.assertEqual(alignment_start["workerStage"], "alignment")

    def test_asr_gpu_failure_is_not_retried_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "audio.wav"
            write_pcm_wav(source)
            model = asr_engine.NewAsrModel("qwen3-asr-1.7b", "vulkan")
            calls = []

            def run(mode, *_args, **_kwargs):
                calls.append(mode)
                raise TaskFailure("Vulkan device lost", "device_error", stage="recognition")

            with patch.object(model, "_run_worker", side_effect=run):
                with self.assertRaises(TaskFailure) as raised:
                    model.transcribe(source)
        self.assertEqual(raised.exception.code, "device_error")
        self.assertEqual(calls, ["transcribe"])

    def test_runtime_override_selects_cpu_aligner_and_gpu_language_id_environment(self):
        model = asr_engine.NewAsrModel("parakeet-tdt-0.6b-v3", "cuda", device_index=5)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            audio = workspace / "audio.wav"
            write_pcm_wav(audio)
            invocations = []

            class SuccessfulProcess:
                def __init__(self, command, **_kwargs):
                    invocations.append(command)
                    output = Path(command[command.index("--output") + 1])
                    device = command[command.index("--device") + 1]
                    output.write_text(json.dumps({"ok": True, "effective_device": device,
                                                  "effective_backend": f"Forced Aligner {device}" if command[command.index("--mode") + 1] == "align" else "mock",
                                                  "precision": "FP32",
                                                  "language_observations": [], "timing_seconds": {}}), encoding="utf-8")
                    self.stdout = io.StringIO("")
                    self.returncode = None

                def wait(self):
                    self.returncode = 0
                    return 0

                def poll(self):
                    return self.returncode

            with patch.object(asr_engine, "runtime_python", side_effect=lambda kind, backend: Path(f"{kind}-{backend}.exe")), \
                 patch.object(asr_engine.subprocess, "Popen", SuccessfulProcess):
                model._run_worker("align", audio, workspace, device="directml", default_stage="alignment")
                model._run_worker("align", audio, workspace, device="cpu", default_stage="alignment")
                model._run_worker("language-id", audio, workspace, device="vulkan", device_index=2)
                model._run_worker("language-id", audio, workspace, device="cpu", device_index=5)
                cpu_model = asr_engine.NewAsrModel("parakeet-tdt-0.6b-v3", "cpu")
                cpu_model._run_worker("language-id", audio, workspace, device="cpu", device_index=5)

        self.assertEqual([command[0] for command in invocations], [
            "aligner-cpu.exe", "aligner-cpu.exe", "native-vulkan.exe", "native-cuda.exe", "native-cpu.exe",
        ])
        self.assertEqual([command[command.index("--device") + 1] for command in invocations],
                         ["directml", "cpu", "vulkan", "cpu", "cpu"])
        indices = [command[command.index("--device-index") + 1] for command in invocations]
        self.assertEqual(indices, ["5", "0", "2", "0", "0"])

    def test_alignment_failures_are_mapped_to_public_recognition_stage(self):
        for message in ("CUDA out of memory", "model.safetensors does not exist", "alignment failed"):
            with self.subTest(message=message):
                failure = asr_engine._map_worker_failure(message, "alignment", device="cuda")
                self.assertEqual(failure.stage, "recognition")

    def test_parakeet_native_words_are_preserved_without_aligner_on_all_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "audio.wav"
            write_pcm_wav(source)
            for device in ("cpu", "cuda", "directml"):
                with self.subTest(device=device):
                    model = asr_engine.NewAsrModel("parakeet-tdt-0.6b-v3", device)
                    calls = []
                    native_words = [{"text": "hello", "start": 0.3, "end": 0.9}]

                    def run(mode, *_args, **_kwargs):
                        calls.append(mode)
                        if mode == "transcribe":
                            return {**fake_worker_result(mode, device),
                                    "segments": [{"start": 0.25, "end": 1.25, "text": "hello",
                                                  "words": native_words}]}
                        raise AssertionError(f"unexpected worker phase: {mode}")

                    with patch.object(model, "_run_worker", side_effect=run):
                        segments, _stats = model.transcribe(source, language="en", word_timestamps=True)

                    self.assertEqual(calls, ["transcribe"])
                    self.assertEqual(segments[0].words, native_words)
                    self.assertNotIn("alignment", model.diagnostics)
                    self.assertNotIn("alignment_fallback", model.diagnostics)

    def test_language_observations_are_weighted_by_recognized_duration(self):
        model = asr_engine.NewAsrModel("parakeet-tdt-0.6b-v3", "cpu")
        segments = [{"start": 0, "end": 10, "text": "words"}]
        observations = [
            {"start": 0, "end": 6, "code": "ru", "probability": 0.8},
            {"start": 6, "end": 8, "code": "en", "probability": 0.9},
            {"start": 8, "end": 10, "code": "en", "probability": 0.7},
        ]
        language, probability = model._language_from_observations(observations, segments)
        self.assertEqual(language, "ru")
        self.assertAlmostEqual(probability, 0.8)

    def test_parakeet_language_id_chooses_best_supported_candidate_and_keeps_raw_top(self):
        model = asr_engine.NewAsrModel("parakeet-tdt-0.6b-v3", "cpu")
        raw = [
            {"start": 0, "end": 5, "code": "be", "probability": 0.55,
             "allowed_mass": 0.96, "candidates": [
                 {"code": "be", "p": 0.55}, {"code": "ru", "p": 0.4}, {"code": "en", "p": 0.05},
             ]},
            {"start": 5, "end": 10, "code": "en", "probability": 0.7,
             "candidates": [{"code": "en", "p": 0.7}, {"code": "be", "p": 0.2}]},
        ]
        supported = model._supported_language_observations(raw)
        segments = [{"start": 0, "end": 5, "text": "russian words"},
                    {"start": 5, "end": 10, "text": "english words"}]
        first_language, first_probability = model._language_from_observations(supported[:1], segments[:1])
        second_language, second_probability = model._language_from_observations(supported[1:], segments[1:])

        self.assertEqual((first_language, first_probability), ("ru", 0.4))
        self.assertEqual((second_language, second_probability), ("en", 0.7))
        self.assertEqual(raw[0]["code"], "be")
        self.assertEqual(raw[0]["probability"], 0.55)
        self.assertEqual(supported[0]["unrestricted_top"], {"code": "be", "probability": 0.55})
        self.assertEqual(supported[0]["allowed_mass"], 0.96)

    def test_transcript_alphabet_conservatively_corrects_russian_and_ukrainian(self):
        model = asr_engine.NewAsrModel("parakeet-tdt-0.6b-v3", "cpu")
        for code, text in (("ru", "Это русский текст, мы говорим о языке и новых возможностях, чтобы проверить ыыэ."),
                           ("uk", "Це український текст, ми говоримо про мову і нові можливості, щоб перевірити їіїєґ.")):
            with self.subTest(code=code):
                rows = [{"start": 0, "end": 5, "text": text}]
                original = [dict(rows[0])]
                raw = [{"start": 0, "end": 5, "code": "uk" if code == "ru" else "ru",
                        "probability": 0.61,
                        "candidates": [{"code": "ru", "p": 0.39}, {"code": "uk", "p": 0.61},
                                       {"code": "bg", "p": 0.2}]}]
                result = model._supported_language_observations(raw, rows)[0]
                self.assertEqual(result["code"], code)
                self.assertEqual(result["probability"], 0.39 if code == "ru" else 0.61)
                self.assertEqual(result["unrestricted_top"]["code"], raw[0]["code"])
                self.assertTrue(result["alphabet_constraint"]["applied"])
                self.assertEqual(rows, original)

    def test_transcript_alphabet_leaves_ambiguous_bulgarian_latin_and_short_text_unchanged(self):
        model = asr_engine.NewAsrModel("parakeet-tdt-0.6b-v3", "cpu")
        candidates = [{"code": "bg", "p": 0.6}, {"code": "ru", "p": 0.4}]
        for text in ("Общият текст говори за език и възможности без отличителни букви",
                     "This is a sufficiently long Latin transcript about language.", "Кратък текст"):
            with self.subTest(text=text):
                result = model._supported_language_observations(
                    [{"code": "ru", "probability": 0.4, "candidates": candidates}],
                    [{"text": text}])[0]
                self.assertEqual((result["code"], result["probability"]), ("bg", 0.6))
                self.assertFalse(result["alphabet_constraint"]["applied"])

    def test_qwen_language_observations_are_duration_weighted(self):
        model = asr_engine.NewAsrModel("qwen3-asr-1.7b", "cpu")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "audio.wav"
            write_pcm_wav(source)
            observations = [
                {"start": 0, "end": 1.5, "language": "ru"},
                {"start": 1.5, "end": 2, "language": "en"},
            ]
            with patch.object(model, "_run_worker", return_value={
                **fake_worker_result("transcribe"), "language": "en", "language_observations": observations,
            }):
                _segments, stats = model.transcribe(source)
        self.assertEqual(stats.language, "ru")

    def test_language_id_selects_matching_native_gpu_for_directml_without_reusing_dxgi_index(self):
        model = asr_engine.NewAsrModel("parakeet-tdt-0.6b-v3", "directml", device_index=7)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            audio = workspace / "audio.wav"
            write_pcm_wav(audio)
            listed = {"devices": [
                {"backend": "vulkan", "index": 0, "name": "Other GPU", "deviceType": "gpu"},
                {"backend": "vulkan", "index": 1, "name": "Vulkan1", "description": "AMD Radeon780M Graphics", "deviceType": "igpu"},
            ]}
            with patch.object(asr_engine, "asr_gpu_candidate", return_value={"backend": "directml", "name": "AMD Radeon 780M Graphics"}), \
                 patch.object(model, "_run_worker", return_value=listed) as run:
                target, index, reason = model._language_id_target(audio, workspace)
        self.assertEqual((target, index, reason), ("vulkan", 1, None))
        run.assert_called_once_with("devices", audio, workspace, device="vulkan")

    def test_missing_worker_result_is_typed_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            audio = workspace / "audio.wav"
            write_pcm_wav(audio)
            model = asr_engine.NewAsrModel("qwen3-asr-1.7b", "cpu")

            class FailedProcess:
                stdout = io.StringIO("ordinary worker log\n")
                returncode = 0
                def __init__(self, *_args, **_kwargs): pass
                def wait(self): return self.returncode
                def poll(self): return self.returncode

            with patch.object(asr_engine, "runtime_python", return_value=Path("python.exe")), \
                 patch.object(asr_engine.subprocess, "Popen", FailedProcess):
                with self.assertRaises(TaskFailure) as raised:
                    model._run_worker("transcribe", audio, workspace)
        self.assertEqual(raised.exception.stage, "recognition")

    def test_non_pcm_media_is_converted_to_pcm16_mono_16k_without_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "source.mp3"
            source.write_bytes(b"encoded")

            def run(command, **_kwargs):
                self.assertIn("-ac", command)
                self.assertEqual(command[command.index("-ac") + 1], "1")
                self.assertEqual(command[command.index("-ar") + 1], "16000")
                target = Path(command[-1])
                write_pcm_wav(target)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("services.whisper_cpp._find_ffmpeg", return_value="ffmpeg.exe"), \
                 patch.object(asr_engine.subprocess, "run", side_effect=run):
                converted = asr_engine._prepare_audio(source, workspace)
            self.assertEqual(asr_engine._audio_duration(converted), 2.0)
            self.assertEqual(converted.name, "input-pcm16k-mono.wav")

    def test_existing_shared_pcm_wav_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "shared.wav"
            write_pcm_wav(wav)
            self.assertIs(asr_engine._prepare_audio(wav, Path(directory)), wav)


if __name__ == "__main__":
    unittest.main()
