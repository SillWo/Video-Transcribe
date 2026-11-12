import importlib.util
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from services.asr_engine import NewAsrModel
from services.diarization import assign_speakers


def load_whisper_gpu_module():
    module_path = Path(__file__).resolve().parents[1] / "whisper-gpu.py"
    spec = importlib.util.spec_from_file_location("whisper_gpu_native_diarization", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


whisper_gpu = load_whisper_gpu_module()


def write_pcm_wav(path):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x01\x00" * 16000 * 4)


class NativeDiarizationRegressionTests(unittest.TestCase):
    def transcribe_with_mock_worker(self, model_id, device, rows):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "audio.wav"
            write_pcm_wav(source)
            events = []
            model = NewAsrModel(model_id, device, progress=events.append)
            calls = []

            def run_worker(mode, _audio, _workspace, **kwargs):
                calls.append((mode, kwargs))
                if mode == "align":
                    raise AssertionError("native word timestamps must not invoke the forced aligner")
                return {"segments": rows, "language": "en", "language_probability": 0.9,
                        "duration": 4.0, "effective_device": device,
                        "effective_backend": f"ASR {device}", "precision": "FP32"}

            with patch.object(model, "_run_worker", side_effect=run_worker):
                segments, stats = model.transcribe(source, language="en", word_timestamps=True)
            return model, segments, stats, calls, events

    def test_parakeet_native_times_reach_speaker_assignment_without_alignment(self):
        rows = [{"start": 0.0, "end": 4.0, "text": "Hello world mystery", "words": [
            {"text": "Hello", "start": 0.2, "end": 0.7},
            {"text": " world", "start": 1.2, "end": 1.7},
            {"text": " mystery", "start": 3.2, "end": 3.7},
        ]}]
        for device in ("cpu", "directml", "cuda"):
            with self.subTest(device=device):
                model, segments, stats, calls, events = self.transcribe_with_mock_worker(
                    "parakeet-tdt-0.6b-v3", device, rows)
                self.assertEqual([mode for mode, _kwargs in calls], ["transcribe"])
                self.assertFalse(any(event.get("phase") == "alignment" for event in events))
                self.assertEqual(stats.language, "en")
                self.assertEqual(model.diagnostics.get("alignment"), None)
                serialized = whisper_gpu.serialize_segments(segments, word_timestamps=True)
                self.assertEqual(serialized[0]["text"], "Hello world mystery")
                self.assertEqual(serialized[0]["words"], rows[0]["words"])
                turns = assign_speakers(serialized, [
                    {"start": 0.0, "end": 1.0, "speaker": 17},
                    {"start": 1.0, "end": 2.0, "speaker": 42},
                ])
                self.assertEqual([(turn["text"], turn["start"], turn["end"], turn["speakerIds"])
                                  for turn in turns], [
                    ("Hello", 0.2, 0.7, [1]),
                    ("world", 1.2, 1.7, [2]),
                    ("mystery", 3.2, 3.7, []),
                ])

    def test_invalid_native_word_times_stay_unknown_and_keep_their_text(self):
        rows = [{"start": 0.0, "end": 2.0, "text": "known broken zero", "words": [
            {"text": "known", "start": 0.2, "end": 0.6},
            {"text": " broken", "start": None, "end": 1.4},
            {"text": " zero", "start": 1.5, "end": 1.5},
        ]}]
        _model, segments, _stats, calls, _events = self.transcribe_with_mock_worker(
            "parakeet-tdt-0.6b-v3", "cpu", rows)
        self.assertEqual([mode for mode, _kwargs in calls], ["transcribe"])
        serialized = whisper_gpu.serialize_segments(segments, word_timestamps=True)
        self.assertEqual(serialized[0]["words"][1], {"text": " broken", "start": None, "end": 1.4})
        self.assertEqual(serialized[0]["words"][2], {"text": " zero", "start": 1.5, "end": 1.5})
        turns = assign_speakers(serialized, [{"start": 0.0, "end": 1.0, "speaker": 5}])
        self.assertEqual(turns[0]["text"], "known")
        self.assertEqual(turns[1]["text"], "broken zero")
        self.assertIsNone(turns[1]["start"])
        self.assertIsNone(turns[1]["end"])
        self.assertFalse(turns[1]["timingKnown"])
        self.assertEqual(turns[1]["speakerIds"], [])

    def test_qwen_still_requests_alignment_and_returns_aligned_words(self):
        rows = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        aligned_rows = [{"start": 0.0, "end": 1.0, "text": "hello", "words": [
            {"text": "hello", "start": 0.25, "end": 0.8},
        ]}]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "audio.wav"
            write_pcm_wav(source)
            model = NewAsrModel("qwen3-asr-1.7b", "cpu")
            calls = []

            def run_worker(mode, _audio, workspace, **kwargs):
                calls.append((mode, kwargs))
                if mode == "align":
                    self.assertEqual(kwargs["language"], "en")
                    self.assertTrue(kwargs["segments_path"].is_file())
                    return {"segments": aligned_rows, "effective_device": "cpu",
                            "effective_backend": "Forced Aligner cpu", "precision": "FP32"}
                return {"segments": rows, "language": "en", "language_probability": 0.9,
                        "duration": 1.0, "effective_device": "cpu",
                        "effective_backend": "ASR cpu", "precision": "F16"}

            with patch.object(model, "_run_worker", side_effect=run_worker):
                segments, _stats = model.transcribe(source, language="en", word_timestamps=True)
        self.assertEqual([mode for mode, _kwargs in calls], ["transcribe", "align"])
        self.assertEqual(segments[0].words, aligned_rows[0]["words"])


if __name__ == "__main__":
    unittest.main()
