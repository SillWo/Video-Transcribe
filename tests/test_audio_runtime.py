import math
import struct
import tempfile
import unittest
import wave
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from services.model_paths import MODEL_TEMP_DIR
from faster_whisper.audio import decode_audio

from utils.get_audio import AudioProcess, ffmpeg


def write_tone(path: Path, duration_seconds: float = 0.25) -> int:
    sample_rate = 16000
    sample_count = int(sample_rate * duration_seconds)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        pcm = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * index / sample_rate)))
            for index in range(sample_count)
        )
        output.writeframes(pcm)
    return sample_count


class FasterWhisperAudioDecodeTests(unittest.TestCase):
    def test_pyav_decodes_real_pcm_wave_without_model_or_network(self):
        self.assertTrue(str(MODEL_TEMP_DIR).startswith(str(Path(__file__).resolve().parents[1])))
        with tempfile.TemporaryDirectory(dir=MODEL_TEMP_DIR) as temp_dir:
            audio_path = Path(temp_dir) / "short-tone.wav"
            expected_samples = write_tone(audio_path)

            samples = decode_audio(str(audio_path))

        self.assertEqual(samples.dtype.name, "float32")
        self.assertGreater(len(samples), 0)
        self.assertAlmostEqual(len(samples), expected_samples, delta=5)
        self.assertGreater(float(abs(samples).max()), 0.05)


class AudioExtractionTests(unittest.TestCase):
    def make_processor(self, *, overwrite=False):
        args = Namespace(
            output_dir=None,
            output_name=None,
            extension="mp3",
            overwrite=overwrite,
            quiet=True,
        )
        return AudioProcess(args)

    def test_extract_audio_honors_explicit_overwrite_argument(self):
        with tempfile.TemporaryDirectory(dir=MODEL_TEMP_DIR) as temp_dir:
            root = Path(temp_dir)
            source = root / "source.webm"
            target = root / "output.mp3"
            source.write_bytes(b"input")
            fake_stream = Mock()
            fake_stream.compile.return_value = ["ffmpeg", "-i", str(source)]

            with patch.object(ffmpeg, "input", return_value=Mock()), \
                    patch.object(ffmpeg, "output", return_value=fake_stream), \
                    patch.object(ffmpeg, "run", return_value=(b"", b"")) as run:
                result = self.make_processor(overwrite=False).extract_audio(
                    str(source), output_filepath=str(target), overwrite=True
                )

        self.assertEqual(result, (0, ""))
        self.assertTrue(run.call_args.kwargs["overwrite_output"])

    def test_extract_audio_raises_readable_error_with_ffmpeg_stderr(self):
        with tempfile.TemporaryDirectory(dir=MODEL_TEMP_DIR) as temp_dir:
            root = Path(temp_dir)
            source = root / "source.webm"
            target = root / "output.mp3"
            source.write_bytes(b"input")
            fake_stream = Mock()
            fake_stream.compile.return_value = ["ffmpeg"]
            error = ffmpeg.Error("ffmpeg", b"", b"decoder rejected this stream")

            with patch.object(ffmpeg, "input", return_value=Mock()), \
                    patch.object(ffmpeg, "output", return_value=fake_stream), \
                    patch.object(ffmpeg, "run", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "FFmpeg audio extraction failed: decoder rejected this stream") as raised:
                    self.make_processor().extract_audio(str(source), output_filepath=str(target), overwrite=True)

        self.assertIs(raised.exception.__cause__, error)

    def test_ffmpeg_setup_errors_are_raised_instead_of_exiting_successfully(self):
        with tempfile.TemporaryDirectory(dir=MODEL_TEMP_DIR) as temp_dir:
            source = Path(temp_dir) / "source.webm"
            target = Path(temp_dir) / "output.mp3"
            source.write_bytes(b"input")
            with patch.object(ffmpeg, "input", side_effect=ValueError("bad ffmpeg input")):
                with self.assertRaisesRegex(RuntimeError, "Unable to prepare FFmpeg audio extraction: bad ffmpeg input") as raised:
                    self.make_processor().extract_audio(str(source), output_filepath=str(target))
        self.assertIsInstance(raised.exception.__cause__, ValueError)


if __name__ == "__main__":
    unittest.main()
