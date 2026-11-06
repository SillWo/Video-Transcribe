import argparse
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.download_best import Download
from services.task_errors import TaskFailure
import importlib.util


def args_for(url="https://example.test/video", single=True):
    return argparse.Namespace(
        url=url, format="bestaudio/best", audio_only=True, video_only=False,
        single_source=single, merge=None, output_dir=".", output_name=None,
        keep=False, verbose=False, overwrite=False, username=None, password=None,
        playlist_start=None, playlist_end=None, audio_format="mp3", restrict_filenames=True,
        timeout=None, bin="yt-dlp",
    )


class SingleSourceValidationTests(unittest.TestCase):
    def construct(self, metadata, *, single=True):
        calls = []

        class YoutubeDL:
            def __init__(self, options):
                calls.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, url, download):
                calls.append((url, download))
                return metadata

        module = types.ModuleType("yt_dlp")
        module.YoutubeDL = YoutubeDL
        with patch.dict(sys.modules, {"yt_dlp": module}), patch.object(Download, "resolve_model_cmd", return_value=["yt-dlp"]):
            downloader = Download(args_for(single=single))
        return downloader, calls

    def test_video_url_with_playlist_parameter_is_single_and_sets_no_playlist(self):
        downloader, calls = self.construct({"id": "video", "title": "Video", "_type": "video"})
        self.assertIn("--no-playlist", downloader.opts)
        self.assertTrue(calls[0]["noplaylist"])
        self.assertEqual(calls[1], ("https://example.test/video", False))

    def test_playlist_only_link_is_rejected_before_download(self):
        with self.assertRaisesRegex(RuntimeError, "contains a playlist or multiple videos"):
            self.construct({"_type": "playlist", "entries": [{"id": "one"}, {"id": "two"}]})

    def test_multiple_video_metadata_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "contains a playlist or multiple videos"):
            self.construct({"_type": "multi_video", "entries": [{"id": "one"}, {"id": "two"}]})

    def test_streaming_live_and_upcoming_are_rejected_but_finished_was_live_is_allowed(self):
        with self.assertRaisesRegex(RuntimeError, "Live streams and upcoming"):
            self.construct({"id": "live", "is_live": True, "live_status": "is_live"})
        with self.assertRaisesRegex(RuntimeError, "Live streams and upcoming"):
            self.construct({"id": "soon", "is_upcoming": True, "live_status": "is_upcoming"})
        downloader, _ = self.construct({"id": "recording", "is_live": False, "was_live": True, "live_status": "was_live"})
        self.assertIn("--no-playlist", downloader.opts)

    def test_legacy_downloader_does_not_probe_or_add_no_playlist(self):
        with patch.object(Download, "resolve_model_cmd", return_value=["yt-dlp"]):
            downloader = Download(args_for(single=False))
        self.assertNotIn("--no-playlist", downloader.opts)

    def test_download_progress_uses_only_exact_totals_and_keeps_output_path_detection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = Download.__new__(Download)
            callbacks = []
            downloader.debug_flag = False
            downloader.model_cmd = ["yt-dlp"]
            downloader.opts = ["https://example.test/video"]
            downloader.output_dir = Path(temp_dir)
            downloader.filepath = None
            downloader.timeout = None
            downloader.progress_callback = callbacks.append

            class FakeProcess:
                stdout = io.StringIO(
                    "__VT_PROGRESS__123\tNA\n"
                    "__VT_PROGRESS__500\t1000\n"
                    f"[download] Destination: {Path(temp_dir) / 'clip.mp4'}\n"
                )

                def wait(self, timeout=None):
                    return 0

                def kill(self):
                    pass

            with patch("utils.download_best.subprocess.Popen", return_value=FakeProcess()):
                files, code = downloader.run()

        self.assertEqual(code, 0)
        self.assertTrue(files[0].endswith("clip.mp4"))
        self.assertEqual(callbacks[0], {"completedBytes": 123, "totalBytes": None, "percent": None})
        self.assertEqual(callbacks[1], {"completedBytes": 500, "totalBytes": 1000, "percent": 50.0})

    def test_source_download_failure_is_typed_as_source_unavailable(self):
        module_path = Path(__file__).resolve().parents[1] / "whisper-gpu.py"
        spec = importlib.util.spec_from_file_location("whisper_gpu_source_error_test", module_path)
        whisper_gpu = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(whisper_gpu)

        class Args(argparse.Namespace):
            def __contains__(self, key):
                return hasattr(self, key)

        args = Args(filename="https://example.test/video", output_name="clip.mp3", output_dir=".",
                    audio_only=True, restrict_filenames=True, overwrite=True, verbose=False,
                    audio_format="mp3", web_json=False, single_source=True)

        class FakeDownload:
            def __init__(self, *_args, **_kwargs):
                pass

            def run(self):
                raise OSError("network unavailable")

        with patch.object(whisper_gpu.validators, "url", return_value=True), \
                patch.object(whisper_gpu, "Download", FakeDownload):
            with self.assertRaises(TaskFailure) as raised:
                whisper_gpu.add_media_files(args, [])

        self.assertEqual(raised.exception.code, "source_unavailable")
        self.assertEqual(raised.exception.stage, "source")


if __name__ == "__main__":
    unittest.main()
