import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from huggingface_hub import DryRunFileInfo
from huggingface_hub.utils._xet_progress_reporting import XetDownloadProgressReporter

from services.model_management import ModelDownloadValidationError, download_registered_model
from services.whisper_cpp import resolve_model_path


def dry_file(name: str, size: int, *, cached: bool = False) -> DryRunFileInfo:
    return DryRunFileInfo("commit", size, name, name, cached, not cached)


class ModelDownloadProgressTests(unittest.TestCase):
    def test_real_tqdm_update_counts_bytes_and_failed_download_never_reports_zero_or_complete(self):
        events = []
        progress_class_state = {}

        def fake_snapshot_download(_repo, **kwargs):
            if kwargs.get("dry_run"):
                return [dry_file("model.bin", 1000)]
            progress_class = kwargs["tqdm_class"]
            bar = progress_class(total=1000, unit="B", desc="model.bin")
            progress_class_state["bar"] = bar
            bar.update(500)
            progress_class_state["n"] = bar.n
            raise RuntimeError("network interrupted")

        output = io.StringIO()
        with patch("services.model_management.get_model_status", return_value={"status": "not_downloaded"}), \
                patch("services.model_management.snapshot_download", side_effect=fake_snapshot_download), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            with self.assertRaisesRegex(RuntimeError, "network interrupted"):
                download_registered_model("tiny", lambda percent, label: events.append((percent, label)))

        self.assertEqual(progress_class_state["n"], 500)
        percentages = [percent for percent, _label in events]
        self.assertIn(50, percentages)
        self.assertNotIn(0, percentages)
        self.assertNotIn(100, percentages)
        self.assertEqual(percentages, sorted(percentages))
        self.assertEqual(output.getvalue(), "")

    def test_cached_bytes_and_multi_file_resume_aggregate_only_reconstruction_progress(self):
        events = []
        call_count = 0

        def fake_snapshot_download(_repo, **kwargs):
            nonlocal call_count
            if kwargs.get("dry_run"):
                return [dry_file("cached.bin", 400, cached=True), dry_file("part-a.bin", 300), dry_file("part-b.bin", 300)]
            call_count += 1
            progress_class = kwargs["tqdm_class"]
            transfer = progress_class(total=0, unit="B", desc="Downloading bytes")
            reconstruct = progress_class(total=0, unit="B", desc="Reconstructing (incomplete total...)")
            # Hugging Face adds each file's total to both shared parent bars.
            transfer.total = (transfer.total or 0) + 600
            reconstruct.total = (reconstruct.total or 0) + 600
            reconstruct.refresh()
            # The Xet transfer bar can move while reconstructed bytes remain cached-only.
            transfer.update(500)
            self.assertEqual(events[-1][0], 40)
            # One file resumes at 100 bytes, then two files add 200 bytes each.
            reconstruct.update(100)
            reconstruct.update(200)
            reconstruct.update(200)
            return "downloaded-snapshot"

        statuses = iter([{"status": "not_downloaded"}, {"status": "downloaded", "cacheLocation": "snapshot"}])
        with patch("services.model_management.get_model_status", side_effect=lambda *_a, **_k: next(statuses)), \
                patch("services.model_management.snapshot_download", side_effect=fake_snapshot_download):
            result = download_registered_model("tiny", lambda percent, label: events.append((percent, label)))

        self.assertEqual(result["snapshotPath"], "downloaded-snapshot")
        self.assertEqual(call_count, 1)
        percentages = [percent for percent, _label in events]
        self.assertIn(90, percentages)  # 400 cached + 500 resumed/downloaded of 1000 total
        self.assertEqual(max(percentages[:-1]), 90)
        self.assertEqual(percentages, sorted(percentages))
        self.assertEqual(percentages[-1], 100)

    def test_full_byte_transfer_without_validated_snapshot_never_reports_100(self):
        events = []

        def fake_snapshot_download(_repo, **kwargs):
            if kwargs.get("dry_run"):
                return [dry_file("model.bin", 1000)]
            bar = kwargs["tqdm_class"](total=1000, unit="B", desc="Reconstructing (incomplete total...)")
            bar.update(1000)
            return "snapshot"

        with patch("services.model_management.get_model_status", return_value={"status": "not_downloaded"}), \
                patch("services.model_management.snapshot_download", side_effect=fake_snapshot_download):
            with self.assertRaises(ModelDownloadValidationError):
                download_registered_model("tiny", lambda percent, label: events.append((percent, label)))

        self.assertTrue(events)
        self.assertLess(max(percent for percent, _label in events), 100)

    def test_success_reports_100_only_after_cache_validation(self):
        sequence = []

        def get_status(*_args, **_kwargs):
            if sequence.count("validated"):
                return {"status": "downloaded", "cacheLocation": "snapshot"}
            if sequence.count("initial"):
                sequence.append("validated")
                return {"status": "downloaded", "cacheLocation": "snapshot"}
            sequence.append("initial")
            return {"status": "not_downloaded"}

        def fake_snapshot_download(_repo, **kwargs):
            if kwargs.get("dry_run"):
                return [dry_file("model.bin", 1000)]
            bar = kwargs["tqdm_class"](total=1000, unit="B", desc="Reconstructing (incomplete total...)")
            bar.update(1000)
            return "snapshot"

        with patch("services.model_management.get_model_status", side_effect=get_status), \
                patch("services.model_management.snapshot_download", side_effect=fake_snapshot_download):
            result = download_registered_model("tiny", lambda percent, _label: sequence.append(percent))

        self.assertEqual(result["status"]["status"], "downloaded")
        self.assertEqual(sequence[-2:], ["validated", 100])
        self.assertNotIn(100, sequence[:-2])


class WhisperCppModelDownloadProgressTests(unittest.TestCase):
    def test_vulkan_model_download_reports_live_percent_and_bytes(self):
        messages = []
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "ggml-tiny.bin"

            def fake_download(**kwargs):
                progress_class = kwargs["tqdm_class"]
                with XetDownloadProgressReporter(
                    reconstruction_desc="Reconstructing model",
                    total=1000,
                    log_level=0,
                    tqdm_class=progress_class,
                ) as reporter:
                    reporter.update_progress(type("Progress", (), {
                        "total_bytes": 1000,
                        "total_bytes_completed": 500,
                        "total_transfer_bytes_completed": 900,
                        "total_bytes_completion_rate": 10,
                        "total_transfer_bytes_completion_rate": 20,
                    })())
                model_path.write_bytes(b"model")
                return str(model_path)

            with patch("huggingface_hub.try_to_load_from_cache", return_value=None), \
                    patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
                resolved = resolve_model_path("tiny", progress=messages.append)

        self.assertEqual(resolved, model_path)
        self.assertTrue(any("50%" in message for message in messages), messages)
        self.assertTrue(any("500 B" in message for message in messages), messages)

    def test_cached_vulkan_model_skips_network_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cached = Path(temp_dir) / "ggml-tiny.bin"
            cached.write_bytes(b"cached")
            progress = []
            with patch("huggingface_hub.try_to_load_from_cache", return_value=str(cached)), \
                    patch("huggingface_hub.hf_hub_download") as download:
                resolved = resolve_model_path("tiny", progress=progress.append)
        self.assertEqual(resolved, cached)
        download.assert_not_called()
        self.assertEqual(progress, ["Using cached Vulkan model tiny."])


if __name__ == "__main__":
    unittest.main()
