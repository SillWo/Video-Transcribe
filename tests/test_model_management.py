import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from huggingface_hub.errors import CacheNotFound

from services.model_management import get_model_panel_payload, list_models_with_status
from web_api import app


def build_revision(snapshot_path: Path, size_on_disk: int = 0, last_modified: float = 0.0):
    return SimpleNamespace(
        snapshot_path=snapshot_path,
        size_on_disk=size_on_disk,
        last_modified=last_modified,
    )


def build_repo(
    repo_id: str,
    repo_path: Path,
    revision: SimpleNamespace | None,
    size_on_disk: int = 0,
    last_modified: float = 0.0,
):
    revisions = (revision,) if revision is not None else ()
    return SimpleNamespace(
        repo_id=repo_id,
        repo_path=repo_path,
        revisions=revisions,
        size_on_disk=size_on_disk,
        last_modified=last_modified,
    )


class ModelManagementTests(unittest.TestCase):
    def test_no_downloaded_models_returns_not_downloaded_for_all(self):
        with patch(
            "services.model_management.scan_cache_dir",
            side_effect=CacheNotFound("missing", cache_dir=Path("missing-cache")),
        ):
            catalog = list_models_with_status()
            panel = get_model_panel_payload()

        self.assertEqual(len(catalog), 15)
        self.assertTrue(all(item["status"] == "not_downloaded" for item in catalog))
        self.assertTrue(all(item["isDownloaded"] is False for item in catalog))
        self.assertEqual(panel["summary"]["downloadedCount"], 0)
        self.assertEqual(panel["summary"]["availableCount"], 15)
        self.assertEqual(panel["summary"]["totalDownloadedSizeBytes"], 0)

    def test_downloaded_model_is_exposed_in_status_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "snapshots" / "tiny"
            snapshot_path.mkdir(parents=True)
            for file_name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
                (snapshot_path / file_name).write_text("ok", encoding="utf-8")

            repo = build_repo(
                repo_id="Systran/faster-whisper-tiny",
                repo_path=Path(temp_dir),
                revision=build_revision(snapshot_path=snapshot_path, size_on_disk=4096, last_modified=1_700_000_000),
                size_on_disk=4096,
                last_modified=1_700_000_000,
            )
            cache_info = SimpleNamespace(repos=(repo,))

            with patch("services.model_management.scan_cache_dir", return_value=cache_info):
                catalog = list_models_with_status()
                panel = get_model_panel_payload()

        tiny = next(item for item in catalog if item["id"] == "tiny")
        self.assertTrue(tiny["isDownloaded"])
        self.assertEqual(tiny["status"], "downloaded")
        self.assertEqual(tiny["cacheLocation"], str(snapshot_path))
        self.assertEqual(tiny["downloadedSizeBytes"], 4096)
        self.assertIsNotNone(tiny["lastModified"])
        self.assertEqual(panel["summary"]["downloadedCount"], 1)
        self.assertEqual(panel["summary"]["availableCount"], 15)
        self.assertEqual(panel["summary"]["totalDownloadedSizeBytes"], 4096)

    def test_incomplete_cached_repo_is_marked_unknown_without_breaking_other_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "snapshots" / "large-v3"
            snapshot_path.mkdir(parents=True)
            for file_name in ("config.json", "tokenizer.json", "vocabulary.json"):
                (snapshot_path / file_name).write_text("partial", encoding="utf-8")

            repo = build_repo(
                repo_id="Systran/faster-whisper-large-v3",
                repo_path=Path(temp_dir),
                revision=build_revision(snapshot_path=snapshot_path, size_on_disk=2048, last_modified=1_700_100_000),
                size_on_disk=2048,
                last_modified=1_700_100_000,
            )
            cache_info = SimpleNamespace(repos=(repo,))

            with patch("services.model_management.scan_cache_dir", return_value=cache_info):
                catalog = list_models_with_status()
                panel = get_model_panel_payload()

        large_v3 = next(item for item in catalog if item["id"] == "large-v3")
        self.assertEqual(large_v3["status"], "unknown")
        self.assertFalse(large_v3["isDownloaded"])
        self.assertEqual(large_v3["cacheLocation"], str(snapshot_path))
        self.assertEqual(panel["summary"]["downloadedCount"], 0)
        self.assertEqual(panel["summary"]["availableCount"], 15)
        self.assertEqual(panel["summary"]["totalDownloadedSizeBytes"], 0)
        self.assertEqual(next(item for item in catalog if item["id"] == "tiny")["status"], "not_downloaded")

    def test_panel_endpoint_returns_error_when_cache_is_unreadable(self):
        client = TestClient(app)
        with patch("services.model_management.scan_cache_dir", side_effect=RuntimeError("permission denied")):
            response = client.get("/api/models/panel")

        self.assertEqual(response.status_code, 503)
        self.assertIn("Unable to inspect Hugging Face cache", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
