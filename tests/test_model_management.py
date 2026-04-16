import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from huggingface_hub import DryRunFileInfo
from huggingface_hub.errors import CacheNotFound

import web_api
from services.model_management import get_model_panel_payload, list_models_with_status
from web_api import app


def build_revision(snapshot_path: Path, size_on_disk: int = 0, last_modified: float = 0.0):
    return SimpleNamespace(
        commit_hash=snapshot_path.name,
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


def create_cached_snapshot(snapshot_path: Path, files: tuple[str, ...]) -> int:
    snapshot_path.mkdir(parents=True, exist_ok=True)
    total_size = 0
    for file_name in files:
        content = f"data:{file_name}"
        file_path = snapshot_path / file_name
        file_path.write_text(content, encoding="utf-8")
        total_size += len(content.encode("utf-8"))
    return total_size


def build_cache_info_for_repo(repo_id: str, snapshot_path: Path):
    size_on_disk = sum(path.stat().st_size for path in snapshot_path.iterdir() if path.is_file())
    repo_path = snapshot_path.parent.parent
    revision = build_revision(
        snapshot_path=snapshot_path,
        size_on_disk=size_on_disk,
        last_modified=snapshot_path.stat().st_mtime,
    )
    repo = build_repo(
        repo_id=repo_id,
        repo_path=repo_path,
        revision=revision,
        size_on_disk=size_on_disk,
        last_modified=snapshot_path.stat().st_mtime,
    )
    return SimpleNamespace(repos=(repo,))


def poll_operation_job(
    client: TestClient,
    operation_type: str,
    job_id: str,
    timeout_seconds: float = 5.0,
):
    deadline = time.time() + timeout_seconds
    latest_payload = None
    while time.time() < deadline:
        response = client.get(f"/api/models/{operation_type}/{job_id}")
        latest_payload = response.json()
        if latest_payload["status"] in {"success", "error"}:
            return latest_payload
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {operation_type} job {job_id}: {latest_payload}")


class ModelManagementTests(unittest.TestCase):
    def setUp(self):
        web_api.model_download_jobs.clear()
        web_api.model_delete_jobs.clear()
        web_api.active_model_operation = None

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
            create_cached_snapshot(
                snapshot_path,
                ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
            )

            cache_info = build_cache_info_for_repo("Systran/faster-whisper-tiny", snapshot_path)

            with patch("services.model_management.scan_cache_dir", return_value=cache_info):
                catalog = list_models_with_status()
                panel = get_model_panel_payload()

        tiny = next(item for item in catalog if item["id"] == "tiny")
        self.assertTrue(tiny["isDownloaded"])
        self.assertEqual(tiny["status"], "downloaded")
        self.assertEqual(tiny["cacheLocation"], str(snapshot_path))
        self.assertIsNotNone(tiny["lastModified"])
        self.assertEqual(panel["summary"]["downloadedCount"], 1)
        self.assertEqual(panel["summary"]["availableCount"], 15)
        self.assertGreater(panel["summary"]["totalDownloadedSizeBytes"], 0)

    def test_incomplete_cached_repo_is_marked_unknown_without_breaking_other_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "snapshots" / "large-v3"
            create_cached_snapshot(
                snapshot_path,
                ("config.json", "tokenizer.json", "vocabulary.json"),
            )

            cache_info = build_cache_info_for_repo("Systran/faster-whisper-large-v3", snapshot_path)

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

    def test_summary_deduplicates_shared_snapshot_sizes_for_alias_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "snapshots" / "large-v3"
            expected_size = create_cached_snapshot(
                snapshot_path,
                ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
            )

            cache_info = build_cache_info_for_repo("Systran/faster-whisper-large-v3", snapshot_path)

            with patch("services.model_management.scan_cache_dir", return_value=cache_info):
                panel = get_model_panel_payload()

        downloaded_items = [item for item in panel["catalog"] if item["status"] == "downloaded"]
        self.assertEqual({item["id"] for item in downloaded_items}, {"large-v3", "large"})
        self.assertEqual(panel["summary"]["downloadedCount"], 2)
        self.assertEqual(panel["summary"]["totalDownloadedSizeBytes"], expected_size)

    def test_panel_endpoint_returns_error_when_cache_is_unreadable(self):
        client = TestClient(app)
        with patch("services.model_management.scan_cache_dir", side_effect=RuntimeError("permission denied")):
            response = client.get("/api/models/panel")

        self.assertEqual(response.status_code, 503)
        self.assertIn("Unable to inspect Hugging Face cache", response.json()["detail"])

    def test_download_endpoint_downloads_missing_model_and_updates_panel(self):
        client = TestClient(app)
        repo_id = "Systran/faster-whisper-tiny"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            snapshot_path = temp_root / "models--Systran--faster-whisper-tiny" / "snapshots" / "commit"

            def fake_scan_cache_dir(*args, **kwargs):
                if snapshot_path.exists():
                    return build_cache_info_for_repo(repo_id, snapshot_path)
                raise CacheNotFound("missing", cache_dir=temp_root)

            def fake_snapshot_download(repo, *args, **kwargs):
                self.assertEqual(repo, repo_id)
                if kwargs.get("dry_run"):
                    return [
                        DryRunFileInfo("commit", 10, "config.json", str(snapshot_path / "config.json"), False, True),
                        DryRunFileInfo("commit", 20, "model.bin", str(snapshot_path / "model.bin"), False, True),
                        DryRunFileInfo("commit", 30, "tokenizer.json", str(snapshot_path / "tokenizer.json"), False, True),
                        DryRunFileInfo("commit", 15, "vocabulary.txt", str(snapshot_path / "vocabulary.txt"), False, True),
                    ]

                create_cached_snapshot(
                    snapshot_path,
                    ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
                )
                return str(snapshot_path)

            with patch("services.model_management.scan_cache_dir", side_effect=fake_scan_cache_dir), patch(
                "services.model_management.snapshot_download",
                side_effect=fake_snapshot_download,
            ):
                response = client.post("/api/models/download", json={"modelId": "tiny"})
                self.assertEqual(response.status_code, 200)
                started_job = response.json()
                self.assertEqual(started_job["modelId"], "tiny")
                finished_job = poll_operation_job(client, "download", started_job["jobId"])
                self.assertEqual(finished_job["status"], "success")

                panel = client.get("/api/models/panel").json()
                tiny = next(item for item in panel["catalog"] if item["id"] == "tiny")
                self.assertEqual(tiny["status"], "downloaded")
                self.assertTrue(tiny["isDownloaded"])
                self.assertEqual(panel["summary"]["downloadedCount"], 1)
                self.assertGreater(panel["summary"]["totalDownloadedSizeBytes"], 0)

    def test_download_endpoint_returns_success_without_redownloading_existing_model(self):
        client = TestClient(app)
        repo_id = "Systran/faster-whisper-tiny"

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "models--Systran--faster-whisper-tiny" / "snapshots" / "commit"
            create_cached_snapshot(
                snapshot_path,
                ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
            )
            cache_info = build_cache_info_for_repo(repo_id, snapshot_path)
            snapshot_mock = Mock()

            with patch("services.model_management.scan_cache_dir", return_value=cache_info), patch(
                "services.model_management.snapshot_download",
                snapshot_mock,
            ):
                response = client.post("/api/models/download", json={"modelId": "tiny"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("already exists", payload["message"])
        snapshot_mock.assert_not_called()

    def test_download_endpoint_reports_job_error_when_download_fails(self):
        client = TestClient(app)
        temp_root = Path(tempfile.mkdtemp())

        def fake_scan_cache_dir(*args, **kwargs):
            raise CacheNotFound("missing", cache_dir=temp_root)

        def fake_snapshot_download(repo, *args, **kwargs):
            if kwargs.get("dry_run"):
                return [
                    DryRunFileInfo("commit", 10, "config.json", "config.json", False, True),
                    DryRunFileInfo("commit", 20, "model.bin", "model.bin", False, True),
                ]
            raise RuntimeError("network down")

        with patch("services.model_management.scan_cache_dir", side_effect=fake_scan_cache_dir), patch(
            "services.model_management.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            response = client.post("/api/models/download", json={"modelId": "tiny"})
            self.assertEqual(response.status_code, 200)
            finished_job = poll_operation_job(client, "download", response.json()["jobId"])

        self.assertEqual(finished_job["status"], "error")
        self.assertIn("network down", finished_job["error"])

    def test_download_endpoint_rejects_second_active_download(self):
        client = TestClient(app)
        release_download = threading.Event()
        repo_map = {
            "tiny": Path(tempfile.mkdtemp()) / "models--Systran--faster-whisper-tiny" / "snapshots" / "commit",
            "small": Path(tempfile.mkdtemp()) / "models--Systran--faster-whisper-small" / "snapshots" / "commit",
        }

        def fake_scan_cache_dir(*args, **kwargs):
            repos = []
            for repo_id, snapshot_path in (
                ("Systran/faster-whisper-tiny", repo_map["tiny"]),
                ("Systran/faster-whisper-small", repo_map["small"]),
            ):
                if snapshot_path.exists():
                    repos.append(build_cache_info_for_repo(repo_id, snapshot_path).repos[0])

            if not repos:
                raise CacheNotFound("missing", cache_dir=Path(tempfile.gettempdir()))

            return SimpleNamespace(repos=tuple(repos))

        def fake_snapshot_download(repo, *args, **kwargs):
            if kwargs.get("dry_run"):
                return [
                    DryRunFileInfo("commit", 10, "config.json", "config.json", False, True),
                    DryRunFileInfo("commit", 20, "model.bin", "model.bin", False, True),
                    DryRunFileInfo("commit", 30, "tokenizer.json", "tokenizer.json", False, True),
                    DryRunFileInfo("commit", 15, "vocabulary.txt", "vocabulary.txt", False, True),
                ]

            release_download.wait(timeout=2)
            target_snapshot = repo_map["tiny"] if repo.endswith("tiny") else repo_map["small"]
            create_cached_snapshot(
                target_snapshot,
                ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
            )
            return str(target_snapshot)

        with patch("services.model_management.scan_cache_dir", side_effect=fake_scan_cache_dir), patch(
            "services.model_management.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            first_response = client.post("/api/models/download", json={"modelId": "tiny"})
            self.assertEqual(first_response.status_code, 200)

            second_response = client.post("/api/models/download", json={"modelId": "small"})
            self.assertEqual(second_response.status_code, 409)
            self.assertIn("already in progress", second_response.json()["detail"])

            release_download.set()
            poll_operation_job(client, "download", first_response.json()["jobId"])

    def test_delete_endpoint_removes_downloaded_model_and_updates_panel(self):
        client = TestClient(app)
        repo_id = "Systran/faster-whisper-tiny"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            snapshot_path = temp_root / "models--Systran--faster-whisper-tiny" / "snapshots" / "commit"
            create_cached_snapshot(
                snapshot_path,
                ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
            )

            class FakeDeleteStrategy:
                def __init__(self, target_snapshot: Path):
                    self.target_snapshot = target_snapshot

                def execute(self):
                    if self.target_snapshot.parent.parent.exists():
                        import shutil

                        shutil.rmtree(self.target_snapshot.parent.parent)

            class FakeCacheInfo:
                def __init__(self, repo_snapshot: Path):
                    self.repos = (build_cache_info_for_repo(repo_id, repo_snapshot).repos[0],)

                def delete_revisions(self, *revisions: str):
                    self.revisions = revisions
                    return FakeDeleteStrategy(snapshot_path)

            def fake_scan_cache_dir(*args, **kwargs):
                if snapshot_path.exists():
                    return FakeCacheInfo(snapshot_path)
                raise CacheNotFound("missing", cache_dir=temp_root)

            with patch("services.model_management.scan_cache_dir", side_effect=fake_scan_cache_dir):
                response = client.post("/api/models/delete", json={"modelId": "tiny"})
                self.assertEqual(response.status_code, 200)
                started_job = response.json()
                self.assertEqual(started_job["modelId"], "tiny")
                finished_job = poll_operation_job(client, "delete", started_job["jobId"])
                self.assertEqual(finished_job["status"], "success")

                panel = client.get("/api/models/panel").json()
                tiny = next(item for item in panel["catalog"] if item["id"] == "tiny")
                self.assertEqual(tiny["status"], "not_downloaded")
                self.assertFalse(tiny["isDownloaded"])
                self.assertIsNone(tiny["cacheLocation"])
                self.assertIsNone(tiny["downloadedSizeBytes"])
                self.assertEqual(panel["summary"]["downloadedCount"], 0)

    def test_delete_endpoint_reports_when_model_is_not_present(self):
        client = TestClient(app)

        with patch(
            "services.model_management.scan_cache_dir",
            side_effect=CacheNotFound("missing", cache_dir=Path("missing-cache")),
        ):
            response = client.post("/api/models/delete", json={"modelId": "tiny"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("Nothing was removed", payload["message"])

    def test_delete_endpoint_rejects_when_another_operation_is_active(self):
        client = TestClient(app)
        release_download = threading.Event()
        repo_map = {
            "tiny": Path(tempfile.mkdtemp()) / "models--Systran--faster-whisper-tiny" / "snapshots" / "commit",
            "small": Path(tempfile.mkdtemp()) / "models--Systran--faster-whisper-small" / "snapshots" / "commit",
        }

        def fake_scan_cache_dir(*args, **kwargs):
            repos = []
            for repo_id, snapshot_path in (
                ("Systran/faster-whisper-tiny", repo_map["tiny"]),
                ("Systran/faster-whisper-small", repo_map["small"]),
            ):
                if snapshot_path.exists():
                    repos.append(build_cache_info_for_repo(repo_id, snapshot_path).repos[0])

            if not repos:
                raise CacheNotFound("missing", cache_dir=Path(tempfile.gettempdir()))

            return SimpleNamespace(repos=tuple(repos))

        def fake_snapshot_download(repo, *args, **kwargs):
            if kwargs.get("dry_run"):
                return [
                    DryRunFileInfo("commit", 10, "config.json", "config.json", False, True),
                    DryRunFileInfo("commit", 20, "model.bin", "model.bin", False, True),
                    DryRunFileInfo("commit", 30, "tokenizer.json", "tokenizer.json", False, True),
                    DryRunFileInfo("commit", 15, "vocabulary.txt", "vocabulary.txt", False, True),
                ]

            release_download.wait(timeout=2)
            create_cached_snapshot(
                repo_map["tiny"],
                ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
            )
            return str(repo_map["tiny"])

        with patch("services.model_management.scan_cache_dir", side_effect=fake_scan_cache_dir), patch(
            "services.model_management.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            first_response = client.post("/api/models/download", json={"modelId": "tiny"})
            self.assertEqual(first_response.status_code, 200)

            second_response = client.post("/api/models/delete", json={"modelId": "small"})
            self.assertEqual(second_response.status_code, 409)
            self.assertIn("already in progress", second_response.json()["detail"])

            release_download.set()
            poll_operation_job(client, "download", first_response.json()["jobId"])


if __name__ == "__main__":
    unittest.main()
