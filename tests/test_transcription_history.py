import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.transcription_history import TranscriptionHistory


class TranscriptionHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "isolated-runtime"
        self.store = TranscriptionHistory(self.root)
        self.input = Path(self.temp.name) / "source.mp4"
        self.input.write_bytes(b"input")

    def tearDown(self):
        self.temp.cleanup()

    def record(self, **extra):
        metadata = {"id": "a" * 32, "sourceType": "file", "source": str(self.input),
                    "profile": "quality", "device": "cpu", "cpuThreads": 2,
                    "restorePunctuation": False, "settings": {"model": "large-v3"},
                    "artifactPaths": {"txt": "transcript.txt"}}
        metadata.update(extra)
        return self.store.create(metadata)

    def test_loads_history_from_new_instance_and_recovers_only_at_startup(self):
        self.record()
        self.store.update("a" * 32, status="running", stage="recognition")
        checkpoints = self.store.get("a" * 32)["checkpoints"]
        checkpoints[3].update(status="active", startedAt="2026-11-05T10:00:00+00:00")
        self.store.update("a" * 32, checkpoints=checkpoints)
        self.assertEqual(self.store.get("a" * 32)["status"], "running")
        reopened = TranscriptionHistory(self.root)
        self.assertEqual(reopened.get("a" * 32)["status"], "interrupted")
        self.assertEqual(reopened.get("a" * 32)["stage"], "recognition")
        recovered = reopened.get("a" * 32)["checkpoints"]
        self.assertEqual(recovered[3]["status"], "failed")
        self.assertIsNotNone(recovered[3]["completedAt"])

    def test_jobs_start_with_stable_six_stage_checkpoint_contract(self):
        record = self.record()
        self.assertEqual([item["stage"] for item in record["checkpoints"]],
                         ["preparation", "source", "audio", "recognition", "postprocess", "result"])
        self.assertTrue(all(item["status"] == "pending" for item in record["checkpoints"]))
        self.assertIsNone(record["progress"])

    def test_latest_matches_newest_history_item(self):
        self.record()
        self.store.create({"id": "b" * 32, "sourceType": "url", "source": "https://example.org/video",
                           "profile": "balanced", "device": "cpu", "cpuThreads": 2,
                           "restorePunctuation": False})
        self.assertEqual(self.store.latest(), self.store.list()[0])

    def test_text_revisions_keep_both_artifacts_in_sync_and_conflict(self):
        self.record()
        self.store.update("a" * 32, status="completed")
        self.store.write_text("a" * 32, "исходный текст", 0)
        result = self.store.replace_text("a" * 32, "новый текст", 0)
        self.assertEqual(result["revision"], 1)
        for ext in ("txt", "md"):
            self.assertEqual(self.store.artifact("a" * 32, ext).read_text(encoding="utf-8"), "новый текст")
        with self.assertRaises(PermissionError):
            self.store.replace_text("a" * 32, "устаревшая правка", 0)

    def test_metadata_never_contains_transcript_and_missing_artifact_is_missing(self):
        self.record()
        self.store.update("a" * 32, status="completed")
        self.store.write_text("a" * 32, "private transcript phrase", 0)
        db_bytes = self.store.db_path.read_bytes()
        self.assertNotIn(b"private transcript phrase", db_bytes)
        self.store.artifact("a" * 32, "txt").unlink()
        self.assertIsNone(self.store.read_text("a" * 32))

    def test_deletion_is_scoped_and_preserves_source_and_other_jobs(self):
        self.record()
        self.store.create({"id": "b" * 32, "sourceType": "file", "source": str(self.input),
                           "profile": "balanced", "device": "cpu", "cpuThreads": 1,
                           "restorePunctuation": False})
        own = self.store.job_dir("a" * 32)
        own.mkdir(exist_ok=True)
        (own / "artifact").write_text("x")
        self.store.update("a" * 32, status="failed")
        self.store.delete("a" * 32)
        self.assertFalse(own.exists())
        self.assertTrue(self.input.exists())
        self.assertIsNotNone(self.store.get("b" * 32))

    def test_store_refuses_to_delete_active_jobs(self):
        self.record()
        with self.assertRaises(PermissionError):
            self.store.delete("a" * 32)

    def test_store_refuses_to_delete_cancelling_jobs(self):
        self.record()
        self.store.update("a" * 32, status="cancelling")
        with self.assertRaises(PermissionError):
            self.store.delete("a" * 32)

    def test_connections_close_after_commit_and_exception(self):
        original_connect = sqlite3.connect
        closed = []

        class TrackingConnection(sqlite3.Connection):
            def close(connection):
                closed.append(True)
                super().close()

        def tracked_connect(*args, **kwargs):
            kwargs["factory"] = TrackingConnection
            return original_connect(*args, **kwargs)

        with patch("services.transcription_history.sqlite3.connect", side_effect=tracked_connect):
            with self.store._connect() as db:
                db.execute("SELECT 1")
            self.assertEqual(len(closed), 1)
            with self.assertRaises(RuntimeError):
                with self.store._connect():
                    raise RuntimeError("forced failure")
            self.assertEqual(len(closed), 2)


if __name__ == "__main__":
    unittest.main()
