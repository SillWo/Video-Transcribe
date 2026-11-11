import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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
        self.assertIsNone(reopened.get("a" * 32)["finishedAt"])
        self.assertIsNone(reopened.get("a" * 32)["durationSeconds"])

    def test_jobs_start_with_stable_checkpoint_contract_including_diarization(self):
        record = self.record()
        self.assertEqual([item["stage"] for item in record["checkpoints"]],
                         ["preparation", "source", "audio", "recognition", "diarization", "postprocess", "result"])
        self.assertTrue(all(item["status"] == "pending" for item in record["checkpoints"]))
        self.assertIsNone(record["progress"])
        self.assertIsNone(record["finishedAt"])
        self.assertIsNone(record["durationSeconds"])

    def test_first_terminal_transition_freezes_timing_across_edits_and_updates(self):
        self.record()
        self.store.update("a" * 32, status="completed")
        first = self.store.get("a" * 32)
        self.assertIsInstance(first["finishedAt"], str)
        self.assertGreaterEqual(first["durationSeconds"], 0)
        self.store.write_text("a" * 32, "initial", 0)
        edited = self.store.replace_text("a" * 32, "updated", 0)
        self.store.update("a" * 32, notice="edited after completion")
        after = self.store.get("a" * 32)
        self.assertEqual(edited["finishedAt"], first["finishedAt"])
        self.assertEqual(edited["durationSeconds"], first["durationSeconds"])
        self.assertEqual(after["finishedAt"], first["finishedAt"])
        self.assertEqual(after["durationSeconds"], first["durationSeconds"])

    def test_failed_and_cancelled_first_terminal_transitions_get_timing(self):
        for job_id, status in (("b" * 32, "failed"), ("c" * 32, "cancelled")):
            self.store.create({"id": job_id, "sourceType": "file", "source": str(self.input),
                               "profile": "quality", "device": "cpu", "cpuThreads": 2,
                               "restorePunctuation": False})
            self.store.update(job_id, status=status)
            record = self.store.get(job_id)
            self.assertIsInstance(record["finishedAt"], str)
            self.assertGreaterEqual(record["durationSeconds"], 0)

    def test_queue_wait_and_processing_durations_are_stable_after_terminal_edits(self):
        self.record()
        started = self.store.mark_started("a" * 32)
        self.assertIsInstance(started["startedAt"], str)
        self.assertGreaterEqual(started["queueWaitSeconds"], 0)
        completed = self.store.update("a" * 32, status="completed")
        self.assertGreaterEqual(completed["processingDurationSeconds"], 0)
        timing = tuple(completed[key] for key in
                       ("startedAt", "finishedAt", "queueWaitSeconds", "processingDurationSeconds", "durationSeconds"))
        edited = self.store.update("a" * 32, revision=2)
        self.assertEqual(tuple(edited[key] for key in
                               ("startedAt", "finishedAt", "queueWaitSeconds", "processingDurationSeconds", "durationSeconds")),
                         timing)

    def test_cancel_while_still_queued_has_zero_processing_duration(self):
        self.record()
        cancelled = self.store.update("a" * 32, status="cancelled")
        self.assertIsNone(cancelled["startedAt"])
        self.assertGreaterEqual(cancelled["queueWaitSeconds"], 0)
        self.assertEqual(cancelled["processingDurationSeconds"], 0)

    def test_invalid_creation_time_keeps_terminal_duration_null(self):
        self.record()
        record = self.store.get("a" * 32)
        record["createdAt"] = "invalid timestamp"
        with self.store._connect() as db:
            db.execute("UPDATE jobs SET metadata=? WHERE id=?", (json.dumps(record), "a" * 32))
        self.store.update("a" * 32, status="failed")
        terminal = self.store.get("a" * 32)
        self.assertIsInstance(terminal["finishedAt"], str)
        self.assertIsNone(terminal["durationSeconds"])

    def test_interrupted_recovery_preserves_null_duration_for_invalid_creation_time(self):
        self.record()
        record = self.store.get("a" * 32)
        record.update(createdAt="invalid timestamp", status="running")
        with self.store._connect() as db:
            db.execute("UPDATE jobs SET status='running', metadata=? WHERE id=?",
                       (json.dumps(record), "a" * 32))
        reopened = TranscriptionHistory(self.root)
        recovered = reopened.get("a" * 32)
        self.assertEqual(recovered["status"], "interrupted")
        self.assertIsNone(recovered["finishedAt"])
        self.assertIsNone(recovered["durationSeconds"])

    def test_legacy_terminal_timing_uses_checkpoint_not_updated_at(self):
        created = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        finished = created + timedelta(seconds=41.5)
        later_updated = finished + timedelta(hours=8)
        self.record()
        record = self.store.get("a" * 32)
        record["createdAt"] = created.isoformat()
        record["status"] = "completed"
        record["updatedAt"] = later_updated.isoformat()
        record.pop("finishedAt", None)
        record.pop("durationSeconds", None)
        record["checkpoints"][-1].update(status="completed", completedAt=finished.isoformat())
        with self.store._connect() as db:
            db.execute("UPDATE jobs SET status='completed', metadata=? WHERE id=?",
                       (json.dumps(record), "a" * 32))
        raw = self.store.get("a" * 32)
        from services.transcription_history import normalize_terminal_timing
        normalized = normalize_terminal_timing(raw)
        self.assertEqual(normalized["finishedAt"], finished.isoformat())
        self.assertAlmostEqual(normalized["durationSeconds"], 41.5)
        self.assertNotEqual(normalized["durationSeconds"], (later_updated - created).total_seconds())

    def test_legacy_terminal_without_valid_checkpoint_does_not_use_updated_at(self):
        created = datetime(2026, 1, 2, tzinfo=timezone.utc)
        self.record()
        record = self.store.get("a" * 32)
        record.update(createdAt=created.isoformat(), status="failed",
                      updatedAt=(created + timedelta(days=3)).isoformat())
        record.pop("finishedAt", None)
        record.pop("durationSeconds", None)
        for checkpoint in record["checkpoints"]:
            checkpoint["completedAt"] = "not-a-date"
        with self.store._connect() as db:
            db.execute("UPDATE jobs SET status='failed', metadata=? WHERE id=?",
                       (json.dumps(record), "a" * 32))
        from services.transcription_history import normalize_terminal_timing
        normalized = normalize_terminal_timing(self.store.get("a" * 32))
        self.assertIsNone(normalized["finishedAt"])
        self.assertIsNone(normalized["durationSeconds"])

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
