import os
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from services.meeting_archive import MeetingArchive
from services import meeting_backup as backup
from services import meeting_backup_api as backup_api


class MeetingBackupTests(unittest.TestCase):
    def test_backup_api_import_is_lightweight_and_exposes_expected_routes(self):
        probe = subprocess.run(
            [sys.executable, "-c",
             "import sys; from services.meeting_backup_api import router; "
             "print('numpy' in sys.modules, 'cv2' in sys.modules, 'onnxruntime' in sys.modules); "
             "print(sorted((route.path, tuple(sorted(route.methods or []))) for route in router.routes))"],
            cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=20, check=False)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        rows = probe.stdout.splitlines()
        self.assertEqual(rows[0], "False False False")
        self.assertIn("/api/meetings/backup/cleanup/run", rows[1])
        self.assertIn("/api/meetings/backup/restore-test", rows[1])

    def make_archive(self, base: Path, *, ready: bool = True, old_original: bool = True):
        archive_root = base / "Архив встреч"
        source_folder = base / "Исходные записи"
        source_folder.mkdir(parents=True, exist_ok=True)
        source = source_folder / "Встреча №1.mp4"
        source.write_bytes(b"synthetic original video bytes")
        if old_original:
            old = (datetime.now(timezone.utc) - timedelta(days=45)).timestamp()
            os.utime(source, (old, old))
        archive = MeetingArchive(archive_root)
        meta = archive.import_file(source, title="Учебная встреча")
        meeting_id = meta["meeting_id"]
        if ready:
            archive.write_transcript(meeting_id, [{"id": "seg-1", "start": 0, "end": 1,
                                                   "speaker": "spk-1", "text": "Проверка архива."}],
                                     raw_text="Проверка архива.")
            archive.write_screen_events(meeting_id, [{"id": "screen-1", "start": 0, "end": 1,
                                                      "frame": "frames/screen.jpg", "text": "Jira"}],
                                        status="ready")
            folder = archive.meeting_dir(meeting_id)
            (folder / "audio").mkdir(parents=True, exist_ok=True)
            (folder / "audio" / "audio-01.opus").write_bytes(b"retained compressed audio")
        archive.finalize(meeting_id)
        return archive_root, source, archive, meeting_id

    def policy(self, backup_root: Path, *, enabled: bool = True, days: int = 30):
        return {"archive_root": "unused", "backup_root": str(backup_root),
                "auto_cleanup": enabled, "original_retention_days": days}

    def test_backup_verify_restore_test_and_restore_keep_unicode_paths_and_stable_id(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root, source, archive, meeting_id = self.make_archive(base)
            backup_root = base / "Резервные копии"
            created = backup.create_backup(archive_root, backup_root)
            receipt = backup._load_receipt(backup_root, created["snapshot_id"])
            self.assertTrue(receipt["verified"])
            self.assertFalse((archive.meeting_dir(meeting_id) / "receipts").exists())
            verified = backup.verify_backup(backup_root, created["snapshot_id"])
            self.assertTrue(verified["verified"])
            tested = backup.restore_test(archive_root, backup_root, created["snapshot_id"])
            self.assertTrue(tested["passed"])
            restored_root = base / "Восстановленный архив"
            restored = backup.restore_backup(archive_root, backup_root, created["snapshot_id"], restored_root)
            self.assertTrue(restored["verified"])
            reopened = MeetingArchive(restored_root, read_only=True)
            self.assertEqual([row["meeting_id"] for row in reopened.list_meetings()], [meeting_id])
            self.assertEqual(reopened.original_path(meeting_id).read_bytes(), source.read_bytes())
            self.assertEqual(backup._entries_fingerprint(backup._current_entries(archive_root)),
                             backup._entries_fingerprint(backup._current_entries(restored_root)))

    def test_restore_refuses_nonempty_target_without_clobbering(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root, _source, _archive, _mid = self.make_archive(base)
            backup_root = base / "backup"
            created = backup.create_backup(archive_root, backup_root)
            target = base / "existing"
            target.mkdir()
            marker = target / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(backup.BackupError, "только в новую или пустую"):
                backup.restore_backup(archive_root, backup_root, created["snapshot_id"], target)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_second_backup_reuses_snapshot_files_but_never_links_live_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root, _source, archive, meeting_id = self.make_archive(base)
            backup_root = base / "backup"
            first = backup.create_backup(archive_root, backup_root)
            relative = f"meetings/{meeting_id}/meeting.md"
            first_file = backup_root / "snapshots" / first["snapshot_id"] / "data" / relative
            live_file = archive.meeting_dir(meeting_id) / "meeting.md"
            original_bytes = live_file.read_bytes()
            second = backup.create_backup(archive_root, backup_root)
            second_file = backup_root / "snapshots" / second["snapshot_id"] / "data" / relative
            self.assertGreater(second["reused_files"], 0)
            try:
                self.assertTrue(os.path.samefile(first_file, second_file))
                self.assertFalse(os.path.samefile(live_file, second_file))
            except OSError as exc:
                self.skipTest(f"Hard links are unavailable: {exc}")
            live_file.write_bytes(b"changed after both snapshots")
            self.assertEqual(first_file.read_bytes(), original_bytes)
            self.assertEqual(second_file.read_bytes(), original_bytes)
            self.assertTrue(backup.verify_backup(backup_root, second["snapshot_id"])["verified"])

    def test_corrupt_old_snapshot_candidate_falls_back_to_current_file_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root, _source, archive, meeting_id = self.make_archive(base)
            backup_root = base / "backup"
            first = backup.create_backup(archive_root, backup_root)
            relative = f"meetings/{meeting_id}/meeting.md"
            first_file = backup_root / "snapshots" / first["snapshot_id"] / "data" / relative
            first_file.write_bytes(b"corrupt old snapshot candidate")
            live_file = archive.meeting_dir(meeting_id) / "meeting.md"
            expected = live_file.read_bytes()
            second = backup.create_backup(archive_root, backup_root)
            second_file = backup_root / "snapshots" / second["snapshot_id"] / "data" / relative
            self.assertEqual(second_file.read_bytes(), expected)
            self.assertFalse(os.path.samefile(first_file, second_file))
            self.assertTrue(backup.verify_backup(backup_root, second["snapshot_id"])["verified"])

    def test_backup_root_inside_archive_and_symlink_roots_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root, _source, _archive, _mid = self.make_archive(base)
            with self.assertRaisesRegex(backup.BackupError, "вне архива"):
                backup.create_backup(archive_root, archive_root / "backup")
            outside = base / "outside"
            outside.mkdir()
            link = base / "linked-backup"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"Directory symlinks unavailable: {exc}")
            with self.assertRaises(backup.BackupError):
                backup.create_backup(archive_root, link)

    def test_interrupted_copy_cleans_staging_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root, source, archive, meeting_id = self.make_archive(base)
            original = archive.original_path(meeting_id)
            backup_root = base / "backup"
            with patch.object(backup.shutil, "copy2", side_effect=OSError("simulated disk interruption")):
                with self.assertRaisesRegex(OSError, "interruption"):
                    backup.create_backup(archive_root, backup_root)
            self.assertTrue(original.is_file())
            self.assertEqual(original.read_bytes(), source.read_bytes())
            self.assertFalse(list(backup_root.glob(".staging-*")))
            self.assertFalse(list((backup_root / "snapshots").glob("*")) if (backup_root / "snapshots").exists() else [])

    def test_corrupt_snapshot_blocks_cleanup_even_if_receipt_was_previously_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root, _source, archive, meeting_id = self.make_archive(base)
            backup_root = base / "backup"
            created = backup.create_backup(archive_root, backup_root)
            backup.restore_test(archive_root, backup_root, created["snapshot_id"])
            manifest = backup._read_manifest(backup_root / "snapshots" / created["snapshot_id"])
            data_file = backup_root / "snapshots" / created["snapshot_id"] / "data" / manifest["files"][0]["path"]
            data_file.write_bytes(data_file.read_bytes() + b"corruption")
            with patch.object(backup, "_policy", return_value=self.policy(backup_root)):
                preview = backup.cleanup_preview(archive_root, now=datetime.now(timezone.utc))
            item = next(row for row in preview["items"] if row["meeting_id"] == meeting_id)
            self.assertFalse(item["eligible"])
            self.assertTrue(item["blocked_reasons"])
            self.assertIsNotNone(archive.original_path(meeting_id))

    def test_archive_edit_after_backup_blocks_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root, _source, archive, meeting_id = self.make_archive(base)
            backup_root = base / "backup"
            created = backup.create_backup(archive_root, backup_root)
            backup.restore_test(archive_root, backup_root, created["snapshot_id"])
            (archive.meeting_dir(meeting_id) / "meeting.md").write_text("edited after backup", encoding="utf-8")
            with patch.object(backup, "_policy", return_value=self.policy(backup_root)):
                preview = backup.cleanup_preview(archive_root, now=datetime.now(timezone.utc))
            item = next(row for row in preview["items"] if row["meeting_id"] == meeting_id)
            self.assertFalse(item["eligible"])
            self.assertTrue(item["blocked_reasons"])
            self.assertIsNotNone(archive.original_path(meeting_id))

    def test_incomplete_archive_blocks_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root, _source, archive, meeting_id = self.make_archive(base, ready=False)
            backup_root = base / "backup"
            created = backup.create_backup(archive_root, backup_root)
            backup.restore_test(archive_root, backup_root, created["snapshot_id"])
            with patch.object(backup, "_policy", return_value=self.policy(backup_root)):
                preview = backup.cleanup_preview(archive_root, now=datetime.now(timezone.utc))
            item = next(row for row in preview["items"] if row["meeting_id"] == meeting_id)
            self.assertFalse(item["eligible"])
            self.assertTrue(any("стенограмма" in reason or "экран" in reason or "сжатый звук" in reason
                                for reason in item["blocked_reasons"]))
            self.assertIsNotNone(archive.original_path(meeting_id))

    def test_cleanup_removes_only_verified_old_original_and_keeps_compact_meeting(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root, user_source, archive, meeting_id = self.make_archive(base)
            backup_root = base / "backup"
            created = backup.create_backup(archive_root, backup_root)
            backup.restore_test(archive_root, backup_root, created["snapshot_id"])
            policy = self.policy(backup_root, enabled=True)
            with patch.object(backup, "_policy", return_value=policy):
                preview = backup.cleanup_preview(archive_root, now=datetime.now(timezone.utc))
                item = next(row for row in preview["items"] if row["meeting_id"] == meeting_id)
                self.assertTrue(item["eligible"], item["blocked_reasons"])
                result = backup.run_cleanup(archive_root, now=datetime.now(timezone.utc))
            self.assertEqual(result["removed_meeting_ids"], [meeting_id])
            self.assertFalse(result["compact_archive_deleted"])
            self.assertIsNone(archive.original_path(meeting_id))
            self.assertTrue(archive.meeting_dir(meeting_id).is_dir())
            self.assertTrue(user_source.is_file())
            self.assertFalse(archive.get_meeting(meeting_id)["source_available"])

    def test_cleanup_stays_blocked_when_auto_cleanup_setting_is_off(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root, _source, archive, meeting_id = self.make_archive(base)
            backup_root = base / "backup"
            created = backup.create_backup(archive_root, backup_root)
            backup.restore_test(archive_root, backup_root, created["snapshot_id"])
            with patch.object(backup, "_policy", return_value=self.policy(backup_root, enabled=False)):
                preview = backup.cleanup_preview(archive_root, now=datetime.now(timezone.utc))
                with self.assertRaisesRegex(backup.BackupError, "не включена"):
                    backup.run_cleanup(archive_root, now=datetime.now(timezone.utc))
            item = next(row for row in preview["items"] if row["meeting_id"] == meeting_id)
            self.assertFalse(item["eligible"])
            self.assertTrue(any("выключена" in reason for reason in item["blocked_reasons"]))
            self.assertIsNotNone(archive.original_path(meeting_id))

    def test_external_source_path_is_listed_but_never_selected_for_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root, source, archive, meeting_id = self.make_archive(base)
            meta_path = archive.meeting_dir(meeting_id) / "meeting.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["original_rel"] = str(source.resolve())
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            with patch.object(backup, "_policy", return_value={"backup_root": None,
                                                                 "auto_cleanup": True,
                                                                 "original_retention_days": 30}):
                preview = backup.cleanup_preview(archive_root, now=datetime.now(timezone.utc))
            item = next(row for row in preview["items"] if row["meeting_id"] == meeting_id)
            self.assertFalse(item["eligible"])
            self.assertTrue(item["blocked_reasons"])
            self.assertTrue(source.is_file())
            self.assertTrue((archive_root / "originals" / meeting_id / source.name).is_file())

    def test_auto_cleanup_off_starts_no_maintenance_thread_or_work(self):
        called = []
        settings = {"auto_cleanup": False, "archive_root": "unused", "backup_root": None}
        with patch.object(backup_api, "_settings", return_value=settings), \
                patch.object(backup, "run_cleanup") as run_cleanup:
            controller = backup_api.start_cleanup_maintenance(
                activity_probe=lambda: called.append("probe") or {"queue_active": False, "recording": False})
            tick = backup_api.maintenance_tick(activity_probe=lambda: called.append("tick"))
        self.assertIsNone(controller)
        self.assertEqual(tick["status"], "disabled")
        self.assertEqual(called, [])
        run_cleanup.assert_not_called()

    def test_maintenance_persists_last_attempt_and_uses_fake_clock(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root = base / "archive"
            archive_root.mkdir()
            backup_root = base / "backup"
            backup_root.mkdir()
            settings = {"auto_cleanup": True, "archive_root": str(archive_root),
                        "backup_root": str(backup_root), "original_retention_days": 30}
            first_time = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)
            activity = lambda: {"queue_active": False, "recording": False}
            with patch.object(backup_api, "_settings", return_value=settings), \
                    patch.object(backup, "run_cleanup", return_value={"removed_count": 0}) as run_cleanup:
                first = backup_api.maintenance_tick(activity_probe=activity, now=first_time)
                same_day = backup_api.maintenance_tick(activity_probe=activity, now=first_time + timedelta(hours=12))
                next_day = backup_api.maintenance_tick(activity_probe=activity, now=first_time + timedelta(days=1, seconds=1))
            self.assertEqual(first["status"], "completed")
            self.assertEqual(same_day["status"], "not_due")
            self.assertEqual(next_day["status"], "completed")
            self.assertEqual(run_cleanup.call_count, 2)
            state = json.loads((backup_root / backup_api._MAINTENANCE_STATE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(state["last_attempt_at"], next_day["last_attempt_at"])

    def test_maintenance_skips_queue_or_recording_without_recording_an_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive_root = base / "archive"
            archive_root.mkdir()
            backup_root = base / "backup"
            backup_root.mkdir()
            settings = {"auto_cleanup": True, "archive_root": str(archive_root),
                        "backup_root": str(backup_root), "original_retention_days": 30}
            with patch.object(backup_api, "_settings", return_value=settings), \
                    patch.object(backup, "run_cleanup") as run_cleanup:
                for state in ({"queue_active": True, "recording": False},
                              {"queue_active": False, "recording": True}):
                    result = backup_api.maintenance_tick(activity_probe=lambda state=state: state)
                    self.assertEqual(result["status"], "skipped")
            run_cleanup.assert_not_called()
            self.assertFalse((backup_root / backup_api._MAINTENANCE_STATE_NAME).exists())


if __name__ == "__main__":
    unittest.main()
