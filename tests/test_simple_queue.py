import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from services.simple_queue import PersistentSimpleQueue, SimpleQueueDispatcher
from services.transcription_history import TranscriptionHistory


class SimpleQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = TranscriptionHistory(Path(self.temp.name) / "runtime")
        self.queue = PersistentSimpleQueue(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def metadata(self, job_id, source):
        return {"id": job_id, "sourceType": "file", "source": source,
                "profile": "balanced", "device": "cpu", "cpuThreads": 2,
                "restorePunctuation": False}

    def test_fifo_preserves_duplicates_and_append_while_dispatching_with_one_worker(self):
        self.queue.enqueue_many([self.metadata("a" * 32, "same.wav"),
                                 self.metadata("b" * 32, "same.wav")])
        entered = threading.Event()
        release = threading.Event()
        seen = []
        active = 0
        maximum = 0
        lock = threading.Lock()

        def run(job_id):
            nonlocal active, maximum
            self.store.mark_started(job_id)
            with lock:
                active += 1
                maximum = max(maximum, active)
                seen.append(job_id)
            if len(seen) == 1:
                entered.set()
                release.wait(2)
            with lock:
                active -= 1
            self.store.update(job_id, status="completed")

        dispatcher = SimpleQueueDispatcher(self.queue.dequeue, lambda: self.queue.state()["paused"], run)
        dispatcher.start()
        self.assertTrue(entered.wait(1))
        self.queue.enqueue_many([self.metadata("c" * 32, "third.wav")])
        release.set()
        deadline = time.time() + 2
        while len(seen) < 3 and time.time() < deadline:
            time.sleep(0.01)
        dispatcher.stop(2)
        self.assertEqual(seen, ["a" * 32, "b" * 32, "c" * 32])
        self.assertEqual(maximum, 1)
        self.assertEqual(self.store.queue_ids(), [])

    def test_group_insert_rolls_back_history_and_fifo_together(self):
        import services.transcription_history as history_module
        original = history_module.json.dumps
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected serialization failure")
            return original(*args, **kwargs)

        with patch.object(history_module.json, "dumps", side_effect=fail_second):
            with self.assertRaises(OSError):
                self.queue.enqueue_many([self.metadata("d" * 32, "one.wav"),
                                         self.metadata("e" * 32, "two.wav")])
        self.assertEqual(self.store.list(), [])
        self.assertEqual(self.queue.ids(), [])

    def test_restart_keeps_waiting_fifo_and_pauses_it_but_interrupts_started_job(self):
        self.queue.enqueue_many([self.metadata("f" * 32, "active.wav"),
                                 self.metadata("1" * 32, "waiting.wav")])
        self.assertIsNotNone(self.store.mark_started("f" * 32))
        reopened = TranscriptionHistory(Path(self.temp.name) / "runtime")
        self.assertEqual(reopened.get("f" * 32)["status"], "interrupted")
        self.assertEqual(reopened.queue_ids(), ["1" * 32])
        self.assertTrue(reopened.queue_state()["paused"])
        self.assertEqual(reopened.queue_state()["pauseReason"]["code"], "recovered_pending")

    def test_restart_clears_only_stale_empty_queue_pauses(self):
        self.queue.set_state(True)
        reopened = TranscriptionHistory(Path(self.temp.name) / "runtime")
        self.assertEqual(reopened.queue_state(), {"paused": False, "pauseReason": None})
        reopened_queue = PersistentSimpleQueue(reopened)
        reopened_queue.enqueue_many([self.metadata("3" * 32, "new.wav")])
        completed = threading.Event()

        def run_new(job_id):
            reopened.mark_started(job_id)
            reopened.update(job_id, status="completed")
            completed.set()

        dispatcher = SimpleQueueDispatcher(reopened_queue.dequeue,
                                           lambda: reopened_queue.state()["paused"], run_new)
        dispatcher.start()
        self.assertTrue(completed.wait(1))
        dispatcher.stop(1)

        reopened.set_queue_state(True, {"code": "recovered_pending", "message": "old recovered queue"})
        reopened_again = TranscriptionHistory(Path(self.temp.name) / "runtime")
        self.assertEqual(reopened_again.queue_state(), {"paused": False, "pauseReason": None})

        reopened_again.set_queue_state(True, {"code": "device_error", "message": "GPU unavailable"})
        still_paused = TranscriptionHistory(Path(self.temp.name) / "runtime")
        self.assertEqual(still_paused.queue_state(),
                         {"paused": True, "pauseReason": {"code": "device_error", "message": "GPU unavailable"}})

    def test_unexpected_worker_exception_latches_dispatcher_until_explicit_resume(self):
        self.queue.enqueue_many([self.metadata("2" * 32, "retry.wav")])
        failed = threading.Event()
        attempts = 0

        def run(job_id):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("unexpected callback failure")
            self.store.mark_started(job_id)
            self.store.update(job_id, status="completed")

        dispatcher = SimpleQueueDispatcher(self.queue.dequeue, lambda: self.queue.state()["paused"],
                                           run, lambda _job_id, _exc: failed.set())
        dispatcher.start()
        self.assertTrue(failed.wait(1))
        time.sleep(0.1)
        self.assertEqual(attempts, 1)
        dispatcher.clear_failure()
        dispatcher.wake()
        deadline = time.time() + 1
        while attempts < 2 and time.time() < deadline:
            time.sleep(0.01)
        dispatcher.stop(1)
        self.assertEqual(attempts, 2)
        self.assertEqual(self.store.get("2" * 32)["status"], "completed")


if __name__ == "__main__":
    unittest.main()
