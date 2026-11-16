import threading
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import services.meeting_screen as screen
from services.meeting_screen import _PrefetchedFrames


class PrefetchedFramesTests(unittest.TestCase):
    def test_preserves_frame_order_metadata_and_variable_native_sizes(self):
        frames = [
            (0.0, np.zeros((8, 12, 3), dtype=np.uint8)),
            (1.0, np.ones((17, 9, 3), dtype=np.uint8)),
            (None, {"decode_complete": False, "error": "fixture end"}),
        ]
        prefetched = _PrefetchedFrames(iter(frames), capacity=2)
        self.assertEqual(prefetched._queue.maxsize, 2)
        actual = list(prefetched)
        self.assertEqual(len(actual), len(frames))
        self.assertIs(actual[0][1], frames[0][1])
        self.assertIs(actual[1][1], frames[1][1])
        self.assertEqual([actual[0][1].shape, actual[1][1].shape], [(8, 12, 3), (17, 9, 3)])
        self.assertEqual(actual[2], frames[2])

    def test_propagates_original_iterator_exception(self):
        failure = RuntimeError("decoder fixture failure")

        def source():
            yield 0.0, np.zeros((2, 3, 3), dtype=np.uint8)
            raise failure

        prefetched = _PrefetchedFrames(source())
        self.assertEqual(next(prefetched)[0], 0.0)
        with self.assertRaises(RuntimeError) as caught:
            next(prefetched)
        self.assertIs(caught.exception, failure)

    def test_early_close_releases_backpressured_producer_and_source(self):
        produced = threading.Event()
        closed = threading.Event()
        count = [0]

        def source():
            try:
                for index in range(100):
                    count[0] += 1
                    if count[0] >= 3:
                        produced.set()
                    yield float(index), np.zeros((3, 4, 3), dtype=np.uint8)
            finally:
                closed.set()

        prefetched = _PrefetchedFrames(source(), capacity=2)
        self.assertTrue(produced.wait(1), "producer did not reach queue backpressure")
        prefetched.close()
        self.assertTrue(closed.wait(1), "producer did not close its source iterator")
        self.assertFalse(prefetched._thread.is_alive())
        self.assertLessEqual(count[0], 4)
        self.assertEqual(prefetched._queue.qsize(), 0)
        with self.assertRaises(StopIteration):
            next(prefetched)

    def test_external_close_releases_consumer_waiting_on_empty_queue(self):
        release_source = threading.Event()
        source_entered = threading.Event()
        source_closed = threading.Event()
        consumer_waiting = threading.Event()
        consumer_done = threading.Event()

        def source():
            try:
                source_entered.set()
                release_source.wait()
                yield 0.0, np.zeros((2, 2, 3), dtype=np.uint8)
            finally:
                source_closed.set()

        prefetched = _PrefetchedFrames(source())
        original_get = prefetched._queue.get

        def observed_get(*args, **kwargs):
            consumer_waiting.set()
            return original_get(*args, **kwargs)

        prefetched._queue.get = observed_get
        consumer = threading.Thread(target=lambda: self._expect_end(prefetched, consumer_done))
        closer = threading.Thread(target=prefetched.close)
        try:
            self.assertTrue(source_entered.wait(1))
            consumer.start()
            self.assertTrue(consumer_waiting.wait(1))
            closer.start()
            self.assertTrue(consumer_done.wait(1), "consumer remained blocked after close")
        finally:
            release_source.set()
            prefetched.close()
            consumer.join(timeout=1)
            closer.join(timeout=1)
        self.assertFalse(consumer.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertTrue(source_closed.wait(1))
        self.assertFalse(prefetched._thread.is_alive())

    def test_extract_screen_keeps_prefetch_opt_in(self):
        frame = np.zeros((12, 20, 3), dtype=np.uint8)
        terminal = {"returncode": 0, "coverage_seconds": 1.0, "decode_complete": True}
        for prefetch, expected_calls in ((0, 0), (2, 1)):
            with self.subTest(prefetch=prefetch), tempfile.TemporaryDirectory() as temp:
                source_frames = iter([(0.0, frame), (None, terminal)])
                with patch.object(screen, "probe_media", return_value={"duration": 1, "video": [{}]}), \
                        patch.object(screen, "_exe", return_value="ffprobe"), \
                        patch.object(screen, "_iter_video_frames", return_value=source_frames), \
                        patch.object(screen, "_load_ocr", return_value=lambda _image: []), \
                        patch.object(screen, "_PrefetchedFrames", wraps=_PrefetchedFrames) as prefetch_ctor:
                    result = screen.extract_screen("fake.mp4", temp, prefetch_frames=prefetch)
                self.assertEqual(prefetch_ctor.call_count, expected_calls)
                self.assertEqual(len(result["events"]), 1)

    @staticmethod
    def _expect_end(prefetched, finished):
        try:
            next(prefetched)
        except StopIteration:
            pass
        finally:
            finished.set()


if __name__ == "__main__":
    unittest.main()
