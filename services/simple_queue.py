"""Persistent FIFO state helpers for simple transcription jobs."""

from __future__ import annotations

import threading
from typing import Callable


class SimpleQueueDispatcher:
    """One condition-driven dispatcher; queued jobs never get one thread each."""

    def __init__(self, dequeue_next: Callable[[], str | None], is_paused: Callable[[], bool],
                 run_job: Callable[[str], None], on_error: Callable[[str, Exception], None] | None = None):
        self._dequeue_next = dequeue_next
        self._is_paused = is_paused
        self._run_job = run_job
        self._on_error = on_error
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._faulted = False
        self.active_job_id: str | None = None

    def start(self) -> None:
        with self._condition:
            self._stopping = False
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="simple-queue-dispatcher", daemon=True)
                self._thread.start()
            self._condition.notify_all()

    def clear_failure(self) -> None:
        with self._condition:
            self._faulted = False
            self._condition.notify_all()

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _loop(self) -> None:
        while True:
            with self._condition:
                if self._stopping:
                    return
                if self._faulted:
                    self._condition.wait(timeout=0.25)
                    continue
                try:
                    if self._is_paused():
                        self._condition.wait(timeout=0.25)
                        continue
                    job_id = self._dequeue_next()
                except Exception as exc:
                    self._faulted = True
                    if self._on_error is not None:
                        try:
                            self._on_error(self.active_job_id or "", exc)
                        except Exception:
                            pass
                    self._condition.wait(timeout=0.25)
                    continue
                if job_id is None:
                    self._condition.wait(timeout=0.25)
                    continue
                self.active_job_id = job_id
            try:
                self._run_job(job_id)
            except Exception as exc:
                with self._condition:
                    self._faulted = True
                if self._on_error is not None:
                    try:
                        self._on_error(job_id, exc)
                    except Exception:
                        # The in-memory fault latch still prevents the next dequeue.
                        pass
            finally:
                with self._condition:
                    self.active_job_id = None
                    self._condition.notify_all()

    def stop(self, timeout: float = 15.0) -> bool:
        with self._condition:
            self._stopping = True
            thread = self._thread
            self._condition.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, timeout))
            if thread.is_alive():
                return False
        with self._condition:
            self._thread = None
            return True

    def reset_for_tests(self, timeout: float = 15.0) -> None:
        if not self.stop(timeout):
            raise RuntimeError("Simple queue dispatcher did not stop in time.")


class PersistentSimpleQueue:
    """Small facade over the queue tables owned by TranscriptionHistory."""

    def __init__(self, history):
        self.history = history

    def enqueue_many(self, metadata_items: list[dict]) -> list[dict]:
        return self.history.create_queued(metadata_items)

    def ids(self) -> list[str]:
        return self.history.queue_ids()

    def records(self) -> list[dict]:
        return self.history.queued_records()

    def dequeue(self) -> str | None:
        return self.history.dequeue_next()

    def mark_started(self, job_id: str):
        return self.history.mark_started(job_id)

    def remove(self, job_id: str) -> bool:
        return self.history.remove_queued(job_id)

    def position(self, job_id: str) -> int | None:
        return self.history.queue_position(job_id)

    def state(self) -> dict:
        return self.history.queue_state()

    def set_state(self, paused: bool, reason: dict | None = None) -> dict:
        return self.history.set_queue_state(paused, reason)
