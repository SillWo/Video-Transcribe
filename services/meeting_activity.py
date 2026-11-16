"""Best-effort, scoped system-awake requests for active meeting work."""
from __future__ import annotations

import os
import threading
from collections.abc import Callable


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _windows_power_request(active: bool) -> None:
    """Set or clear the system execution state; never request display power."""
    if os.name != "nt":
        return
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_state = kernel32.SetThreadExecutionState
    set_state.argtypes = [ctypes.c_uint]
    set_state.restype = ctypes.c_uint
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if active else 0)
    if not set_state(flags):
        raise ctypes.WinError(ctypes.get_last_error())


class MeetingActivity:
    """Serialize Windows power API calls on one lazily started daemon thread.

    Each key represents one logical holder. Repeated acquisition of a key is
    idempotent, so polling recorder status cannot inflate the holder count.
    """

    def __init__(self, backend: Callable[[bool], None] | None = None):
        self._backend = backend or _windows_power_request
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._holders: set[str] = set()
        self._thread: threading.Thread | None = None

    def acquire(self, key: str) -> None:
        with self._lock:
            self._holders.add(key)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="meeting-activity", daemon=True)
                self._thread.start()
            self._wake.set()

    def release(self, key: str) -> None:
        with self._lock:
            self._holders.discard(key)
            self._wake.set()

    @property
    def active_keys(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._holders)

    def _run(self) -> None:
        applied = False
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                desired = bool(self._holders)
            if desired != applied:
                try:
                    self._backend(desired)
                except Exception:
                    # Power management is optional and must never fail a job.
                    pass
                applied = desired
            with self._lock:
                if not self._holders and not applied:
                    # Clear the thread reference while holding the same lock
                    # used by acquire(), avoiding a lost wakeup at shutdown.
                    self._thread = None
                    return


meeting_activity = MeetingActivity()


def hold_activity(key: str) -> None:
    meeting_activity.acquire(key)


def release_activity(key: str) -> None:
    meeting_activity.release(key)
