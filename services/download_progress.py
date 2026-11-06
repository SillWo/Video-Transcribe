"""Quiet tqdm adapters that preserve Hugging Face byte counters."""

from __future__ import annotations

import threading
from typing import Callable

from tqdm.auto import tqdm as base_tqdm

ProgressCallback = Callable[[int, str], None]
ByteProgressCallback = Callable[[int, int | None], None]


class _NullProgressFile:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def format_bytes(value: int) -> str:
    amount = max(0, int(value))
    if amount < 1000:
        return f"{amount} B"
    units = ("kB", "MB", "GB", "TB")
    scaled = float(amount)
    for unit in units:
        scaled /= 1000
        if scaled < 1000 or unit == units[-1]:
            return f"{scaled:.1f} {unit}"
    return f"{amount} B"


def create_download_progress_tqdm(
    *,
    total_bytes: int | None = None,
    cached_bytes: int = 0,
    initial_percent: int = 10,
    callback: ProgressCallback | None = None,
    byte_callback: ByteProgressCallback | None = None,
    label: str = "Downloading model files",
):
    """Create a tqdm class with real counters and no terminal output.

    Hugging Face snapshot downloads can create separate transfer and disk
    reconstruction bars. Only reconstruction contributes to completion,
    because it includes resumed bytes and avoids counting the same transfer
    twice.
    """
    cached_bytes = max(0, int(cached_bytes))
    initial_percent = max(0, min(int(initial_percent), 99))
    state = {"highest_n": 0, "last_percent": None, "last_bucket": None, "last_bytes": 0}
    state_lock = threading.RLock()

    class SilentDownloadProgressTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            unit = kwargs.get("unit", "it")
            desc = str(kwargs.get("desc") or "").lower()
            bar_format = str(kwargs.get("bar_format") or "").lower()
            transfer_bar = (
                "downloading bytes" in desc
                or ".transfer" in str(kwargs.get("name") or "").lower()
                or ("{bar}" in bar_format and "{total_fmt" not in bar_format and unit == "B")
            )
            self._tracks_progress = unit == "B" and not transfer_bar
            self._transfer_bar = transfer_bar
            kwargs["disable"] = False
            kwargs["file"] = _NullProgressFile()
            kwargs["leave"] = False
            super().__init__(*args, **kwargs)
            self._emit_progress()

        def display(self, msg=None, pos=None):
            # Keep tqdm enabled so update() advances n, but suppress rendering.
            return None

        def _emit_progress(self):
            with state_lock:
                self._emit_progress_locked()

        def _emit_progress_locked(self):
            if not self._tracks_progress or (callback is None and byte_callback is None):
                return

            current_n = max(0, int(getattr(self, "n", 0) or 0))
            state["highest_n"] = max(state["highest_n"], current_n)
            completed = cached_bytes + state["highest_n"]
            denominator = int(total_bytes or getattr(self, "total", 0) or 0)
            actual_completed = min(completed, denominator) if denominator > 0 else completed
            if actual_completed > state["last_bytes"]:
                state["last_bytes"] = actual_completed
                if byte_callback is not None:
                    byte_callback(actual_completed, denominator or None)
            if denominator > 0:
                completed = min(completed, denominator)
                percent = max(
                    initial_percent,
                    state["last_percent"] or 0,
                    int(completed * 100 / denominator),
                )
                percent = min(99, percent)
                if percent == initial_percent and completed <= cached_bytes:
                    return
                # One callback per percentage point keeps UI updates useful and bounded.
                if state["last_percent"] == percent:
                    return
                state["last_percent"] = percent
                message = f"{label} ({percent}%: {format_bytes(completed)} / {format_bytes(denominator)})"
                if callback is not None:
                    callback(percent, message)
                return

            # If the server did not disclose a content length, report in 8 MiB
            # increments; Hugging Face's file metadata normally provides totals.
            bucket = completed // (8 * 1024 * 1024)
            if completed <= cached_bytes or state["last_bucket"] == bucket:
                return
            state["last_bucket"] = bucket
            message = f"{label}: {format_bytes(completed)} received"
            if callback is not None:
                callback(initial_percent, message)

        def update(self, n=1):
            with self.get_lock(), state_lock:
                result = super().update(n)
                self._emit_progress()
                return result

        def refresh(self, *args, **kwargs):
            with self.get_lock(), state_lock:
                result = super().refresh(*args, **kwargs)
                self._emit_progress()
                return result

        def set_description(self, desc=None, refresh=True):
            with self.get_lock(), state_lock:
                result = super().set_description(desc, refresh)
                self._emit_progress()
                return result

    return SilentDownloadProgressTqdm
