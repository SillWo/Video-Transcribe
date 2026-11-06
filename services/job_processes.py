"""Cancellation control for subprocesses owned by a transcription job."""

from __future__ import annotations

import math
import subprocess
import threading

import psutil


class JobCancelled(RuntimeError):
    """Raised when work continues after its owning job was cancelled."""


class JobProcessController:
    """Own and stop one job subprocess and its descendants safely.

    Only ``subprocess.Popen`` handles can be registered. Cancellation never
    targets arbitrary process IDs or removes job data.
    """

    _TERMINATE_WAIT_SECONDS = 1.25
    _KILL_WAIT_SECONDS = 0.35
    _CREATE_TIME_TOLERANCE = 0.01

    def __init__(self) -> None:
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._stopping_handles: set[int] = set()
        self._process: subprocess.Popen | None = None
        self._root_create_time: float | None = None
        self._pending_descendants: list[tuple[psutil.Process, float]] = []
        self._pending_tree_incomplete = False

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled("This job was cancelled.")

    def register(self, process: subprocess.Popen) -> None:
        """Register an owned subprocess, stopping it if cancellation raced launch."""
        create_time = self._capture_create_time(process)
        with self._lock:
            cancelled = self._cancel_event.is_set()
            self._process = process
            self._root_create_time = create_time
        if cancelled:
            self._stop_process_tree(process, create_time)
            self.unregister(process)
            raise JobCancelled("This job was cancelled before its process could start.")

    def unregister(self, process: subprocess.Popen) -> None:
        """Clear ownership only if this exact handle is still registered."""
        with self._lock:
            if self._process is not process or id(process) in self._stopping_handles:
                return
            pending = list(self._pending_descendants)
            pending_incomplete = self._pending_tree_incomplete
        if pending_incomplete or any(self._is_alive(child, create_time) is not False
                                      for child, create_time in pending):
            return
        with self._lock:
            if self._process is process and id(process) not in self._stopping_handles:
                self._process = None
                self._root_create_time = None
                self._pending_descendants = []
                self._pending_tree_incomplete = False

    def cancel(self) -> None:
        """Cancel the job and stop its currently owned process tree."""
        with self._lock:
            self._cancel_event.set()
            process = self._process
            create_time = self._root_create_time
        if process is not None:
            self._stop_process_tree(process, create_time)

    def stop_owned(self) -> None:
        """Stop the currently owned process tree without marking the job cancelled.

        Returns only after all known owned processes are confirmed stopped.
        Raises RuntimeError when the tree cannot be verified or stopped.
        """
        with self._lock:
            process = self._process
            create_time = self._root_create_time
        if process is not None:
            self._stop_process_tree(process, create_time)

    @staticmethod
    def _capture_create_time(process: subprocess.Popen) -> float | None:
        try:
            if process.poll() is not None:
                return None
            identity = psutil.Process(process.pid).create_time()
            # The Popen handle is authoritative for whether its child still
            # exists. Recheck after psutil lookup to avoid a reused PID.
            if process.poll() is not None:
                return None
            return float(identity)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _matches_identity(process: psutil.Process, expected: float | None) -> bool:
        if expected is None:
            return False
        try:
            actual = float(process.create_time())
            return math.isclose(actual, expected, rel_tol=0.0, abs_tol=JobProcessController._CREATE_TIME_TOLERANCE)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, AttributeError, TypeError, ValueError):
            return False

    @classmethod
    def _snapshot_descendants(cls, root: psutil.Process, root_create_time: float | None):
        """Return a deepest-first snapshot with creation-time guards."""
        if not cls._matches_identity(root, root_create_time):
            return [], False

        result: list[tuple[psutil.Process, float]] = []
        visited = {root.pid}
        complete = True

        def visit(parent: psutil.Process) -> None:
            nonlocal complete
            try:
                children = parent.children(recursive=False)
            except psutil.NoSuchProcess:
                return
            except (psutil.AccessDenied, OSError):
                complete = False
                return
            for child in children:
                if child.pid in visited:
                    continue
                visited.add(child.pid)
                try:
                    child_create_time = float(child.create_time())
                except psutil.NoSuchProcess:
                    continue
                except (psutil.AccessDenied, OSError, TypeError, ValueError):
                    complete = False
                    continue
                visit(child)
                result.append((child, child_create_time))

        visit(root)
        return result, complete

    @staticmethod
    def _is_alive(process: psutil.Process, create_time: float) -> bool | None:
        try:
            actual = float(process.create_time())
            if not math.isclose(actual, create_time, rel_tol=0.0,
                                abs_tol=JobProcessController._CREATE_TIME_TOLERANCE):
                return False
            return process.is_running()
        except psutil.NoSuchProcess:
            return False
        except (psutil.AccessDenied, OSError):
            return None

    def _stop_process_tree(self, process: subprocess.Popen, root_create_time: float | None) -> None:
        with self._stop_lock:
            with self._lock:
                self._stopping_handles.add(id(process))
            try:
                self._stop_process_tree_locked(process, root_create_time)
            finally:
                with self._lock:
                    self._stopping_handles.discard(id(process))

    def _stop_process_tree_locked(self, process: subprocess.Popen, root_create_time: float | None) -> None:
        root_alive = process.poll() is None
        with self._lock:
            pending = list(self._pending_descendants)
            pending_incomplete = self._pending_tree_incomplete
        if not root_alive and not pending:
            if pending_incomplete:
                raise RuntimeError("Cancellation was requested, but the owned process tree could not be verified.")
            return

        root_ps = None
        descendants: list[tuple[psutil.Process, float]] = pending
        tree_verified = not root_alive
        tree_complete = not pending_incomplete
        if root_alive and root_create_time is not None:
            try:
                candidate = psutil.Process(process.pid)
                if self._matches_identity(candidate, root_create_time):
                    root_ps = candidate
                    discovered, tree_complete = self._snapshot_descendants(candidate, root_create_time)
                    known = {(child.pid, create_time) for child, create_time in descendants}
                    descendants.extend((child, create_time) for child, create_time in discovered
                                       if (child.pid, create_time) not in known)
                    tree_verified = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                root_ps = None

        if root_alive and (not tree_verified or not tree_complete):
            pending_incomplete = True
        with self._lock:
            self._pending_descendants = list(descendants)
            self._pending_tree_incomplete = pending_incomplete

        # The traversal is postorder, so descendants are terminated deepest-first.
        identity_uncertain = pending_incomplete or (root_alive and (not tree_verified or not tree_complete))
        for child, create_time in descendants:
            child_alive = self._is_alive(child, create_time)
            if child_alive is None:
                identity_uncertain = True
            elif child_alive:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    pass

        if root_alive and process.poll() is None:
            try:
                process.terminate()
            except (OSError, ValueError):
                pass

        waitables = [child for child, _ in descendants]
        if root_ps is not None and root_alive:
            waitables.append(root_ps)
        if waitables:
            psutil.wait_procs(waitables, timeout=self._TERMINATE_WAIT_SECONDS)

        for child, create_time in descendants:
            child_alive = self._is_alive(child, create_time)
            if child_alive is None:
                identity_uncertain = True
            elif child_alive:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    pass

        root_alive = process.poll() is None
        if root_alive:
            try:
                process.kill()
            except (OSError, ValueError):
                pass

        alive_waitables = [child for child, create_time in descendants if self._is_alive(child, create_time) is not False]
        if root_ps is not None and root_alive:
            alive_waitables.append(root_ps)
        if alive_waitables:
            psutil.wait_procs(alive_waitables, timeout=self._KILL_WAIT_SECONDS)

        root_alive = process.poll() is None
        descendants_alive = any(self._is_alive(child, create_time) is not False for child, create_time in descendants)
        remaining = [(child, create_time) for child, create_time in descendants
                     if self._is_alive(child, create_time) is not False]
        with self._lock:
            self._pending_descendants = remaining
        if root_alive or descendants_alive or identity_uncertain:
            raise RuntimeError("Cancellation was requested, but the owned process tree could not be stopped.")
