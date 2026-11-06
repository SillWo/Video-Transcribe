import math
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import psutil

from services import job_processes


class FakePsutilProcess:
    def __init__(self, pid, create_time, *, children=(), events=None, name=None):
        self.pid = pid
        self._create_time = create_time
        self._children = list(children)
        self.events = events if events is not None else []
        self.name = name or str(pid)
        self.alive = True
        self.deny_terminate = False
        self.deny_kill = False

    def create_time(self):
        return self._create_time

    def children(self, recursive=False):
        return list(self._children)

    def terminate(self):
        self.events.append(self.name)
        if not self.deny_terminate:
            self.alive = False

    def kill(self):
        self.events.append(f"{self.name}:kill")
        if not self.deny_kill:
            self.alive = False

    def is_running(self):
        return self.alive


class FakePopen:
    def __init__(self, pid, root_process, *, ignore_terminate=False, ignore_kill=False):
        self.pid = pid
        self.root_process = root_process
        self.returncode = None
        self.events = root_process.events
        self.ignore_terminate = ignore_terminate
        self.ignore_kill = ignore_kill

    def poll(self):
        return self.returncode

    def terminate(self):
        self.events.append("root")
        if not self.ignore_terminate:
            self.returncode = 0
            self.root_process.alive = False

    def kill(self):
        self.events.append("root:kill")
        if not self.ignore_kill:
            self.returncode = -9
            self.root_process.alive = False


def fake_wait_procs(processes, timeout):
    alive = [process for process in processes if process.is_running()]
    gone = [process for process in processes if not process.is_running()]
    return gone, alive


class JobProcessControllerTests(unittest.TestCase):
    def make_fake_tree(self):
        events = []
        grandchild = FakePsutilProcess(303, 30.3, events=events, name="grandchild")
        child = FakePsutilProcess(202, 20.2, children=[grandchild], events=events, name="child")
        root = FakePsutilProcess(101, 10.1, children=[child], events=events, name="root-psutil")
        popen = FakePopen(101, root)
        processes = {101: root, 202: child, 303: grandchild}
        return popen, root, child, grandchild, processes, events

    def patch_process_table(self, processes):
        def lookup(pid):
            return processes[pid]

        return (patch.object(job_processes.psutil, "Process", side_effect=lookup),
                patch.object(job_processes.psutil, "wait_procs", side_effect=fake_wait_procs))

    def test_cancel_stops_only_owned_process_tree_deepest_first(self):
        popen, root, child, grandchild, processes, events = self.make_fake_tree()
        unrelated = FakePsutilProcess(404, 40.4, events=events, name="unrelated")
        processes[404] = unrelated
        controller = job_processes.JobProcessController()
        patches = self.patch_process_table(processes)
        with patches[0], patches[1]:
            controller.register(popen)
            controller.cancel()

        self.assertEqual(events, ["grandchild", "child", "root"])
        self.assertFalse(root.alive)
        self.assertFalse(child.alive)
        self.assertFalse(grandchild.alive)
        self.assertTrue(unrelated.alive)
        self.assertTrue(controller.cancelled)

    def test_stop_owned_stops_process_without_marking_job_cancelled(self):
        popen, root, child, grandchild, processes, _events = self.make_fake_tree()
        controller = job_processes.JobProcessController()
        with patch.object(job_processes.psutil, "Process", side_effect=processes.__getitem__), \
                patch.object(job_processes.psutil, "wait_procs", side_effect=fake_wait_procs):
            controller.register(popen)
            controller.stop_owned()

        self.assertFalse(controller.cancelled)
        self.assertEqual(popen.poll(), 0)
        self.assertFalse(root.alive)
        self.assertFalse(child.alive)
        self.assertFalse(grandchild.alive)

    def test_failed_descendant_kill_is_retained_and_retryable_after_root_exits(self):
        popen, root, child, grandchild, processes, _events = self.make_fake_tree()
        child.deny_terminate = True
        child.deny_kill = True
        controller = job_processes.JobProcessController()
        with patch.object(job_processes.psutil, "Process", side_effect=processes.__getitem__), \
                patch.object(job_processes.psutil, "wait_procs", side_effect=fake_wait_procs):
            controller.register(popen)
            with self.assertRaisesRegex(RuntimeError, "could not be stopped"):
                controller.cancel()
            self.assertEqual(popen.poll(), 0)
            self.assertTrue(child.alive)
            self.assertFalse(grandchild.alive)
            self.assertEqual(len(controller._pending_descendants), 1)
            child.deny_kill = False
            controller.stop_owned()

        self.assertFalse(child.alive)
        self.assertFalse(grandchild.alive)
        self.assertEqual(controller._pending_descendants, [])
        self.assertTrue(controller.cancelled)

    def test_cancel_before_register_stops_late_process_and_raises_job_cancelled(self):
        popen, root, child, grandchild, processes, _events = self.make_fake_tree()
        controller = job_processes.JobProcessController()
        patches = self.patch_process_table(processes)
        with patches[0], patches[1]:
            controller.cancel()
            with self.assertRaises(job_processes.JobCancelled):
                controller.register(popen)

        self.assertTrue(controller.cancelled)
        self.assertEqual(popen.poll(), 0)
        self.assertFalse(root.alive)
        self.assertFalse(child.alive)
        self.assertFalse(grandchild.alive)
        self.assertIsNone(controller._process)
        # Register retains ownership when a stop fails, so cancel can retry.

    def test_cancel_before_register_retains_handle_when_stopping_fails(self):
        popen, root, child, grandchild, processes, events = self.make_fake_tree()
        popen.ignore_terminate = True
        popen.ignore_kill = True
        root.deny_terminate = True
        root.deny_kill = True
        controller = job_processes.JobProcessController()
        timeouts = []

        def wait_procs(process_list, timeout):
            timeouts.append(timeout)
            return [], list(process_list)

        with patch.object(job_processes.psutil, "Process", side_effect=processes.__getitem__), \
                patch.object(job_processes.psutil, "wait_procs", side_effect=wait_procs):
            controller.cancel()
            with self.assertRaisesRegex(RuntimeError, "could not be stopped"):
                controller.register(popen)
            self.assertIs(controller._process, popen)
            self.assertLessEqual(sum(timeouts), 2.0)
            timeouts.clear()
            with self.assertRaisesRegex(RuntimeError, "could not be stopped"):
                controller.cancel()

        self.assertLessEqual(sum(timeouts), 2.0)

    def test_unregister_old_handle_does_not_clear_new_owner(self):
        popen1, root1, *_rest1, processes1, _events1 = self.make_fake_tree()
        popen2, root2, *_rest2, processes2, events2 = self.make_fake_tree()
        popen2.pid, root2.pid = 501, 501
        root2._create_time = 50.1
        processes = {**processes1, **{501: root2}}
        controller = job_processes.JobProcessController()
        with patch.object(job_processes.psutil, "Process", side_effect=processes.__getitem__), \
                patch.object(job_processes.psutil, "wait_procs", side_effect=fake_wait_procs):
            controller.register(popen1)
            controller.register(popen2)
            controller.unregister(popen1)
            controller.cancel()

        self.assertIsNone(popen1.poll())
        self.assertEqual(popen2.poll(), 0)
        self.assertIn("root", events2)
        self.assertTrue(root1.alive)

    def test_exited_process_is_a_benign_cancel_race(self):
        popen, root, _child, _grandchild, processes, events = self.make_fake_tree()
        popen.returncode = 0
        controller = job_processes.JobProcessController()
        with patch.object(job_processes.psutil, "Process", side_effect=processes.__getitem__):
            controller.register(popen)
            controller.cancel()
        self.assertEqual(events, [])
        self.assertTrue(root.alive)

    def test_root_pid_reuse_mismatch_never_signals_the_reused_process(self):
        popen, root, _child, _grandchild, processes, events = self.make_fake_tree()
        controller = job_processes.JobProcessController()
        owned_root = FakePsutilProcess(101, 10.1, events=events, name="owned-root")
        popen.root_process = owned_root
        with patch.object(job_processes.psutil, "Process", side_effect=processes.__getitem__), \
                patch.object(job_processes.psutil, "wait_procs", side_effect=fake_wait_procs):
            controller.register(popen)
            root._create_time += 100
            with self.assertRaisesRegex(RuntimeError, "could not be stopped"):
                controller.cancel()
        self.assertEqual(events, ["root"])
        self.assertTrue(root.alive)
        self.assertFalse(owned_root.alive)
        self.assertEqual(popen.poll(), 0)

    def test_normal_register_check_and_unregister_do_not_cancel(self):
        popen, root, _child, _grandchild, processes, events = self.make_fake_tree()
        controller = job_processes.JobProcessController()
        with patch.object(job_processes.psutil, "Process", side_effect=processes.__getitem__):
            controller.register(popen)
            controller.check_cancelled()
            controller.unregister(popen)
            controller.check_cancelled()
        self.assertFalse(controller.cancelled)
        self.assertIsNone(popen.poll())
        self.assertTrue(root.alive)
        self.assertEqual(events, [])

    def test_cancel_and_late_registration_race_has_no_deadlock(self):
        popen, root, _child, _grandchild, processes, _events = self.make_fake_tree()
        controller = job_processes.JobProcessController()
        barrier = threading.Barrier(2)
        errors = []

        def cancel_job():
            barrier.wait()
            controller.cancel()

        def register_process():
            barrier.wait()
            try:
                controller.register(popen)
            except job_processes.JobCancelled:
                pass
            except Exception as exc:  # surfaced in the test thread
                errors.append(exc)

        patches = self.patch_process_table(processes)
        with patches[0], patches[1]:
            cancel_thread = threading.Thread(target=cancel_job)
            register_thread = threading.Thread(target=register_process)
            cancel_thread.start()
            register_thread.start()
            cancel_thread.join(timeout=2)
            register_thread.join(timeout=2)

        self.assertFalse(cancel_thread.is_alive())
        self.assertFalse(register_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(controller.cancelled)
        self.assertEqual(popen.poll(), 0)
        self.assertFalse(root.alive)

    def test_unregister_during_cancel_does_not_hold_registry_lock_while_stopping(self):
        popen, root, _child, _grandchild, processes, _events = self.make_fake_tree()
        controller = job_processes.JobProcessController()
        with patch.object(job_processes.psutil, "Process", side_effect=processes.__getitem__), \
                patch.object(job_processes.psutil, "wait_procs", side_effect=fake_wait_procs):
            controller.register(popen)
            stopping = threading.Event()
            continue_stop = threading.Event()
            original_terminate = popen.terminate

            def delayed_terminate():
                stopping.set()
                if not continue_stop.wait(timeout=1):
                    raise TimeoutError("test did not release the process stop")
                original_terminate()

            popen.terminate = delayed_terminate
            cancel_thread = threading.Thread(target=controller.cancel)
            cancel_thread.start()
            self.assertTrue(stopping.wait(timeout=1))

            unregister_thread = threading.Thread(target=controller.unregister, args=(popen,))
            unregister_thread.start()
            unregister_thread.join(timeout=0.5)
            self.assertFalse(unregister_thread.is_alive(), "unregister must not block behind process I/O")
            continue_stop.set()
            cancel_thread.join(timeout=2)

        self.assertFalse(cancel_thread.is_alive())
        self.assertTrue(controller.cancelled)
        self.assertEqual(popen.poll(), 0)
        self.assertFalse(root.alive)

    def test_real_owned_process_tree_cancellation_leaves_unrelated_process_alive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            child_pid_file = Path(temp_dir) / "child.pid"
            grandchild_pid_file = Path(temp_dir) / "grandchild.pid"
            grandchild_code = "import time; time.sleep(60)"
            child_code = (
                "import pathlib,subprocess,sys,time; "
                "p=subprocess.Popen([sys.executable,'-c',sys.argv[2]]); "
                "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
            )
            root_code = (
                "import pathlib,subprocess,sys,time; "
                "p=subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[3],sys.argv[4]], "
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
            )
            # The child receives its own code plus the marker path and grandchild code.
            child_code_with_args = child_code
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            controller = job_processes.JobProcessController()
            root = subprocess.Popen(
                [sys.executable, "-c", root_code, str(child_pid_file), child_code_with_args,
                 str(grandchild_pid_file), grandchild_code],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                controller.register(root)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not child_pid_file.exists():
                    time.sleep(0.02)
                self.assertTrue(child_pid_file.exists(), "root should launch its child")
                child_pid = int(child_pid_file.read_text())
                while time.monotonic() < deadline and not grandchild_pid_file.exists():
                    time.sleep(0.02)
                self.assertTrue(grandchild_pid_file.exists(), "child should launch its grandchild")
                grandchild_pid = int(grandchild_pid_file.read_text())
                child_identity = psutil.Process(child_pid).create_time()
                grandchild_identity = psutil.Process(grandchild_pid).create_time()

                controller.cancel()
                root.wait(timeout=1)
                self.assertTrue(controller.cancelled)
                self.assertIsNone(unrelated.poll(), "cancellation must not signal an unrelated process")
                self.assertFalse(self._identity_is_running(child_pid, child_identity))
                self.assertFalse(self._identity_is_running(grandchild_pid, grandchild_identity))
            finally:
                if root.poll() is None:
                    controller.cancel()
                if unrelated.poll() is None:
                    unrelated.terminate()
                    unrelated.wait(timeout=2)

    @staticmethod
    def _identity_is_running(pid, expected_create_time):
        try:
            current = psutil.Process(pid)
            return (math.isclose(current.create_time(), expected_create_time, abs_tol=0.01)
                    and current.status() != psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            return False


if __name__ == "__main__":
    unittest.main()
