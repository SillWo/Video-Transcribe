import os
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(os.name == "nt", "Windows Job Objects are Windows-specific")
class WindowsJobTests(unittest.TestCase):
    def test_closing_owner_kills_child_and_grandchild_but_not_unrelated_process(self):
        helper = r'''
import subprocess, sys, time
from services.windows_job import own_current_process
job = own_current_process()
child = subprocess.Popen([sys.executable, "-c", "import subprocess,sys,time; p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); print(p.pid, flush=True); time.sleep(120)"], stdout=subprocess.PIPE, text=True)
grandchild = int(child.stdout.readline().strip())
print(child.pid, grandchild, flush=True)
time.sleep(120)
'''
        base_python = getattr(sys, "_base_executable", sys.executable)
        owner = subprocess.Popen([base_python, "-c", helper], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        unrelated = subprocess.Popen([base_python, "-c", "import time; time.sleep(120)"])
        def clean_unrelated():
            if unrelated.poll() is None:
                unrelated.kill(); unrelated.wait(timeout=5)
        self.addCleanup(clean_unrelated)
        try:
            line = owner.stdout.readline()
            if not line:
                self.fail(f"Job owner exited before reporting descendants: {owner.stderr.read()}")
            child_id, grandchild_id = map(int, line.split())
            owner.kill(); owner.wait(timeout=5)
            deadline = time.time() + 8
            while time.time() < deadline and any(_alive(pid) for pid in (child_id, grandchild_id)):
                time.sleep(.1)
            self.assertFalse(_alive(child_id))
            self.assertFalse(_alive(grandchild_id))
            self.assertIsNone(unrelated.poll())
        finally:
            if owner.poll() is None: owner.kill()
            owner.wait(timeout=5)
            owner.stdout.close(); owner.stderr.close()


def _alive(pid):
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


if __name__ == "__main__":
    unittest.main()
