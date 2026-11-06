import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


class AppHostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vt-host-test-")
        self.runtime = Path(self.temp.name)
        self.env = os.environ.copy()
        self.env["VT_SIMPLE_RUNTIME_DIR"] = str(self.runtime)

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, port):
        proc = subprocess.Popen([str(PYTHON), "-m", "services.app_host", "--host", "127.0.0.1", "--port", str(port)],
                                cwd=ROOT, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        deadline = time.time() + 20
        while time.time() < deadline and proc.poll() is None:
            descriptor = self.runtime / "desktop-host.json"
            if descriptor.exists():
                try:
                    return proc, json.loads(descriptor.read_text(encoding="utf-8"))
                except ValueError:
                    pass
            time.sleep(.1)
        self.fail("host failed to publish its descriptor")

    def test_host_info_hides_token_wrong_token_denied_and_stop_is_authenticated(self):
        proc, record = self.launch(0)
        url = record["backendUrl"]
        with urllib.request.urlopen(url + "/api/desktop/host", timeout=4) as response:
            info = json.load(response)
        self.assertEqual(info["appId"], "video-transcribe")
        self.assertNotIn("token", info)
        request = urllib.request.Request(url + "/api/desktop/quit", method="POST", headers={"X-VT-Host-Token": "wrong"})
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=4)
        self.assertEqual(raised.exception.code, 403)
        self.assertIsNone(proc.poll())
        request = urllib.request.Request(url + "/api/desktop/quit", method="POST", headers={"X-VT-Host-Token": record["token"]})
        with urllib.request.urlopen(request, timeout=10) as response:
            self.assertEqual(response.status, 202)
        proc.wait(timeout=10)

    def test_occupied_port_does_not_initialize_history(self):
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen()
        port = listener.getsockname()[1]
        proc = subprocess.run([str(PYTHON), "-m", "services.app_host", "--port", str(port)], cwd=ROOT,
                              env=self.env, capture_output=True, text=True, timeout=10)
        listener.close()
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse((self.runtime / "history.sqlite3").exists())

    def test_second_host_lock_prevents_second_instance(self):
        proc, record = self.launch(0)
        second = subprocess.run([str(PYTHON), "-m", "services.app_host", "--port", "0"], cwd=ROOT,
                                env=self.env, capture_output=True, text=True, timeout=10)
        self.assertEqual(second.returncode, 2)
        request = urllib.request.Request(record["backendUrl"] + "/api/desktop/quit", method="POST",
                                         headers={"X-VT-Host-Token": record["token"]})
        urllib.request.urlopen(request, timeout=10).close()
        proc.wait(timeout=10)

    def test_qa_runtime_and_port_8003_start_and_stop(self):
        runtime = ROOT / ".runtime" / "qa-desktop-runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", 8003))
            except OSError:
                self.skipTest("127.0.0.1:8003 is already in use")
        env = self.env.copy(); env["VT_SIMPLE_RUNTIME_DIR"] = str(runtime)
        proc = subprocess.Popen([str(PYTHON), "-m", "services.app_host", "--host", "127.0.0.1", "--port", "8003"],
                                cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.time() + 20
            while time.time() < deadline and proc.poll() is None:
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8003/api/desktop/host", timeout=1) as response:
                        info = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(.1)
            else:
                self.fail("QA host did not become ready on port 8003")
            descriptor = json.loads((runtime / "desktop-host.json").read_text(encoding="utf-8"))
            self.assertEqual(info["projectRoot"], str(ROOT))
            self.assertEqual(info["backendUrl"], "http://127.0.0.1:8003")
            request = urllib.request.Request(info["backendUrl"] + "/api/desktop/quit", method="POST",
                                             headers={"X-VT-Host-Token": descriptor["token"]})
            with urllib.request.urlopen(request, timeout=10) as response:
                self.assertEqual(response.status, 202)
            proc.wait(timeout=10)
            self.assertEqual(proc.returncode, 0)
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait(timeout=5)

    def test_shutdown_failure_returns_503_without_scheduling_server_exit(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from services.app_host import _make_asgi_app
        server = SimpleNamespace(should_exit=False)
        app = _make_asgi_app("expected-secret", [server, "instance", str(ROOT), "http://127.0.0.1:8000", None])
        with patch("web_api.prepare_simple_shutdown", side_effect=RuntimeError("queue did not stop")):
            response = TestClient(app).post("/api/desktop/quit", headers={"X-VT-Host-Token": "expected-secret"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("queue did not stop", response.json()["detail"])
        self.assertFalse(server.should_exit)


if __name__ == "__main__":
    unittest.main()
