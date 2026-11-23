import json
import base64
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

    @unittest.skipUnless(os.name == "nt", "Windows anonymous pipe behavior")
    def test_managed_stdin_watcher_allows_numpy_import_and_exits_on_pipe_close(self):
        marker = self.runtime / "managed-stdin-state.txt"
        child = r'''
import os
import threading
import time
from pathlib import Path
from services.app_host import _wait_for_parent_stdin_close

marker = Path(os.environ["VT_TEST_MARKER"])
closed = threading.Event()
watching = threading.Event()
def watch_parent():
    watching.set()
    _wait_for_parent_stdin_close(0)
    closed.set()

watcher = threading.Thread(target=watch_parent, name="desktop-parent-watch", daemon=True)
watcher.start()
if not watching.wait(5):
    raise SystemExit("watcher did not start")
time.sleep(0.2)
import numpy
if not watcher.is_alive():
    raise SystemExit("watcher exited before the parent pipe was closed")
marker.write_text("imported", encoding="ascii")
if not closed.wait(10):
    raise SystemExit("watcher did not detect stdin EOF")
marker.write_text("closed", encoding="ascii")
'''
        env = self.env.copy()
        env["VT_TEST_MARKER"] = str(marker)
        proc = subprocess.Popen([str(PYTHON), "-c", child], cwd=ROOT, env=env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())

        deadline = time.monotonic() + 12
        while time.monotonic() < deadline and not marker.exists() and proc.poll() is None:
            time.sleep(0.05)
        self.assertTrue(marker.exists(), "numpy import did not finish while the idle parent pipe remained open")
        self.assertEqual(marker.read_text(encoding="ascii"), "imported")
        self.assertIsNone(proc.poll(), "child exited before parent stdin was closed")

        proc.stdin.close()
        try:
            return_code = proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            self.fail(f"managed stdin watcher did not exit after EOF; stderr={stderr.decode(errors='replace')}")
        stdout, stderr = proc.communicate()
        self.assertEqual(return_code, 0, stderr.decode(errors="replace") or stdout.decode(errors="replace"))
        self.assertEqual(marker.read_text(encoding="ascii"), "closed")

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
        from services.app_host import verify_host_descriptor
        self.assertTrue(verify_host_descriptor(record, project_root=ROOT))
        self.assertFalse(verify_host_descriptor({**record, "instanceId": "wrong"}, project_root=ROOT))
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

    def test_ensure_headless_host_starts_in_isolated_runtime(self):
        from services import app_host
        record = app_host.ensure_headless_host(directory=self.runtime, project_root=ROOT, timeout=20)
        self.assertTrue(app_host.verify_host_descriptor(record, project_root=ROOT))
        self.assertEqual(Path(record["projectRoot"]), ROOT.resolve())
        self.assertTrue(app_host._request_quit(record, timeout=20))
        deadline = time.time() + 15
        while time.time() < deadline and (self.runtime / "desktop-host.json").exists():
            time.sleep(.1)
        self.assertFalse((self.runtime / "desktop-host.json").exists())

    def test_local_wmi_fallback_uses_hidden_host_startup_and_minimal_environment(self):
        from unittest.mock import patch
        from services import app_host
        env = {"PATH": "C:\\Windows\\System32", "VT_DICTIONARY_PATH": str(self.runtime / "dictionary.json"),
               "MEETING_EMBEDDING_MODEL": str(self.runtime / "model"), "SECRET_TOKEN": "must-not-pass"}
        with patch.object(app_host.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "4321", "")) as run:
            pid = app_host._start_host_via_local_wmi(
                [str(PYTHON), "-m", "services.app_host", "--runtime-dir", str(self.runtime)], ROOT, env)
        self.assertEqual(pid, 4321)
        encoded = run.call_args.args[0][-1]
        script = base64.b64decode(encoded).decode("utf-16le")
        self.assertIn("[wmiclass]'Win32_ProcessStartup'", script)
        self.assertIn("ShowWindow = [uint16]0", script)
        self.assertNotIn("CreateFlags", script)
        self.assertIn("VT_DICTIONARY_PATH=", script)
        self.assertIn("MEETING_EMBEDDING_MODEL=", script)
        self.assertNotIn("must-not-pass", script)

    def test_windows_detached_host_uses_local_wmi_outside_mcp_job(self):
        from unittest.mock import patch
        from services import app_host
        env = {"VT_MEETING_SETTINGS": str(self.runtime / "settings.json"),
               "VT_MEETING_ROOT": str(self.runtime / "archive")}
        with patch.object(app_host, "os") as fake_os, \
                patch.object(app_host.subprocess, "Popen") as popen, \
                patch.object(app_host, "_start_host_via_local_wmi", return_value=4321) as wmi:
            fake_os.name = "nt"
            fake_os.environ = os.environ
            fake_os.path = os.path
            app_host._start_detached_host(self.runtime, ROOT, env)
        popen.assert_not_called()
        self.assertTrue(wmi.called)
        command = wmi.call_args.args[0]
        self.assertIn("--settings", command)
        self.assertIn("--root", command)

    def test_conversion_routes_require_token_and_match_archive_root(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from services.app_host import _make_asgi_app
        expected = self.runtime / "archive"
        expected.mkdir()
        source = self.runtime / "source.wav"
        source.write_bytes(b"source")
        import web_api
        queued = []
        app = _make_asgi_app("secret", [None, "instance", str(ROOT), "http://127.0.0.1:8000", None])
        client = TestClient(app)
        with patch.dict(os.environ, {"VT_MEETING_ROOT": str(expected)}), \
                patch("services.meeting_settings.load_settings", return_value={"archive_root": str(expected)}), \
                patch.object(web_api, "enqueue_media_conversion", side_effect=lambda path, request_id: queued.append((path, request_id)) or {"job_id": "job-1", "id": "job-1", "status": "queued"}), \
                patch.object(web_api, "media_conversion_status", side_effect=lambda job_id: {"job_id": job_id, "status": "running", "archiveRoot": str(expected)}) as status_lookup:
            bad = client.post("/api/media/conversions", json={"path": str(source), "request_id": "req"})
            self.assertEqual(bad.status_code, 403)
            headers = {"X-VT-Host-Token": "secret"}
            wrong_root = client.post("/api/media/conversions", headers=headers,
                                     json={"path": str(source), "request_id": "req", "archive_root": str(self.runtime / "other")})
            self.assertEqual(wrong_root.status_code, 409)
            self.assertEqual(queued, [])
            response = client.post("/api/media/conversions", headers=headers,
                                   json={"path": str(source), "request_id": "req", "archive_root": str(expected)})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["job_id"], "job-1")
            self.assertEqual(queued, [(str(source), "req")])
            status = client.get("/api/media/conversions/job-1", headers=headers,
                                params={"archive_root": str(expected)})
            self.assertEqual(status.json()["status"], "running")
            status_lookup.side_effect = lambda job_id: {"job_id": job_id, "status": "completed",
                                                         "archiveRoot": str(self.runtime / "foreign")}
            hidden_status = client.get("/api/media/conversions/job-1", headers=headers,
                                       params={"archive_root": str(expected)})
            self.assertEqual(hidden_status.status_code, 404)
            self.assertEqual(client.get("/api/media/conversions/job-1").status_code, 403)

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
