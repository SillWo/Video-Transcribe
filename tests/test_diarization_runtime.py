import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services import diarization_runtime


class DiarizationRuntimeTests(unittest.TestCase):
    def test_windows_inventory_does_not_inherit_host_stdin(self):
        result = SimpleNamespace(returncode=0, stdout="[]")
        with (patch.object(diarization_runtime.os, "name", "nt"),
              patch.object(diarization_runtime.subprocess, "run", return_value=result) as run):
            self.assertEqual(diarization_runtime._windows_inventory(), [])
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_cuda_inventory_does_not_inherit_host_stdin(self):
        result = SimpleNamespace(returncode=0, stdout="")
        with patch.object(diarization_runtime.subprocess, "run", return_value=result) as run:
            self.assertEqual(diarization_runtime._cuda_inventory(), [])
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_runtime_doctor_does_not_inherit_host_stdin(self):
        result = SimpleNamespace(returncode=0, stdout='{"version":"0.3.0"}', stderr="")
        with patch.object(diarization_runtime.subprocess, "run", return_value=result) as run:
            self.assertEqual(diarization_runtime._doctor("test-runtime", refresh=True), {"version": "0.3.0"})
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()
