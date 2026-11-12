import unittest
from unittest.mock import call, patch

from services.diarization import run_diarization
from services.task_errors import TaskFailure


GPU = {"id": "gpu", "name": "Test GPU", "backend": "cuda", "deviceIndex": 0}
CPU = {"id": "cpu", "name": "CPU", "backend": "cpu", "deviceIndex": 0}


class DiarizationFallbackTests(unittest.TestCase):
    def setup_mocks(self, requested, gpu_failure=None, cpu_failure=None):
        resolver = patch("services.diarization_runtime.resolve_diarization_device",
                         side_effect=lambda requested_device: CPU if requested_device == "cpu" else GPU)
        resolver.start()
        self.addCleanup(resolver.stop)
        runner = patch("services.diarization._run_diarization_device")
        run_mock = runner.start()
        self.addCleanup(runner.stop)
        outcomes = []
        if gpu_failure:
            outcomes.append(gpu_failure)
        elif requested != "cpu":
            outcomes.append({"effectiveDevice": {"id": "gpu"}, "intervals": []})
        if cpu_failure:
            outcomes.append(cpu_failure)
        else:
            outcomes.append({"effectiveDevice": {"id": "cpu", "name": "CPU", "backend": "cpu"}, "intervals": []})
        def run_side_effect(_path, device, _resolve_model):
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        run_mock.side_effect = run_side_effect
        return run_mock

    def test_auto_gpu_failure_retries_cpu_once_logs_reason_and_returns_cpu_metadata(self):
        failure = TaskFailure("GPU inference crashed", "diarization_failed", stage="diarization")
        runner = self.setup_mocks("auto", failure)
        with patch("builtins.print") as log:
            result = run_diarization("audio.wav", "auto")
        self.assertEqual(runner.call_count, 2)
        self.assertEqual([c.args[1]["id"] for c in runner.call_args_list], ["gpu", "cpu"])
        self.assertEqual(result["effectiveDevice"]["id"], "cpu")
        self.assertEqual(result["fallbackReason"], str(failure))
        self.assertIn(str(failure), log.call_args.args[0])

    def test_explicit_gpu_failure_does_not_retry_cpu(self):
        runner = self.setup_mocks("cuda:0", TaskFailure("GPU failed", "device_error"))
        with self.assertRaises(TaskFailure):
            run_diarization("audio.wav", "cuda:0")
        runner.assert_called_once()

    def test_auto_cpu_failure_does_not_retry(self):
        resolver = patch("services.diarization_runtime.resolve_diarization_device", return_value=CPU)
        resolver.start(); self.addCleanup(resolver.stop)
        runner = patch("services.diarization._run_diarization_device", side_effect=TaskFailure("CPU failed", "device_error"))
        run_mock = runner.start(); self.addCleanup(runner.stop)
        with self.assertRaises(TaskFailure):
            run_diarization("audio.wav", "auto")
        run_mock.assert_called_once()

    def test_model_and_source_failures_do_not_retry(self):
        for code in ("model_unavailable", "source_invalid"):
            with self.subTest(code=code):
                runner = self.setup_mocks("auto", TaskFailure("not retryable", code))
                with self.assertRaises(TaskFailure):
                    run_diarization("audio.wav", "auto")
                runner.assert_called_once()

    def test_cpu_retry_failure_propagates(self):
        cpu_failure = TaskFailure("CPU failed", "device_error", stage="diarization")
        runner = self.setup_mocks("auto", TaskFailure("GPU failed", "device_error"), cpu_failure)
        with self.assertRaises(TaskFailure) as raised:
            run_diarization("audio.wav", "auto")
        self.assertIs(raised.exception, cpu_failure)
        self.assertEqual(runner.call_count, 2)


if __name__ == "__main__":
    unittest.main()
