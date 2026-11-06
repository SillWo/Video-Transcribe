import unittest
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

from services import resource_preparation as resources


class ResourcePreparationTests(unittest.TestCase):
    def test_cached_model_does_not_download_or_make_hub_calls(self):
        with (patch.object(resources, "get_model_status", return_value={"status": "downloaded"}),
              patch.object(resources, "download_registered_model") as download,
              patch("services.prepare_models._punctuation_cached", return_value=True)):
            resources.ensure_selected_resources({"model": "small", "device": "cpu",
                                                  "restorePunctuation": True})
        download.assert_not_called()

    def test_missing_selected_model_downloads_and_reports_actual_bytes(self):
        events = []
        with (patch.object(resources, "get_model_status", return_value={"status": "not_downloaded"}),
              patch.object(resources, "download_registered_model") as download):
            def download_side_effect(model_id, byte_progress_callback=None):
                self.assertEqual(model_id, "large-v3-turbo")
                byte_progress_callback(512, 2048)
            download.side_effect = download_side_effect
            resources.ensure_selected_resources({"model": "large-v3-turbo", "device": "cuda",
                                                  "restorePunctuation": False}, events.append)
        self.assertEqual(events[0]["message"], "Загрузка модели распознавания")
        self.assertEqual(events[1]["completedBytes"], 512)
        self.assertEqual(events[1]["totalBytes"], 2048)
        self.assertEqual(events[-1]["message"], "Подготовка завершена")

    def test_vulkan_model_receives_byte_callback_and_is_revalidated(self):
        events = []
        with (patch("services.prepare_models._vulkan_cached", side_effect=[False, True]),
              patch("services.whisper_cpp.resolve_model_path") as resolve):
            def resolve_side_effect(model_id, *, byte_progress=None):
                self.assertEqual(model_id, "small")
                byte_progress(10, 100)
            resolve.side_effect = resolve_side_effect
            resources.ensure_selected_resources({"model": "small", "device": "vulkan",
                                                  "restorePunctuation": False}, events.append)
        self.assertEqual(events[1]["completedBytes"], 10)

    def test_vulkan_model_validation_failure_is_retryable_preparation_error(self):
        with (patch("services.prepare_models._vulkan_cached", return_value=False),
              patch("services.whisper_cpp.resolve_model_path", return_value="/model.bin")):
            with self.assertRaisesRegex(resources.ResourcePreparationError, "Не удалось загрузить модель") as raised:
                resources.ensure_selected_resources({"model": "small", "device": "vulkan",
                                                      "restorePunctuation": False})
        self.assertEqual(raised.exception.code, "model_unavailable")

    def test_missing_punctuation_dependencies_give_retry_action_without_install_attempt(self):
        with (patch.object(resources, "_punctuation_dependencies_ready", return_value=False),
              patch.object(resources, "snapshot_download") as snapshot):
            with self.assertRaisesRegex(resources.ResourcePreparationError, "Отключите пунктуацию") as raised:
                resources._download_punctuation(None)
        self.assertEqual(raised.exception.code, "dependency_missing")
        snapshot.assert_not_called()

    def test_model_network_failure_has_systemic_download_code(self):
        with (patch.object(resources, "get_model_status", return_value={"status": "not_downloaded"}),
              patch.object(resources, "download_registered_model", side_effect=OSError("offline"))):
            with self.assertRaises(resources.ResourcePreparationError) as raised:
                resources.ensure_selected_resources({"model": "small", "device": "cpu",
                                                      "restorePunctuation": False})
        self.assertEqual(raised.exception.code, "model_download_failed")

    def test_cached_model_validation_failure_has_model_unavailable_code(self):
        with (patch.object(resources, "get_model_status", return_value={"status": "not_downloaded"}),
              patch.object(resources, "download_registered_model",
                           side_effect=resources.ModelDownloadValidationError("bad cache"))):
            with self.assertRaises(resources.ResourcePreparationError) as raised:
                resources.ensure_selected_resources({"model": "small", "device": "cpu",
                                                      "restorePunctuation": False})
        self.assertEqual(raised.exception.code, "model_unavailable")

    def test_punctuation_download_is_filtered_and_validated(self):
        files = [type("Dry", (), {"file_size": 100, "is_cached": False, "will_download": True})()]
        with (patch.object(resources, "_punctuation_dependencies_ready", return_value=True),
              patch("services.prepare_models._punctuation_cached", side_effect=[False, True]),
              patch.object(resources, "snapshot_download", side_effect=[files, "/cache/snapshot"]) as snapshot):
            resources._download_punctuation(None)
        self.assertEqual(snapshot.call_count, 2)
        kwargs = snapshot.call_args.kwargs
        self.assertIn("model.safetensors", kwargs["allow_patterns"])
        self.assertNotIn("*.py", kwargs["allow_patterns"])

    def test_cli_emits_utf8_json_progress_and_selected_only_arguments(self):
        output = io.StringIO()

        def prepare(settings, on_progress):
            self.assertEqual(settings, {"model": "small", "device": "cpu", "restorePunctuation": True})
            on_progress({"completedBytes": 25, "totalBytes": 100, "message": "Загрузка модели распознавания"})

        with (patch.object(resources, "ensure_selected_resources", side_effect=prepare),
              redirect_stdout(output)):
            result = resources.main(["--model", "small", "--device", "cpu", "--restore-punctuation"])
        self.assertEqual(result, 0)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.startswith(resources.PREPARATION_EVENT_PREFIX) for line in lines))
        progress = json.loads(lines[0][len(resources.PREPARATION_EVENT_PREFIX):])
        self.assertEqual(progress["percent"], 25)
        self.assertEqual(progress["message"], "Загрузка модели распознавания")

    def test_cli_error_event_contains_stable_error_code(self):
        output = io.StringIO()
        with (patch.object(resources, "ensure_selected_resources",
                           side_effect=resources.ResourcePreparationError("Нет зависимости.", "dependency_missing")),
              redirect_stdout(output)):
            result = resources.main(["--model", "small", "--device", "cpu"])
        self.assertEqual(result, 1)
        event = json.loads(output.getvalue().strip()[len(resources.PREPARATION_EVENT_PREFIX):])
        self.assertEqual(event["type"], "error")
        self.assertEqual(event["stage"], "preparation")
        self.assertEqual(event["code"], "dependency_missing")
        self.assertEqual(event["message"], "Нет зависимости.")


if __name__ == "__main__":
    unittest.main()
