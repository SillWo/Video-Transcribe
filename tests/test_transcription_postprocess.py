import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import web_api
from utils import text_postprocess
from web_api import JobState, app, build_command


class TextPostprocessTests(unittest.TestCase):
    def test_restore_russian_segment_text_lowercases_input_and_normalizes_output(self):
        model = Mock()
        model.punctuate.return_value = "  Привет,   мир.  "

        with patch("utils.text_postprocess.load_punctuation_model", return_value=model):
            result = text_postprocess.restore_russian_segment_text("  ПрИВЕТ,   МИР!!!  ")

        model.punctuate.assert_called_once_with("привет мир")
        self.assertEqual(result, "Привет, мир.")

    def test_build_polished_russian_text_from_segments_drops_empty_items_and_returns_single_line(self):
        mock_sentences = [
            SimpleNamespace(text=" Привет, мир. "),
            SimpleNamespace(text=" Как дела? "),
        ]

        with patch("utils.text_postprocess.razdel") as razdel_mock:
            razdel_mock.sentenize.return_value = mock_sentences
            result = text_postprocess.build_polished_russian_text_from_segments(
                [" Привет, мир. ", "", "   ", "Как дела? "]
            )

        razdel_mock.sentenize.assert_called_once_with("Привет, мир. Как дела?")
        self.assertEqual(result, "Привет, мир. Как дела?")


class WebApiCommandTests(unittest.TestCase):
    def build_job(self, *, language: str, restore_punctuation: bool) -> JobState:
        return JobState(
            id="job-1",
            settings={
                "sourceType": "url",
                "url": "https://example.com/video",
                "language": language,
                "model": "small",
                "device": "cpu",
                "nproc": 4,
                "outputFormat": "json",
                "saveAudio": False,
                "useTimestamps": True,
                "restorePunctuation": restore_punctuation,
            },
        )

    def test_build_command_includes_restore_punctuation_for_ru_only(self):
        command = build_command(self.build_job(language="ru", restore_punctuation=True), "input.mp3")
        self.assertIn("--restore_punctuation", command)

    def test_build_command_ignores_restore_punctuation_for_non_ru_language(self):
        command = build_command(self.build_job(language="en", restore_punctuation=True), "input.mp3")
        self.assertNotIn("--restore_punctuation", command)

    def test_build_command_ignores_restore_punctuation_when_flag_is_disabled(self):
        command = build_command(self.build_job(language="ru", restore_punctuation=False), "input.mp3")
        self.assertNotIn("--restore_punctuation", command)


class CreateTranscriptionApiTests(unittest.TestCase):
    def setUp(self):
        web_api.jobs.clear()

    def test_create_transcription_disables_restore_punctuation_for_non_ru_language(self):
        client = TestClient(app)

        with patch("web_api.threading.Thread") as thread_mock:
            response = client.post(
                "/api/transcriptions",
                data={
                    "sourceType": "url",
                    "url": "https://example.com/video",
                    "language": "en",
                    "model": "small",
                    "device": "cpu",
                    "outputFormat": "json",
                    "saveAudio": "false",
                    "useTimestamps": "true",
                    "restorePunctuation": "true",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["settings"]["restorePunctuation"])
        thread_mock.return_value.start.assert_called_once()

    def test_create_transcription_keeps_restore_punctuation_for_ru_language(self):
        client = TestClient(app)

        with patch("web_api.threading.Thread") as thread_mock:
            response = client.post(
                "/api/transcriptions",
                data={
                    "sourceType": "url",
                    "url": "https://example.com/video",
                    "language": "ru",
                    "model": "small",
                    "device": "cpu",
                    "outputFormat": "json",
                    "saveAudio": "false",
                    "useTimestamps": "true",
                    "restorePunctuation": "true",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["settings"]["restorePunctuation"])
        thread_mock.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
