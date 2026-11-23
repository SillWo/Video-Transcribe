import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from services import whisper_cpp
from services.task_errors import TaskFailure


class WhisperCppJsonUtf8Tests(unittest.TestCase):
    def load_raw_json(self, raw_json):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        output_base = Path(temp_dir.name) / "transcript"
        Path(str(output_base) + ".json").write_bytes(raw_json)
        return whisper_cpp._load_whisper_result(output_base, word_timestamps=True)

    def test_split_multibyte_tokens_reassemble_words_punctuation_and_timestamps(self):
        raw_json = (
            b'{"result":{"language":"ru"},"transcription":[{"offsets":{"from":0,"to":1000},'
            b'"text":" \xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82, \xd0\xbc\xd0\xb8\xd1\x80!",'
            b'"tokens":['
            b'{"text":" \xd0","offsets":{"from":0,"to":200}},'
            b'{"text":"\xbf","offsets":{"from":200,"to":250}},'
            b'{"text":"\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82","offsets":{"from":250,"to":500}},'
            b'{"text":",","offsets":{"from":500,"to":550}},'
            b'{"text":" \xd0","offsets":{"from":550,"to":600}},'
            b'{"text":"\xbc","offsets":{"from":600,"to":650}},'
            b'{"text":"\xd0\xb8\xd1\x80","offsets":{"from":650,"to":950}},'
            b'{"text":"!","offsets":{"from":950,"to":1000}}]}]}'
        )

        segments, info = self.load_raw_json(raw_json)

        self.assertEqual(info.language, "ru")
        self.assertEqual(segments[0].text, "привет, мир!")
        self.assertEqual([word["text"] for word in segments[0].words], [" привет,", " мир!"])
        self.assertEqual(segments[0].words[0]["start"], 0.0)
        self.assertEqual(segments[0].words[0]["end"], 0.55)
        self.assertEqual(segments[0].words[1]["start"], 0.55)
        self.assertEqual(segments[0].words[1]["end"], 1.0)
        for word in segments[0].words:
            self.assertFalse(any(0xD800 <= ord(char) <= 0xDFFF for char in word["text"]))

    def test_unrecoverable_token_bytes_fall_back_to_valid_segment_text(self):
        raw_json = (
            b'{"result":{"language":"ru"},"transcription":[{"offsets":{"from":0,"to":400},'
            b'"text":"\xd1\x8f","tokens":[{"text":" \xd1",'
            b'"offsets":{"from":0,"to":400}}]}]}'
        )

        segments, _info = self.load_raw_json(raw_json)

        self.assertEqual(segments[0].text, "я")
        self.assertIsNone(segments[0].words)

    def test_invalid_segment_utf8_is_a_typed_recognition_failure(self):
        raw_json = (
            b'{"result":{"language":"ru"},"transcription":[{"offsets":{"from":0,"to":400},'
            b'"text":"\xd1","tokens":[]}]}'
        )

        with self.assertRaises(TaskFailure) as raised:
            self.load_raw_json(raw_json)

        self.assertEqual(raised.exception.code, "transcription_failed")
        self.assertEqual(raised.exception.stage, "recognition")

    def test_invalid_language_utf8_is_a_typed_recognition_failure(self):
        raw_json = (
            b'{"result":{"language":"\xd1"},"transcription":[{"offsets":{"from":0,"to":400},'
            b'"text":"\xd1\x8f","tokens":[]}]}'
        )

        with self.assertRaises(TaskFailure) as raised:
            self.load_raw_json(raw_json)

        self.assertEqual(raised.exception.code, "transcription_failed")
        self.assertEqual(raised.exception.stage, "recognition")

    def test_genuinely_malformed_json_is_a_typed_recognition_failure(self):
        with self.assertRaises(TaskFailure) as raised:
            self.load_raw_json(b'{"transcription":}')

        self.assertEqual(raised.exception.code, "transcription_failed")
        self.assertEqual(raised.exception.stage, "recognition")


class WhisperCppTypedFailureTests(unittest.TestCase):
    def make_model(self):
        model = whisper_cpp.WhisperCppModel.__new__(whisper_cpp.WhisperCppModel)
        model.executable = "whisper-cli"
        model.model_path = Path("model.bin")
        model.threads = 1
        return model

    def run_failure(self, runtime_detail):
        model = self.make_model()

        def fake_run(command, **_kwargs):
            if command[0] == "ffmpeg.exe":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr=runtime_detail)

        with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                patch.object(whisper_cpp.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(TaskFailure) as raised:
                model.transcribe("damaged.wav")
        return raised.exception

    def test_corrupt_input_decode_failure_is_source_invalid_even_with_vulkan_logs(self):
        failure = self.run_failure("ggml_vk_buffer initialized successfully; input decode failed")
        self.assertEqual(failure.code, "source_invalid")
        self.assertEqual(failure.stage, "source")

    def test_explicit_oom_code_takes_priority_over_decode_failure_text(self):
        failure = self.run_failure("input decode failed: CUDA_ERROR_OUT_OF_MEMORY")
        self.assertEqual(failure.code, "out_of_memory")
        self.assertEqual(failure.stage, "recognition")

    def test_native_cuda_oom_signature_is_typed_as_systemic_oom(self):
        failure = self.run_failure("CUDA_ERROR_OUT_OF_MEMORY while allocating decoder buffers")
        self.assertEqual(failure.code, "out_of_memory")
        self.assertEqual(failure.stage, "recognition")

    def test_ffmpeg_input_conversion_failure_is_source_invalid(self):
        model = self.make_model()
        with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                patch.object(whisper_cpp.subprocess, "run",
                             return_value=SimpleNamespace(returncode=1, stdout="", stderr="Invalid data found when processing input")):
            with self.assertRaises(TaskFailure) as raised:
                model.transcribe("damaged.wav")
        self.assertEqual(raised.exception.code, "source_invalid")

    def test_missing_vulkan_runtime_is_typed_device_unavailable(self):
        with patch.object(whisper_cpp, "find_whisper_cpp_executable", return_value=None):
            with self.assertRaises(TaskFailure) as raised:
                whisper_cpp.WhisperCppModel("small")
        self.assertEqual(raised.exception.code, "device_unavailable")


class WhisperCppRepetitionGuardTests(unittest.TestCase):
    def make_model(self, callback=None):
        model = whisper_cpp.WhisperCppModel.__new__(whisper_cpp.WhisperCppModel)
        model.executable = "whisper-cli"
        model.model_path = Path("model.bin")
        model.threads = 2
        model.model_name = "large-v3"
        model.gpu_index = 3
        model.transcription_progress = callback
        return model

    @staticmethod
    def repeated_payload(count=7):
        variants = ["We are going to continue!", "WE, are going to continue."]
        return {"result": {"language": "ru"}, "transcription": [
            {"offsets": {"from": index * 1500, "to": (index + 1) * 1500},
             "text": variants[index % len(variants)]}
            for index in range(count)
        ]}

    @staticmethod
    def clean_payload():
        return {"result": {"language": "ru"}, "transcription": [
            {"offsets": {"from": 0, "to": 2100}, "text": "The transcript is clean."}
        ]}

    @staticmethod
    def brief_and_zero_duration_payload():
        rows = [
            {"offsets": {"from": 0, "to": 1500}, "text": "Brief phrase should pass."},
            {"offsets": {"from": 1500, "to": 3000}, "text": "Brief phrase should pass!"},
        ]
        rows.extend({"offsets": {"from": 3000, "to": 3000},
                     "text": "Zero duration phrase remains harmless."} for _ in range(7))
        return {"result": {"language": "ru"}, "transcription": rows}

    @staticmethod
    def write_payload(command, payload):
        output_base = command[command.index("-of") + 1]
        Path(output_base + ".json").write_text(json.dumps(payload), encoding="utf-8")

    def test_valid_large_v3_output_keeps_the_native_vulkan_command(self):
        model = self.make_model()
        inference_commands = []

        def fake_run(command, **_kwargs):
            if command[0] == "ffmpeg.exe":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            inference_commands.append(command)
            self.write_payload(command, self.brief_and_zero_duration_payload())
            return SimpleNamespace(returncode=0, stdout="using Vulkan", stderr="")

        with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                patch.object(whisper_cpp.subprocess, "run", side_effect=fake_run):
            segments, info = model.transcribe("valid.wav", beam_size=7, language="ru", word_timestamps=True)

        self.assertEqual(len(inference_commands), 1)
        command = inference_commands[0]
        self.assertEqual(command[command.index("-l") + 1], "ru")
        self.assertEqual(command[command.index("-bs") + 1], "7")
        self.assertEqual(command[command.index("-dev") + 1], "3")
        self.assertIn("-ojf", command)
        self.assertIn("-dtw", command)
        self.assertIn("--no-flash-attn", command)
        self.assertEqual(segments[0].text, "Brief phrase should pass.")
        self.assertEqual(info.language, "ru")

    def test_json_repeat_retries_once_on_same_gpu_without_dictionary_prompt(self):
        model = self.make_model()
        inference_commands = []

        def fake_run(command, **_kwargs):
            if command[0] == "ffmpeg.exe":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            inference_commands.append(command)
            self.write_payload(command, self.repeated_payload() if len(inference_commands) == 1
                               else self.clean_payload())
            return SimpleNamespace(returncode=0, stdout="using Vulkan", stderr="")

        with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                patch.object(whisper_cpp, "bound_whisper_vocabulary", return_value=("Vocabulary: codex", {})), \
                patch.object(whisper_cpp.subprocess, "run", side_effect=fake_run):
            segments, _info = model.transcribe("valid.wav", beam_size=7, language="ru",
                                               word_timestamps=True, vocabulary=("codex",))

        self.assertEqual(len(inference_commands), 2)
        initial, retry = inference_commands
        self.assertNotIn("--max-context", initial)
        self.assertIn("--max-context", retry)
        self.assertEqual(retry[retry.index("--max-context") + 1], "0")
        self.assertIn("--no-flash-attn", initial)
        self.assertIn("--no-flash-attn", retry)
        for flag in ("-l", "-bs", "-dev"):
            self.assertEqual(retry[retry.index(flag) + 1], initial[initial.index(flag) + 1])
        self.assertIn("-ojf", retry)
        self.assertIn("-dtw", retry)
        self.assertNotIn("--prompt", retry)
        self.assertEqual(segments[0].text, "The transcript is clean.")

    def test_flash_attention_workaround_is_limited_to_large_models(self):
        for model_name, should_retry in (("large", True), ("small", False), ("large-v3-turbo", False)):
            with self.subTest(model_name=model_name):
                model = self.make_model()
                model.model_name = model_name
                inference_commands = []

                def fake_run(command, **_kwargs):
                    if command[0] == "ffmpeg.exe":
                        return SimpleNamespace(returncode=0, stdout="", stderr="")
                    inference_commands.append(command)
                    payload = self.repeated_payload() if len(inference_commands) == 1 else self.clean_payload()
                    self.write_payload(command, payload)
                    return SimpleNamespace(returncode=0, stdout="using Vulkan", stderr="")

                with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                        patch.object(whisper_cpp.subprocess, "run", side_effect=fake_run):
                    model.transcribe("valid.wav")

                expected_attempts = 2 if should_retry else 1
                self.assertEqual(len(inference_commands), expected_attempts)
                self.assertEqual(
                    ["--no-flash-attn" in command for command in inference_commands],
                    [should_retry] * expected_attempts,
                )

    def test_repeat_after_retry_is_a_typed_recognition_failure(self):
        model = self.make_model()
        inference_commands = []

        def fake_run(command, **_kwargs):
            if command[0] == "ffmpeg.exe":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            inference_commands.append(command)
            self.write_payload(command, self.repeated_payload())
            return SimpleNamespace(returncode=0, stdout="using Vulkan", stderr="")

        with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                patch.object(whisper_cpp.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(TaskFailure) as raised:
                model.transcribe("valid.wav")

        self.assertEqual(len(inference_commands), 2)
        self.assertEqual(raised.exception.code, "repetitive_transcription")
        self.assertEqual(raised.exception.stage, "recognition")

    def test_streaming_repeat_kills_child_retries_and_holds_completion_until_clean(self):
        progress = []
        model = self.make_model(progress.append)
        processes = []
        commands = []

        class FakeProcess:
            def __init__(self, lines):
                self.stdout = io.StringIO(lines)
                self.killed = False
                self.waited = False

            def kill(self):
                self.killed = True

            def wait(self):
                self.waited = True
                return -9 if self.killed else 0

        def fake_run(command, **_kwargs):
            if command[0] == "ffmpeg.exe":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError("recognition with a callback should use Popen")

        def fake_popen(command, **_kwargs):
            commands.append(command)
            if len(commands) == 1:
                text_rows = ["We are going to continue!", "WE, are going to continue."] * 4
                lines = "using Vulkan backend\n"
                for index, text in enumerate(text_rows):
                    start_min, start_sec = divmod(index * 1.5, 60)
                    end_min, end_sec = divmod((index + 1) * 1.5, 60)
                    lines += (f"[00:{start_min:02.0f}:{start_sec:06.3f} --> "
                              f"00:{end_min:02.0f}:{end_sec:06.3f}] {text}\n")
            else:
                lines = "using Vulkan backend\nprogress = 70%\n"
                self.write_payload(command, self.clean_payload())
            process = FakeProcess(lines)
            processes.append(process)
            return process

        with patch.object(whisper_cpp, "_find_ffmpeg", return_value="ffmpeg.exe"), \
                patch.object(whisper_cpp.subprocess, "run", side_effect=fake_run), \
                patch.object(whisper_cpp.subprocess, "Popen", side_effect=fake_popen):
            segments, _info = model.transcribe("valid.wav", beam_size=7, language="ru",
                                               word_timestamps=True)

        self.assertEqual(len(commands), 2)
        self.assertTrue(processes[0].killed)
        self.assertTrue(processes[0].waited)
        self.assertTrue(processes[1].waited)
        self.assertIn("--max-context", commands[1])
        self.assertNotIn("--prompt", commands[1])
        self.assertEqual(segments[0].text, "The transcript is clean.")
        percentages = [event["percent"] for event in progress if "percent" in event]
        self.assertEqual(percentages[-1], 100.0)
        self.assertTrue(any("повтор" in str(event).lower() for event in progress))


if __name__ == "__main__":
    unittest.main()
