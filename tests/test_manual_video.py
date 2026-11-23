import json
import wave
from unittest.mock import patch

import pytest

from services import manual_video
from services.task_errors import TaskFailure


class _ProbeProcess:
    def __init__(self, payload, *, returncode=0, stderr=b""):
        self.payload = payload
        self.returncode = returncode
        self.stderr = stderr
        self.killed = False

    def communicate(self, timeout=None):
        return json.dumps(self.payload).encode(), self.stderr

    def poll(self):
        return -9 if self.killed else self.returncode

    def kill(self):
        self.killed = True


def test_probe_excludes_attached_picture_and_classifies_audio(tmp_path):
    media = tmp_path / "cover-art.m4a"
    media.write_bytes(b"fixture")
    process = _ProbeProcess({"streams": [
        {"codec_type": "video", "disposition": {"attached_pic": 1}},
        {"codec_type": "audio", "disposition": {"attached_pic": 0}},
    ]})
    with patch.object(manual_video, "_ffprobe_path", return_value="ffprobe"), \
            patch.object(manual_video, "_ffmpeg_path", return_value="ffmpeg"), \
            patch.object(manual_video, "_decoded_streams", return_value=["audible"]), \
            patch.object(manual_video.subprocess, "Popen", return_value=process) as popen:
        result = manual_video.probe_local_media(media)

    assert result["isVideo"] is False
    assert result["hasAudio"] is True
    assert popen.call_args.kwargs["shell"] is False
    assert popen.call_args.kwargs["stdin"] is manual_video.subprocess.DEVNULL
    assert popen.call_args.kwargs["creationflags"] == getattr(manual_video.subprocess, "CREATE_NO_WINDOW", 0)


def test_media_decoder_probe_detaches_stdin():
    process = _ProbeProcess({})
    with patch.object(manual_video.subprocess, "Popen", return_value=process) as popen:
        code, stdout, stderr = manual_video._run_media_probe(["ffmpeg", "-i", "input.webm"], timeout=1)

    assert (code, stdout, stderr) == (0, json.dumps({}).encode(), b"")
    assert popen.call_args.kwargs["stdin"] is manual_video.subprocess.DEVNULL


def test_probe_reports_missing_ffprobe_without_downgrading(tmp_path):
    media = tmp_path / "recording.mp4"
    media.write_bytes(b"fixture")
    with patch.object(manual_video, "_ffprobe_path", return_value=None):
        with pytest.raises(TaskFailure) as exc:
            manual_video.probe_local_media(media)
    assert exc.value.code == "media_probe_unavailable"


def test_zero_byte_file_is_the_only_media_empty_case(tmp_path):
    media = tmp_path / "empty.wav"
    media.write_bytes(b"")

    with pytest.raises(TaskFailure) as exc:
        manual_video.probe_local_media(media)

    assert exc.value.code == "media_empty"
    assert "ни одного байта" in exc.value.message


def test_non_media_text_reports_unsupported_format_not_empty(tmp_path):
    media = tmp_path / "not-media.txt"
    media.write_bytes(b"this is a text file, not media".ljust(52, b"!"))
    assert media.stat().st_size == 52
    process = _ProbeProcess({}, returncode=1, stderr=b"Invalid data found when processing input")

    with patch.object(manual_video, "_ffprobe_path", return_value="ffprobe"), \
            patch.object(manual_video.subprocess, "Popen", return_value=process):
        with pytest.raises(TaskFailure) as exc:
            manual_video.probe_local_media(media)

    assert exc.value.code == "media_format_unsupported"
    assert "поддерживаемый медиаформат" in exc.value.message
    assert "пустой" not in exc.value.message.casefold()


def test_decoded_digital_silence_reports_silence_not_decode_error(tmp_path):
    media = tmp_path / "silent.wav"
    with wave.open(str(media), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * (16000 * 8))
    assert media.stat().st_size > 0
    process = _ProbeProcess({
        "streams": [{"index": 0, "codec_type": "audio"}],
        "format": {"format_name": "wav"},
    })

    with patch.object(manual_video, "_ffprobe_path", return_value="ffprobe"), \
            patch.object(manual_video, "_ffmpeg_path", return_value="ffmpeg"), \
            patch.object(manual_video, "_run_media_probe", return_value=(0, b"", b"Peak level dB: -inf")), \
            patch.object(manual_video.subprocess, "Popen", return_value=process):
        with pytest.raises(TaskFailure) as exc:
            manual_video.probe_local_media(media)

    assert exc.value.code == "audio_silent"
    assert "цифровую тишину" in exc.value.message


def test_video_with_silent_primary_and_audible_secondary_audio_is_usable(tmp_path):
    media = tmp_path / "multitrack.mkv"
    media.write_bytes(b"fixture")
    process = _ProbeProcess({"streams": [
        {"index": 0, "codec_type": "video", "disposition": {"attached_pic": 0}},
        {"index": 1, "codec_type": "audio"},
        {"index": 2, "codec_type": "audio"},
    ]})

    def decoded(_path, streams, *, kind, controller=None):
        return ["decoded"] if kind == "video" else ["silent", "audible"]

    with patch.object(manual_video, "_ffprobe_path", return_value="ffprobe"), \
            patch.object(manual_video, "_decoded_streams", side_effect=decoded), \
            patch.object(manual_video.subprocess, "Popen", return_value=process):
        result = manual_video.probe_local_media(media)

    assert result["isVideo"] is True
    assert result["hasAudio"] is True
    assert [stream["index"] for stream in result["decodedAudioStreams"]] == [2]
