from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from services.meeting_archive import MeetingArchive
import services.meeting_recorder as meeting_recorder_module
from services.meeting_recorder import (MeetingRecorder, RecorderError, input_track_mask,
                                       meter_value, validate_selection)


def test_validate_selection_requires_three_exact_ids():
    assert validate_selection({"monitor_id": "монитор #1", "browser_id": "msedge #3A", "microphone_id": "default"}) == {
        "monitor_id": "монитор #1", "browser_id": "msedge #3A", "microphone_id": "default"}
    for value in ({}, {"monitor_id": "m", "browser_id": "b"},
                  {"monitor_id": "\x00", "browser_id": "b", "microphone_id": "m"}):
        with pytest.raises(ValueError):
            validate_selection(value)


def test_prepare_waits_for_obs_frontend_and_only_enumerates_sources(tmp_path, monkeypatch):
    class NotReady(Exception):
        def __init__(self):
            self.code = 207

    class Client:
        def __init__(self):
            self.calls = 0
            self.record_calls = 0

        def get_scene_list(self):
            self.calls += 1
            if self.calls < 3:
                raise NotReady()
            return {"scenes": [{"sceneName": "VT Meeting Capture"}]}

        def get_record_status(self):
            self.record_calls += 1
            return {"outputActive": False}

    now = {"value": 0.0}
    sleeps = []
    monkeypatch.setattr(meeting_recorder_module.time, "monotonic", lambda: now["value"])
    def fake_sleep(delay):
        sleeps.append(delay)
        now["value"] += delay
    monkeypatch.setattr(meeting_recorder_module.time, "sleep", fake_sleep)

    recorder = MeetingRecorder(runtime_root=tmp_path)
    recorder.exe.parent.mkdir(parents=True)
    recorder.exe.touch()
    client = Client()
    recorder._is_owned_process = lambda: False
    recorder._launch = lambda: None
    recorder._connect = lambda attempts=16: client
    enumerated = []
    monkeypatch.setattr(recorder, "_enumerate_monitors", lambda _client, scenes: (
        enumerated.append(("monitors", scenes)) or [{"id": "monitor"}]))
    monkeypatch.setattr(recorder, "_enumerate_windows", lambda _client: (
        enumerated.append(("browsers", None)) or [{"id": "browser"}]))
    monkeypatch.setattr(recorder, "_enumerate_microphones", lambda _client: (
        enumerated.append(("microphones", None)) or [{"id": "mic"}]))
    monkeypatch.setattr(recorder, "_status_from_obs", lambda: {"state": "idle", "connected": True})

    result = recorder.prepare()

    assert result["state"] == "idle"
    assert client.calls == 3
    assert sleeps == [0.3, 0.3]
    assert [name for name, _ in enumerated] == ["monitors", "browsers", "microphones"]
    assert enumerated[0][1] == {"scenes": [{"sceneName": "VT Meeting Capture"}]}
    assert client.record_calls == 0
    assert recorder._prepared is True


def test_frontend_readiness_propagates_non_207_without_retry(tmp_path, monkeypatch):
    class RequestFailure(Exception):
        code = 500

    class Client:
        calls = 0

        def get_scene_list(self):
            self.calls += 1
            raise error

    error = RequestFailure("bad request")
    recorder = MeetingRecorder(runtime_root=tmp_path)
    sleeps = []
    monkeypatch.setattr(meeting_recorder_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(meeting_recorder_module.time, "sleep", sleeps.append)
    client = Client()

    with pytest.raises(RequestFailure) as caught:
        recorder._wait_for_frontend_ready(client)

    assert caught.value is error
    assert client.calls == 1
    assert sleeps == []


def test_frontend_readiness_207_timeout_is_bounded(tmp_path, monkeypatch):
    class NotReady(Exception):
        code = 207

    class Client:
        calls = 0

        def get_scene_list(self):
            self.calls += 1
            raise NotReady()

    now = {"value": 0.0}
    sleeps = []
    monkeypatch.setattr(meeting_recorder_module.time, "monotonic", lambda: now["value"])
    def fake_sleep(delay):
        sleeps.append(delay)
        now["value"] += delay
    monkeypatch.setattr(meeting_recorder_module.time, "sleep", fake_sleep)
    client = Client()
    recorder = MeetingRecorder(runtime_root=tmp_path)

    with pytest.raises(RecorderError, match="не завершил подготовку интерфейса за 30 секунд"):
        recorder._wait_for_frontend_ready(client)

    assert now["value"] == pytest.approx(30.0)
    assert all(0 < delay <= 0.3 for delay in sleeps)
    assert client.calls < 110


def test_monitor_property_id_and_browser_window_are_passed_verbatim():
    class Client:
        def __init__(self):
            self.created = []
            self.settings = []
            self.tracks = []
            self.muted = []
            self.profile = []
            self.video = None
            self.current_scene = "Previous scene"
            self.scene_items = [{"sceneItemId": 7, "sourceName": "VT Meeting Monitor", "sceneItemEnabled": False}]
            self.scene_transform = {}

        def get_scene_list(self): return {"scenes": [{"sceneName": "VT Meeting Capture"}]}
        def get_input_list(self): return {"inputs": []}
        def create_input(self, scene, name, kind, settings, enabled): self.created.append((scene, name, kind, settings, enabled))
        def get_input_default_settings(self, kind): return {"defaultInputSettings": {}}
        def get_input_properties_list_property_items(self, name, property_name):
            if property_name == "priority":
                return {"propertyItems": [
                    {"itemName": "Заголовок окна должен совпадать", "itemValue": 1},
                    {"itemName": "Заголовок совпадает, иначе найти окно того же типа", "itemValue": 0},
                    {"itemName": "Заголовок совпадает, иначе найти окно того же исполняемого файла", "itemValue": 2},
                ]}
            return {"propertyItems": []}
        def remove_input(self, name): pass
        def set_input_audio_tracks(self, name, tracks): self.tracks.append((name, tracks))
        def get_input_audio_tracks(self, name):
            if name == "VT Meeting Monitor":
                raise AssertionError("video-only monitor must not receive audio tracks")
            return {"inputAudioTracks": next(tracks for source, tracks in self.tracks if source == name)}
        def get_special_inputs(self): return {"desktop1": "Desktop Audio"}
        def set_input_mute(self, name, muted): self.muted.append((name, muted))
        def set_video_settings(self, *args): self.video = args
        def set_input_settings(self, *args): self.settings.append(args)
        def set_profile_parameter(self, *args): self.profile.append(args)
        def get_profile_parameter(self, category, name):
            return {"parameterValue": {"Mode": "Advanced", "RecFormat2": "mkv", "RecTracks": "7",
                                        "RecEncoder": "obs_x264"}.get(name, "Standard")}
        def get_scene_item_list(self, scene): return {"sceneItems": self.scene_items}
        def set_scene_item_enabled(self, scene, item_id, enabled): self.scene_items[0]["sceneItemEnabled"] = enabled
        def set_scene_item_transform(self, scene, item_id, transform):
            self.scene_transform = transform
        def get_scene_item_transform(self, scene, item_id):
            assert self.current_scene == "VT Meeting Capture"
            return {"sceneItemTransform": {**self.scene_transform, "sourceWidth": 2880, "sourceHeight": 1800,
                                            "width": 2880, "height": 1800}}
        def set_current_program_scene(self, scene): self.current_scene = scene
        def get_current_program_scene(self): return {"currentProgramSceneName": self.current_scene}

    recorder = MeetingRecorder(runtime_root="unused")
    client = Client()
    monitor_id = "1: DISPLAY\\?MONITOR#A"
    browser_window = "Chrome_WidgetWin_1: Google Chrome: msedge.exe:1234:#3A"
    selection = {"monitor_id": monitor_id, "browser_id": browser_window, "microphone_id": "default"}
    recorder._ensure_capture_sources(client, selection,
        [{"id": monitor_id, "name": "Primary", "width": 2880, "height": 1800}],
        [{"id": browser_window, "obs_window": browser_window}],
        [{"id": "default", "name": "Default"}])
    by_name = {row[1]: row for row in client.created}
    assert by_name["VT Meeting Monitor"][2] == "monitor_capture"
    assert by_name["VT Meeting Monitor"][3]["monitor_id"] == monitor_id
    assert by_name["VT Meeting Browser Audio"][2] == "wasapi_process_output_capture"
    assert by_name["VT Meeting Browser Audio"][3]["window"] == browser_window
    assert by_name["VT Meeting Browser Audio"][3]["priority"] == 2
    assert all("window_capture" != row[2] for row in client.created)
    assert client.video == (30, 1, 2880, 1800, 2880, 1800)
    assert client.current_scene == "VT Meeting Capture"
    assert client.scene_items[0]["sceneItemEnabled"] is True
    assert client.scene_transform["scaleX"] == client.scene_transform["scaleY"] == 1.0
    assert client.scene_transform["positionX"] == client.scene_transform["positionY"] == 0.0
    assert "boundsWidth" not in client.scene_transform and "boundsHeight" not in client.scene_transform
    assert client.tracks[0][1] == input_track_mask(1, 2)
    assert client.tracks[1][1] == input_track_mask(1, 3)
    assert [row[0] for row in client.tracks] == ["VT Meeting Browser Audio", "VT Meeting Microphone"]
    assert client.muted == [("Desktop Audio", True)]
    assert ("AdvOut", "RecTracks", "7") in client.profile
    assert ("AdvOut", "RecFormat2", "mkv") in client.profile
    assert ("AdvOut", "RecEncoder", "obs_x264") in client.profile


def test_audio_track_masks_and_invalid_track_numbers():
    assert input_track_mask(1, 2) == {str(i): i in {1, 2} for i in range(1, 7)}
    assert input_track_mask(1, 3)["3"] is True
    with pytest.raises(ValueError):
        input_track_mask(0)


def test_browser_selection_survives_title_changes_only_for_same_verified_window_identity():
    old = "Проверка записи #3A Личный Microsoft\u200b Edge:Chrome_WidgetWin_1:msedge.exe"
    current = "Jira — 5 вкладок:Chrome_WidgetWin_1:msedge.exe"
    row = {"id": current, "obs_window": current, "browser": "msedge"}
    assert MeetingRecorder._resolve_current_browser(old, [row]) == row

    changed_executable = "Jira:Chrome_WidgetWin_1:chrome.exe"
    changed_class = "Jira:Chrome_WidgetWin_2:msedge.exe"
    with pytest.raises(ValueError, match="окно браузера"):
        MeetingRecorder._resolve_current_browser(old, [{"id": changed_executable, "obs_window": changed_executable}])
    with pytest.raises(ValueError, match="окно браузера"):
        MeetingRecorder._resolve_current_browser(old, [{"id": changed_class, "obs_window": changed_class}])
    assert MeetingRecorder._obs_window_identity(old) == ("chrome_widgetwin_1", "msedge.exe", None)
    assert MeetingRecorder._obs_window_identity("Заголовок:Chrome_WidgetWin_1:unknown.exe") is None
    assert MeetingRecorder._obs_window_identity("Заголовок:Chrome_WidgetWin_1") is None


def test_monitor_enumeration_filters_dummy_and_uses_native_resolution():
    class Client:
        def get_scene_list(self): return {"scenes": []}
        def create_scene(self, name): pass
        def get_input_list(self): return {"inputs": []}
        def create_input(self, *args): pass
        def get_monitor_list(self):
            return {"monitors": [{"monitorName": "Primary", "monitorWidth": 1440, "monitorHeight": 900}]}
        def get_input_properties_list_property_items(self, name, property_name):
            assert property_name == "monitor_id"
            return {"propertyItems": [
                {"itemName": "[Выберите экран для захвата]", "itemValue": "DUMMY"},
                {"itemName": ": 2880x1800 @ 0,0 (Основной монитор)",
                 "itemValue": r"\\?\DISPLAY#LEN8ABA#5&1&UID1"},
            ]}
        def remove_input(self, name): pass

    recorder = MeetingRecorder(runtime_root="unused")
    assert recorder._enumerate_monitors(Client()) == [{
        "id": r"\\?\DISPLAY#LEN8ABA#5&1&UID1",
        "name": ": 2880x1800 @ 0,0 (Основной монитор)", "width": 2880, "height": 1800,
    }]


def test_meter_freshness_is_distinct_from_volume_configuration():
    event = {"inputName": "VT Meeting Browser Audio", "_received_at": 100, "inputLevelsMul": [0.0, 0.42]}
    assert meter_value(event, "VT Meeting Browser Audio", now=101) == pytest.approx(0.42)
    assert meter_value(event, "VT Meeting Browser Audio", now=105) is None
    assert meter_value(event, "another input", now=101) is None
    assert meter_value({"inputName": "VT Meeting Browser Audio", "inputVolume": 0.8, "_received_at": 100},
                       "VT Meeting Browser Audio", now=101) is None
    assert meter_value({"inputName": "VT Meeting Browser Audio", "_received_at": 100,
                        "inputLevelsMul": [[0.12, 0.42, 0.99], [0.03, 0.11, 0.88]]},
                       "VT Meeting Browser Audio", now=101) == pytest.approx(0.12)


def test_volume_meter_event_contains_multiple_active_inputs():
    recorder = MeetingRecorder(runtime_root="unused")
    recorder.on_input_volume_meters({"inputs": [
        {"input_name": "VT Meeting Browser Audio", "input_levels_mul": [[0.22, 0.44, 0.66]]},
        {"input_name": "VT Meeting Microphone", "input_levels_mul": [[0.31, 0.51, 0.71]]},
        {"input_name": "Desktop Audio", "input_levels_mul": [[1.0, 1.0, 1.0]]},
    ]})
    assert set(recorder._meters) == {"VT Meeting Browser Audio", "VT Meeting Microphone"}
    assert meter_value(recorder._meters["VT Meeting Microphone"], "VT Meeting Microphone") == pytest.approx(0.31)


def test_stop_imports_recording_without_scheduling_processing(tmp_path, monkeypatch):
    archive = MeetingArchive(tmp_path / "archive")
    capture_dir = tmp_path / "meeting-recorder" / "recordings"
    capture_dir.mkdir(parents=True)
    capture = capture_dir / "запись test.mkv"
    capture.write_bytes(b"test mkv with three logical audio streams")
    recorder = MeetingRecorder(archive=archive, runtime_root=tmp_path,
                               ffprobe=lambda path: {"audio": [{}, {}, {}], "video": [{}]})
    recorder._session = {"recording_path": str(capture), "title": "Тестовая встреча",
                         "started_at": "2026-11-14T10:00:00+00:00"}
    class Client:
        def get_record_status(self): return {"outputActive": False, "outputPath": str(capture)}
    recorder._connect = lambda attempts=1: Client()
    monkeypatch.setattr(recorder, "_status_from_obs", lambda: {"state": "idle", "connected": True})

    # A successful stop imports/finalizes an archive record; it never calls the queue.
    result = recorder.stop()
    meeting = result["meeting"]
    assert meeting["title"] == "Тестовая встреча"
    assert meeting["audio_track_roles"] == {"1": "mix", "2": "browser", "3": "microphone"}
    assert archive.original_path(meeting["meeting_id"]).exists()
    assert not capture.exists()
    assert not (tmp_path / "archive" / "meeting-recorder").exists()


def test_failed_probe_keeps_owned_mkv_for_recovery(tmp_path):
    archive = MeetingArchive(tmp_path / "archive")
    capture_dir = tmp_path / "meeting-recorder" / "recordings"
    capture_dir.mkdir(parents=True)
    capture = capture_dir / "incomplete.mkv"
    capture.write_bytes(b"incomplete")
    recorder = MeetingRecorder(archive=archive, runtime_root=tmp_path,
                               ffprobe=lambda path: {"audio": [{}, {}]})
    recorder._session = {"recording_path": str(capture)}
    class Client:
        def get_record_status(self): return {"outputActive": False, "outputPath": str(capture)}
    recorder._connect = lambda attempts=1: Client()
    with pytest.raises(RecorderError, match="вместо трёх"):
        recorder.stop()
    assert capture.read_bytes() == b"incomplete"
    assert archive.list_meetings() == []


def test_reconnect_error_keeps_interrupted_recording_visible_and_never_attaches_elsewhere(tmp_path):
    opened = []
    recorder = MeetingRecorder(runtime_root=tmp_path, client_factory=lambda session: opened.append(session))
    recorder._session = {"recording_path": str(tmp_path / "capture.mkv"), "pid": 991231}
    recorder._is_owned_process = lambda: False
    status = recorder.status()
    assert status["state"] == "recording"
    assert status["connected"] is False
    assert "потеряна" in status["error"]
    assert opened == []


def test_private_obs_download_is_pinned_and_not_reported_as_installed(tmp_path):
    recorder = MeetingRecorder(runtime_root=tmp_path)
    expected = recorder.expected_download()
    assert expected["version"] == "33.0.1"
    assert expected["size_bytes"] == 223786669
    assert len(expected["sha256"]) == 64
    status = recorder.status()
    assert status["state"] == "unavailable"
    assert status["available"] is False
    assert status["connected"] is False
    assert status["error"] is None


def test_recording_encoder_profile_is_written_only_under_private_obs_root(tmp_path):
    recorder = MeetingRecorder(runtime_root=tmp_path / ".runtime")
    recorder._prepare_private_encoder_profile()
    config = tmp_path / ".runtime" / "obs-studio" / "config" / "obs-studio" / "basic" / "profiles" / "Video Transcribe Meeting"
    settings = json.loads((config / "recordEncoder.json").read_text(encoding="utf-8"))
    assert settings == {"rate_control": "CRF", "crf": 18, "preset": "ultrafast",
                        "profile": "high", "keyint_sec": 2}
    assert (config / "basic.ini").is_file()


def test_recording_session_marker_is_read_without_obs_or_models(tmp_path):
    from services.meeting_recorder import recording_session_active
    session = tmp_path / "meeting-recorder" / "session.json"
    session.parent.mkdir(parents=True)
    session.write_text(json.dumps({"recording_path": "D:/capture.mkv", "password": "hidden"}), encoding="utf-8")
    assert recording_session_active(tmp_path) is True
    session.write_text(json.dumps({"recording_path": None}), encoding="utf-8")
    assert recording_session_active(tmp_path) is False


def test_recorder_status_includes_active_meeting_title(tmp_path):
    class Client:
        def get_record_status(self): return {"outputActive": True, "outputDuration": 62000}

    recorder = MeetingRecorder(runtime_root=tmp_path)
    recorder._session = {"title": "Планирование Jira"}
    recorder._connect = lambda attempts=1: Client()
    recorder._start_event_client = lambda: None
    status = recorder._status_from_obs()
    assert status["title"] == "Планирование Jira"
    assert status["duration_seconds"] == 62


def test_recorder_uses_configured_meeting_archive_and_private_obs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("VT_MEETING_ROOT", str(tmp_path / "chosen archive"))
    monkeypatch.setenv("VT_SIMPLE_RUNTIME_DIR", str(tmp_path / "host-runtime"))
    monkeypatch.setenv("VT_MEETING_OBS_ROOT", str(tmp_path / ".runtime" / "obs-studio"))
    recorder = MeetingRecorder(project_root=tmp_path, runtime_root=tmp_path / "host-runtime")
    assert recorder.archive.root == (tmp_path / "chosen archive")
    assert recorder.obs_root == (tmp_path / ".runtime" / "obs-studio")
    monkeypatch.setenv("VT_MEETING_OBS_ROOT", str(tmp_path / "outside" / "obs"))
    with pytest.raises(ValueError, match="inside the project's private"):
        MeetingRecorder(project_root=tmp_path, runtime_root=tmp_path / "host-runtime")


def test_stale_unclean_markers_are_moved_only_when_private_obs_is_stopped(tmp_path):
    obs_root = tmp_path / ".runtime" / "obs-studio"
    sentinel = obs_root / "config" / "obs-studio" / ".sentinel"
    sentinel.mkdir(parents=True)
    stale = sentinel / "run_old-session"
    stale.write_text("stale marker", encoding="utf-8")
    unrelated = sentinel / "diagnostic.json"
    unrelated.write_text("preserve", encoding="utf-8")
    capture = tmp_path / ".runtime" / "meeting-recorder" / "recordings" / "interrupted.mkv"
    capture.parent.mkdir(parents=True)
    capture.write_bytes(b"recoverable recording")
    recorder = MeetingRecorder(project_root=tmp_path, runtime_root=tmp_path / ".runtime")
    recorder._private_obs_process_running = lambda: False
    moved = recorder._recover_stale_unclean_markers()
    assert len(moved) == 1
    assert moved[0].read_text(encoding="utf-8") == "stale marker"
    assert not stale.exists()
    assert unrelated.read_text(encoding="utf-8") == "preserve"
    assert capture.read_bytes() == b"recoverable recording"


def test_live_private_obs_blocks_sentinel_recovery_without_mutation(tmp_path):
    obs_root = tmp_path / ".runtime" / "obs-studio"
    sentinel = obs_root / "config" / "obs-studio" / ".sentinel"
    sentinel.mkdir(parents=True)
    marker = sentinel / "run_live"
    marker.write_text("leave", encoding="utf-8")
    recorder = MeetingRecorder(project_root=tmp_path, runtime_root=tmp_path / ".runtime")
    recorder._private_obs_process_running = lambda: True
    with pytest.raises(RecorderError, match="уже запущен"):
        recorder._recover_stale_unclean_markers()
    assert marker.read_text(encoding="utf-8") == "leave"


def test_obs_sentinel_junction_escape_is_rejected_before_mutation(tmp_path, monkeypatch):
    obs_root = tmp_path / ".runtime" / "obs-studio"
    config = obs_root / "config" / "obs-studio"
    config.mkdir(parents=True)
    outside = tmp_path / "outside-sentinel"
    outside.mkdir()
    marker = outside / "run_external"
    marker.write_text("do not move", encoding="utf-8")
    sentinel = config / ".sentinel"
    sentinel.mkdir()
    marker_inside = sentinel / "run_local"
    marker_inside.write_text("also preserve", encoding="utf-8")
    original_resolve = Path.resolve
    def resolve_with_junction(self, *args, **kwargs):
        if self == sentinel:
            return outside
        return original_resolve(self, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve_with_junction)
    recorder = MeetingRecorder(project_root=tmp_path, runtime_root=tmp_path / ".runtime")
    recorder._private_obs_process_running = lambda: False
    with pytest.raises(RecorderError, match="выходит за границы|ссылкой"):
        recorder._recover_stale_unclean_markers()
    assert marker.read_text(encoding="utf-8") == "do not move"
    assert marker_inside.read_text(encoding="utf-8") == "also preserve"


def test_screen_gate_unpauses_and_honors_cancellation(tmp_path):
    from services.meeting_worker import wait_for_recording_clear
    session = tmp_path / "meeting-recorder" / "session.json"
    session.parent.mkdir(parents=True)
    session.write_text(json.dumps({"recording_path": "capture.mkv"}), encoding="utf-8")
    waits = []
    wait_for_recording_clear(tmp_path, on_wait=waits.append,
        sleep=lambda _delay: session.write_text(json.dumps({"recording_path": None}), encoding="utf-8"))
    assert len(waits) == 1
    assert waits[0]["recording_paused"] is True

    session.write_text(json.dumps({"recording_path": "capture.mkv"}), encoding="utf-8")
    counter = {"value": 0}
    def on_wait(_payload): counter["value"] += 1
    with pytest.raises(InterruptedError, match="cancelled"):
        wait_for_recording_clear(tmp_path, cancelled=lambda: counter["value"] > 0,
                                 on_wait=on_wait, sleep=lambda _delay: None)
