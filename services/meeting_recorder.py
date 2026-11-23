"""Private, local OBS recording for meeting archives.

Public entry point: ``MeetingRecorder(project_root=None, archive=None, ...)``.
Call ``status()``, ``prepare()``, ``configure(selection)``, ``start(title)`` and
``stop()``. OBS credentials and process ownership are persisted only below the
private runtime directory and are never included in status responses.

The module keeps OBS/image/ML imports out of the API import path. The optional
``obsws-python`` dependency is loaded when an OBS connection is first needed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from services.meeting_archive import MeetingArchive
from services.meeting_activity import hold_activity, release_activity


OBS_VERSION = "33.0.1"
OBS_ZIP_SIZE = 223786669
OBS_ZIP_SHA256 = "0d9436451c2d772b33428e3fdae9776f071895c335de17283176306fb72651d3"
OBS_ZIP_FILENAME = f"OBS-Studio-{OBS_VERSION}-Windows-x64.zip"
OBS_ZIP_URL = f"https://github.com/obsproject/obs-studio/releases/download/{OBS_VERSION}/{OBS_ZIP_FILENAME}"
_BROWSER_EXES = {"msedge.exe", "chrome.exe", "firefox.exe"}
OBS_RECORD_PROFILE = "Video Transcribe Meeting"
_VALID_SOURCE_KINDS = {"monitor_capture", "wasapi_process_output_capture", "wasapi_input_capture"}
_OBS_FRONTEND_READY_TIMEOUT = 30.0
_OBS_FRONTEND_READY_POLL_INTERVAL = 0.3
_LOCK = threading.RLock()


def recording_session_active(runtime_root: str | Path | None = None) -> bool:
    """Read the persisted recording marker without connecting to OBS or starting it."""
    if runtime_root is None:
        configured = os.environ.get("VT_SIMPLE_RUNTIME_DIR")
        root = Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[1] / ".runtime"
    else:
        root = Path(runtime_root).expanduser().resolve()
    try:
        session = json.loads((root / "meeting-recorder" / "session.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(isinstance(session, dict) and session.get("recording_path"))


class RecorderError(RuntimeError):
    """An OBS operation failed without discarding recoverable recordings."""


def validate_selection(selection: dict[str, Any]) -> dict[str, str]:
    """Validate the three selected source identifiers; no OBS mutation occurs."""
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    result = {}
    for key in ("monitor_id", "browser_id", "microphone_id"):
        value = selection.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 2048:
            raise ValueError(f"{key} must be a non-empty string")
        if "\x00" in value:
            raise ValueError(f"{key} is invalid")
        result[key] = value
    return result


def input_track_mask(*tracks: int) -> dict[str, bool]:
    """Return OBS SetInputAudioTracks settings with one-based track numbers."""
    selected = set(tracks)
    if any(isinstance(track, bool) or not isinstance(track, int) or track < 1 or track > 6 for track in selected):
        raise ValueError("OBS audio tracks must be integers from 1 through 6")
    return {str(number): number in selected for number in range(1, 7)}


def _verify_input_audio_tracks(client, name: str, expected: dict[str, bool]) -> None:
    getter = getattr(client, "get_input_audio_tracks", None)
    if not callable(getter):
        raise RecorderError("OBS не предоставляет проверку назначенных аудиодорожек.")
    response = _response_dict(getter(name))
    actual = _response_dict(response.get("inputAudioTracks", response.get("input_audio_tracks", {})))
    normalized = {str(key).removeprefix("track"): bool(value) for key, value in actual.items()
                  if str(key).removeprefix("track") in {str(n) for n in range(1, 7)}}
    if normalized != expected:
        raise RecorderError(f"OBS не принял назначение аудиодорожек для {name}.")


def meter_value(event: dict[str, Any], expected_input: str, now: float | None = None,
                max_age: float = 3.0) -> float | None:
    """Extract a fresh linear meter level (0..1); return None for stale events."""
    if not isinstance(event, dict) or event.get("inputName") != expected_input:
        return None
    timestamp = event.get("_received_at", event.get("received_at"))
    try:
        age = (time.time() if now is None else now) - float(timestamp)
        if age < -1 or age > max_age:
            return None
        values = event.get("inputLevelsMul", event.get("levels"))
        if isinstance(values, (int, float)):
            value = float(values)
        elif isinstance(values, (list, tuple)) and values:
            numeric = []
            pending = list(values)
            while pending:
                value = pending.pop()
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric.append(float(value))
                elif isinstance(value, (list, tuple)):
                    if value and isinstance(value[0], (int, float)) and not isinstance(value[0], bool):
                        # OBS emits [magnitude, peak, inputPeak] per channel;
                        # magnitude reflects the actual fader/mute state.
                        numeric.append(float(value[0]))
                    else:
                        pending.extend(value)
            if not numeric:
                return None
            value = max(numeric)
        else:
            return None
        if not 0 <= value <= 1:
            return None
        return value
    except (TypeError, ValueError):
        return None


def _response_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        data = dict(response)
    elif hasattr(response, "__dict__"):
        data = vars(response).copy()
    else:
        return {}
    # obsws-python exposes protocol camelCase keys as snake_case dataclass fields.
    for key, value in list(data.items()):
        if "_" in key:
            head, *tail = key.split("_")
            data.setdefault(head + "".join(part[:1].upper() + part[1:] for part in tail), value)
    return data


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MeetingRecorder:
    """Own one isolated OBS process and queue completed capture conversions."""

    def __init__(self, project_root: str | Path | None = None, archive: MeetingArchive | None = None,
                 *, client_factory=None, process_factory=None, runtime_root: str | Path | None = None,
                 ffprobe=None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        configured_runtime = os.environ.get("VT_SIMPLE_RUNTIME_DIR")
        self.runtime_root = Path(runtime_root or configured_runtime or self.project_root / ".runtime").resolve()
        configured_obs_root = os.environ.get("VT_MEETING_OBS_ROOT")
        self.obs_root = Path(configured_obs_root).expanduser().resolve() if configured_obs_root else self.runtime_root / "obs-studio"
        if configured_obs_root:
            private_boundary = (self.project_root / ".runtime").resolve()
            if not self.obs_root.is_relative_to(private_boundary):
                raise ValueError("VT_MEETING_OBS_ROOT must point inside the project's private .runtime directory")
        self.exe = self.obs_root / "bin" / "64bit" / "obs64.exe"
        self.download_path = self.runtime_root / "downloads" / OBS_ZIP_FILENAME
        if archive is not None:
            self.archive = archive
        else:
            from services.meeting_settings import load_settings
            self.archive = MeetingArchive(load_settings(self.project_root)["archive_root"])
        self.session_path = self.runtime_root / "meeting-recorder" / "session.json"
        self._client_factory = client_factory
        self._process_factory = process_factory or subprocess.Popen
        self._ffprobe = ffprobe
        self._client = None
        self._event_client = None
        self._process = None
        self._session = self._read_session()
        self._selection = self._session.get("selection", {}) if isinstance(self._session, dict) else {}
        self._meters: dict[str, dict[str, Any]] = {}
        self._error: str | None = None
        self._state = "idle"
        self._prepared = False
        self._preparing = threading.Event()
        self._prepare_request_lock = threading.Lock()
        self._prepare_requests = 0

    def _read_session(self) -> dict[str, Any]:
        try:
            value = json.loads(self.session_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_session(self, **values) -> None:
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        session = {**self._session, **values}
        temp = self.session_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, self.session_path)
        self._session = session

    def _client_new(self):
        if self._client_factory:
            return self._client_factory(self._session)
        try:
            import obsws_python
        except ImportError as exc:
            raise RecorderError("Для локальной записи не установлен компонент OBS WebSocket.") from exc
        return obsws_python.ReqClient(host="127.0.0.1", port=int(self._session["port"]),
                                      password=self._session["password"], timeout=5)

    def _reset_clients(self) -> None:
        for client in (self._client, self._event_client):
            if client is None:
                continue
            with __import__("contextlib").suppress(Exception):
                client.disconnect()
        self._client = None
        self._event_client = None

    def _is_owned_process(self) -> bool:
        pid = self._session.get("pid")
        expected = str(self.exe.resolve()).casefold()
        if not isinstance(pid, int) or pid <= 0:
            return False
        if self._process is not None and getattr(self._process, "pid", None) == pid:
            return getattr(self._process, "poll", lambda: None)() is None
        if os.name != "nt":
            return False
        try:
            import ctypes
            from ctypes import wintypes
            query_limited = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(query_limited, False, pid)
            if not handle:
                return False
            try:
                buf = ctypes.create_unicode_buffer(32768)
                size = wintypes.DWORD(len(buf))
                if not ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    return False
                return str(Path(buf.value).resolve()).casefold() == expected
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except (OSError, ValueError):
            return False

    def _private_obs_process_running(self) -> bool:
        """True if any process image lives under our private portable install."""
        try:
            import psutil
        except ImportError:
            return self._is_owned_process()
        expected_root = self.obs_root.resolve()
        for proc in psutil.process_iter(attrs=["exe"]):
            try:
                executable = proc.info.get("exe")
                if executable and Path(executable).resolve().is_relative_to(expected_root):
                    return True
            except (OSError, ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def _recover_stale_unclean_markers(self) -> list[Path]:
        """Move only stale private OBS run_* sentinels; preserve recordings/logs."""
        if self._private_obs_process_running():
            raise RecorderError("Приватный OBS уже запущен; восстановительные маркеры не менялись.")
        sentinel = self.obs_root / "config" / "obs-studio" / ".sentinel"
        obs_root = self.obs_root.resolve()
        if (sentinel.is_symlink() or (hasattr(sentinel, "is_junction") and sentinel.is_junction())):
            raise RecorderError("OBS recovery marker является ссылкой; запуск остановлен для защиты данных.")
        if not sentinel.exists():
            return []
        try:
            resolved_sentinel = sentinel.resolve(strict=True)
            if not resolved_sentinel.is_relative_to(obs_root):
                raise RecorderError("Путь OBS recovery marker выходит за границы private OBS root; запуск остановлен.")
            if hasattr(sentinel, "is_junction") and sentinel.is_junction():
                raise RecorderError("OBS recovery marker является junction; запуск остановлен для защиты данных.")
            if sentinel.is_symlink():
                raise RecorderError("OBS recovery marker является symlink; запуск остановлен для защиты данных.")
        except OSError as exc:
            raise RecorderError("Не удалось безопасно проверить OBS recovery marker.") from exc
        markers = []
        for entry in resolved_sentinel.iterdir():
            if not entry.name.startswith("run_"):
                continue
            try:
                if entry.is_symlink() or (hasattr(entry, "is_junction") and entry.is_junction()):
                    raise RecorderError("OBS run marker является ссылкой; запуск остановлен для защиты данных.")
                resolved_entry = entry.resolve(strict=True)
                if not resolved_entry.is_relative_to(obs_root) or not resolved_entry.is_file():
                    raise RecorderError("OBS run marker выходит за границы private OBS root; запуск остановлен.")
            except OSError as exc:
                raise RecorderError("Не удалось безопасно проверить OBS run marker.") from exc
            markers.append(resolved_entry)
        if not markers:
            return []
        recovery_parent = self.obs_root / "recovery-diagnostics" / "stale-sentinels"
        if not recovery_parent.resolve().is_relative_to(obs_root):
            raise RecorderError("Путь OBS recovery diagnostics выходит за границы private OBS root.")
        recovery_parent.mkdir(parents=True, exist_ok=True)
        if not recovery_parent.resolve().is_relative_to(obs_root):
            raise RecorderError("Путь OBS recovery diagnostics вышел за границы private OBS root.")
        recovery = recovery_parent / str(time.time_ns())
        recovery.mkdir()
        moved = []
        for marker in markers:
            destination = recovery / marker.name
            shutil.move(str(marker), str(destination))
            moved.append(destination)
        return moved

    def _find_obs(self) -> Path:
        if self.exe.is_file():
            return self.exe
        raise RecorderError("OBS не установлен. Используйте подготовку, чтобы загрузить официальный пакет.")

    @staticmethod
    def expected_download() -> dict[str, Any]:
        return {"version": OBS_VERSION, "size_bytes": OBS_ZIP_SIZE, "sha256": OBS_ZIP_SHA256,
                "url": OBS_ZIP_URL}

    def _download_and_install(self) -> None:
        self.download_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(OBS_ZIP_URL, timeout=45) as response, self.download_path.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > OBS_ZIP_SIZE:
                        raise RecorderError("Загруженный пакет OBS превышает ожидаемый размер.")
                    digest.update(block)
                    out.write(block)
        except Exception:
            self.download_path.unlink(missing_ok=True)
            raise
        if size != OBS_ZIP_SIZE or digest.hexdigest() != OBS_ZIP_SHA256:
            self.download_path.unlink(missing_ok=True)
            raise RecorderError("Проверка официального пакета OBS не прошла.")
        staging = self.runtime_root / "obs-studio.installing"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with zipfile.ZipFile(self.download_path) as archive:
            for item in archive.infolist():
                path = PurePosixPath(item.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise RecorderError("Архив OBS содержит недопустимый путь.")
            archive.extractall(staging)
        candidates = list(staging.rglob("obs64.exe"))
        if len(candidates) != 1:
            raise RecorderError("В официальном пакете не найден ожидаемый OBS executable.")
        candidate = candidates[0]
        if len(candidate.parents) < 3 or candidate.parents[1].name.lower() != "bin":
            raise RecorderError("В официальном пакете неверная структура OBS.")
        extracted_root = candidate.parents[2]
        if self.obs_root.exists():
            shutil.rmtree(self.obs_root)
        os.replace(extracted_root, self.obs_root)
        shutil.rmtree(staging, ignore_errors=True)

    def _create_private_configuration(self) -> None:
        """Persist credentials for this runtime's portable OBS only."""
        config = self.obs_root / "config" / "obs-studio" / "plugin_config" / "obs-websocket" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        try:
            current = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        port = self._session.get("port")
        password = self._session.get("password")
        if not isinstance(port, int) or not isinstance(password, str):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            password = uuid.uuid4().hex + uuid.uuid4().hex
        current.update({"server_enabled": True, "server_port": port,
                        "auth_required": True, "server_password": password})
        temp = config.with_suffix(".json.tmp")
        temp.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, config)
        self._save_session(port=port, password=password, exe=str(self.exe.resolve()))

    def _prepare_private_encoder_profile(self) -> None:
        """Write the x264 recording settings to our private profile before OBS starts."""
        obs_root = self.obs_root.resolve()
        profile = self.obs_root / "config" / "obs-studio" / "basic" / "profiles" / OBS_RECORD_PROFILE
        if not profile.resolve().is_relative_to(obs_root):
            raise RecorderError("OBS encoder profile path leaves the private OBS root.")
        profile.mkdir(parents=True, exist_ok=True)
        if not profile.resolve().is_relative_to(obs_root):
            raise RecorderError("OBS encoder profile path leaves the private OBS root.")
        encoder_file = profile / "recordEncoder.json"
        basic_file = profile / "basic.ini"
        if encoder_file.is_symlink() or (hasattr(encoder_file, "is_junction") and encoder_file.is_junction()):
            raise RecorderError("OBS encoder profile is a link; recording startup was stopped.")
        if basic_file.is_symlink() or (hasattr(basic_file, "is_junction") and basic_file.is_junction()):
            raise RecorderError("OBS profile configuration is a link; recording startup was stopped.")
        if encoder_file.exists() and not encoder_file.resolve().is_relative_to(obs_root):
            raise RecorderError("OBS encoder profile file leaves the private OBS root.")
        encoder_file.parent.mkdir(parents=True, exist_ok=True)
        content = {"rate_control": "CRF", "crf": 18, "preset": "ultrafast",
                   "profile": "high", "keyint_sec": 2}
        temp = encoder_file.with_suffix(".json.tmp")
        temp.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, encoder_file)
        if not basic_file.exists():
            basic_file.write_text("[Output]\nMode=Advanced\n\n[AdvOut]\nRecType=Standard\nRecFormat2=mkv\nRecTracks=7\nRecEncoder=obs_x264\n", encoding="utf-8")

    def _launch(self) -> None:
        if self._is_owned_process():
            return
        if self._private_obs_process_running():
            raise RecorderError("Найден процесс OBS в private install без текущей session ownership; он не изменён.")
        self._recover_stale_unclean_markers()
        obs = self._find_obs()
        self._prepare_private_encoder_profile()
        self._create_private_configuration()
        args = [str(obs), "--portable", "--profile", OBS_RECORD_PROFILE,
                "--minimize-to-tray", "--disable-updater", "--disable-shutdown-check"]
        kwargs = {"cwd": str(obs.parent), "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            kwargs["startupinfo"] = subprocess.STARTUPINFO()
            kwargs["startupinfo"].dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kwargs["startupinfo"].wShowWindow = 0
        self._process = self._process_factory(args, **kwargs)
        self._save_session(pid=int(self._process.pid), exe=str(obs.resolve()), started_at=datetime.now(timezone.utc).isoformat())

    def _connect(self, attempts: int = 16):
        if not self._is_owned_process():
            if self._client is not None or self._event_client is not None:
                self._reset_clients()
            raise RecorderError("Не удалось подтвердить владение локальным OBS; подключение к другим экземплярам запрещено.")
        if self._client is not None:
            return self._client
        last_error = None
        for _ in range(attempts):
            try:
                self._client = self._client_new()
                return self._client
            except Exception as exc:
                last_error = exc
                time.sleep(0.35)
        raise RecorderError(f"Не удалось подключиться к приватному OBS: {last_error}")

    def _ensure_scene(self, client, scenes_response: dict[str, Any] | None = None):
        scenes = _response_dict(scenes_response if scenes_response is not None
                                else client.get_scene_list()).get("scenes", [])
        names = {row.get("sceneName") for row in scenes if isinstance(row, dict)}
        scene = "VT Meeting Capture"
        if scene not in names:
            client.create_scene(scene)
        return scene

    def _wait_for_frontend_ready(self, client) -> dict[str, Any]:
        """Wait for OBS frontend readiness after WebSocket connect, retrying only error 207."""
        deadline = time.monotonic() + _OBS_FRONTEND_READY_TIMEOUT
        while True:
            try:
                return _response_dict(client.get_scene_list())
            except Exception as exc:
                if getattr(exc, "code", None) != 207:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RecorderError(
                        "OBS подключён, но не завершил подготовку интерфейса за 30 секунд. "
                        "Попробуйте подготовить OBS снова."
                    ) from exc
                time.sleep(min(_OBS_FRONTEND_READY_POLL_INTERVAL, remaining))

    def _existing_inputs(self, client) -> set[str]:
        rows = _response_dict(client.get_input_list()).get("inputs", [])
        return {row.get("inputName") for row in rows if isinstance(row, dict)}

    @staticmethod
    def _obs_window_identity(value: str) -> tuple[str, str, str | None] | None:
        """Extract class and executable from OBS 33's title:class:executable value."""
        if not isinstance(value, str):
            return None
        fields = value.rsplit(":", 2)
        if len(fields) != 3:
            return None
        _title, window_class, executable = (part.strip() for part in fields)
        if not window_class:
            return None
        match = re.fullmatch(r"(msedge|chrome|firefox)\.exe", executable, re.IGNORECASE)
        if match is None:
            return None
        # OBS's advertised value has no process ID; do not infer one from title text.
        return window_class.casefold(), f"{match.group(1).casefold()}.exe", None

    @classmethod
    def _resolve_current_browser(cls, browser_id: str, browsers: list[dict]) -> dict:
        exact = next((row for row in browsers if row.get("id") == browser_id or
                      row.get("obs_window") == browser_id), None)
        if exact is not None:
            return exact
        previous = cls._obs_window_identity(browser_id)
        if previous is None:
            raise ValueError("Выбранное окно браузера больше недоступно.")
        class_name, executable, process_id = previous
        matches = []
        for row in browsers:
            identity = cls._obs_window_identity(row.get("obs_window", row.get("id", "")))
            if identity and identity[0] == class_name and identity[1] == executable:
                matches.append((row, identity))
        if process_id:
            same_process = [row for row, identity in matches if identity[2] == process_id]
            if same_process:
                return same_process[0]
        if matches:
            # Every candidate is a currently advertised OBS browser window with
            # the same validated class+executable; title changes are harmless.
            return matches[0][0]
        raise ValueError("Выбранное окно браузера больше недоступно.")

    def _ensure_capture_sources(self, client, selection: dict[str, str], monitors: list[dict],
                                browsers: list[dict], microphones: list[dict]):
        selected_monitor = next((item for item in monitors if item["id"] == selection["monitor_id"]), None)
        if selected_monitor is None:
            raise ValueError("Выбранный монитор больше недоступен.")
        browser = self._resolve_current_browser(selection["browser_id"], browsers)
        selection = dict(selection)
        selection["browser_id"] = browser["id"]
        microphone = next((item for item in microphones if item["id"] == selection["microphone_id"]), None)
        if microphone is None:
            raise ValueError("Выбранный микрофон больше недоступен.")
        scene = self._ensure_scene(client)
        existing = self._existing_inputs(client)
        priority = self._executable_match_priority(client)
        specs = [
            ("VT Meeting Monitor", "monitor_capture", {"monitor_id": selection["monitor_id"], "capture_cursor": True}),
            # Keep matching the browser process when its title changes as users
            # move between meeting and Jira tabs. OBS's exact window value is
            # preserved while its documented executable priority is explicit.
            ("VT Meeting Browser Audio", "wasapi_process_output_capture",
             {"window": browser["obs_window"], "priority": priority}),
            ("VT Meeting Microphone", "wasapi_input_capture", {"device_id": selection["microphone_id"]}),
        ]
        for name, kind, settings in specs:
            if kind not in _VALID_SOURCE_KINDS:
                raise ValueError("Unsupported OBS source kind")
            if name not in existing:
                client.create_input(scene, name, kind, settings, True)
            else:
                client.set_input_settings(name, settings, True)
        # Monitor capture is video-only and OBS rejects audio-track requests for it.
        track_roles = (("VT Meeting Browser Audio", input_track_mask(1, 2)),
                       ("VT Meeting Microphone", input_track_mask(1, 3)))
        for name, tracks in track_roles:
            client.set_input_audio_tracks(name, tracks)
            _verify_input_audio_tracks(client, name, tracks)
        # Disable global desktop audio: the selected browser + microphone define the mix.
        special = _response_dict(client.get_special_inputs())
        for key in ("desktop1", "desktop2", "mic1", "mic2", "mic3", "mic4"):
            name = special.get(key)
            if name:
                client.set_input_mute(name, True)
        width, height = int(selected_monitor.get("width", 0)), int(selected_monitor.get("height", 0))
        if width <= 0 or height <= 0:
            raise RecorderError("OBS не сообщил физическое разрешение выбранного монитора; захват не запущен.")
        client.set_video_settings(30, 1, width, height, width, height)
        self._select_and_verify_capture_scene(client, scene, width, height)
        self._configure_recording(client)
        return selection

    @staticmethod
    def _select_and_verify_capture_scene(client, scene: str, width: int, height: int) -> None:
        """Make the native-resolution monitor scene the actual recording scene."""
        items_response = _response_dict(client.get_scene_item_list(scene))
        items = items_response.get("sceneItems", items_response.get("scene_items", []))
        monitor = next((item for item in items if isinstance(item, dict) and
                        item.get("sourceName", item.get("source_name")) == "VT Meeting Monitor"), None)
        if monitor is None:
            create_item = getattr(client, "create_scene_item", None)
            if not callable(create_item):
                raise RecorderError("OBS не добавил захват монитора в сцену записи.")
            created = _response_dict(create_item(scene, "VT Meeting Monitor", True))
            item_id = created.get("sceneItemId", created.get("scene_item_id"))
        else:
            item_id = monitor.get("sceneItemId", monitor.get("scene_item_id"))
        if item_id is None:
            raise RecorderError("OBS не сообщил идентификатор элемента захвата монитора.")
        client.set_scene_item_enabled(scene, int(item_id), True)
        transform = {"positionX": 0.0, "positionY": 0.0, "rotation": 0.0,
                     "scaleX": 1.0, "scaleY": 1.0, "alignment": 5,
                     "boundsType": "OBS_BOUNDS_NONE", "boundsAlignment": 0,
                     "cropLeft": 0, "cropRight": 0, "cropTop": 0, "cropBottom": 0}
        client.set_scene_item_transform(scene, int(item_id), transform)
        # A monitor source reports 0x0 until OBS activates the scene and the
        # renderer initializes it. Select first, then wait briefly for real
        # dimensions; never replace an unavailable measurement with a guess.
        client.set_current_program_scene(scene)
        current = _response_dict(client.get_current_program_scene())
        if current.get("currentProgramSceneName", current.get("current_program_scene_name")) != scene:
            raise RecorderError("OBS не переключился на сцену захвата встречи.")
        deadline = time.monotonic() + 5.0
        observed = {}
        while time.monotonic() < deadline:
            response = _response_dict(client.get_scene_item_transform(scene, int(item_id)))
            observed = _response_dict(response.get("sceneItemTransform", response.get("scene_item_transform", response)))
            source_width = observed.get("sourceWidth", observed.get("source_width"))
            source_height = observed.get("sourceHeight", observed.get("source_height"))
            if source_width and source_height:
                break
            time.sleep(0.1)
        for key, expected in (("sourceWidth", width), ("sourceHeight", height),
                              ("width", width), ("height", height)):
            value = observed.get(key, observed.get(re.sub(r"[A-Z]", lambda m: "_" + m.group(0).lower(), key)))
            if value is None or abs(float(value) - float(expected)) > 0.5:
                raise RecorderError("OBS не подтвердил полноразмерный захват монитора в native resolution.")
        if observed.get("boundsType", observed.get("bounds_type")) not in {None, "OBS_BOUNDS_NONE"}:
            raise RecorderError("OBS изменил размер захвата монитора через границы сцены.")

    def _executable_match_priority(self, client) -> int:
        """Resolve OBS's executable-name match from its advertised properties."""
        get_defaults = getattr(client, "get_input_default_settings", None)
        if not callable(get_defaults):
            raise RecorderError("OBS не предоставляет проверку настроек захвата звука приложения.")
        defaults = _response_dict(get_defaults("wasapi_process_output_capture"))
        # An empty defaultInputSettings object is valid; the property list is
        # authoritative for available priority choices.
        _response_dict(defaults.get("defaultInputSettings", defaults.get("default_input_settings", {})))
        probe = "VT Meeting Audio Priority Probe"
        if probe not in self._existing_inputs(client):
            client.create_input(self._ensure_scene(client), probe, "wasapi_process_output_capture", {}, False)
        try:
            values = _response_dict(client.get_input_properties_list_property_items(probe, "priority"))
            items = values.get("propertyItems", values.get("property_items", []))
            parsed_items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("itemName", item.get("item_name", ""))).casefold()
                value = item.get("itemValue", item.get("item_value"))
                try:
                    parsed_items.append((name, int(value)))
                except (TypeError, ValueError):
                    continue
            for name, value in parsed_items:
                if ("executable" in name or re.search(r"\bexe\b", name) or
                        re.search(r"исполняем|выполняем", name)):
                    return value
            # OBS 33.0.1 adds title, class, executable in this exact order. If its
            # labels use an unknown locale, accept that documented final option
            # only when all three distinct integer choices are advertised.
            advertised = [value for _, value in parsed_items]
            if len(advertised) == 3 and len(set(advertised)) == 3:
                return advertised[2]
            raise RecorderError("OBS не объявил вариант приоритета по имени исполняемого файла браузера.")
        finally:
            with __import__("contextlib").suppress(Exception):
                client.remove_input(probe)

    @staticmethod
    def _configure_recording(client):
        """Set a private multitrack MKV profile; check each setting is accepted."""
        settings = (("Output", "Mode", "Advanced"),
                    ("AdvOut", "RecType", "Standard"),
                    ("AdvOut", "RecFormat2", "mkv"),
                    ("AdvOut", "RecTracks", "7"),
                    ("AdvOut", "RecEncoder", "obs_x264"))
        for category, name, value in settings:
            client.set_profile_parameter(category, name, value)
        # Verify the basic profile contract instead of claiming a multitrack success.
        current = _response_dict(client.get_profile_parameter("Output", "Mode"))
        if current and str(current.get("parameterValue", "Advanced")).casefold() != "advanced":
            raise RecorderError("OBS не принял расширенный профиль записи.")
        for name, expected in (("RecFormat2", "mkv"), ("RecTracks", "7"), ("RecEncoder", "obs_x264")):
            value = _response_dict(client.get_profile_parameter("AdvOut", name)).get("parameterValue")
            if value is not None and str(value).casefold() != expected.casefold():
                raise RecorderError(f"OBS не принял настройку {name}.")

    def _enumerate_monitors(self, client, scenes_response: dict[str, Any] | None = None) -> list[dict]:
        self._ensure_scene(client, scenes_response)
        rows = _response_dict(client.get_monitor_list()).get("monitors", [])
        properties = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            name = str(item.get("monitorName", item.get("monitor_name", "Display")))
            properties.append({"name": name,
                "width": int(item.get("monitorWidth", item.get("width", 0)) or 0),
                "height": int(item.get("monitorHeight", item.get("height", 0)) or 0)})
        probe = "VT Meeting Monitor Probe"
        if probe not in self._existing_inputs(client):
            client.create_input("VT Meeting Capture", probe, "monitor_capture", {}, False)
        values = _response_dict(client.get_input_properties_list_property_items(probe, "monitor_id")).get("propertyItems", [])
        result = []
        for item in values:
            if not isinstance(item, dict) or item.get("itemValue") is None:
                continue
            value = str(item["itemValue"])
            name = str(item.get("itemName", "")).strip()
            if (not value or value.casefold() == "dummy" or
                    re.search(r"\b(?:select|choose)\b|выберите|экран для захвата", name, re.IGNORECASE)):
                continue
            # GetMonitorList may report logical DPI-scaled dimensions while
            # monitor_capture exposes the actual native dimensions in its item label.
            dimensions = re.search(r"(?<!\d)(\d{3,5})\s*[x×]\s*(\d{3,5})(?!\d)", name)
            width = int(dimensions.group(1)) if dimensions else 0
            height = int(dimensions.group(2)) if dimensions else 0
            matched = next((row for row in properties if row["name"] == name), None)
            if width <= 0 or height <= 0:
                if matched and matched["width"] > 0 and matched["height"] > 0:
                    width, height = matched["width"], matched["height"]
            if width <= 0 or height <= 0:
                raise RecorderError("OBS не сообщил физическое разрешение выбранного монитора; захват не запущен.")
            result.append({"id": value, "name": name, "width": width, "height": height})
        with __import__("contextlib").suppress(Exception):
            client.remove_input(probe)
        return result

    def _enumerate_windows(self, client) -> list[dict]:
        """Use OBS's window property values verbatim; never treat them as video sources."""
        self._ensure_scene(client)
        all_inputs = self._existing_inputs(client)
        probe = "VT Meeting Window Probe"
        if probe not in all_inputs:
            client.create_input("VT Meeting Capture", probe, "window_capture", {}, False)
        values = _response_dict(client.get_input_properties_list_property_items(probe, "window")).get("propertyItems", [])
        result, seen = [], set()
        for item in values:
            value = str(item.get("itemValue", "")) if isinstance(item, dict) else ""
            label = str(item.get("itemName", "")) if isinstance(item, dict) else ""
            match = re.search(r"(?:^|:)(msedge|chrome|firefox)\.exe", value, re.IGNORECASE)
            if not match or value in seen:
                continue
            seen.add(value)
            result.append({"id": value, "name": label or value, "obs_window": value,
                           "browser": match.group(1).lower()})
        # Probe is properties-only and has no capture output; remove it immediately.
        with __import__("contextlib").suppress(Exception):
            client.remove_input(probe)
        return result

    def _enumerate_microphones(self, client) -> list[dict]:
        self._ensure_scene(client)
        all_inputs = self._existing_inputs(client)
        probe = "VT Meeting Microphone Probe"
        if probe not in all_inputs:
            scene = self._ensure_scene(client)
            client.create_input(scene, probe, "wasapi_input_capture", {"device_id": "default"}, False)
        values = _response_dict(client.get_input_properties_list_property_items(probe, "device_id")).get("propertyItems", [])
        result = [{"id": str(item.get("itemValue", "")), "name": str(item.get("itemName", "Microphone"))}
                  for item in values if isinstance(item, dict) and item.get("itemValue") is not None]
        if not any(row["id"] == "default" for row in result):
            result.insert(0, {"id": "default", "name": "Устройство по умолчанию"})
        with __import__("contextlib").suppress(Exception):
            client.remove_input(probe)
        return result

    def on_input_volume_meters(self, event):
        data = _response_dict(event)
        received_at = time.time()
        for raw in data.get("inputs", []):
            item = _response_dict(raw)
            name = item.get("inputName")
            if name in {"VT Meeting Browser Audio", "VT Meeting Microphone"}:
                item["_received_at"] = received_at
                self._meters[name] = item

    def _start_event_client(self):
        if self._event_client is not None or self._client_factory:
            return
        try:
            import obsws_python
            from obsws_python import Subs
            event_client = obsws_python.EventClient(host="127.0.0.1", port=int(self._session["port"]),
                password=self._session["password"], subs=Subs.INPUTVOLUMEMETERS)
            event_client.callback.register(self.on_input_volume_meters)
            self._event_client = event_client
        except Exception as exc:
            self._error = f"Индикаторы уровня недоступны: {exc}"

    def _status_template(self, *, state: str | None = None) -> dict[str, Any]:
        return {"available": self.exe.is_file(), "connected": False, "state": state or self._state,
                  "duration_seconds": None, "monitors": [], "browsers": [], "microphones": [],
                  "title": self._session.get("title"),
                  "selection": {key: self._selection.get(key) for key in ("monitor_id", "browser_id", "microphone_id")},
                  "preview_url": "/api/meetings/recorder/preview",
                  "levels": {"browser": None, "microphone": None, "stale": True}, "error": self._error}

    def _status_during_prepare(self) -> dict[str, Any]:
        status = self._status_template()
        if self._session.get("recording_path") or self._state == "recording":
            status["state"] = "recording"
            status["error"] = "Найдена незавершённая запись; остановите её, чтобы проверить файл и сохранить встречу."
        elif self._state == "stopping":
            status["state"] = "stopping"
        else:
            status["state"] = "preparing"
            status["error"] = None
        return status

    def _begin_prepare_request(self) -> None:
        with self._prepare_request_lock:
            self._prepare_requests += 1
            self._preparing.set()

    def _end_prepare_request(self) -> None:
        with self._prepare_request_lock:
            self._prepare_requests -= 1
            if self._prepare_requests == 0:
                self._preparing.clear()

    def _status_from_obs(self) -> dict[str, Any]:
        status = self._status_template()
        try:
            client = self._connect(attempts=1)
            status["connected"] = True
            record = _response_dict(client.get_record_status())
            active = bool(record.get("outputActive", record.get("output_active", False)))
            if active:
                hold_activity("meeting-recorder")
                status["state"] = "recording"
                duration = record.get("outputDuration", record.get("output_duration"))
                if isinstance(duration, (int, float)):
                    status["duration_seconds"] = max(0, int(duration / 1000))
            elif self._session.get("recording_path"):
                release_activity("meeting-recorder")
                status["state"] = "recording"
                status["error"] = "Найдена незавершённая запись; остановите её, чтобы проверить файл и сохранить встречу."
            else:
                release_activity("meeting-recorder")
            status["monitors"] = list(self._session.get("monitors", []))
            status["browsers"] = list(self._session.get("browsers", []))
            status["microphones"] = list(self._session.get("microphones", []))
            browser = meter_value(self._meters.get("VT Meeting Browser Audio", {}), "VT Meeting Browser Audio")
            mic = meter_value(self._meters.get("VT Meeting Microphone", {}), "VT Meeting Microphone")
            # The private UI contract uses a 0..100 progress value; OBS supplies
            # linear multipliers in the 0..1 range.
            status["levels"] = {"browser": browser * 100 if browser is not None else None,
                                "microphone": mic * 100 if mic is not None else None,
                                "stale": browser is None or mic is None}
            self._start_event_client()
            if self._error:
                status["error"] = self._error
        except Exception as exc:
            status["connected"] = False
            process_running = self._private_obs_process_running()
            if not process_running:
                # A persisted path is recoverable media, not evidence that OBS
                # is still producing a recording.
                release_activity("meeting-recorder")
            if self._state == "recording" or self._session.get("recording_path"):
                status["state"] = "recording"
                status["error"] = "Связь с OBS потеряна; файл записи сохранён для восстановления."
            else:
                status["state"] = "unavailable" if not status["available"] else "idle"
                if process_running:
                    status["error"] = str(exc)
                else:
                    # Before the user prepares OBS, this is the normal state.
                    status["error"] = None
        self._state = status["state"]
        return status

    def status(self) -> dict[str, Any]:
        """Return UI-safe status. May reconnect only to a previously owned OBS PID."""
        while True:
            if self._preparing.is_set():
                return self._status_during_prepare()
            if _LOCK.acquire(timeout=0.05):
                try:
                    if self._preparing.is_set():
                        return self._status_during_prepare()
                    return self._status_from_obs()
                finally:
                    _LOCK.release()

    def prepare(self) -> dict[str, Any]:
        """Install pinned official OBS if needed, launch our private copy, and connect."""
        if self._state == "recording" or self._session.get("recording_path"):
            raise RecorderError("Нельзя подготовить OBS, пока есть незавершённая запись. Сначала остановите или сохраните её.")
        self._begin_prepare_request()
        try:
            with _LOCK:
                if self._state == "recording" or self._session.get("recording_path"):
                    raise RecorderError("Нельзя подготовить OBS, пока есть незавершённая запись. Сначала остановите или сохраните её.")
                self._state = "preparing"
                self._error = None
                try:
                    if not self.exe.is_file():
                        self._download_and_install()
                    if not self._is_owned_process():
                        self._reset_clients()
                    self._launch()
                    self._state = "starting"
                    client = self._connect()
                    scenes_response = self._wait_for_frontend_ready(client)
                    self._save_session(monitors=self._enumerate_monitors(client, scenes_response),
                                       browsers=self._enumerate_windows(client),
                                       microphones=self._enumerate_microphones(client))
                    self._prepared = True
                    self._error = None
                    self._state = "idle"
                    return self._status_from_obs()
                except Exception as exc:
                    self._error = str(exc)
                    self._state = "unavailable" if not self.exe.is_file() else "idle"
                    raise RecorderError(self._error) from exc
        finally:
            self._end_prepare_request()

    def configure(self, selection: dict[str, Any]) -> dict[str, Any]:
        """Configure the selected monitor/browser/mic, refusing changes during recording."""
        with _LOCK:
            selection = validate_selection(selection)
            status = self._status_from_obs()
            if status["state"] == "recording":
                raise RecorderError("Нельзя менять источники во время записи.")
            if not status["connected"]:
                raise RecorderError(status.get("error") or "Сначала подготовьте OBS.")
            selection = self._ensure_capture_sources(self._client, selection, status["monitors"], status["browsers"], status["microphones"])
            self._selection = selection
            self._save_session(selection=selection)
            return self._status_from_obs()

    def _preview_bytes(self) -> bytes:
        client = self._connect(attempts=1)
        value = _response_dict(client.get_source_screenshot("VT Meeting Capture", "jpg", 960, 600, 85))
        image = value.get("imageData", value.get("image_data", ""))
        if isinstance(image, str) and image.startswith("data:image/"):
            image = image.split(",", 1)[1]
        import base64
        raw = base64.b64decode(image, validate=True)
        if not raw.startswith(b"\xff\xd8"):
            raise RecorderError("OBS вернул недопустимый preview frame.")
        return raw

    def preview(self) -> bytes:
        """Return a JPEG preview from the owned OBS capture scene."""
        with _LOCK:
            return self._preview_bytes()

    def start(self, title: str = "Meeting") -> dict[str, Any]:
        """Start MKV recording after sources have been explicitly configured."""
        with _LOCK:
            if not isinstance(title, str) or not title.strip():
                raise ValueError("title must be a non-empty string")
            status = self._status_from_obs()
            if not status["connected"]:
                raise RecorderError(status.get("error") or "OBS не подключён.")
            if not all(self._selection.get(key) for key in ("monitor_id", "browser_id", "microphone_id")):
                raise RecorderError("Перед записью выберите монитор, браузер и микрофон.")
            # Ensure the screen source and multitrack profile actually exist before recording.
            selection = self._ensure_capture_sources(self._client, self._selection, status["monitors"], status["browsers"], status["microphones"])
            if selection != self._selection:
                self._selection = selection
                self._save_session(selection=selection)
            record = _response_dict(self._client.get_record_status())
            if record.get("outputActive", record.get("output_active", False)):
                raise RecorderError("OBS уже ведёт запись.")
            directory = self.runtime_root / "meeting-recorder" / "recordings"
            directory.mkdir(parents=True, exist_ok=True)
            # OBS's private record directory and name format persist with the private profile.
            self._client.set_record_directory(str(directory))
            filename = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
            self._client.set_profile_parameter("Output", "FilenameFormatting", filename)
            expected = directory / f"{filename}.mkv"
            self._save_session(recording_path=str(expected), started_at=datetime.now(timezone.utc).isoformat(), title=title,
                audio_track_roles={"1": "mix", "2": "browser", "3": "microphone"})
            try:
                self._client.start_record()
            except Exception:
                if not expected.exists():
                    self._session.pop("recording_path", None)
                    self._save_session(recording_path=None)
                raise
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                info = _response_dict(self._client.get_record_status())
                if info.get("outputActive", info.get("output_active", False)):
                    hold_activity("meeting-recorder")
                    output = info.get("outputPath", info.get("output_path"))
                    expected = Path(output) if output else directory / f"{filename}.mkv"
                    self._save_session(recording_path=str(expected), started_at=datetime.now(timezone.utc).isoformat(), title=title,
                        audio_track_roles={"1": "mix", "2": "browser", "3": "microphone"})
                    self._state = "recording"
                    return self._status_from_obs()
                time.sleep(0.2)
            if expected.exists():
                self._state = "recording"
                raise RecorderError("OBS не подтвердил состояние, но файл записи найден и сохранён для восстановления.")
            self._session.pop("recording_path", None)
            self._save_session(recording_path=None)
            raise RecorderError("OBS не подтвердил начало записи.")

    def stop(self) -> dict[str, Any]:
        """Stop the capture and queue its validated media conversion."""
        with _LOCK:
            self._state = "stopping"
            path = Path(str(self._session.get("recording_path", ""))).resolve()
            try:
                capture_root = (self.runtime_root / "meeting-recorder" / "recordings").resolve()
                if self._is_owned_process():
                    client = self._connect(attempts=1)
                    status = _response_dict(client.get_record_status())
                    if status.get("outputActive", status.get("output_active", False)):
                        hold_activity("meeting-recorder")
                        client.stop_record()
                    else:
                        release_activity("meeting-recorder")
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        status = _response_dict(client.get_record_status())
                        if not status.get("outputActive", status.get("output_active", False)):
                            release_activity("meeting-recorder")
                            break
                        time.sleep(0.2)
                    else:
                        raise RecorderError("OBS продолжает закрывать файл; повторите остановку через несколько секунд.")
                    output = status.get("outputPath", status.get("output_path"))
                    if output:
                        path = Path(output).resolve()
                elif not path.is_file():
                    release_activity("meeting-recorder")
                    raise RecorderError("Приватный OBS уже завершён, файл записи не найден; сеанс сохранён для восстановления.")
                else:
                    release_activity("meeting-recorder")
                # If our process has ended, this is an interrupted but recoverable recording.
                # Keep the staged MKV until its conversion is durably published.
                if not path.is_file() or not path.is_relative_to(capture_root):
                    raise RecorderError("Не найден собственный MKV файл; он не удалён и сохранён для восстановления.")
                title = self._session.get("title") or path.stem
                started_at = self._session.get("started_at")
                track_roles = {"1": "mix", "2": "browser", "3": "microphone"}
                import web_api
                job = web_api.enqueue_recording_conversion(str(path), title=title,
                                                           started_at=started_at,
                                                           track_roles=track_roles,
                                                           cleanup_root=str(capture_root))
                self._session.pop("recording_path", None)
                self._session.pop("started_at", None)
                self._session.pop("title", None)
                self._save_session(recording_path=None, started_at=None, title=None)
                self._state = "idle"
                self._error = None
                return {"job": job, "recorder": self._status_from_obs()}
            except Exception as exc:
                self._state = "recording" if path.exists() else "idle"
                self._error = str(exc)
                raise


_DEFAULT_RECORDER: MeetingRecorder | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_recorder() -> MeetingRecorder:
    global _DEFAULT_RECORDER
    with _DEFAULT_LOCK:
        from services.meeting_settings import load_settings
        project_root = Path(__file__).resolve().parents[1]
        archive_root = Path(load_settings(project_root)["archive_root"]).resolve()
        old_root = _DEFAULT_RECORDER.archive.root.resolve() if _DEFAULT_RECORDER else None
        if _DEFAULT_RECORDER is None or old_root != archive_root:
            if _DEFAULT_RECORDER and recording_session_active(_DEFAULT_RECORDER.runtime_root):
                raise RecorderError("Archive location cannot change during a recording.")
            _DEFAULT_RECORDER = MeetingRecorder(project_root, archive=MeetingArchive(archive_root))
        return _DEFAULT_RECORDER

