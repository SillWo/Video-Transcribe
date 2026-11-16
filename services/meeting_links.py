"""Small, read-only localhost viewer for evidence in the meeting archive.

The module deliberately uses only the Python standard library at import time.
The HTTP server is started on demand by the MCP tool, never at application boot.
"""
from __future__ import annotations

import html
import json
import math
import mimetypes
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVIDENCE_SCHEME = "video-transcribe"
_PROTOCOL_REGISTRY_KEY = r"Software\Classes\video-transcribe"
_PROTOCOL_OWNER_MARKER = "VideoTranscribeEvidenceHandler"
_PROTOCOL_SETTINGS_VALUE = "VideoTranscribeSettingsPath"
_PROTOCOL_PROJECT_VALUE = "VideoTranscribeProjectRoot"


def evidence_protocol_status() -> dict[str, Any]:
    if os.name != "nt":
        return {"supported": False, "registered": False, "owned": False, "conflict": False}
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _PROTOCOL_REGISTRY_KEY, 0, winreg.KEY_READ) as key:
            try:
                marker, _ = winreg.QueryValueEx(key, _PROTOCOL_OWNER_MARKER)
            except FileNotFoundError:
                return {"supported": True, "registered": True, "owned": False, "conflict": True}
            if marker != "1":
                return {"supported": True, "registered": True, "owned": False, "conflict": True}
        command_path = _PROTOCOL_REGISTRY_KEY + r"\shell\open\command"
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, command_path, 0, winreg.KEY_READ) as command_key:
                command, _ = winreg.QueryValueEx(command_key, "")
        except FileNotFoundError:
            return {"supported": True, "registered": True, "owned": False, "conflict": True}
        owned = isinstance(command, str) and "meeting_links.py" in command.casefold() and "--open-uri" in command and '"%1"' in command
        return {"supported": True, "registered": True, "owned": owned, "conflict": not owned}
    except FileNotFoundError:
        return {"supported": True, "registered": False, "owned": False, "conflict": False}


def register_evidence_protocol(python_executable: str | Path,
                               settings: str | Path | None = None,
                               project_root: str | Path | None = None) -> dict[str, Any]:
    status = evidence_protocol_status()
    if not status["supported"] or status["conflict"]:
        return status
    import winreg
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    settings_file = Path(settings).expanduser().resolve() if settings else _default_settings_path()
    python_path = Path(python_executable).resolve()
    pythonw = python_path.with_name("pythonw.exe")
    executable = pythonw if pythonw.is_file() else python_path
    script = Path(__file__).resolve()
    settings_arg = f' --settings "{Path(settings).resolve()}"' if settings else ""
    command = f'"{executable}" "{script}" --open-uri "%1"{settings_arg} --project-root "{root}"'
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, _PROTOCOL_REGISTRY_KEY, 0, winreg.KEY_WRITE) as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "URL:Video Transcribe Evidence")
        winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
        winreg.SetValueEx(key, _PROTOCOL_OWNER_MARKER, 0, winreg.REG_SZ, "1")
        winreg.SetValueEx(key, _PROTOCOL_SETTINGS_VALUE, 0, winreg.REG_SZ, str(settings_file))
        winreg.SetValueEx(key, _PROTOCOL_PROJECT_VALUE, 0, winreg.REG_SZ, str(root))
    command_key = _PROTOCOL_REGISTRY_KEY + r"\shell\open\command"
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, command_key, 0, winreg.KEY_WRITE) as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, command)
    return evidence_protocol_status()


def _default_settings_path() -> Path:
    configured = os.environ.get("VT_MEETING_SETTINGS")
    if configured:
        return Path(configured).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return (Path(local) / "VideoTranscribe" / "meetings-settings.json").resolve()
    return (Path.home() / ".video-transcribe" / "meetings-settings.json").resolve()


def evidence_protocol_binding() -> dict[str, str] | None:
    """Return the owned URI handler's settings binding, if one is registered."""
    if os.name != "nt":
        return None
    status = evidence_protocol_status()
    if not status.get("owned"):
        return None
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _PROTOCOL_REGISTRY_KEY, 0, winreg.KEY_READ) as key:
            settings, _ = winreg.QueryValueEx(key, _PROTOCOL_SETTINGS_VALUE)
            project_root, _ = winreg.QueryValueEx(key, _PROTOCOL_PROJECT_VALUE)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(settings, str) or not settings or not isinstance(project_root, str) or not project_root:
        return None
    return {"settings_path": settings, "project_root": project_root}


def _configured_archive_root(binding: dict[str, str]) -> Path:
    settings_file = Path(binding["settings_path"]).expanduser().resolve()
    try:
        config = json.loads(settings_file.read_text(encoding="utf-8"))
        configured = config.get("archive_root") if isinstance(config, dict) else None
        if configured:
            return Path(configured).expanduser().resolve()
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return (Path(binding["project_root"]).expanduser().resolve() / ".runtime" / "meetings").resolve()


def evidence_protocol_matches_archive(root: str | Path) -> bool:
    """Check whether the registered URI handler currently resolves to this archive."""
    binding = evidence_protocol_binding()
    if binding is None:
        return False
    try:
        return _configured_archive_root(binding) == Path(root).expanduser().resolve()
    except (OSError, ValueError, TypeError, KeyError):
        return False


def unregister_evidence_protocol() -> dict[str, Any]:
    status = evidence_protocol_status()
    if not status["owned"]:
        return status
    import winreg

    root_children = _registry_children(winreg, _PROTOCOL_REGISTRY_KEY)
    shell_children = _registry_children(winreg, _PROTOCOL_REGISTRY_KEY + r"\shell")
    open_children = _registry_children(winreg, _PROTOCOL_REGISTRY_KEY + r"\shell\open")
    command_children = _registry_children(winreg, _PROTOCOL_REGISTRY_KEY + r"\shell\open\command")
    if root_children != ["shell"] or shell_children != ["open"] or open_children != ["command"] or command_children:
        return status
    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, _PROTOCOL_REGISTRY_KEY + r"\shell\open\command")
    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, _PROTOCOL_REGISTRY_KEY + r"\shell\open")
    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, _PROTOCOL_REGISTRY_KEY + r"\shell")
    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, _PROTOCOL_REGISTRY_KEY)
    return evidence_protocol_status()


def _registry_children(winreg: Any, path: str) -> list[str]:
    children = []
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_READ) as key:
            index = 0
            while True:
                try:
                    children.append(winreg.EnumKey(key, index))
                    index += 1
                except OSError:
                    break
    except FileNotFoundError:
        return []
    return children


def _inside(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Evidence path is outside the configured archive")
    return candidate


def _get_evidence(root: Path, meeting_id: str, at: float) -> dict[str, Any]:
    # Import only when a human opens an evidence link.
    try:
        from services.meeting_archive import MeetingArchive
    except ImportError:
        from services.meetings import MeetingArchive  # type: ignore[no-redef]
    archive = MeetingArchive(root, read_only=True)
    meeting = archive.get_meeting(meeting_id)
    if not meeting:
        raise FileNotFoundError("Meeting not found")
    meeting_dir = _inside(root, Path(archive.meeting_dir(meeting_id)))
    original = archive.original_path(meeting_id)
    if original:
        original = _inside(root, Path(original))
        if not original.is_file():
            original = None
    page = archive.timeline(meeting_id, start=max(0.0, at - 20), end=at + 20, limit=100, offset=0)
    items = page.get("items", []) if isinstance(page, dict) else page
    frames = [item for item in items if item.get("kind") == "screen" and item.get("frame")]
    visible: list[tuple[float, dict[str, Any]]] = []
    for event in frames:
        try:
            if event.get("start") is None:
                continue
            event_start = float(event.get("start", 0))
            event_end = float(event.get("end")) if event.get("end") is not None else float("inf")
        except (TypeError, ValueError):
            continue
        if event_start <= at < event_end:
            visible.append((event_start, event))
    nearest = max(visible, key=lambda row: row[0], default=(0.0, None))[1]
    frame = None
    if nearest and nearest.get("frame"):
        relative = Path(str(nearest["frame"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Frame paths must remain inside the meeting directory")
        frame = _inside(meeting_dir, meeting_dir / relative)
        if not frame.is_file():
            frame = None
    audio_dir = meeting_dir / "audio"
    audio = next(iter(sorted(audio_dir.glob("*.opus"))), None) if audio_dir.is_dir() else None
    if audio:
        audio = _inside(meeting_dir, audio)
    return {"meeting": meeting, "source": original, "audio": audio, "frame": frame, "at": max(0.0, at)}


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Viewer:
    def __init__(self, root: Path, port: int = 0, token: str | None = None):
        self.root = root.resolve()
        self.token = token or secrets.token_urlsafe(24)
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                return

            def _fail(self, status: int, message: str) -> None:
                data = message.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                host = self.headers.get("Host", "").split(":", 1)[0].strip("[]").lower()
                if host not in {"127.0.0.1", "localhost", "::1"}:
                    return self._fail(403, "Loopback access only")
                parts = urllib.parse.urlsplit(self.path)
                prefix = "/" + viewer.token + "/"
                if not parts.path.startswith(prefix):
                    return self._fail(404, "Not found")
                route = parts.path[len(prefix):]
                query = urllib.parse.parse_qs(parts.query)
                try:
                    if route == "health":
                        payload = b"ok"
                        self.send_response(200)
                        self.send_header("Content-Type", "text/plain")
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        return
                    if route.startswith("evidence/"):
                        meeting_id = urllib.parse.unquote(route[len("evidence/"):])
                        if not meeting_id or "/" in meeting_id or "\\" in meeting_id:
                            return self._fail(400, "Invalid meeting id")
                        at = float(query.get("t", ["0"])[0])
                        if at < 0 or at > 7 * 24 * 3600:
                            return self._fail(400, "Invalid time")
                        evidence = _get_evidence(viewer.root, meeting_id, at)
                        title = html.escape(str(evidence["meeting"].get("title", meeting_id)))
                        media = ""
                        if evidence["source"]:
                            source_route = "source/" + urllib.parse.quote(meeting_id, safe="")
                            source_url = prefix + source_route
                            ext = evidence["source"].suffix.lower()
                            tag = "audio" if ext in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus"} else "video"
                            seek_script = "<script>const m=document.querySelector('video,audio');m.addEventListener('loadedmetadata',()=>{m.currentTime=" + str(evidence["at"]) + "})</script>"
                            media = f'<{tag} controls preload="metadata" src="{html.escape(source_url, quote=True)}"></{tag}>' + seek_script
                        elif evidence.get("audio"):
                            audio_route = "audio/" + urllib.parse.quote(meeting_id, safe="")
                            audio_url = prefix + audio_route
                            seek_script = "<script>const m=document.querySelector('audio');m.addEventListener('loadedmetadata',()=>{m.currentTime=" + str(evidence["at"]) + "})</script>"
                            media = f'<audio controls preload="metadata" src="{html.escape(audio_url, quote=True)}"></audio>' + seek_script
                        if evidence["frame"]:
                            frame_route = "frame/" + urllib.parse.quote(meeting_id, safe="")
                            media += f'<img alt="Saved screen evidence" src="{prefix}{frame_route}?t={evidence["at"]}" />'
                        if not media:
                            media = "<p>The original recording and a retained frame are unavailable.</p>"
                        body = f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{title}</title><style>body{{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}}video,audio,img{{display:block;max-width:100%;margin:1rem 0}}</style><h1>{title}</h1><p>Evidence at {evidence["at"]:.2f} seconds</p>{media}'
                        data = body.encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(data)))
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    if route.startswith("source/") or route.startswith("frame/") or route.startswith("audio/"):
                        kind, encoded_id = route.split("/", 1)
                        meeting_id = urllib.parse.unquote(encoded_id)
                        if not meeting_id or "/" in meeting_id or "\\" in meeting_id:
                            return self._fail(400, "Invalid meeting id")
                        evidence = _get_evidence(viewer.root, meeting_id, float(query.get("t", ["0"])[0]))
                        path = evidence["source"] if kind == "source" else evidence["frame"] if kind == "frame" else evidence.get("audio")
                        if path is None:
                            return self._fail(404, "Evidence is no longer available")
                        data_size = path.stat().st_size
                        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                        range_header = self.headers.get("Range", "")
                        start_byte, end_byte, status = 0, data_size - 1, 200
                        if range_header.startswith("bytes=") and "," not in range_header:
                            try:
                                left, right = range_header[6:].split("-", 1)
                                if left:
                                    start_byte = int(left)
                                    end_byte = min(data_size - 1, int(right)) if right else data_size - 1
                                else:
                                    suffix_length = int(right)
                                    start_byte = max(0, data_size - suffix_length)
                                if start_byte < 0 or start_byte > end_byte or start_byte >= data_size:
                                    raise ValueError
                                status = 206
                            except ValueError:
                                self.send_response(416)
                                self.send_header("Content-Range", f"bytes */{data_size}")
                                self.end_headers()
                                return
                        self.send_response(status)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Content-Length", str(end_byte - start_byte + 1))
                        self.send_header("Accept-Ranges", "bytes")
                        if status == 206:
                            self.send_header("Content-Range", f"bytes {start_byte}-{end_byte}/{data_size}")
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        with path.open("rb") as stream:
                            stream.seek(start_byte)
                            remaining = end_byte - start_byte + 1
                            while remaining and (chunk := stream.read(min(128 * 1024, remaining))):
                                self.wfile.write(chunk)
                                remaining -= len(chunk)
                        return
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    return self._fail(404, str(exc))
                return self._fail(404, "Not found")

        self.server = _Server(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="meeting-evidence-viewer", daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/{self.token}/"

    def evidence_url(self, meeting_id: str, at: float = 0) -> str:
        return self.base_url + "evidence/" + urllib.parse.quote(meeting_id, safe="") + "?t=" + urllib.parse.quote(str(float(at)))

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def start_viewer(root: str | Path) -> "ViewerHandle":
    """Start/reuse a detached viewer whose discovery state lives outside archive."""
    return _start_detached_viewer(root)


class ViewerHandle:
    def __init__(self, root: Path, token: str, port: int, descriptor: Path):
        self.root, self.token, self.port, self.descriptor = root, token, port, descriptor

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/{self.token}/"

    def evidence_url(self, meeting_id: str, at: float = 0) -> str:
        return self.base_url + "evidence/" + urllib.parse.quote(meeting_id, safe="") + "?t=" + urllib.parse.quote(str(float(at)))


def _viewer_state_dir(settings: str | Path | None = None) -> Path:
    if settings:
        return Path(settings).expanduser().resolve().parent
    configured = os.environ.get("VT_MEETING_SETTINGS")
    if configured:
        return Path(configured).expanduser().resolve().parent
    local = os.environ.get("LOCALAPPDATA")
    return (Path(local) / "VideoTranscribe" if local else Path.home() / ".video-transcribe").resolve()


def _start_detached_viewer(root: str | Path, settings: str | Path | None = None,
                           project_root: str | Path | None = None) -> ViewerHandle:
    resolved = Path(root).resolve()
    state_dir = _viewer_state_dir(settings)
    state_dir.mkdir(parents=True, exist_ok=True)
    descriptor = state_dir / "read-viewer.json"
    state: dict[str, Any] = {}
    state_key = str(resolved)
    try:
        state = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        state = {}
    viewers = state.get("viewers", {}) if isinstance(state, dict) else {}
    prior = viewers.get(state_key) if isinstance(viewers, dict) else None
    if not isinstance(prior, dict) and Path(state.get("root", "")).resolve() == resolved:
        prior = state
    if isinstance(prior, dict):
        try:
            handle = ViewerHandle(resolved, str(prior["token"]), int(prior["port"]), descriptor)
            with urllib.request.urlopen(handle.base_url + "health", timeout=.4) as response:
                if response.status == 200:
                    return handle
        except (OSError, ValueError, KeyError):
            pass
    token = str(prior["token"]) if isinstance(prior, dict) and prior.get("token") else secrets.token_urlsafe(24)
    if isinstance(prior, dict) and prior.get("port"):
        port = int(prior["port"])
    else:
        # Reserve an ephemeral loopback port before launching the independent process.
        probe = _Server(("127.0.0.1", 0), BaseHTTPRequestHandler)
        port = probe.server_port
        probe.server_close()
    app_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    diagnostic_log = state_dir / "viewer-startup.log"
    creationflags = 0
    for flag_name in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP", "CREATE_NO_WINDOW"):
        creationflags |= int(getattr(subprocess, flag_name, 0))
    command = [sys.executable, "-m", "services.meeting_links", "--serve", "--root", str(resolved),
               "--port", str(port), "--token=" + token, "--diagnostic-log", str(diagnostic_log)]
    launch_mode = "detached"
    breakaway_flag = int(getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0))

    def launch(flags: int) -> subprocess.Popen[Any]:
        diagnostic_log.parent.mkdir(parents=True, exist_ok=True)
        with diagnostic_log.open("ab") as diagnostic_stream:
            try:
                os.chmod(diagnostic_log, 0o600)
            except OSError:
                pass
            return subprocess.Popen(
                command, cwd=str(app_root), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=diagnostic_stream, close_fds=True, creationflags=flags,
            )

    try:
        process = launch(creationflags | breakaway_flag)
        if breakaway_flag:
            launch_mode = "breakaway_attempted"
    except OSError:
        # Windows only permits breakaway when the parent job allows it. Keep a
        # visible fallback mode so callers do not mistake a child-job server
        # for a process that survives the client.
        launch_mode = "inherited_job_fallback"
        process = launch(creationflags)
    handle = ViewerHandle(resolved, token, port, descriptor)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(_viewer_start_failure(diagnostic_log, exit_code, token))
        try:
            with urllib.request.urlopen(handle.base_url + "health", timeout=.4) as response:
                if response.status == 200:
                    viewers = state.get("viewers", {}) if isinstance(state.get("viewers", {}), dict) else {}
                    viewers[state_key] = {"root": str(resolved), "port": port, "token": token,
                                          "pid": process.pid, "launch_mode": launch_mode}
                    temp = descriptor.with_suffix(".tmp")
                    temp.write_text(json.dumps({"root": str(resolved), "port": port, "token": token,
                                                "pid": process.pid, "launch_mode": launch_mode,
                                                "viewers": viewers}, ensure_ascii=False), encoding="utf-8")
                    temp.replace(descriptor)
                    return handle
        except OSError:
            time.sleep(.05)
    raise RuntimeError(_viewer_start_failure(diagnostic_log, process.poll(), token))


def _viewer_start_failure(diagnostic_log: Path, exit_code: int | None, token: str) -> str:
    try:
        contents = diagnostic_log.read_text(encoding="utf-8", errors="replace")
        safe_contents = contents.replace(token, "[redacted]") if token else contents
        if safe_contents != contents:
            diagnostic_log.write_text(safe_contents, encoding="utf-8")
        diagnostic = safe_contents[-4096:].strip()
    except OSError:
        diagnostic = ""
    if token:
        diagnostic = diagnostic.replace(token, "[redacted]")
    detail = f"; startup log: {diagnostic}" if diagnostic else "; startup log is empty"
    return f"Could not start the local evidence viewer (exit code: {exit_code}){detail}"


def _record_viewer_start_failure(log_path: str | None, token: str | None, exc: Exception) -> None:
    if not log_path:
        return
    # Keep this diagnostic intentionally compact: never record command-line arguments,
    # credentials, or the viewer token.
    message = f"{type(exc).__name__}: {exc}"
    if token:
        message = message.replace(token, "[redacted]")
    try:
        path = Path(log_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(message[:1000] + "\n")
    except OSError:
        pass


def open_evidence_url(root: str | Path, meeting_id: str, at: float = 0,
                      settings: str | Path | None = None,
                      project_root: str | Path | None = None) -> str:
    return _start_detached_viewer(root, settings, project_root).evidence_url(meeting_id, at)


def evidence_uri(meeting_id: str, at: float = 0) -> str:
    if not meeting_id or "/" in meeting_id or "\\" in meeting_id:
        raise ValueError("Invalid meeting id")
    if not math.isfinite(float(at)) or float(at) < 0:
        raise ValueError("Invalid evidence time")
    return f"video-transcribe://evidence/{urllib.parse.quote(meeting_id, safe='')}?t={float(at):g}"


def open_evidence_uri(uri: str, settings: str | Path | None = None,
                      project_root: str | Path | None = None,
                      launch_browser: bool = True) -> str:
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme != "video-transcribe" or parsed.netloc != "evidence":
        raise ValueError("Unsupported evidence link")
    encoded_id = parsed.path.lstrip("/")
    meeting_id = urllib.parse.unquote(encoded_id)
    if not encoded_id or "/" in meeting_id or "\\" in meeting_id or meeting_id in {".", ".."}:
        raise ValueError("Invalid meeting id in evidence link")
    values = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
    if set(values) - {"t"} or len(values.get("t", [])) > 1:
        raise ValueError("Invalid evidence link parameters")
    try:
        at = float(values.get("t", ["0"])[0])
    except ValueError as exc:
        raise ValueError("Invalid evidence time") from exc
    if not math.isfinite(at) or not 0 <= at <= 7 * 24 * 3600:
        raise ValueError("Invalid evidence time")
    from services.meeting_mcp import resolve_archive_root
    root = resolve_archive_root(project_root=project_root, settings=settings)
    url = open_evidence_url(root, meeting_id, at, settings=settings, project_root=project_root)
    if launch_browser:
        webbrowser.open(url)
    return url


def viewer_launch_mode(root: str | Path, settings: str | Path | None = None) -> str | None:
    descriptor = _viewer_state_dir(settings) / "read-viewer.json"
    try:
        state = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    resolved = str(Path(root).resolve())
    per_root = state.get("viewers", {}).get(resolved) if isinstance(state, dict) else None
    return per_root.get("launch_mode") if isinstance(per_root, dict) else state.get("launch_mode") if isinstance(state, dict) and state.get("root") == resolved else None


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--root")
    parser.add_argument("--port", type=int)
    parser.add_argument("--token")
    parser.add_argument("--open-uri")
    parser.add_argument("--no-browser", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--settings")
    parser.add_argument("--project-root")
    parser.add_argument("--diagnostic-log", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.open_uri:
        open_evidence_uri(args.open_uri, settings=args.settings, project_root=args.project_root,
                          launch_browser=not args.no_browser)
        return 0
    if not args.serve or not args.root or args.port is None or not args.token:
        parser.error("--serve requires --root, --port, and --token; alternatively pass --open-uri")
    try:
        Viewer(Path(args.root), port=args.port, token=args.token).server.serve_forever()
        return 0
    except Exception as exc:
        _record_viewer_start_failure(args.diagnostic_log, args.token, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
