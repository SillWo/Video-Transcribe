"""Single-instance local FastAPI host for the Video Transcribe desktop shell."""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

APP_ID = "video-transcribe"
PROTOCOL_VERSION = 1


def _loopback_url(value: Any) -> bool:
    try:
        parsed = urllib.parse.urlsplit(str(value))
        return (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1"}
                and parsed.port is not None and not parsed.username and not parsed.password
                and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment)
    except (ValueError, TypeError):
        return False


def verify_host_descriptor(record: dict[str, Any], *, project_root: str | Path,
                           timeout: float = 2.0) -> bool:
    """Accept a runtime descriptor only when its loopback host proves the same identity."""
    if (record.get("appId") != APP_ID or record.get("protocolVersion") != PROTOCOL_VERSION
            or not isinstance(record.get("token"), str) or not record.get("token")
            or not isinstance(record.get("instanceId"), str) or not record.get("instanceId")
            or not isinstance(record.get("pid"), int) or isinstance(record.get("pid"), bool)
            or record.get("pid", 0) < 1 or not _loopback_url(record.get("backendUrl"))):
        return False
    expected_root = str(Path(project_root).resolve())
    try:
        if not isinstance(record.get("projectRoot"), str) or not Path(record["projectRoot"]).is_absolute():
            return False
        if str(Path(record["projectRoot"]).resolve()) != expected_root:
            return False
    except (OSError, RuntimeError, ValueError):
        return False
    request = urllib.request.Request(str(record["backendUrl"]).rstrip("/") + "/api/desktop/host")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            info = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False
    if not isinstance(info, dict) or not isinstance(info.get("projectRoot"), str):
        return False
    try:
        reported_root = str(Path(info["projectRoot"]).resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return (info.get("appId") == APP_ID
            and info.get("protocolVersion") == PROTOCOL_VERSION
            and info.get("instanceId") == record.get("instanceId")
            and info.get("pid") == record.get("pid")
            and reported_root == expected_root
            and info.get("backendUrl") == record.get("backendUrl"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def _start_detached_host(directory: Path, project_root: Path,
                         environment: dict[str, str] | None = None) -> None:
    command = [sys.executable, "-m", "services.app_host", "--host", "127.0.0.1", "--port", "0"]
    env = os.environ.copy()
    env["VT_SIMPLE_RUNTIME_DIR"] = str(directory.resolve())
    if environment:
        env.update(environment)
    command.extend(["--runtime-dir", str(directory.resolve()), "--quiet"])
    if env.get("VT_MEETING_SETTINGS"):
        command.extend(["--settings", env["VT_MEETING_SETTINGS"]])
    if env.get("VT_MEETING_ROOT"):
        command.extend(["--root", env["VT_MEETING_ROOT"]])
    if os.name == "nt":
        # MCP clients commonly place their child trees in kill-on-close jobs.
        # Win32_Process.Create runs through the local provider, outside that job.
        _start_host_via_local_wmi(command, project_root, env)
        return
    subprocess.Popen(command, cwd=str(project_root), env=env, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
                     start_new_session=True)


def _start_host_via_local_wmi(command: list[str], project_root: Path,
                              environment: dict[str, str] | None = None,
                              timeout: float = 15.0) -> int:
    """Create the host through the local WMI provider, outside the caller's job."""
    powershell = "powershell.exe"
    cmdline = subprocess.list2cmdline(command)
    cwd = str(project_root.resolve())
    # PowerShell single-quoted strings escape embedded apostrophes by doubling them.
    def ps_quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    ps_cmdline = ps_quote(cmdline)
    ps_cwd = ps_quote(cwd)
    env_keys = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LOCALAPPDATA", "APPDATA",
                "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "VT_SIMPLE_RUNTIME_DIR",
                "VT_MEETING_SETTINGS", "VT_MEETING_ROOT", "VT_DICTIONARY_PATH",
                "MEETING_EMBEDDING_MODEL", "CUDA_PATH", "CUDA_HOME")
    env_upper = {str(key).upper(): value for key, value in (environment or {}).items()}
    environment_values = [f"{key}={env_upper[key]}" for key in env_keys
                          if isinstance(env_upper.get(key), str)]
    env_array = "@(" + ",".join(ps_quote(value) for value in environment_values) + ")"
    startup = ("$startup = ([wmiclass]'Win32_ProcessStartup').CreateInstance(); "
               "$startup.ShowWindow = [uint16]0; "
               f"$startup.EnvironmentVariables = [string[]]{env_array}; ")
    script = (startup + "$processClass = [wmiclass]'Win32_Process'; "
              f"$r = $processClass.Create({ps_cmdline}, {ps_cwd}, $startup); "
              "if ($r.ReturnValue -ne 0) { [Console]::Error.Write('Win32_Process.Create returned ' + $r.ReturnValue); exit 1 }; "
              "[Console]::Out.Write($r.ProcessId)")
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                            cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or "Local Windows process creation failed."
        raise RuntimeError(detail)
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("Local Windows process creation returned no process id.") from exc


def ensure_headless_host(*, directory: Path | None = None, project_root: str | Path | None = None,
                         environment: dict[str, str] | None = None,
                         timeout: float = 20.0) -> dict[str, Any]:
    """Reuse a verified host, otherwise start an unmanaged detached host and await it."""
    directory = (directory or runtime_dir()).resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    record = read_descriptor(directory)
    if record and verify_host_descriptor(record, project_root=project_root):
        return record
    _start_detached_host(directory, project_root, environment)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = read_descriptor(directory)
        if record and verify_host_descriptor(record, project_root=project_root):
            return record
        time.sleep(0.1)
    raise RuntimeError("The local Video Transcribe app host did not become ready with a verified identity.")


def runtime_path() -> Path:
    configured = os.environ.get("VT_SIMPLE_RUNTIME_DIR")
    path = Path(configured) if configured else Path(__file__).resolve().parents[1] / ".runtime"
    return path.resolve()


def runtime_dir() -> Path:
    path = runtime_path()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _acquire_lock(path: Path):
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if stream.read(1) == b"":
                stream.seek(0); stream.write(b"0"); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return stream
    except (OSError, BlockingIOError):
        stream.close()
        return None


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def read_descriptor(directory: Path | None = None) -> dict[str, Any] | None:
    path = (directory or runtime_dir()) / "desktop-host.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _request_quit(descriptor: dict[str, Any], timeout: float = 20) -> bool:
    token = descriptor.get("token")
    url = descriptor.get("backendUrl")
    if not isinstance(token, str) or not isinstance(url, str):
        return False
    request = urllib.request.Request(url.rstrip("/") + "/api/desktop/quit", method="POST",
                                     headers={"X-VT-Host-Token": token})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 202
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _wait_for_parent_stdin_close(fd: int = 0) -> None:
    """Wait for stdin EOF without blocking a Windows pipe reader thread."""
    if os.name == "nt":
        try:
            import ctypes
            import msvcrt
            from ctypes import wintypes

            handle = msvcrt.get_osfhandle(fd)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            get_file_type = kernel32.GetFileType
            get_file_type.argtypes = [wintypes.HANDLE]
            get_file_type.restype = wintypes.DWORD
            if get_file_type(handle) == 3:  # FILE_TYPE_PIPE
                peek_named_pipe = kernel32.PeekNamedPipe
                peek_named_pipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                            wintypes.LPDWORD, wintypes.LPDWORD, wintypes.LPDWORD]
                peek_named_pipe.restype = wintypes.BOOL
                while True:
                    available = wintypes.DWORD()
                    if not peek_named_pipe(handle, None, 0, None, ctypes.byref(available), None):
                        # Broken/closed pipes report EOF here. Other pipe errors
                        # had the same shutdown effect as the old os.read path.
                        return
                    if available.value:
                        chunk = os.read(fd, min(available.value, 4096))
                        if not chunk:
                            return
                    else:
                        time.sleep(0.15)

        except (OSError, AttributeError):
            # If stdin is not a queryable pipe, retain the original read path.
            pass

    while os.read(fd, 1):
        pass


def stop_host(directory: Path | None = None) -> int:
    record = read_descriptor(directory)
    if not record or not _request_quit(record):
        print("No authenticated Video Transcribe host stopped.", file=sys.stderr)
        return 1
    print("Video Transcribe host stopped.")
    return 0


def _make_asgi_app(token: str, server_ref: list[Any]):
    from starlette.applications import Starlette
    from starlette.background import BackgroundTask
    from starlette.concurrency import run_in_threadpool
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    # Imported only after the runtime lock and listening socket are both owned.
    import web_api

    async def host_info(request: Request):
        return JSONResponse({"appId": APP_ID, "instanceId": server_ref[1], "projectRoot": server_ref[2],
                             "pid": os.getpid(), "backendUrl": server_ref[3],
                             "protocolVersion": PROTOCOL_VERSION, "launchId": server_ref[4]})

    async def quit_host(request: Request):
        if not secrets.compare_digest(request.headers.get("X-VT-Host-Token", ""), token):
            return JSONResponse({"detail": "Invalid host token."}, status_code=403)
        try:
            await run_in_threadpool(web_api.prepare_simple_shutdown, 15.0)
        except Exception as exc:
            return JSONResponse({"detail": str(exc) or "Queue did not stop cleanly."}, status_code=503)

        def request_server_exit():
            time.sleep(0.15)
            server_ref[0].should_exit = True
        return JSONResponse({"status": "stopping"}, status_code=202,
                            background=BackgroundTask(request_server_exit))

    async def convert_media(request: Request):
        if not secrets.compare_digest(request.headers.get("X-VT-Host-Token", ""), token):
            return JSONResponse({"detail": "Invalid host token."}, status_code=403)
        try:
            payload = await request.json()
            if not isinstance(payload, dict) or set(payload) - {"path", "request_id", "archive_root"}:
                raise ValueError("Expected path and optional request_id.")
            path = payload.get("path")
            if (not isinstance(path, str) or not path or "\x00" in path
                    or not Path(path).is_absolute() or Path(path).resolve() != Path(path)):
                raise ValueError("path must be one absolute local file path")
            request_id = payload.get("request_id")
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("request_id must be a non-empty idempotency key")
            expected_root = payload.get("archive_root")
            if not isinstance(expected_root, str) or not Path(expected_root).is_absolute():
                raise ValueError("archive_root must be absolute")
            from services.meeting_settings import load_settings
            actual_root = str(Path(load_settings(Path(__file__).resolve().parents[1])["archive_root"]).resolve())
            if str(Path(expected_root).resolve()) != actual_root:
                return JSONResponse({"detail": "The running app host is connected to a different archive root."}, status_code=409)
            import web_api
            result = await run_in_threadpool(web_api.enqueue_media_conversion, path, request_id)
            return JSONResponse(result, status_code=202)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except RuntimeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)
        except Exception as exc:
            return JSONResponse({"detail": str(exc) or "Media conversion could not be queued."}, status_code=500)

    async def media_conversion_status(request: Request):
        if not secrets.compare_digest(request.headers.get("X-VT-Host-Token", ""), token):
            return JSONResponse({"detail": "Invalid host token."}, status_code=403)
        job_id = request.path_params["job_id"]
        try:
            expected_root = request.query_params.get("archive_root")
            if not expected_root or not Path(expected_root).is_absolute():
                return JSONResponse({"detail": "archive_root must be absolute."}, status_code=400)
            from services.meeting_settings import load_settings
            actual_root = str(Path(load_settings(Path(__file__).resolve().parents[1])["archive_root"]).resolve())
            if str(Path(expected_root).resolve()) != actual_root:
                return JSONResponse({"detail": "The running app host is connected to a different archive root."}, status_code=409)
            import web_api
            result = await run_in_threadpool(web_api.media_conversion_status, job_id)
            if result is None:
                return JSONResponse({"detail": "Conversion job not found.", "job_id": job_id}, status_code=404)
            record_root = result.get("archiveRoot") if isinstance(result, dict) else None
            if not isinstance(record_root, str):
                return JSONResponse({"detail": "Conversion job not found.", "job_id": job_id}, status_code=404)
            try:
                if Path(record_root).expanduser().resolve() != Path(expected_root).expanduser().resolve():
                    return JSONResponse({"detail": "Conversion job not found.", "job_id": job_id}, status_code=404)
            except (OSError, RuntimeError, ValueError):
                return JSONResponse({"detail": "Conversion job not found.", "job_id": job_id}, status_code=404)
            return JSONResponse(result)
        except Exception as exc:
            return JSONResponse({"detail": str(exc) or "Conversion status is unavailable."}, status_code=500)

    return Starlette(routes=[Route("/api/desktop/host", host_info, methods=["GET"]),
                             Route("/api/desktop/quit", quit_host, methods=["POST"]),
                             Route("/api/media/conversions", convert_media, methods=["POST"]),
                             Route("/api/media/conversions/{job_id:str}", media_conversion_status, methods=["GET"]),
                             Mount("/", app=web_api.app)])


def serve(host: str = "127.0.0.1", port: int = 8000, *, managed: bool = False,
          directory: Path | None = None, quiet: bool = False) -> int:
    from services.windows_job import own_current_process
    from uvicorn import Config, Server

    directory = (directory or runtime_dir()).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(directory / "desktop-host.lock")
    if lock is None:
        print("A Video Transcribe host already owns this project runtime.", file=sys.stderr)
        return 2
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(128)
    except OSError as exc:
        sock.close(); lock.close()
        print(f"Cannot listen on {host}:{port}: {exc}", file=sys.stderr)
        return 3

    job = own_current_process()
    token = secrets.token_urlsafe(32)
    instance_id = str(uuid.uuid4())
    launch_id = os.environ.get("VT_DESKTOP_LAUNCH_ID")
    backend_url = f"http://{host}:{sock.getsockname()[1]}"
    descriptor_path = directory / "desktop-host.json"
    descriptor = {"projectRoot": str(Path(__file__).resolve().parents[1]), "appId": APP_ID,
                  "instanceId": instance_id, "pid": os.getpid(), "backendUrl": backend_url,
                  "token": token, "protocolVersion": PROTOCOL_VERSION}
    if launch_id:
        descriptor["launchId"] = launch_id
    _atomic_json(descriptor_path, descriptor)
    server = Server(Config(app=None, host=host, port=sock.getsockname()[1], lifespan="on",
                           timeout_graceful_shutdown=5, log_level="warning" if quiet else "info"))
    server_ref = [server, instance_id, descriptor["projectRoot"], backend_url, launch_id]
    server.config.app = _make_asgi_app(token, server_ref)

    # The FastAPI Mount does not run web_api's lifespan. Resume only persisted
    # queued work here; the helper leaves a deliberately paused queue paused.
    try:
        import web_api
        web_api.ensure_media_conversion_dispatcher_started()
    except Exception as exc:
        print(f"Could not resume persisted conversion queue: {exc}", file=sys.stderr)

    if managed:
        def watch_parent_pipe():
            try:
                _wait_for_parent_stdin_close(0)
            except OSError:
                pass
            try:
                import web_api
                web_api.prepare_simple_shutdown(timeout=15)
            except Exception as exc:
                print(f"Managed shutdown did not complete: {exc}", file=sys.stderr)
            server.should_exit = True
        threading.Thread(target=watch_parent_pipe, name="desktop-parent-watch", daemon=True).start()
    if not quiet:
        print(f"Video Transcribe listening at {backend_url}", flush=True)
    try:
        server.run(sockets=[sock])
    finally:
        sock.close()
        with contextlib.suppress(OSError, ValueError):
            current = json.loads(descriptor_path.read_text(encoding="utf-8"))
            if current.get("instanceId") == instance_id:
                descriptor_path.unlink()
        # Keep the kill-on-close handle until process teardown. Closing it here would
        # terminate this host immediately because the host itself is a job member.
        lock.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--managed", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--runtime-dir", help="Override the app runtime directory")
    parser.add_argument("--settings", help="Override the shared meeting settings JSON")
    parser.add_argument("--root", help="Override the meeting archive root")
    parser.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.runtime_dir:
        os.environ["VT_SIMPLE_RUNTIME_DIR"] = args.runtime_dir
    if args.settings:
        os.environ["VT_MEETING_SETTINGS"] = args.settings
    if args.root:
        os.environ["VT_MEETING_ROOT"] = args.root
    directory = runtime_dir()
    if args.stop:
        return stop_host(directory)
    if args.quiet:
        with (directory / "headless-host.log").open("a", encoding="utf-8") as diagnostic:
            sys.stdout = diagnostic
            sys.stderr = diagnostic
            return serve(args.host, args.port, managed=args.managed, directory=directory, quiet=True)
    return serve(args.host, args.port, managed=args.managed, directory=directory, quiet=False)


if __name__ == "__main__":
    raise SystemExit(main())
