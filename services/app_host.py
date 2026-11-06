"""Single-instance local FastAPI host for the Video Transcribe desktop shell."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

APP_ID = "video-transcribe"
PROTOCOL_VERSION = 1


def runtime_dir() -> Path:
    configured = os.environ.get("VT_SIMPLE_RUNTIME_DIR")
    path = Path(configured) if configured else Path(__file__).resolve().parents[1] / ".runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


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

    return Starlette(routes=[Route("/api/desktop/host", host_info, methods=["GET"]),
                             Route("/api/desktop/quit", quit_host, methods=["POST"]),
                             Mount("/", app=web_api.app)])


def serve(host: str = "127.0.0.1", port: int = 8000, *, managed: bool = False,
          directory: Path | None = None) -> int:
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
                           timeout_graceful_shutdown=5, log_level="info"))
    server_ref = [server, instance_id, descriptor["projectRoot"], backend_url, launch_id]
    server.config.app = _make_asgi_app(token, server_ref)

    if managed:
        def watch_parent_pipe():
            try:
                while os.read(0, 1):
                    pass
            except OSError:
                pass
            try:
                import web_api
                web_api.prepare_simple_shutdown(timeout=15)
            except Exception as exc:
                print(f"Managed shutdown did not complete: {exc}", file=sys.stderr)
            server.should_exit = True
        threading.Thread(target=watch_parent_pipe, name="desktop-parent-watch", daemon=True).start()
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
    args = parser.parse_args(argv)
    directory = runtime_dir()
    if args.stop:
        return stop_host(directory)
    return serve(args.host, args.port, managed=args.managed, directory=directory)


if __name__ == "__main__":
    raise SystemExit(main())
