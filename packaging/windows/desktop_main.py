from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from runtime_config import prepare_runtime_environment, resolve_runtime_config


def _load_worker_module(worker_path: Path):
    spec = importlib.util.spec_from_file_location("whisper_gpu_worker", worker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load worker entrypoint from {worker_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_worker_runtime(worker_module, config) -> None:
    original_resolve_model_cmd = worker_module.Download.resolve_model_cmd
    original_adjust_format = worker_module.Download.adjust_format

    def packaged_resolve_model_cmd(self, candidate: str | None) -> list[str]:
        if config.yt_dlp_path.exists():
            return [str(config.yt_dlp_path)]
        return original_resolve_model_cmd(self, candidate)

    def packaged_adjust_format(self, args):
        original_adjust_format(self, args)
        if config.deno_path.exists():
            self.opts = [
                "--js-runtimes",
                f"deno:{config.deno_path}",
                "--remote-components",
                "ejs:github",
                *self.opts,
            ]

    worker_module.Download.resolve_model_cmd = packaged_resolve_model_cmd
    worker_module.Download.adjust_format = packaged_adjust_format


def _run_transcribe_worker(argv: list[str]) -> int:
    config = resolve_runtime_config()
    prepare_runtime_environment(config)
    worker_path = config.worker_script_path
    if not worker_path.exists():
        worker_path = Path(__file__).resolve().parent / "whisper-gpu.py"
    worker_module = _load_worker_module(worker_path)
    _patch_worker_runtime(worker_module, config)
    sys.argv = ["whisper-gpu.py", *argv]
    return int(worker_module.main())


def _patch_backend_runtime(config) -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import web_api

    web_api.RUNS_DIR = config.data_root / "runs"
    web_api.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def packaged_build_command(job, source_path: str) -> list[str]:
        settings = job.settings
        command = [
            *config.worker_command,
            "-f",
            source_path,
            "-od",
            str(web_api.RUNS_DIR / job.id / "output"),
            "-l",
            settings["language"],
            "-s",
            settings["model"],
            "-d",
            settings["device"],
            "--output_format",
            settings["outputFormat"],
            "--web_json",
        ]

        if settings["device"] == "cpu":
            command.extend(["-n", str(settings["nproc"])])

        if settings["saveAudio"]:
            command.append("-k")

        if not settings["useTimestamps"] and settings["outputFormat"] != "srt":
            command.append("--no_timestamps")

        return command

    web_api.build_command = packaged_build_command
    return web_api.app


def _configure_static_routes(app, config) -> None:
    if getattr(app.state, "desktop_static_configured", False):
        return

    assets_dir = config.static_build_dir / "assets"
    if config.static_build_dir.exists() and assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(config.static_build_dir / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            if full_path.startswith("api/"):
                return PlainTextResponse("Not found", status_code=404)

            candidate = config.static_build_dir / full_path
            if candidate.exists() and candidate.is_file():
                return FileResponse(candidate)

            return FileResponse(config.static_build_dir / "index.html")

    app.state.desktop_static_configured = True


def _wait_for_health(url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urlopen(f"{url}/api/health", timeout=2) as response:
                if response.status == 200:
                    return
        except URLError:
            time.sleep(0.25)
    raise TimeoutError(f"Backend did not become healthy within {timeout_seconds} seconds")


def _run_server(app, config, smoke_test: bool) -> int:
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.host,
            port=config.effective_port,
            log_level="info",
            log_config=None,
            access_log=False,
        )
    )
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    _wait_for_health(config.app_url)

    if smoke_test:
        server.should_exit = True
        server_thread.join(timeout=10)
        return 0

    return 0


def _serve_ui(no_browser: bool, smoke_test: bool) -> int:
    config = resolve_runtime_config()
    prepare_runtime_environment(config)
    app = _patch_backend_runtime(config)
    _configure_static_routes(app, config)

    result = _run_server(app, config, smoke_test)
    if smoke_test:
        return result

    if config.open_browser and not no_browser:
        webbrowser.open(config.app_url)

    while True:
        time.sleep(1)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--transcribe-worker":
        exit_code = _run_transcribe_worker(args[1:])
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    parser = argparse.ArgumentParser("VideoTranscribe desktop launcher")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parsed = parser.parse_args(args)
    return _serve_ui(parsed.no_browser, parsed.smoke_test)


if __name__ == "__main__":
    raise SystemExit(main())
