from __future__ import annotations

import json
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "VideoTranscribe"


@dataclass(frozen=True)
class RuntimeConfig:
    install_root: Path
    resource_root: Path
    static_build_dir: Path
    bundled_bin_dir: Path
    yt_dlp_path: Path
    deno_path: Path
    ffmpeg_path: Path
    ffprobe_path: Path
    config_path: Path
    worker_script_path: Path
    user_data_root: Path
    logs_root: Path
    data_root: Path
    huggingface_root: Path
    host: str
    preferred_port: int
    effective_port: int
    open_browser: bool

    @property
    def app_url(self) -> str:
        return f"http://{self.host}:{self.effective_port}"

    @property
    def worker_command(self) -> list[str]:
        return [sys.executable, "--transcribe-worker"]


def _default_install_root() -> Path:
    executable_path = Path(sys.executable).resolve()
    if getattr(sys, "frozen", False):
        return executable_path.parent
    return Path(__file__).resolve().parent


def _default_resource_root(install_root: Path) -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return (install_root / "_internal").resolve()
    return Path(__file__).resolve().parent


def _user_data_root() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / APP_NAME
    return Path.home() / "AppData" / "Local" / APP_NAME


def _load_runtime_defaults(config_path: Path) -> dict[str, object]:
    candidate_paths = [config_path, Path(__file__).resolve().with_name("runtime.default.json")]
    for candidate in candidate_paths:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _find_free_port(host: str, preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, preferred_port))
        except OSError:
            sock.bind((host, 0))
        return int(sock.getsockname()[1])


def resolve_runtime_config() -> RuntimeConfig:
    install_root = Path(os.environ.get("VT_INSTALL_ROOT", _default_install_root())).resolve()
    resource_root = _default_resource_root(install_root)
    config_path = resource_root / "config" / "runtime.default.json"
    defaults = _load_runtime_defaults(config_path)

    host = str(defaults.get("host", "127.0.0.1"))
    preferred_port = int(defaults.get("preferredPort", 8765))
    effective_port = _find_free_port(host, preferred_port)
    open_browser = bool(defaults.get("openBrowser", True))

    user_data_root = Path(os.environ.get("VT_USER_DATA_ROOT", _user_data_root())).resolve()
    logs_root = user_data_root / "logs"
    data_root = user_data_root / "data"
    huggingface_root = user_data_root / "huggingface"
    static_build_dir = resource_root / "ui_dist"
    bundled_bin_dir = install_root / "bin"

    return RuntimeConfig(
        install_root=install_root,
        resource_root=resource_root,
        static_build_dir=static_build_dir,
        bundled_bin_dir=bundled_bin_dir,
        yt_dlp_path=bundled_bin_dir / "yt-dlp.exe",
        deno_path=bundled_bin_dir / "deno.exe",
        ffmpeg_path=bundled_bin_dir / "ffmpeg.exe",
        ffprobe_path=bundled_bin_dir / "ffprobe.exe",
        config_path=config_path,
        worker_script_path=resource_root / "whisper-gpu.py",
        user_data_root=user_data_root,
        logs_root=logs_root,
        data_root=data_root,
        huggingface_root=huggingface_root,
        host=host,
        preferred_port=preferred_port,
        effective_port=effective_port,
        open_browser=open_browser,
    )


def prepare_runtime_environment(config: RuntimeConfig) -> None:
    for path in (config.user_data_root, config.logs_root, config.data_root, config.huggingface_root):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(config.huggingface_root)
    os.environ["HF_HUB_CACHE"] = str(config.huggingface_root / "hub")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ["PATH"] = str(config.bundled_bin_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ["FFMPEG_BINARY"] = str(config.ffmpeg_path)
    os.environ["FFPROBE_BINARY"] = str(config.ffprobe_path)
    os.environ["VT_BUNDLED_YT_DLP"] = str(config.yt_dlp_path)
