# -*- mode: python ; coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


source_dir = Path(os.environ["VT_SOURCE_DIR"]).resolve()
packaging_dir = Path(os.environ["VT_PACKAGING_DIR"]).resolve()
frontend_dir = Path(os.environ["VT_FRONTEND_BUILD_DIR"]).resolve()
build_root = Path(os.environ["VT_BUILD_ROOT"]).resolve()

hiddenimports = sorted(
    set(
        collect_submodules("services")
        + collect_submodules("utils")
        + collect_submodules("uvicorn")
        + collect_submodules("faster_whisper")
        + collect_submodules("sbert_punc_case_ru")
        + [
            "psutil",
            "validators",
            "web_api",
            "source_check",
            "ffmpeg",
            "yt_dlp",
            "razdel",
            "sbert_punc_case_ru",
        ]
    )
)

datas = [
    (str(frontend_dir), "ui_dist"),
    (str(packaging_dir / "runtime.default.json"), "config"),
    (str(source_dir / "whisper-gpu.py"), "."),
    (str(source_dir / "README.md"), "docs"),
]

a = Analysis(
    [str(packaging_dir / "desktop_main.py")],
    pathex=[str(packaging_dir), str(source_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VideoTranscribe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="VideoTranscribe",
)
