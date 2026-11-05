"""Keep model weights and model-related caches inside this project by default.

Import this module before huggingface_hub, transformers, torch, or faster_whisper.
The VT_MODEL_ROOT setting allows an explicit alternate root; incidental user
HF_HOME settings are intentionally overridden to prevent large downloads on C:.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_STORAGE_ROOT = Path(os.environ.get("VT_MODEL_ROOT", str(APP_ROOT / ".runtime" / "models"))).expanduser().resolve()
HF_HOME = MODEL_STORAGE_ROOT / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
HF_XET_CACHE = HF_HOME / "xet"
HF_ASSETS_CACHE = HF_HOME / "assets"
HF_MODULES_CACHE = HF_HOME / "modules"
TORCH_HOME = MODEL_STORAGE_ROOT / "torch"
XDG_CACHE_HOME = MODEL_STORAGE_ROOT.parent / "cache"
MODEL_TEMP_DIR = MODEL_STORAGE_ROOT.parent / "tmp"

for directory in (HF_HUB_CACHE, HF_XET_CACHE, HF_ASSETS_CACHE, HF_MODULES_CACHE,
                  TORCH_HOME, XDG_CACHE_HOME, MODEL_TEMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ["VT_MODEL_ROOT"] = str(MODEL_STORAGE_ROOT)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["HF_XET_CACHE"] = str(HF_XET_CACHE)
os.environ["HF_ASSETS_CACHE"] = str(HF_ASSETS_CACHE)
os.environ["HF_MODULES_CACHE"] = str(HF_MODULES_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_HUB_CACHE)
os.environ["TORCH_HOME"] = str(TORCH_HOME)
os.environ["XDG_CACHE_HOME"] = str(XDG_CACHE_HOME)
os.environ["TEMP"] = str(MODEL_TEMP_DIR)
os.environ["TMP"] = str(MODEL_TEMP_DIR)
os.environ["TMPDIR"] = str(MODEL_TEMP_DIR)

# Some launchers import tempfile before loading the application module. Drop
# its cached C: choice so subsequent temporary model/download files use D:.
tempfile_module = sys.modules.get("tempfile")
if tempfile_module is not None:
    tempfile_module.tempdir = None
