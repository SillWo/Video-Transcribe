"""Prepare the bundled app's required model cache before it is marked ready."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from services.model_management import _build_download_plan, download_registered_model, get_model_status
from services.model_paths import HF_HUB_CACHE
from services.runtime_devices import APP_ROOT, _windows_gpu_inventory, enumerate_vulkan_gpus, get_simple_gpu_details
from services.whisper_cpp import MODEL_REPO, model_filename, resolve_model_path
from utils.text_postprocess import HF_HOME

FASTER_MODELS = ("small", "large-v3-turbo", "large-v3")
VULKAN_MODELS = ("small", "large-v3-turbo", "large-v3")


def _report_size(model_id: str) -> str:
    try:
        status = get_model_status(model_id)
        if status.get("status") == "downloaded":
            return "already cached"
        entry_id = model_id
        from services.model_registry import get_model_registry_entry
        entry = get_model_registry_entry(entry_id)
        plan = _build_download_plan(entry.hfRepoId)
        total = sum(max(int(item.file_size or 0), 0) for item in plan if item.will_download)
        return f"about {total / (1024 ** 3):.1f} GB" if total else "size not reported by the model host"
    except Exception:
        return "size unavailable (will be shown by the host during download)"


def _vulkan_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(repo_id=MODEL_REPO, filename=model_filename(model_id), cache_dir=str(HF_HUB_CACHE))
        return isinstance(cached, str) and Path(cached).is_file()
    except Exception:
        return False


def _punctuation_cached() -> bool:
    """Check the local HF snapshot contents without importing the model or contacting HF."""
    try:
        from huggingface_hub import try_to_load_from_cache
        if not all(importlib.util.find_spec(name) is not None for name in ("razdel", "sbert_punc_case_ru", "torch", "transformers")):
            return False
        cache_dir = str(HF_HOME / "hub")
        def cached(name: str) -> bool:
            value = try_to_load_from_cache(repo_id="kontur-ai/sbert_punc_case_ru", filename=name, cache_dir=cache_dir)
            return isinstance(value, str) and Path(value).is_file()
        has_tokenizer = any(cached(name) for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt"))
        has_weights = any(cached(name) for name in ("model.safetensors", "pytorch_model.bin", "model.bin"))
        return cached("config.json") and has_tokenizer and has_weights
    except Exception:
        return False


def _vulkan_expected_size(model_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
        plan = snapshot_download(MODEL_REPO, allow_patterns=[model_filename(model_id)], dry_run=True)
        total = sum(max(int(item.file_size or 0), 0) for item in plan if item.will_download)
        return f"about {total / (1024 ** 3):.1f} GB" if total else "size unavailable"
    except Exception:
        return "size unavailable"


def _punctuation_expected_size() -> str:
    try:
        from huggingface_hub import snapshot_download
        plan = snapshot_download("kontur-ai/sbert_punc_case_ru", dry_run=True, cache_dir=str(HF_HUB_CACHE))
        total = sum(max(int(item.file_size or 0), 0) for item in plan if item.will_download)
        return f"about {total / (1024 ** 3):.1f} GB" if total else "size unavailable"
    except Exception:
        return "size unavailable"


def prepare(*, check_only: bool = False, skip_postprocess: bool = False, progress=print) -> int:
    gpu = get_simple_gpu_details()
    need_vulkan = (gpu.get("backend") == "vulkan"
                   or any(item.get("vendorId") == 0x1002 for item in enumerate_vulkan_gpus())
                   or any(item.get("vendorId") == 0x1002 for item in _windows_gpu_inventory()))
    missing: list[str] = []
    tasks = [("faster-whisper", model_id, lambda model_id=model_id: get_model_status(model_id).get("status") == "downloaded",
              lambda model_id=model_id: download_registered_model(model_id, lambda _p, text: progress(f"{model_id}: {text}")))
             for model_id in FASTER_MODELS]
    if need_vulkan:
        tasks.extend(("whisper.cpp", model_id, lambda model_id=model_id: _vulkan_cached(model_id),
                      lambda model_id=model_id: resolve_model_path(model_id, progress=progress)) for model_id in VULKAN_MODELS)

    progress(f"Model cache: {HF_HUB_CACHE}")
    progress(f"Project runtime: {APP_ROOT / '.runtime'}")
    for backend, model_id, cached, install in tasks:
        if cached():
            progress(f"{backend} {model_id}: already cached.")
        elif check_only:
            estimate = "size not checked in offline mode"
            missing.append(f"{backend} {model_id} ({estimate})")
        else:
            estimate = _report_size(model_id) if backend == "faster-whisper" else _vulkan_expected_size(model_id)
            progress(f"Preparing {backend} model {model_id}; uncached size {estimate}.")
            install()

    if not skip_postprocess:
        if check_only:
            if not _punctuation_cached():
                missing.append("Russian punctuation model (kontur-ai/sbert_punc_case_ru)")
            else:
                progress("Russian punctuation model: validated cached snapshot.")
        else:
            if not _punctuation_cached():
                progress(f"Preparing Russian punctuation model; uncached size {_punctuation_expected_size()}.")
            else:
                progress("Loading the Russian punctuation model from cache.")
            from utils.text_postprocess import load_punctuation_model
            load_punctuation_model()
            progress("Russian punctuation model is ready.")

    if missing:
        progress("Missing model data: " + "; ".join(missing))
        progress("Run start.ps1 to download the required models before launching the app.")
        return 1
    progress("All required models are ready.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare required transcription model caches")
    parser.add_argument("--check-only", action="store_true", help="Report missing models without downloading")
    parser.add_argument("--skip-postprocess", action="store_true", help="Skip the optional Russian punctuation model")
    args = parser.parse_args()
    try:
        return prepare(check_only=args.check_only, skip_postprocess=args.skip_postprocess)
    except Exception as exc:
        print(f"Model preparation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
