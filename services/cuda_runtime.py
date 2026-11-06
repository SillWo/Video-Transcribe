"""Project-local, checksum-pinned CUDA runtime files for faster-whisper on Windows."""

from __future__ import annotations

import hashlib
import os
import shutil
import argparse
import urllib.request
import zipfile
from pathlib import Path

from services.runtime_devices import APP_ROOT, _windows_gpu_inventory

CUDA_ROOT = APP_ROOT / ".runtime" / "cuda"
ARCHIVES = (
    {
        "name": "cuBLAS 12.4.5.8",
        "url": "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.4.5.8-archive.zip",
        "sha256": "698140f12da055a3709eee2e022fcfe7bc8edf31f30115e3f7a5c877a9491de5",
        "size": 391538487,
    },
    {
        "name": "cuDNN 9.1.1.17 for CUDA 12",
        "url": "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.1.1.17_cuda12-archive.zip",
        "sha256": "3f1efb1a82e480594594e8b1fc8075a4852b4ada5a1fea3c8ec4b3ed1e5122ed",
        "size": 669632100,
    },
)
REQUIRED_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")
_DLL_HANDLES = []


def _dll_dirs(root: Path = CUDA_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted({path.parent for name in REQUIRED_DLLS for path in root.rglob(name)})


def local_cuda_ready(root: Path = CUDA_ROOT) -> bool:
    return all(any(path.is_file() for path in root.rglob(name)) for name in REQUIRED_DLLS) if root.exists() else False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def register_local_cuda_runtime(root: Path = CUDA_ROOT) -> bool:
    """Put local DLL directories ahead of imports, preserving add_dll_directory handles."""
    directories = _dll_dirs(root)
    if not directories:
        return False
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        known = {str(path) for path in directories}
        for directory in directories:
            if str(directory) not in getattr(register_local_cuda_runtime, "_registered", set()):
                _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
        register_local_cuda_runtime._registered = known
    current = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([str(path) for path in directories] + [p for p in current if p and p not in {str(x) for x in directories}])
    return local_cuda_ready(root)


def _download_archive(spec: dict, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / (spec["name"].replace(" ", "-") + ".zip")
    if target.exists() and target.stat().st_size == spec["size"]:
        digest = _sha256_file(target)
        if digest == spec["sha256"]:
            return target
        target.unlink()
    partial = target.with_suffix(target.suffix + ".part")
    current_size = partial.stat().st_size if partial.exists() else 0
    if current_size >= spec["size"]:
        partial.unlink()
        current_size = 0
    headers = {"User-Agent": "Video-Transcribe/1.0"}
    if current_size:
        headers["Range"] = f"bytes={current_size}-"
    request = urllib.request.Request(spec["url"], headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            append = current_size > 0 and getattr(response, "status", 200) == 206
            if not append:
                current_size = 0
            with partial.open("ab" if append else "wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
    except Exception:
        # Preserve verified bytes in the partial file so the next setup can resume.
        raise
    if partial.stat().st_size != spec["size"] or _sha256_file(partial) != spec["sha256"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum or size validation failed for {spec['name']}; the incomplete archive was removed.")
    partial.replace(target)
    return target


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise RuntimeError(f"Unsafe path in CUDA runtime archive: {member.filename}")
        bundle.extractall(destination)


def ensure_local_cuda_runtime(*, force_download: bool = False, progress=None, root: Path = CUDA_ROOT) -> Path | None:
    """Install pinned NVIDIA runtime archives only when NVIDIA hardware is present."""
    if os.name != "nt" or not any(item["vendorId"] == 0x10DE for item in _windows_gpu_inventory()):
        return None
    backup = root.parent / "cuda.previous"
    if not root.exists() and local_cuda_ready(backup):
        backup.rename(root)
    if local_cuda_ready(root) and not force_download:
        register_local_cuda_runtime(root)
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    archive_dir = root.parent / "downloads"
    staging = root.parent / "cuda.extracting"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for index, spec in enumerate(ARCHIVES, start=1):
            if progress:
                progress(f"Downloading verified {spec['name']} runtime ({index}/{len(ARCHIVES)}).")
            archive = _download_archive(spec, archive_dir)
            extract_root = staging / f"extract-{index}"
            extract_root.mkdir()
            _safe_extract(archive, extract_root)
        # Validate the complete extraction before replacing a previously working runtime.
        if not local_cuda_ready(staging):
            raise RuntimeError("Verified CUDA archives did not contain the required cuBLAS and cuDNN DLLs.")
        if backup.exists():
            shutil.rmtree(backup)
        if root.exists():
            root.rename(backup)
        try:
            staging.rename(root)
        except Exception:
            if backup.exists() and not root.exists():
                backup.rename(root)
            raise
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    if not local_cuda_ready(root):
        if root.exists():
            shutil.rmtree(root)
        if backup.exists():
            backup.rename(root)
        raise RuntimeError("CUDA runtime archives were verified, but required cuBLAS/cuDNN DLLs were not found.")
    if backup.exists():
        shutil.rmtree(backup)
    register_local_cuda_runtime(root)
    if progress:
        progress(f"Project-local CUDA runtime ready in {root}.")
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the project-local CUDA runtime")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    nvidia = os.name == "nt" and any(item["vendorId"] == 0x10DE for item in _windows_gpu_inventory())
    if not nvidia:
        print("No NVIDIA GPU detected; project-local CUDA runtime is not needed.")
        return 0
    if args.check_only:
        if local_cuda_ready():
            print(f"Project-local CUDA runtime is ready: {CUDA_ROOT}")
            return 0
        print("NVIDIA GPU detected, but the project-local CUDA 12/cuDNN 9 runtime is missing. Run start.ps1 to prepare it.")
        return 1
    try:
        ensure_local_cuda_runtime(progress=print)
        return 0
    except Exception as exc:
        print(f"CUDA runtime preparation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
