"""Pinned, isolated resources for the additional ASR profiles.

The module deliberately imports only the Python standard library.  Heavy ML
packages are installed in profile-specific project environments and are never
imported by the application process.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from services.task_errors import TaskFailure

APP_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = APP_ROOT / ".runtime"
MODEL_ROOT = RUNTIME / "models" / "asr"
ENV_ROOT = RUNTIME / "asr"
NEW_MODELS = frozenset({"parakeet-tdt-0.6b-v3", "qwen3-asr-1.7b"})
Progress = Callable[[dict], None]
_HASH_CACHE: dict[tuple[str, int, int, int], str] = {}
_ENV_CACHE: dict[tuple, bool] = {}


def invalidate_runtime_cache() -> None:
    """Forget runtime validation results after another process prepares environments."""
    _ENV_CACHE.clear()

PARAKEET_REV = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"
QWEN_REV = "0e0a596e3959c3f3f2ab7d443a5b6f0f4a4bba6f"
LANGID_REV = "e59f0f3847e77723b7d58447ef9456fa0c7e9436"
ALIGNER_REV = "c07281df297b9905d24a508279258cccf987a064"
VALIDATION_URL = "https://raw.githubusercontent.com/handy-computer/transcribe.cpp/v0.4.1/samples/fleurs-en.wav"
VALIDATION_FILES = {"fleurs-en.wav": (337964, "7835bd6ffb54ce38a2a9bcde3905ba424faed94d50a474f21a9cbe9209b869df")}

# Every artifact used by an installer is pinned to a commit and verified by
# both its advertised size and digest before it becomes visible as a final file.
PARAKEET_FILES = {
    "config.json": (97, "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466"),
    "encoder-model.onnx": (41770866, "98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1"),
    "encoder-model.onnx.data": (2435420160, "9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36"),
    "decoder_joint-model.onnx": (72520893, "e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1"),
    "nemo128.onnx": (139764, "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f"),
    "vocab.txt": (93939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
}
QWEN_FILES = {
    "Qwen3-ASR-1.7B-F16.gguf": (4091390944, "edb09c29b8f73822c639168d5ef72aa2dccdf8b4e48fc4b8518885352ff62c71"),
}
LANGID_FILES = {
    "lang-id-voxlingua107-ecapa-F32.gguf": (84993184, "d0420e485112ca05e31aa06e1615556102083a9725e89b55a1f0963c0c0ead66"),
}
ALIGNER_FILES = {
    "config.json": (248029, "82ac390b36a8f80c5a0c4e202367debd149e5d8c30abe4cf5eed439101b1c5ce"),
    "processor_config.json": (487, "bc0b230081b44e629dd5b9045b78495615c1831b4b9f4cffe97bd37e82a6156a"),
    "tokenizer.json": (11429842, "cadfb9cbd9ff8f4e309075ec6a71b063295ff796afb4b5dbc0fd2873fbb301f2"),
    "tokenizer_config.json": (998, "945e980986de2ca7768f3326bfdbb4fbea3406f972b8ae0be233089f2b253c11"),
    "chat_template.jinja": (551, "59b5ae119bda7c00089bba5f49af775791ea67c7aeb7d2b828cddda9463121bf"),
    "model.safetensors": (1835545960, "00568245ceca5af1991d28562a75fe1ddc9bfeb041c27fda66947ea05c47fb86"),
}

PACKAGE_VERSIONS = {
    ("native", "cpu"): {"transcribe-cpp": "0.4.1", "transcribe-cpp-native": "0.4.1", "numpy": "2.4.4"},
    ("native", "vulkan"): {"transcribe-cpp": "0.4.1", "transcribe-cpp-native": "0.4.1", "numpy": "2.4.4"},
    ("native", "cuda"): {"transcribe-cpp": "0.4.1", "transcribe-cpp-native-cu12": "0.4.1",
                           "transcribe-cpp-native": "0.4.1", "nvidia-cuda-runtime-cu12": "12.9.79",
                           "nvidia-cuda-nvrtc-cu12": "12.9.86", "nvidia-cublas-cu12": "12.9.2.10",
                           "numpy": "2.4.4"},
    ("parakeet", "cpu"): {"onnx-asr": "0.12.0", "numpy": "2.4.4"},
    ("parakeet", "cuda"): {"onnx-asr": "0.12.0", "onnxruntime-gpu": "1.24.4", "numpy": "2.4.4"},
    ("parakeet", "directml"): {"onnx-asr": "0.12.0", "onnxruntime-directml": "1.24.4", "numpy": "2.4.4"},
    ("aligner", "cpu"): {"torch": "2.15.0+cpu", "transformers": "5.22.1", "safetensors": "0.8.0"},
    ("aligner", "cuda"): {"torch": "2.15.0", "transformers": "5.22.1", "safetensors": "0.8.0"},
}

_CANDIDATES = {
    ("native", "cpu"): (RUNTIME / "asr-validation" / "native-venv",),
    ("native", "vulkan"): (RUNTIME / "asr-validation" / "native-venv",),
    ("native", "cuda"): (),
    ("parakeet", "cpu"): (RUNTIME / "asr-evaluation" / "parakeet-venv",),
    ("parakeet", "directml"): (RUNTIME / "asr-evaluation" / "parakeet-venv",),
    ("aligner", "cpu"): (RUNTIME / "asr-evaluation" / "nemotron-venv",),
    ("aligner", "cuda"): (RUNTIME / "asr-evaluation" / "aligner-cuda-venv",
                            RUNTIME / "asr-validation" / "aligner-cuda-venv"),
}


def is_new_asr(model_id: str) -> bool:
    return model_id in NEW_MODELS


def _uses_qwen_aligner(model_id: str, diarization: bool) -> bool:
    return diarization and model_id == "qwen3-asr-1.7b"


def _python_at(env: Path) -> Path:
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _kind_backend(kind: str, backend: str) -> tuple[str, str]:
    if kind not in {"native", "parakeet", "aligner"}:
        raise ValueError(f"Unknown ASR runtime: {kind}")
    if (kind, backend) not in PACKAGE_VERSIONS:
        raise ValueError(f"Unsupported backend {backend!r} for runtime {kind!r}")
    return kind, backend


def _environment_fingerprint(python: Path, kind: str, backend: str) -> tuple:
    resolved = python.resolve()
    try:
        exe_stat = resolved.stat()
        exe_fingerprint = (exe_stat.st_size, exe_stat.st_mtime_ns, exe_stat.st_ctime_ns)
    except OSError:
        exe_fingerprint = None
    site_packages = (resolved.parent / ".." / "Lib" / "site-packages" if os.name == "nt"
                     else resolved.parent / ".." / "lib")
    try:
        env_stat = site_packages.resolve().stat()
        env_fingerprint = (env_stat.st_mtime_ns, env_stat.st_ctime_ns)
    except OSError:
        env_fingerprint = None
    req = _requirements_file(kind, backend)
    try:
        req_stat = req.stat()
        req_fingerprint = (req_stat.st_mtime_ns, req_stat.st_size)
    except OSError:
        req_fingerprint = None
    return (str(resolved), exe_fingerprint, env_fingerprint, req_fingerprint, kind, backend)


def _installed_exact(python: Path, kind: str, backend: str) -> bool:
    cache_key = _environment_fingerprint(python, kind, backend)
    if cache_key in _ENV_CACHE:
        return _ENV_CACHE[cache_key]
    script = (
        "import importlib.metadata as m,json\n"
        "names=('transcribe-cpp','transcribe-cpp-native','transcribe-cpp-native-cu12','nvidia-cuda-runtime-cu12','nvidia-cuda-nvrtc-cu12','nvidia-cublas-cu12','onnx-asr','onnxruntime','onnxruntime-directml','onnxruntime-gpu','numpy','torch','transformers','safetensors')\n"
        "got={}\n"
        "for n in names:\n try: got[n]=m.version(n)\n except m.PackageNotFoundError: pass\n"
        "providers=None\n"
        "try:\n import onnxruntime as ort; providers=ort.get_available_providers()\nexcept Exception: pass\n"
        "torch_state=None\n"
        "try:\n import torch; torch_state={'version':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available()}\nexcept Exception: pass\n"
        "print(json.dumps({'versions':got,'providers':providers,'torch':torch_state}))"
    )
    try:
        result = subprocess.run([str(python), "-c", script], cwd=APP_ROOT,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                timeout=20, check=False)
        try:
            state = json.loads(result.stdout or "{}") if result.returncode == 0 else {}
        except (TypeError, ValueError):
            state = {}
        _ENV_CACHE[cache_key] = _runtime_versions_match(kind, backend, state)
        return _ENV_CACHE[cache_key]
    except (OSError, subprocess.SubprocessError):
        _ENV_CACHE[cache_key] = False
        return False


def _runtime_versions_match(kind: str, backend: str, state: dict) -> bool:
    versions = state.get("versions") or {}
    want = PACKAGE_VERSIONS[(kind, backend)]
    if kind == "parakeet" and backend == "cpu":
        runtime_ok = any(versions.get(name) == "1.24.4"
                         for name in ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu"))
        providers = state.get("providers") or ()
        return (versions.get("onnx-asr") == "0.12.0" and versions.get("numpy") == "2.4.4"
                and runtime_ok and "CPUExecutionProvider" in providers)
    for name, version in want.items():
        actual = versions.get(name)
        cpu_torch_alias = (kind == "aligner" and backend == "cpu" and name == "torch"
                           and version == "2.15.0+cpu" and actual == "2.15.0")
        cuda_torch_alias = (kind == "aligner" and backend == "cuda" and name == "torch"
                            and actual is not None and (actual == version or actual.startswith(version + "+")))
        if actual != version and not cpu_torch_alias and not cuda_torch_alias:
            return False
    providers = state.get("providers") or ()
    if kind == "parakeet":
        needed = {"directml": {"CPUExecutionProvider", "DmlExecutionProvider"},
                  "cuda": {"CPUExecutionProvider", "CUDAExecutionProvider"}}.get(backend, set())
        return needed.issubset(set(providers))
    if kind == "aligner":
        torch = state.get("torch") or {}
        if backend == "cpu":
            return (torch.get("version") in {"2.15.0", "2.15.0+cpu"}
                    and torch.get("cuda") is None)
        return (str(torch.get("version", "")).startswith("2.15.0")
                and torch.get("cuda") is not None and torch.get("available") is True)
    return True


def _canonical_env(kind: str, backend: str) -> Path:
    suffix = f"-{backend}" if kind in {"native", "parakeet"} or (kind == "aligner" and backend == "cuda") else ""
    return ENV_ROOT / f"{kind}{suffix}"


def runtime_python(kind: str, backend: str = "cpu") -> Path:
    kind, backend = _kind_backend(kind, backend)
    canonical = _canonical_env(kind, backend)
    if _installed_exact(_python_at(canonical), kind, backend):
        return _python_at(canonical)
    for candidate in _CANDIDATES.get((kind, backend), ()):
        if _inside_project(candidate) and _installed_exact(_python_at(candidate), kind, backend):
            return _python_at(candidate)
    return _python_at(canonical)


def model_paths(model_id: str) -> dict[str, Path]:
    if model_id == "parakeet-tdt-0.6b-v3":
        return {"model_dir": MODEL_ROOT / "parakeet-tdt-0.6b-v3", "language_model": MODEL_ROOT / "lang-id-voxlingua107-ecapa-F32.gguf"}
    if model_id == "qwen3-asr-1.7b":
        return {"model": MODEL_ROOT / "Qwen3-ASR-1.7B-F16.gguf"}
    raise ValueError(f"Unknown new ASR model: {model_id}")


def alignment_model_dir() -> Path:
    return MODEL_ROOT / "qwen3-forced-aligner-0.6b"


def validation_audio_path() -> Path:
    return MODEL_ROOT / "validation" / "fleurs-en.wav"


def ensure_validation_audio(on_progress: Progress | None = None) -> None:
    """Prepare only the pinned GPU verification fixture used by legacy Whisper."""
    total = sum(size for size, _ in VALIDATION_FILES.values())
    _ensure_files(VALIDATION_URL, "v0.4.1", VALIDATION_FILES,
                  validation_audio_path().parent, "validation", on_progress, 0, total)
    if not validation_audio_path().is_file():
        raise TaskFailure("Не удалось подготовить аудиофайл проверки GPU.", "model_unavailable", stage="preparation")


def _inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(APP_ROOT.resolve())
        return True
    except (OSError, ValueError):
        return False


def _hash_file(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    cache_key = (str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if cache_key in _HASH_CACHE:
        return _HASH_CACHE[cache_key]
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _HASH_CACHE[cache_key] = value
    return value


def _valid(path: Path, expected: tuple[int, str]) -> bool:
    size, digest = expected
    return path.is_file() and path.stat().st_size == size and _hash_file(path) == digest


def _copy_verified(source: Path, target: Path, expected: tuple[int, str]) -> bool:
    if not _valid(source, expected):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".copy.part")
    try:
        # Reuse the bytes without doubling multi-gigabyte weights when the
        # existing validation cache and canonical model directory share a volume.
        os.link(source, partial)
    except OSError:
        shutil.copyfile(source, partial)
    if not _valid(partial, expected):
        partial.unlink(missing_ok=True)
        return False
    os.replace(partial, target)
    return True


def _known_local_sources(model_id: str, file_name: str) -> tuple[Path, ...]:
    if model_id == "parakeet-tdt-0.6b-v3":
        return (RUNTIME / "asr-evaluation" / "models" / "parakeet-v3-fp32" / file_name,)
    if model_id == "qwen3-asr-1.7b":
        return (RUNTIME / "asr-validation" / "models" / "qwen-f16" / file_name,)
    if model_id == "validation":
        return (RUNTIME / "asr-validation" / "audio" / file_name,)
    if model_id == "aligner":
        return (RUNTIME / "asr-evaluation" / "models" / "qwen-forced-aligner-0.6b-hf" / file_name,)
    return (RUNTIME / "asr-validation" / "models" / "langid" / file_name,)


def _emit(progress: Progress | None, done: int | None, total: int | None, message: str) -> None:
    if progress:
        progress({"stage": "preparation", "message": message,
                  "completedBytes": done, "totalBytes": total})


def _download(repo: str, revision: str, name: str, target: Path,
              expected: tuple[int, str], progress: Progress | None, base_done: int, total: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if _valid(target, expected):
        return
    partial = target.with_name(target.name + ".part")
    size, digest = expected
    if partial.exists() and partial.stat().st_size > size:
        partial.unlink()
    if repo.startswith("https://"):
        url = repo
    else:
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}?download=true"
    failure: BaseException | None = None
    for attempt in range(5):
        start = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "Video-Transcribe/1.0"} | ({"Range": f"bytes={start}-"} if start else {}))
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                status = getattr(response, "status", 200)
                if start and status == 206:
                    content_range = getattr(response, "headers", {}).get("Content-Range", "")
                    try:
                        byte_range, resource_size = content_range.split(" ", 1)[1].split("/")
                        range_start = int(byte_range.split("-", 1)[0])
                        declared_size = int(resource_size)
                    except (ValueError, IndexError):
                        raise ValueError("Invalid Content-Range for resumed download")
                    if range_start != start or declared_size != size:
                        raise ValueError("Unexpected Content-Range for resumed download")
                elif start:
                    start = 0
                    partial.unlink(missing_ok=True)
                mode = "ab" if start else "wb"
                completed = start
                last_report = 0.0
                with partial.open(mode) as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        if completed + len(block) > size:
                            raise ValueError(f"Download exceeded expected size for {name}")
                        output.write(block)
                        completed += len(block)
                        now = time.monotonic()
                        if now - last_report >= 0.3:
                            _emit(progress, min(total, base_done + completed), total, f"Загрузка {name}")
                            last_report = now
                if partial.stat().st_size == size and _hash_file(partial) == digest:
                    os.replace(partial, target)
                    _emit(progress, min(total, base_done + size), total, f"Проверен файл {name}")
                    return
                if partial.stat().st_size == size:
                    partial.unlink()
                    raise ValueError(f"SHA-256 mismatch for {name}")
                raise OSError(f"Incomplete download for {name}: {partial.stat().st_size}/{size}")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            failure = exc
            if isinstance(exc, ValueError):
                partial.unlink(missing_ok=True)
                break
            if attempt < 4:
                time.sleep(min(2 ** attempt, 8))
    raise TaskFailure(f"Не удалось загрузить или проверить файл {name}.", "model_download_failed", stage="preparation") from failure


def _ensure_files(repo: str, revision: str, manifest: dict[str, tuple[int, str]], directory: Path,
                  source_model: str, progress: Progress | None, total_before: int, overall: int) -> int:
    done = total_before
    for name, expected in manifest.items():
        target = directory / name
        if _valid(target, expected):
            done += expected[0]
            continue
        adopted = any(_copy_verified(source, target, expected) for source in _known_local_sources(source_model, name))
        if adopted:
            done += expected[0]
            _emit(progress, done, overall, f"Использован проверенный файл {name}")
            continue
        _download(repo, revision, name, target, expected, progress, done, overall)
        done += expected[0]
    return done


def _manifest_for(model_id: str, diarization: bool, backend: str = "cpu") -> tuple[int, list[tuple[str, str, dict, Path, str]]]:
    paths = model_paths(model_id)
    groups: list[tuple[str, str, dict, Path, str]] = []
    if model_id == "parakeet-tdt-0.6b-v3":
        groups.append(("istupakov/parakeet-tdt-0.6b-v3-onnx", PARAKEET_REV, PARAKEET_FILES, paths["model_dir"], model_id))
        groups.append(("handy-computer/lang-id-voxlingua107-ecapa-gguf", LANGID_REV, LANGID_FILES, paths["language_model"].parent, "langid"))
    else:
        groups.append(("handy-computer/Qwen3-ASR-1.7B-gguf", QWEN_REV, QWEN_FILES, paths["model"].parent, model_id))
    if backend != "cpu":
        groups.append((VALIDATION_URL, "v0.4.1", VALIDATION_FILES, validation_audio_path().parent, "validation"))
    if _uses_qwen_aligner(model_id, diarization):
        # Qwen uses the separate aligner for requested word timestamps/diarization.
        groups.append(("Qwen/Qwen3-ForcedAligner-0.6B-hf", ALIGNER_REV, ALIGNER_FILES, alignment_model_dir(), "aligner"))
    byte_total = sum(size for _, _, manifest, _, _ in groups for size, _ in manifest.values())
    return byte_total, groups


def _fingerprint(model_id: str, backend: str, diarization: bool) -> str:
    total, groups = _manifest_for(model_id, diarization, backend)
    artifacts = []
    for _repo, revision, manifest, directory, _source in groups:
        for name, expected in sorted(manifest.items()):
            path = directory / name
            try:
                resolved = path.resolve()
                stat = resolved.stat()
                signature = (str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            except OSError:
                signature = (str(path.resolve()), None, None, None)
            artifacts.append((revision, name, expected, signature))
    runtimes = []
    kind = "native" if model_id == "qwen3-asr-1.7b" else "parakeet"
    runtime_specs = [(kind, backend)]
    if model_id == "parakeet-tdt-0.6b-v3":
        runtime_specs.append(("native", {"directml": "vulkan"}.get(backend, backend)))
    if _uses_qwen_aligner(model_id, diarization):
        runtime_specs.append(("aligner", "cpu"))
    for runtime_kind, runtime_backend in runtime_specs:
        python = runtime_python(runtime_kind, runtime_backend)
        runtimes.append((runtime_kind, runtime_backend,
                         _environment_fingerprint(python, runtime_kind, runtime_backend),
                         _installed_exact(python, runtime_kind, runtime_backend)))
    data = {"model": model_id, "backend": backend, "diarization": diarization, "total": total,
            "groups": [(repo, rev, sorted((name, *value) for name, value in manifest.items()))
                       for repo, rev, manifest, _, _ in groups],
            "artifacts": artifacts, "runtimes": runtimes,
            "packages": {f"{kind}:{backend}": PACKAGE_VERSIONS.get((kind, backend), {})
                         for kind, backend in runtime_specs}}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def _requirements_file(kind: str, backend: str) -> Path:
    suffix = "-cuda" if backend == "cuda" else "-directml" if backend == "directml" else ""
    path = APP_ROOT / f"requirements-asr-{kind}{suffix}.txt"
    if not path.is_file() and kind == "native":
        path = APP_ROOT / "requirements-asr-native.txt"
    return path


def _ensure_runtime(kind: str, backend: str, progress: Progress | None) -> None:
    python = runtime_python(kind, backend)
    canonical = _canonical_env(kind, backend)
    if _inside_project(python) and _installed_exact(python, kind, backend):
        return
    if not _python_at(canonical).is_file():
        canonical.parent.mkdir(parents=True, exist_ok=True)
        _emit(progress, None, None, f"Создание окружения {kind} ({backend})")
        result = subprocess.run([sys.executable, "-m", "venv", str(canonical)], cwd=APP_ROOT,
                                stdin=subprocess.DEVNULL, check=False)
        if result.returncode:
            raise TaskFailure("Не удалось создать изолированное окружение ASR.", "dependency_missing", stage="preparation")
        python = _python_at(canonical)
    req = _requirements_file(kind, backend)
    if not req.is_file():
        raise TaskFailure(f"Не найден файл зависимостей: {req.name}", "dependency_missing", stage="preparation")
    cache = RUNTIME / "cache" / "pip-asr"
    temp = RUNTIME / "temp" / "asr"
    cache.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"PIP_CACHE_DIR": str(cache), "TEMP": str(temp), "TMP": str(temp), "HF_HOME": str(RUNTIME / "cache" / "hf-asr")})
    _emit(progress, None, None, f"Установка зависимостей {kind} ({backend})")
    result = subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)],
                            cwd=APP_ROOT, env=env, stdin=subprocess.DEVNULL, check=False)
    if result.returncode or not _installed_exact(python, kind, backend):
        raise TaskFailure("Не удалось подготовить точные версии зависимостей ASR.", "dependency_missing", stage="preparation")


def _resource_status(model_id: str, backend: str, diarization: bool) -> dict:
    missing: list[str] = []
    _, groups = _manifest_for(model_id, diarization, backend)
    for _, _, manifest, directory, _ in groups:
        for name, expected in manifest.items():
            if not _valid(directory / name, expected):
                missing.append(str(directory / name))
    kind = "native" if model_id == "qwen3-asr-1.7b" else "parakeet"
    runtime = runtime_python(kind, backend)
    if not _installed_exact(runtime, kind, backend):
        missing.append(f"{kind} environment ({backend})")
    if model_id == "parakeet-tdt-0.6b-v3":
        native_backend = {"directml": "vulkan"}.get(backend, backend)
        if not _installed_exact(runtime_python("native", native_backend), "native", native_backend):
            missing.append(f"native language-id environment ({native_backend})")
    if _uses_qwen_aligner(model_id, diarization) and not _installed_exact(runtime_python("aligner", "cpu"), "aligner", "cpu"):
        missing.append("aligner environment (cpu)")
    return {"ready": not missing, "missing": missing, "fingerprint": _fingerprint(model_id, backend, diarization)}


def asr_resource_status(model_id: str, backend: str, *, diarization: bool = False) -> dict:
    if not is_new_asr(model_id):
        raise ValueError(f"Not a new ASR model: {model_id}")
    return _resource_status(model_id, backend, diarization)


def ensure_asr_resources(model_id: str, backend: str, *, diarization: bool = False,
                         on_progress: Progress | None = None) -> None:
    if not is_new_asr(model_id):
        raise ValueError(f"Not a new ASR model: {model_id}")
    kind = "native" if model_id == "qwen3-asr-1.7b" else "parakeet"
    _kind_backend(kind, backend)
    use_aligner = _uses_qwen_aligner(model_id, diarization)
    if use_aligner:
        _kind_backend("aligner", "cuda" if backend == "cuda" else "cpu")
    total, groups = _manifest_for(model_id, diarization, backend)
    done = 0
    for repo, revision, manifest, directory, source_model in groups:
        done = _ensure_files(repo, revision, manifest, directory, source_model, on_progress, done, total)
    _ensure_runtime(kind, backend, on_progress)
    if model_id == "parakeet-tdt-0.6b-v3":
        native_backend = {"directml": "vulkan"}.get(backend, backend)
        _ensure_runtime("native", native_backend, on_progress)
    if use_aligner:
        if backend == "cuda":
            try:
                _ensure_runtime("aligner", "cuda", on_progress)
            except Exception as exc:
                _emit(on_progress, None, None,
                      f"CUDA aligner недоступен ({exc}); будет использован CPU aligner")
        _ensure_runtime("aligner", "cpu", on_progress)
    status = _resource_status(model_id, backend, diarization)
    if not status["ready"]:
        raise TaskFailure("Проверка подготовленных ресурсов ASR не пройдена.", "model_unavailable", stage="preparation")
