"""Runtime discovery for the transcription backends exposed by the API."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]
_CUDA_DLL_HANDLES = []
CUDA_RUNTIME_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")
VULKAN_MODELS = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo", "large",
]


def find_whisper_cpp_executable() -> str | None:
    configured = os.environ.get("VT_WHISPER_CPP_EXE")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    runtime = APP_ROOT / ".runtime" / "whisper-vulkan"
    if runtime.exists():
        candidates.extend(sorted(runtime.rglob("whisper-cli.exe")))
        candidates.extend(sorted(runtime.rglob("whisper-cli")))
    for name in ("whisper-cli.exe", "whisper-cli"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


class _VkPhysicalDevicePropertiesPrefix(ctypes.Structure):
    _fields_ = [
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_uint32),
        ("deviceName", ctypes.c_char * 256),
    ]


def enumerate_vulkan_gpus() -> list[dict[str, Any]]:
    """Return Vulkan physical devices excluding CPU/software adapters."""
    library_name = ctypes.util.find_library("vulkan")
    candidates = [library_name] if library_name else []
    candidates += ["vulkan-1.dll", "libvulkan.so.1", "libvulkan.dylib"]
    library = None
    for candidate in candidates:
        try:
            if candidate:
                library = ctypes.CDLL(candidate)
                break
        except OSError:
            continue
    if library is None:
        return []
    try:
        create_instance = library.vkCreateInstance
        destroy_instance = library.vkDestroyInstance
        enumerate_devices = library.vkEnumeratePhysicalDevices
    except AttributeError:
        return []

    # Vulkan 1.0 structures. The application requests no extensions or layers.
    class AppInfo(ctypes.Structure):
        _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p),
                    ("pApplicationName", ctypes.c_char_p), ("applicationVersion", ctypes.c_uint32),
                    ("pEngineName", ctypes.c_char_p), ("engineVersion", ctypes.c_uint32),
                    ("apiVersion", ctypes.c_uint32)]

    class InstanceCreateInfo(ctypes.Structure):
        _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                    ("pApplicationInfo", ctypes.POINTER(AppInfo)), ("enabledLayerCount", ctypes.c_uint32),
                    ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
                    ("enabledExtensionCount", ctypes.c_uint32),
                    ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p))]

    info = AppInfo(0, None, b"Video Transcribe", 1, b"Video Transcribe", 1, (1 << 22))
    create = InstanceCreateInfo(1, None, 0, ctypes.pointer(info), 0, None, 0, None)
    instance = ctypes.c_void_p()
    create_instance.argtypes = [ctypes.POINTER(InstanceCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    create_instance.restype = ctypes.c_int32
    if create_instance(ctypes.byref(create), None, ctypes.byref(instance)) != 0:
        return []
    try:
        enumerate_devices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        enumerate_devices.restype = ctypes.c_int32
        count = ctypes.c_uint32()
        if enumerate_devices(instance, ctypes.byref(count), None) != 0 or count.value == 0:
            return []
        devices = (ctypes.c_void_p * count.value)()
        if enumerate_devices(instance, ctypes.byref(count), devices) != 0:
            return []
        get_properties = library.vkGetPhysicalDeviceProperties
        get_properties.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        get_properties.restype = None
        found = []
        for device in devices[:count.value]:
            raw = ctypes.create_string_buffer(1024)
            get_properties(device, raw)
            props = ctypes.cast(raw, ctypes.POINTER(_VkPhysicalDevicePropertiesPrefix)).contents
            # Device type 4 is CPU (lavapipe/software). Prefer discrete and integrated GPUs.
            if props.deviceType == 4:
                continue
            name = props.deviceName.split(b"\0", 1)[0].decode("utf-8", "replace") or "Vulkan GPU"
            found.append({"name": name, "type": props.deviceType, "vendorId": props.vendorID})
        return found
    finally:
        destroy_instance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        destroy_instance(instance, None)


def cuda_device_count() -> int:
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _is_windows() -> bool:
    return os.name == "nt"


def cuda_runtime_status(device_count: int) -> tuple[bool, str | None]:
    if device_count <= 0:
        return False, "No CUDA device is available through CTranslate2."
    if not _is_windows():
        return True, None
    try:
        for library in CUDA_RUNTIME_DLLS:
            _CUDA_DLL_HANDLES.append(ctypes.WinDLL(library))
    except (AttributeError, OSError) as exc:
        return False, f"CUDA libraries are missing ({exc}). Use Vulkan or install CUDA 12 and cuDNN 9."
    return True, None


def get_device_details() -> list[dict[str, Any]]:
    vulkan_exe = find_whisper_cpp_executable()
    vulkan_gpus = enumerate_vulkan_gpus()
    vulkan_available = bool(vulkan_exe and vulkan_gpus)
    cuda_count = cuda_device_count()
    cuda_available, cuda_reason = cuda_runtime_status(cuda_count)
    cpu_name = "CPU"
    try:
        import psutil
        cpu_name = f"CPU ({psutil.cpu_count(logical=True) or os.cpu_count() or 1} threads)"
    except ImportError:
        pass
    gpu_names = ", ".join(gpu["name"] for gpu in vulkan_gpus)
    return [
        {"id": "auto", "available": True, "name": "Automatic", "reason": "Uses Vulkan GPU, then CUDA GPU, then CPU."},
        {"id": "cpu", "available": True, "name": cpu_name, "reason": "Always available."},
        {"id": "vulkan", "available": vulkan_available,
         "name": gpu_names or "Vulkan GPU",
         "reason": (None if vulkan_available else
                    "Vulkan Whisper runtime is missing." if not vulkan_exe else "No non-CPU Vulkan GPU was detected.")},
        {"id": "cuda", "available": cuda_available,
         "name": f"CUDA GPU ({cuda_count})" if cuda_available else "CUDA GPU",
         "reason": cuda_reason},
    ]
