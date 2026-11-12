"""Runtime discovery for the transcription backends exposed by the API."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]
_CUDA_DLL_HANDLES = []
CUDA_RUNTIME_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")
VULKAN_MODELS = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo", "large",
]
_GPU_INVENTORY_CACHE: tuple[float, tuple[int, int, int], list, list, list] | None = None


def invalidate_gpu_inventory_cache() -> None:
    global _GPU_INVENTORY_CACHE
    _GPU_INVENTORY_CACHE = None


def _gpu_inventory_snapshot() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Share short-lived adapter discovery across profile rows in one options request."""
    global _GPU_INVENTORY_CACHE
    providers = (enumerate_vulkan_gpus, _dxgi_gpu_inventory, _windows_gpu_inventory)
    now = time.monotonic()
    if (_GPU_INVENTORY_CACHE is not None
            and now - _GPU_INVENTORY_CACHE[0] < 2
            and _GPU_INVENTORY_CACHE[1] == tuple(map(id, providers))):
        return _GPU_INVENTORY_CACHE[2], _GPU_INVENTORY_CACHE[3], _GPU_INVENTORY_CACHE[4]
    snapshot = (enumerate_vulkan_gpus(), _dxgi_gpu_inventory(), _windows_gpu_inventory())
    _GPU_INVENTORY_CACHE = (now, tuple(map(id, providers)), *snapshot)
    return snapshot


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

    # Vulkan 1.1 exposes deviceUUID through vkGetPhysicalDeviceProperties2.
    info = AppInfo(0, None, b"Video Transcribe", 1, b"Video Transcribe", 1, (1 << 22) | (1 << 12))
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
        get_properties2 = getattr(library, "vkGetPhysicalDeviceProperties2", None)
        if get_properties2 is None:
            get_properties2 = getattr(library, "vkGetPhysicalDeviceProperties2KHR", None)
        if get_properties2 is not None:
            class DeviceIdProperties(ctypes.Structure):
                _fields_ = [
                    ("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p),
                    ("deviceUUID", ctypes.c_uint8 * 16), ("driverUUID", ctypes.c_uint8 * 16),
                    ("deviceLUID", ctypes.c_uint8 * 8), ("deviceNodeMask", ctypes.c_uint32),
                    ("deviceLUIDValid", ctypes.c_uint32),
                ]

            class PhysicalDeviceProperties2Buffer(ctypes.Structure):
                _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p),
                            ("properties", ctypes.c_uint8 * 1024)]

            get_properties2.argtypes = [ctypes.c_void_p, ctypes.POINTER(PhysicalDeviceProperties2Buffer)]
            get_properties2.restype = None
        else:
            DeviceIdProperties = None
            PhysicalDeviceProperties2Buffer = None
        found = []
        for physical_index, device in enumerate(devices[:count.value]):
            raw = ctypes.create_string_buffer(1024)
            get_properties(device, raw)
            props = ctypes.cast(raw, ctypes.POINTER(_VkPhysicalDevicePropertiesPrefix)).contents
            # Device type 4 is CPU (lavapipe/software). Prefer discrete and integrated GPUs.
            if props.deviceType == 4:
                continue
            name = props.deviceName.split(b"\0", 1)[0].decode("utf-8", "replace") or "Vulkan GPU"
            entry = {"name": name, "type": props.deviceType, "vendorId": props.vendorID,
                     "deviceId": props.deviceID, "deviceIndex": physical_index,
                     "runtimeIndex": len(found)}
            if get_properties2 is not None:
                identity = DeviceIdProperties(1000071004, None)
                properties2 = PhysicalDeviceProperties2Buffer(1000059001, ctypes.cast(ctypes.pointer(identity), ctypes.c_void_p))
                get_properties2(device, ctypes.byref(properties2))
                device_uuid = bytes(identity.deviceUUID)
                if any(device_uuid):
                    entry["deviceUUID"] = device_uuid.hex()
            found.append(entry)
        return found
    finally:
        destroy_instance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        destroy_instance(instance, None)


def cuda_device_count() -> int:
    try:
        from services.cuda_runtime import register_local_cuda_runtime
        register_local_cuda_runtime()
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _is_windows() -> bool:
    return os.name == "nt"


def _windows_gpu_inventory() -> list[dict[str, Any]]:
    if not _is_windows():
        return []
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            return []
        import json
        rows = json.loads(result.stdout or "[]")
        if isinstance(rows, dict):
            rows = [rows]
        found = []
        for row in rows:
            name = str(row.get("Name") or "").strip()
            pnp = str(row.get("PNPDeviceID") or "").upper()
            vendor_id = 0x10DE if "VEN_10DE" in pnp else 0x1002 if "VEN_1002" in pnp else 0x8086 if "VEN_8086" in pnp else None
            if name and vendor_id:
                import re
                device_match = re.search(r"&DEV_([0-9A-F]{4})", pnp)
                found.append({"name": name, "vendorId": vendor_id,
                              "deviceId": int(device_match.group(1), 16) if device_match else None,
                              "driverVersion": row.get("DriverVersion"), "pnpDeviceId": pnp})
        return found
    except Exception:
        return []


def _dxgi_gpu_inventory() -> list[dict[str, Any]]:
    """Enumerate hardware DXGI adapters and preserve their DirectML device IDs.

    DirectML's device_id is the DXGI adapter ordinal. Vulkan's physical-device
    index is a different namespace, so it must never be copied here.
    """
    if not _is_windows():
        return []
    try:
        class Luid(ctypes.Structure):
            _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]

        class AdapterDesc1(ctypes.Structure):
            _fields_ = [
                ("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint32),
                ("DeviceId", ctypes.c_uint32), ("SubSysId", ctypes.c_uint32),
                ("Revision", ctypes.c_uint32), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", Luid), ("Flags", ctypes.c_uint32),
            ]

        dxgi = ctypes.WinDLL("dxgi")
        create = dxgi.CreateDXGIFactory1
        create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_long
        iid = (ctypes.c_ubyte * 16).from_buffer_copy(
            __import__("uuid").UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le
        )
        factory = ctypes.c_void_p()
        if create(ctypes.byref(iid), ctypes.byref(factory)) < 0 or not factory:
            return []
        try:
            vtable = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            enum_adapters1 = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
            )(vtable[12])
            release_factory = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
            adapters: list[dict[str, Any]] = []
            ordinal = 0
            while True:
                adapter = ctypes.c_void_p()
                hr = enum_adapters1(factory, ordinal, ctypes.byref(adapter))
                if hr < 0 or not adapter:
                    break
                try:
                    adapter_vtable = ctypes.cast(adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
                    get_desc1 = ctypes.WINFUNCTYPE(
                        ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(AdapterDesc1)
                    )(adapter_vtable[10])
                    release_adapter = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(adapter_vtable[2])
                    desc = AdapterDesc1()
                    if get_desc1(adapter, ctypes.byref(desc)) >= 0 and not (desc.Flags & 2):
                        luid = (int(desc.AdapterLuid.HighPart) & 0xFFFFFFFF) << 32 | int(desc.AdapterLuid.LowPart)
                        adapters.append({
                            "name": desc.Description,
                            "vendorId": int(desc.VendorId),
                            "deviceId": int(desc.DeviceId),
                            "deviceIndex": ordinal,
                            "luid": f"{luid:016x}",
                            "dedicatedVideoMemory": int(desc.DedicatedVideoMemory),
                        })
                    release_adapter(adapter)
                except Exception:
                    # Release the COM adapter even if parsing its description fails.
                    try:
                        adapter_vtable = ctypes.cast(adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
                        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(adapter_vtable[2])(adapter)
                    except Exception:
                        pass
                ordinal += 1
            release_factory(factory)
            return adapters
        except Exception:
            try:
                release_factory(factory)
            except Exception:
                pass
            return []
    except Exception:
        return []


def asr_gpu_candidate(model_id: str) -> dict[str, Any]:
    """Return the best supported physical GPU candidate for one ASR model.

    The selected index belongs to the candidate backend. In particular,
    Parakeet DirectML uses DXGI's adapter ordinal, never a Vulkan ordinal.
    """
    qwen_id = "qwen3-asr-1.7b"
    parakeet_id = "parakeet-tdt-0.6b-v3"
    is_new = model_id in {qwen_id, parakeet_id}
    if model_id not in {qwen_id, parakeet_id, "small"}:
        return {"available": False, "canPrepare": False, "backend": None, "deviceIndex": None,
                "name": None, "reason": "Для этой модели GPU backend не настроен."}

    vulkan, dxgi, windows = _gpu_inventory_snapshot()
    adapters = dxgi or windows
    def enrich_windows_identity(selected: dict[str, Any]) -> dict[str, Any]:
        """Attach driver identity only when one WMI row maps unambiguously."""
        matches = [item for item in windows
                   if item.get("vendorId") == selected.get("vendorId")
                   and str(item.get("name", "")).casefold() == str(selected.get("name", "")).casefold()
                   and (selected.get("deviceId") is None or item.get("deviceId") is None
                        or item.get("deviceId") == selected.get("deviceId"))]
        if len(matches) == 1:
            return {**selected,
                    "driverVersion": matches[0].get("driverVersion"),
                    "pnpDeviceId": matches[0].get("pnpDeviceId")}
        # If same-name adapters are ambiguous, do not attach a guessed driver
        # identity or allow a capability cache to survive a driver change.
        return {**selected, "driverVersion": None, "pnpDeviceId": None,
                "identity": None if len(matches) > 1 else selected.get("identity")}

    by_vendor: dict[int, list[dict[str, Any]]] = {0x10DE: [], 0x1002: [], 0x8086: []}
    for item in adapters + vulkan:
        vendor = item.get("vendorId")
        if vendor in by_vendor:
            key = (vendor, item.get("deviceId"), item.get("name"))
            if not any((vendor, existing.get("deviceId"), existing.get("name")) == key for existing in by_vendor[vendor]):
                by_vendor[vendor].append(item)

    # NVIDIA is the preferred candidate when it has a CUDA-visible device.
    if by_vendor[0x10DE]:
        cuda_count = cuda_device_count()
        cuda_ready, cuda_reason = cuda_runtime_status(cuda_count)
        if cuda_ready:
            selected = enrich_windows_identity(by_vendor[0x10DE][0])
            return {"available": True, "canPrepare": True, "backend": "cuda", "deviceIndex": 0,
                    "name": selected.get("name") or "NVIDIA GPU", "vendorId": 0x10DE,
                    "deviceId": selected.get("deviceId"), "driverVersion": selected.get("driverVersion"),
                    "identity": selected.get("luid") or selected.get("pnpDeviceId") or selected.get("name"),
                    "reason": None}
        # The physical device is still a preparable candidate; exact runtime
        # readiness is established during owned preparation and GPU smoke.
        selected = enrich_windows_identity(by_vendor[0x10DE][0])
        return {"available": False, "canPrepare": True, "backend": "cuda", "deviceIndex": 0,
                "name": selected.get("name") or "NVIDIA GPU", "vendorId": 0x10DE,
                "deviceId": selected.get("deviceId"), "driverVersion": selected.get("driverVersion"),
                "identity": selected.get("luid") or selected.get("pnpDeviceId") or selected.get("name"),
                "reason": cuda_reason or "NVIDIA CUDA runtime will be checked during preparation."}

    if is_new and model_id == parakeet_id:
        supported = by_vendor[0x1002] + by_vendor[0x8086]
        if supported:
            selected = enrich_windows_identity(supported[0])
            index = selected.get("deviceIndex")
            if index is None:
                return {"available": False, "canPrepare": False, "backend": "directml", "deviceIndex": None,
                        "name": selected.get("name"), "reason": "Не удалось сопоставить GPU с индексом DXGI для DirectML."}
            return {"available": False, "canPrepare": True, "backend": "directml", "deviceIndex": int(index),
                    "name": selected.get("name"), "vendorId": selected.get("vendorId"),
                    "deviceId": selected.get("deviceId"), "driverVersion": selected.get("driverVersion"),
                    "identity": selected.get("luid") or selected.get("pnpDeviceId") or selected.get("name"),
                    "reason": "DirectML GPU capability has not been verified for this model yet."}

    if by_vendor[0x1002] or by_vendor[0x8086]:
        supported = [gpu for gpu in vulkan if gpu.get("vendorId") in {0x1002, 0x8086}]
        if supported:
            selected = enrich_windows_identity(supported[0])
            index = selected.get("runtimeIndex", selected.get("deviceIndex"))
            return {"available": False, "canPrepare": True, "backend": "vulkan", "deviceIndex": int(index or 0),
                    "name": selected.get("name"), "vendorId": selected.get("vendorId"),
                    "deviceId": selected.get("deviceId"), "driverVersion": selected.get("driverVersion"),
                    "identity": selected.get("deviceUUID") or selected.get("name"),
                    "reason": "Vulkan GPU capability has not been verified for this model yet."}
        selected = enrich_windows_identity((by_vendor[0x1002] + by_vendor[0x8086])[0])
        return {"available": False, "canPrepare": False, "backend": "vulkan", "deviceIndex": None,
                "name": selected.get("name"), "reason": "Физический GPU обнаружен, но Vulkan adapter сейчас недоступен."}

    return {"available": False, "canPrepare": False, "backend": None, "deviceIndex": None,
            "name": None, "reason": "Совместимая видеокарта не обнаружена."}


def get_simple_gpu_details() -> dict[str, Any]:
    """Choose the supported GPU for the simple UI, separating hardware from runtime readiness."""
    vulkan = enumerate_vulkan_gpus()
    inventory = _windows_gpu_inventory()
    candidates = vulkan + [item for item in inventory if not any(item["name"] == v["name"] for v in vulkan)]
    nvidia = next((gpu for gpu in candidates if gpu["vendorId"] == 0x10DE), None)
    amd = next((gpu for gpu in candidates if gpu["vendorId"] == 0x1002), None)
    if nvidia:
        count = cuda_device_count()
        ready, reason = cuda_runtime_status(count)
        if ready:
            return {"available": True, "name": nvidia["name"], "reason": None, "backend": "cuda"}
        return {"available": False, "name": nvidia["name"], "reason": reason or "NVIDIA CUDA runtime is unavailable.", "backend": "cuda"}
    if amd:
        vk = next((gpu for gpu in vulkan if gpu["vendorId"] == 0x1002), None)
        if not vk:
            return {"available": False, "name": amd["name"], "reason": "AMD GPU was detected, but no Vulkan runtime device is available.", "backend": "vulkan"}
        exe = find_whisper_cpp_executable()
        if not exe:
            return {"available": False, "name": amd["name"], "reason": "AMD GPU was detected, but the Vulkan Whisper runtime is missing.", "backend": "vulkan", "deviceIndex": vk.get("deviceIndex")}
        return {"available": True, "name": vk["name"], "reason": None, "backend": "vulkan", "deviceIndex": vk.get("deviceIndex")}
    return {"available": False, "name": None, "reason": "No supported NVIDIA or AMD GPU was detected; transcription will use CPU.", "backend": None}


def cuda_runtime_status(device_count: int) -> tuple[bool, str | None]:
    if device_count <= 0:
        if _is_windows():
            try:
                from services.cuda_runtime import local_cuda_ready
                if not local_cuda_ready():
                    return False, "CTranslate2 reports no CUDA device and the project-local CUDA 12/cuDNN 9 runtime is missing. Run start.ps1; an NVIDIA driver is also required."
            except Exception:
                pass
            return False, "No CUDA device is available through CTranslate2. Confirm the NVIDIA driver is installed and active."
        return False, "No CUDA device is available through CTranslate2."
    if not _is_windows():
        return True, None
    try:
        from services.cuda_runtime import register_local_cuda_runtime
        register_local_cuda_runtime()
        for library in CUDA_RUNTIME_DLLS:
            _CUDA_DLL_HANDLES.append(ctypes.WinDLL(library))
    except (AttributeError, OSError) as exc:
        return False, f"CUDA libraries are missing ({exc}). Run start.ps1 to prepare the project-local CUDA 12 and cuDNN 9 runtime, or use Vulkan."
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
