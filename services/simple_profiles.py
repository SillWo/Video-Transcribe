"""Small, server-owned configuration for the simple transcription experience."""

from __future__ import annotations

from services.runtime_devices import get_device_details, get_simple_gpu_details

PROFILES = {
    "quality": ("Качество", "large-v3"),
    "balanced": ("Баланс", "large-v3-turbo"),
    "economy": ("Экономный", "small"),
}


def gpu_details() -> dict:
    """Return the selected simple-mode GPU, including Vulkan physical index when applicable."""
    return get_simple_gpu_details()


def _public_gpu_reason(details: dict) -> str | None:
    reason = details.get("reason")
    if not reason:
        return None
    backend = details.get("backend")
    name = details.get("name")
    if backend == "cuda":
        return f"Обнаружена видеокарта {name}, но CUDA сейчас недоступна. Выберите процессор или обновите драйвер видеокарты."
    if backend == "vulkan":
        return f"Обнаружена видеокарта {name}, но графический runtime сейчас недоступен. Выберите процессор или обновите драйвер видеокарты."
    return "Совместимая видеокарта NVIDIA или AMD не обнаружена. Модель готова работать на CPU."


def options(max_cpu_threads: int) -> dict:
    details = {item["id"]: item for item in get_device_details()}
    gpu = gpu_details()
    devices = {
        "cpu": {"available": True, "name": details.get("cpu", {}).get("name", "CPU")},
        "gpu": {
            "available": bool(gpu.get("available")),
            "name": gpu.get("name") or "GPU",
            "reason": _public_gpu_reason(gpu),
        },
    }
    return {
        "profiles": [{"id": key, "label": value[0], "model": value[1]} for key, value in PROFILES.items()],
        "devices": devices,
        "defaultDevice": "gpu" if devices["gpu"]["available"] else "cpu",
        "maxCpuThreads": max_cpu_threads,
    }


def resolve(profile: str, device: str) -> tuple[str, str]:
    if profile not in PROFILES:
        raise ValueError("Неизвестный профиль расшифровки.")
    if device not in {"cpu", "gpu"}:
        raise ValueError("Устройство должно быть cpu или gpu.")
    model = PROFILES[profile][1]
    if device == "cpu":
        return model, "cpu"
    gpu = gpu_details()
    if gpu.get("available") and gpu.get("backend") in {"cuda", "vulkan"}:
        return model, gpu["backend"]
    raise ValueError(_public_gpu_reason(gpu) or "Совместимый GPU сейчас недоступен.")
