"""Small, server-owned configuration for the simple transcription experience."""

from __future__ import annotations

from services.asr_capabilities import profile_device_options
from services.runtime_devices import asr_gpu_candidate, get_simple_gpu_details

PROFILES = {
    "quality": ("Качество", "qwen3-asr-1.7b"),
    "balanced": ("Баланс", "parakeet-tdt-0.6b-v3"),
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
    return profile_device_options(PROFILES, max_cpu_threads)


def resolve(profile: str, device: str) -> tuple[str, str]:
    if profile not in PROFILES:
        raise ValueError("Неизвестный профиль расшифровки.")
    if device not in {"cpu", "gpu"}:
        raise ValueError("Устройство должно быть cpu или gpu.")
    model = PROFILES[profile][1]
    if device == "cpu":
        return model, "cpu"
    candidate = asr_gpu_candidate(model)
    if candidate.get("canPrepare") and candidate.get("backend"):
        return model, candidate["backend"]
    gpu = gpu_details()
    if model == "small" and gpu.get("backend") in {"cuda", "vulkan"}:
        return model, gpu["backend"]
    raise ValueError(_public_gpu_reason(gpu) or "Совместимый GPU сейчас недоступен.")
