"""Typed failures shared by preparation and transcription subprocesses."""

from __future__ import annotations


SYSTEMIC_ERROR_CODES = frozenset({
    "model_download_failed",
    "model_unavailable",
    "dependency_missing",
    "device_unavailable",
    "device_error",
    "out_of_memory",
    "storage_unavailable",
})


class TaskFailure(RuntimeError):
    """An expected task failure with a stable machine-readable category."""

    def __init__(self, message: str, code: str = "internal_error", *, stage: str | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.stage = stage


def failure_stage(exc: BaseException, default: str = "result") -> str:
    """Return the operation stage attached to a failure, if any."""
    stage = getattr(exc, "stage", None)
    return stage if isinstance(stage, str) and stage else default


def is_out_of_memory_error(exc: BaseException | str) -> bool:
    text = str(exc).lower().replace("-", "_")
    signatures = (
        "out of memory",
        "out_of_memory",
        "cuda_error_out_of_memory",
        "cudnn_status_alloc_failed",
        "vk_error_out_of_device_memory",
        "vkallocatememory failed",
        "failed to allocate",
    )
    return any(signature in text for signature in signatures)


def is_invalid_source_error(exc: BaseException | str) -> bool:
    text = str(exc).lower()
    signatures = (
        "invalid data found when processing input",
        "error opening input",
        "could not open input",
        "failed to open input",
        "failed to decode",
        "decode failed",
        "input decode failed",
        "decoding failed",
        "error while decoding",
        "unable to read audio",
        "not a valid media file",
        "no such file or directory",
        "does not exist",
        "unexpected end of file",
        "truncated input",
    )
    return any(signature in text for signature in signatures)
