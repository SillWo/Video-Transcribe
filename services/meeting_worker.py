"""Child-process entry point for heavyweight meeting stages.

The module imports only the standard library and archive code at startup. OCR,
OpenCV, NumPy and ONNX dependencies are imported only inside their worker stage.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from services.meeting_archive import MeetingArchive

EVENT_PREFIX = "__VT_MEETING__ "


def _emit(event: dict[str, Any]) -> None:
    sys.stdout.write(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def _inside(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("worker path resolves outside meeting archive")
    return resolved


def wait_for_recording_clear(runtime_root: str | Path, *, cancelled=lambda: False,
                             on_wait=None, interval: float = 0.25, sleep=time.sleep) -> float:
    """Yield between screen samples while the meeting recorder owns resources."""
    from services.meeting_recorder import recording_session_active
    started = None
    last_report = None
    while recording_session_active(runtime_root):
        if cancelled():
            raise InterruptedError("Screen stage was cancelled while waiting for the recording to finish.")
        if started is None:
            started = time.monotonic()
        if on_wait and (last_report is None or time.monotonic() - last_report >= 2.0):
            on_wait({"stage": "screen", "recording_paused": True, "message": "Waiting for meeting recording to finish."})
            last_report = time.monotonic()
        sleep(max(0.05, interval))
    if started is None:
        return 0.0
    return max(0.0, time.monotonic() - started)


def _validated_paths(args) -> tuple[Path, Path, Path | None, Path]:
    root = Path(args.root).expanduser().resolve(strict=True)
    archive = MeetingArchive(root, read_only=True)
    meeting_dir = archive.meeting_dir(args.meeting_id)
    staging_parent = meeting_dir / ".staging"
    resolved_staging_parent = staging_parent.resolve()
    if not resolved_staging_parent.is_relative_to(meeting_dir.resolve()):
        raise ValueError("meeting staging directory resolves outside meeting")
    staging = _inside(staging_parent, Path(args.result_path).expanduser().resolve(strict=False).parent)
    if staging != staging_parent.resolve() and staging_parent.resolve() not in staging.parents:
        raise ValueError("result path must be under meeting staging directory")
    if not staging.is_dir():
        raise ValueError("staging directory must be created by the parent coordinator")
    result_path = _inside(staging, Path(args.result_path).expanduser().resolve(strict=False))
    source = None
    if args.source:
        originals_root = (root / "originals").resolve()
        if not originals_root.is_relative_to(root):
            raise ValueError("originals directory resolves outside archive")
        source = _inside(originals_root, Path(args.source).expanduser().resolve(strict=True))
        if not source.is_file():
            raise ValueError("source recording is unavailable")
    return root, meeting_dir, source, result_path


def run_stage(stage: str, root: Path, meeting_dir: Path, source: Path | None,
              staging: Path, device: str, runtime_root: Path | None = None) -> dict[str, Any]:
    if stage == "screen":
        if source is None:
            raise ValueError("screen stage requires an original recording")
        # Heavy screen extraction imports happen only inside this child process.
        from services.meeting_screen import extract_screen
        runtime_root = runtime_root or Path(os.environ.get("VT_SIMPLE_RUNTIME_DIR") or Path(__file__).resolve().parents[1] / ".runtime")
        def report(payload):
            wait_for_recording_clear(runtime_root, on_wait=lambda waiting: _emit(
                {"type": "progress", "stage": stage, "progress": waiting}))
            _emit({"type": "progress", "stage": stage, "progress": payload})
        result = extract_screen(source, staging,
            progress=report,
            device=device)
        return {"stage": stage, "device": device, "result": result}
    if stage == "audio":
        if source is None:
            raise ValueError("audio stage requires an original recording")
        from services.meeting_screen import compress_audio
        outputs = compress_audio(source, staging,
            progress=lambda payload: _emit({"type": "progress", "stage": stage, "progress": payload}))
        return {"stage": stage, "device": device, "outputs": outputs,
                "status": "ready" if outputs else "unavailable",
                "reason": None if outputs else "No audio stream or FFmpeg unavailable."}
    if stage == "index":
        # ONNX/model modules and the SQLite writer are confined to this child.
        from services.meeting_index import MeetingIndex
        indexer = MeetingIndex(root)
        indexed = indexer.index_meeting(meeting_dir.name)
        return {"stage": stage, "device": device, "indexed_items": indexed,
                "embeddings": indexer.capabilities.get("embeddings")}
    raise ValueError("Unsupported meeting stage")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one isolated meeting archive stage")
    parser.add_argument("--stage", choices=("screen", "audio", "index"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--meeting-id", required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--device", choices=("cpu", "directml"), default="cpu")
    parser.add_argument("--runtime-root", default="")
    args = parser.parse_args(argv)
    result_path: Path | None = None
    try:
        root, meeting_dir, source, result_path = _validated_paths(args)
        staging = result_path.parent
        runtime_root = Path(args.runtime_root).expanduser().resolve() if args.runtime_root else None
        output = run_stage(args.stage, root, meeting_dir, source, staging, args.device, runtime_root)
        envelope = {"ok": True, **output}
        _write_json(result_path, envelope)
        _emit({"type": "result", "stage": args.stage, "status": output.get("status", "ready"),
               "result_path": str(result_path)})
        return 0
    except Exception as exc:
        if result_path is not None:
            try:
                _write_json(result_path, {"ok": False, "stage": args.stage, "error": str(exc),
                                          "error_type": type(exc).__name__})
            except Exception:
                pass
        _emit({"type": "error", "stage": args.stage, "error": str(exc),
               "error_type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
