"""Meeting-specific persistence and optional stages for the shared job queue."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from services.job_processes import JobCancelled, JobProcessController
from services.meeting_archive import MeetingArchive

STAGES = ("speech", "screen", "audio", "index")


class MeetingStageFailure(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)


_STAGE_TIMEOUT = {"screen": 8 * 60 * 60, "audio": 2 * 60 * 60, "index": 2 * 60 * 60}
_EVENT_PREFIX = "__VT_MEETING__ "


def _worker_command(stage: str, root: Path, meeting_id: str, source: str | Path,
                    staging: Path, device: str) -> list[str]:
    return [sys.executable, "-m", "services.meeting_worker", "--stage", stage,
            "--root", str(root), "--meeting-id", meeting_id,
            "--source", str(source or ""), "--result-path", str(staging / "result.json"),
            "--device", device]


def _standalone_worker(stage: str, root: Path, meeting_id: str, source: str | Path,
                       staging: Path, device: str,
                       progress: Callable[[dict], None] | None,
                       cancel: Callable[[], bool] | None) -> dict:
    """Run the same isolated CLI worker outside FastAPI with safe tree cancellation."""
    command = _worker_command(stage, root, meeting_id, source, staging, device)
    controller = JobProcessController()
    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1, env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    controller.register(process)
    messages: queue.Queue[str | None] = queue.Queue()
    def read_output():
        try:
            for line in process.stdout or ():
                messages.put(line.rstrip())
        finally:
            messages.put(None)
    reader = threading.Thread(target=read_output, name="meeting-worker-output", daemon=True)
    reader.start()
    deadline = time.monotonic() + _STAGE_TIMEOUT.get(stage, 3600)
    ended = False
    try:
        while not ended:
            if cancel and cancel():
                controller.cancel()
                process.wait()
                raise JobCancelled("Meeting stage was cancelled.")
            if time.monotonic() >= deadline:
                controller.stop_owned()
                process.wait()
                raise TimeoutError(f"{stage} worker exceeded its time limit")
            try:
                line = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                ended = True
            elif line.startswith(_EVENT_PREFIX):
                try:
                    event = json.loads(line[len(_EVENT_PREFIX):])
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "progress" and progress:
                    progress(event.get("progress") or {})
        exit_code = process.wait()
    finally:
        if process.poll() is None:
            controller.stop_owned()
            process.wait()
        if process.stdout:
            process.stdout.close()
        controller.unregister(process)
        reader.join(timeout=1)
    result_path = staging / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{stage} worker returned no valid result: {exc}") from exc
    if exit_code != 0 or not result.get("ok"):
        raise RuntimeError(result.get("error") or f"{stage} worker failed with exit code {exit_code}")
    return result


def _commit_files(staging: Path, meeting_dir: Path, relative_paths: list[str]) -> None:
    root = staging.resolve()
    meeting_root = meeting_dir.resolve()
    for relative_text in relative_paths:
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Worker returned an unsafe artifact path")
        source = (staging / relative).resolve()
        if source != root and root not in source.parents:
            raise ValueError("Worker artifact resolves outside staging")
        if not source.is_file():
            raise FileNotFoundError(source)
        target = meeting_dir / relative
        resolved_target_parent = target.parent.resolve()
        if resolved_target_parent != meeting_root and meeting_root not in resolved_target_parent.parents:
            raise ValueError("Artifact target resolves outside meeting directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temp)
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)


def meeting_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer diarization turns; retain native timestamped ASR as a fallback."""
    raw_segments = result.get("segments")
    if not isinstance(raw_segments, list):
        return []
    diarization = result.get("diarization") or {}
    turns = diarization.get("turns", []) if isinstance(diarization, dict) else []
    if turns:
        source_rows = [t for t in turns if isinstance(t, dict) and t.get("text")]
    else:
        source_rows = [s for s in raw_segments if isinstance(s, dict)]
    prepared = []
    for raw in source_rows:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", ""))
        if not text:
            continue
        start, end = raw.get("start"), raw.get("end")
        # Never estimate timestamps from text length. Preserve native values or None.
        try:
            start = float(start) if start is not None else None
            end = float(end) if end is not None else None
        except (TypeError, ValueError):
            start, end = None, None
        identity = f"{start!r}\0{end!r}" if start is not None and end is not None else f"untimed\0{text}"
        evidence_id = "speech:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        speaker_ids = raw.get("speakerIds") or []
        speaker = raw.get("speaker")
        segment = {"id": evidence_id, "start": start, "end": end, "text": text}
        if speaker_ids:
            segment["raw_speaker_ids"] = [str(s) for s in speaker_ids]
        if speaker is not None:
            speaker = str(speaker) if str(speaker).startswith("speaker_") else f"speaker_{speaker}"
            segment["speaker"] = speaker
        elif len(speaker_ids) > 1:
            segment["speaker_ambiguous"] = True
        prepared.append(segment)
    return prepared


def persist_speech(root: str | Path, meeting_id: str, result: dict[str, Any]) -> dict:
    archive = MeetingArchive(root)
    raw_turns = (((result.get("diarization") or {}).get("turns") or [])
                 if isinstance(result.get("diarization"), dict) else [])
    prior_path = archive.meeting_dir(meeting_id) / "speaker-turns.json"
    try:
        prior_turns = json.loads(prior_path.read_text(encoding="utf-8")).get("turns", [])
    except (OSError, json.JSONDecodeError):
        prior_turns = []
    try:
        confirmed_speakers = json.loads((archive.meeting_dir(meeting_id) / "speakers.json").read_text(encoding="utf-8")).get("confirmed", {})
    except (OSError, json.JSONDecodeError):
        confirmed_speakers = {}
    old_by_new: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    new_clusters = {}
    for turn in raw_turns:
        if not isinstance(turn, dict) or turn.get("start") is None or turn.get("end") is None:
            continue
        for raw_speaker in turn.get("speakerIds", []) or []:
            key = str(raw_speaker)
            new_clusters.setdefault(key, []).append(turn)
    old_clusters = {}
    for turn in prior_turns:
        sid = turn.get("speaker_id")
        if sid and turn.get("start") is not None and turn.get("end") is not None:
            old_clusters.setdefault(str(sid), []).append(turn)
    claimed_old: set[str] = set()
    for new_id, turns in new_clusters.items():
        scores: dict[str, float] = {}
        for new_turn in turns:
            a, b = float(new_turn["start"]), float(new_turn["end"])
            for old_id, old_turns in old_clusters.items():
                scores[old_id] = scores.get(old_id, 0.0) + sum(
                    max(0.0, min(b, float(o["end"])) - max(a, float(o["start"])))
                    for o in old_turns if o.get("start") is not None and o.get("end") is not None)
        ranked = sorted(((score, sid) for sid, score in scores.items()), reverse=True)
        if ranked and ranked[0][0] > 0:
            best = ranked[0][0]
            tied = [sid for score, sid in ranked if score == best]
            if len(tied) == 1 and tied[0] not in claimed_old and (len(ranked) == 1 or ranked[1][0] < best):
                old_by_new[new_id] = tied[0]
                claimed_old.add(tied[0])
            else:
                ambiguous[new_id] = [sid for score, sid in ranked if score > 0]
        elif not prior_turns and not confirmed_speakers:
            old_by_new[new_id] = f"speaker_{new_id}"
        elif not prior_turns and confirmed_speakers:
            ambiguous[new_id] = list(confirmed_speakers)
        else:
            onset = min(float(t["start"]) for t in turns)
            digest = hashlib.sha256(f"{onset:.3f}".encode()).hexdigest()[:10]
            old_by_new[new_id] = f"speaker_auto_{digest}"
    segments = meeting_segments(result)
    for seg in segments:
        turn = next((candidate for candidate in raw_turns
                     if isinstance(candidate, dict) and candidate.get("start") == seg.get("start")
                     and candidate.get("end") == seg.get("end")
                     and candidate.get("text") == seg.get("text")), seg)
        raw_ids = [str(v) for v in (turn.get("speakerIds", []) if isinstance(turn, dict) else [])]
        if len(raw_ids) == 1:
            mapped = old_by_new.get(raw_ids[0])
            if mapped:
                seg["speaker"] = mapped
            elif raw_ids[0] in ambiguous:
                seg["speaker_ambiguous"] = True
                seg["speaker_candidates"] = ambiguous[raw_ids[0]]
    stored_turns = []
    for turn in raw_turns:
        if not isinstance(turn, dict):
            continue
        row = dict(turn)
        raw_ids = [str(v) for v in (turn.get("speakerIds") or [])]
        row["speaker_id"] = old_by_new.get(raw_ids[0]) if len(raw_ids) == 1 else None
        row["speaker_mapping_ambiguous"] = bool(raw_ids and raw_ids[0] in ambiguous)
        row["speaker_candidates"] = ambiguous.get(raw_ids[0], []) if raw_ids else []
        stored_turns.append(row)
    raw_text = result.get("text")
    stored = archive.write_transcript(meeting_id, segments, raw_text if isinstance(raw_text, str) else None)
    archive.write_source_artifacts(meeting_id,
        raw_asr={"segments": result.get("segments", []), "text": raw_text,
                 "diarization": result.get("diarization", {})},
        speaker_turns=stored_turns)
    archive.update_meeting(meeting_id, status="speech_ready", speech_status="ready")
    return stored


def run_stages(root: str | Path, meeting_id: str, source_path: str | Path,
               stages: list[str], *, progress: Callable[[str, str, dict], None] | None = None,
               cancel: Callable[[], bool] | None = None,
               runner: Callable[[str, Path, Callable[[dict], None] | None], dict] | None = None,
               device: str = "cpu") -> dict:
    """Run selected post-ASR stages in workers and finalize archive output."""
    archive = MeetingArchive(root)
    folder = archive.meeting_dir(meeting_id)
    root_path = Path(root).resolve()
    meta = archive.get_meeting(meeting_id)
    selected = [stage for stage in STAGES if stage in stages]
    state = meta.get("stage_state", {})
    for stage in selected:
        started = time.monotonic()
        entry = {"status": "active", "started_at": time.time()}
        state[stage] = entry
        archive.update_meeting(meeting_id, stage_state=state, status=f"{stage}_active")
        if progress:
            progress(stage, "active", {})
        staging = folder / ".staging" / f"{stage}-{uuid.uuid4().hex}"
        try:
            staging.mkdir(parents=True, exist_ok=False)
            if not staging.resolve().is_relative_to(folder.resolve()):
                raise ValueError("Meeting staging path resolves outside meeting directory")
            if cancel and cancel():
                raise InterruptedError("stage cancelled")
            report_progress = (lambda payload: progress(stage, "active", payload)) if progress else None
            execute = runner or (lambda current_stage, workdir, callback: _standalone_worker(
                current_stage, root_path, meeting_id, source_path, workdir, device, callback, cancel))
            if stage == "index":
                archive.refresh_editorial_exports(meeting_id)
            output = execute(stage, staging, report_progress)
            if not isinstance(output, dict) or output.get("ok") is False:
                raise RuntimeError((output or {}).get("error") or f"{stage} worker returned an invalid result")
            entry["device"] = output.get("device", device)
            if stage == "screen":
                extracted = output.get("result", {})
                events = extracted.get("events", [])
                frame_paths = [str(event.get("frame")) for event in events if event.get("frame")]
                _commit_files(staging, folder, frame_paths)
                archive.write_screen_events(meeting_id, extracted.get("events", []),
                                            status=extracted.get("status", "unavailable"),
                                            limitations=extracted.get("limitations", []))
                entry.update(status=extracted.get("status", "unavailable"), stats=extracted.get("stats", {}),
                             limitations=extracted.get("limitations", []))
            elif stage == "audio":
                outputs = output.get("outputs", [])
                _commit_files(staging, folder, [str(row["path"]) for row in outputs if row.get("path")])
                entry.update(status=output.get("status", "unavailable"), outputs=outputs,
                             reason=output.get("reason"))
            elif stage == "index":
                entry.update(status="ready", indexed_items=output.get("indexed_items"),
                             embeddings=output.get("embeddings", {"status": "unavailable"}))
            if cancel and cancel():
                raise InterruptedError(f"{stage} stage cancelled")
            entry["elapsed_seconds"] = round(time.monotonic() - started, 3)
            entry["completed_at"] = time.time()
            if progress:
                progress(stage, entry["status"], entry)
            shutil.rmtree(staging, ignore_errors=True)
        except JobCancelled as exc:
            shutil.rmtree(staging, ignore_errors=True)
            entry.update(status="cancelled", reason=str(exc), elapsed_seconds=round(time.monotonic() - started, 3),
                         completed_at=time.time())
            state[stage] = entry
            archive.update_meeting(meeting_id, stage_state=state, status="interrupted")
            archive.finalize(meeting_id)
            raise
        except InterruptedError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            entry.update(status="cancelled", reason=str(exc), elapsed_seconds=round(time.monotonic() - started, 3),
                         completed_at=time.time())
            state[stage] = entry
            archive.update_meeting(meeting_id, stage_state=state, status="interrupted")
            archive.finalize(meeting_id)
            raise
        except Exception as exc:
            if cancel and cancel():
                shutil.rmtree(staging, ignore_errors=True)
                entry.update(status="cancelled", reason=str(exc), elapsed_seconds=round(time.monotonic() - started, 3),
                             completed_at=time.time())
                state[stage] = entry
                archive.update_meeting(meeting_id, stage_state=state, status="interrupted")
                archive.finalize(meeting_id)
                raise JobCancelled("Meeting stage was cancelled.") from exc
            entry.update(status="failed", reason=str(exc), elapsed_seconds=round(time.monotonic() - started, 3),
                         completed_at=time.time())
            if progress:
                progress(stage, "failed", entry)
            state[stage] = entry
            archive.update_meeting(meeting_id, stage_state=state, status="partial")
            archive.finalize(meeting_id)
            raise MeetingStageFailure(stage, str(exc)) from exc
        state[stage] = entry
        archive.update_meeting(meeting_id, stage_state=state, status="partial")
    aggregate = "ready" if all(state.get(s, {}).get("status") in {"ready", "completed"} for s in selected) else "partial"
    archive.update_meeting(meeting_id, stage_state=state, status=aggregate, manifest_ready=True)
    manifest = archive.finalize(meeting_id)
    if manifest.get("status", {}).get("overall") != aggregate:
        archive.update_meeting(meeting_id, status=manifest.get("status", {}).get("overall", "partial"))
        manifest = archive.finalize(meeting_id)
    return {"stages": state, "manifest": manifest}
