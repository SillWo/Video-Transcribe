"""Nemotron diarization runtime and word-to-speaker turn assembly."""

from __future__ import annotations

import heapq
import json
import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path

from services.model_paths import APP_ROOT, MODEL_TEMP_DIR
from services.task_errors import TaskFailure, is_out_of_memory_error


def run_diarization(wav_path, requested_device="auto") -> dict:
    """Run one Nemotron pass over a complete PCM WAV and return intervals."""
    from services.diarization_runtime import resolve_diarization_device, resolve_diarization_model

    device = resolve_diarization_device(requested_device)
    try:
        return _run_diarization_device(wav_path, device, resolve_diarization_model)
    except TaskFailure as exc:
        retryable_codes = {"device_unavailable", "device_error", "out_of_memory", "diarization_failed"}
        if requested_device != "auto" or device.get("id") == "cpu" or exc.code not in retryable_codes:
            raise
        print(f"Nemotron {device.get('backend')} failed ({exc}); retrying diarization on CPU.", flush=True)
        cpu_device = resolve_diarization_device("cpu")
        result = _run_diarization_device(wav_path, cpu_device, resolve_diarization_model)
        result["fallbackReason"] = str(exc)
        return result


def _run_diarization_device(wav_path, device: dict, resolve_model) -> dict:
    model_path = Path(resolve_model())
    info = Path(device["executable"])
    if not info.is_file():
        raise TaskFailure("Nemotron diarization executable is unavailable.", "device_unavailable", stage="diarization")
    if not model_path.is_file():
        raise TaskFailure("Nemotron diarization model is unavailable.", "model_unavailable", stage="diarization")

    fd, output_name = tempfile.mkstemp(prefix="vt-diarization-", suffix=".json", dir=MODEL_TEMP_DIR)
    os.close(fd)
    output = Path(output_name)
    selected = device["id"]
    runtime_device = "cpu" if selected == "cpu" else f"{device['backend']}:{device['deviceIndex']}"
    command = [str(info), "diarize", str(Path(wav_path).resolve()), "--model", str(model_path),
               "--device", runtime_device, "--preset", "v3-offline", "--format", "json", "--force",
               "--output", str(output)]
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    try:
        completed = subprocess.run(command, cwd=str(APP_ROOT), capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", env=env)
        log = f"{completed.stdout or ''}\n{completed.stderr or ''}".strip()
        if completed.returncode:
            if is_out_of_memory_error(log):
                code = "out_of_memory"
            elif any(term in log.lower() for term in ("device lost", "device error", "vulkan", "cuda")):
                code = "device_error"
            else:
                code = "diarization_failed"
            raise TaskFailure(f"Nemotron diarization failed (exit {completed.returncode}): {log[-1600:]}",
                              code, stage="diarization")
        try:
            result = json.loads(output.read_text(encoding="utf-8"))
            intervals = result["segments"]
            if not isinstance(intervals, list):
                raise ValueError("segments is not an array")
            with wave.open(str(wav_path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            intervals = validate_intervals(intervals, duration)
        except (OSError, ValueError, KeyError, TypeError, wave.Error, ZeroDivisionError) as exc:
            raise TaskFailure(f"Nemotron returned invalid JSON output: {exc}", "diarization_failed",
                              stage="diarization") from exc
        return {"effectiveDevice": {key: device[key] for key in ("id", "name", "backend")},
                "intervals": intervals}
    except TaskFailure:
        raise
    except OSError as exc:
        raise TaskFailure(f"Could not start Nemotron diarization: {exc}", "device_error",
                          stage="diarization") from exc
    finally:
        output.unlink(missing_ok=True)


def validate_intervals(intervals, duration: float) -> list[dict]:
    """Reject malformed native segments and clamp harmless 20ms duration overshoot."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("input WAV has invalid duration")
    result = []
    for index, item in enumerate(intervals):
        if not isinstance(item, dict):
            raise ValueError(f"segment {index} is not an object")
        try:
            raw_start, raw_end, speaker = item["start"], item["end"], item["speaker"]
            if (isinstance(raw_start, bool) or not isinstance(raw_start, (int, float))
                    or isinstance(raw_end, bool) or not isinstance(raw_end, (int, float))):
                raise ValueError("start/end must be JSON numbers")
            start, end = float(raw_start), float(raw_end)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"segment {index} has invalid start/end/speaker fields") from exc
        if isinstance(speaker, bool) or not isinstance(speaker, int) or not 1 <= speaker <= 8:
            raise ValueError(f"segment {index} has invalid speaker id")
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"segment {index} has invalid time range")
        if start > duration + 0.02 or end > duration + 0.02:
            raise ValueError(f"segment {index} exceeds audio duration")
        start = min(start, duration)
        end = min(end, duration)
        if end <= start:
            raise ValueError(f"segment {index} is empty after duration clamping")
        result.append({"start": start, "end": end, "speaker": speaker})
    return result


def _valid_interval(interval):
    try:
        start, end = float(interval["start"]), float(interval["end"])
        channel = interval["speaker"]
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        return None
    return start, end, channel


def _label(speaker_ids):
    if not speaker_ids:
        return "Говорящий не определён"
    if len(speaker_ids) == 1:
        return f"Говорящий {speaker_ids[0]}"
    return "Говорящие " + " и ".join(map(str, speaker_ids))


def _join_text(words):
    return "".join(str(word.get("text", "")) for word in words).strip()


def assign_speakers(segments, intervals, *, restore_punctuation=False) -> list[dict]:
    """Group word-timed ASR segments into turns, preserving every source word."""
    valid = [entry for item in intervals if (entry := _valid_interval(item)) is not None]
    appearances = {}
    for start, _end, channel in valid:
        appearances[channel] = min(start, appearances.get(channel, start))
    def channel_order(channel):
        try:
            return 0, float(channel)
        except (TypeError, ValueError):
            return 1, str(channel)

    ordered_speakers = sorted(appearances, key=lambda ch: (appearances[ch], channel_order(ch)))
    speaker_numbers = {channel: number for number, channel in enumerate(ordered_speakers, 1)}
    valid.sort(key=lambda item: (item[0], item[1], str(item[2])))

    entries = []
    for segment_index, segment in enumerate(segments):
        text = str(segment.get("text", ""))
        raw_words = segment.get("words") or []
        if not raw_words:
            entries.append({"start": segment.get("start", 0), "end": segment.get("end", 0),
                            "text": text, "_fallback": True, "_segment_index": segment_index})
            continue
        clean_words = []
        for word in raw_words:
            try:
                start, end = float(word["start"]), float(word["end"])
                valid_time = math.isfinite(start) and math.isfinite(end) and 0 <= start < end
                clean_words.append({"start": start if valid_time else None, "end": end if valid_time else None,
                                    "text": str(word.get("text", "")), "_fallback": not valid_time,
                                    "_segment_index": segment_index})
            except (KeyError, TypeError, ValueError):
                clean_words.append({"start": None, "end": None, "text": str(word.get("text", "")),
                                    "_fallback": True, "_segment_index": segment_index})
        if not clean_words or _join_text(clean_words) != text.strip():
            entries.append({"start": segment.get("start", 0), "end": segment.get("end", 0),
                            "text": text, "_fallback": True, "_segment_index": segment_index})
        else:
            entries.extend(clean_words)

    # Sweep sorted midpoints against interval start/end events.
    indexed = sorted(((index, word) for index, word in enumerate(entries) if not word.get("_fallback")),
                     key=lambda pair: (pair[1]["start"] + pair[1]["end"], pair[0]))
    heap = []
    active = {}
    next_interval = 0
    speakers_for_entry = {}
    for original_index, word in indexed:
        midpoint = (word["start"] + word["end"]) / 2
        while next_interval < len(valid) and valid[next_interval][0] <= midpoint:
            start, end, channel = valid[next_interval]
            serial = next_interval
            active[serial] = (end, channel)
            heapq.heappush(heap, (end, serial))
            next_interval += 1
        while heap and heap[0][0] <= midpoint:
            _end, serial = heapq.heappop(heap)
            active.pop(serial, None)
        channels = {channel for _end, channel in active.values()}
        ids = sorted({speaker_numbers[channel] for channel in channels})
        speakers_for_entry[original_index] = ids

    turns = []
    for index, item in enumerate(entries):
        start, end = item["start"], item["end"]
        ids = [] if item.get("_fallback") else speakers_for_entry.get(index, [])
        text_part = str(item.get("text", ""))
        untimed_unknown = item.get("_fallback") and start is None and end is None
        previous_untimed_unknown = bool(turns and turns[-1].get("timingKnown") is False)
        merge_untimed = untimed_unknown and previous_untimed_unknown
        merge_timed = (turns and not untimed_unknown and turns[-1].get("timingKnown") is not False
                       and start is not None and end is not None and start - turns[-1]["end"] <= 1.5)
        if turns and turns[-1]["speakerIds"] == ids and (merge_untimed or merge_timed):
            if not merge_untimed:
                turns[-1]["end"] = max(turns[-1]["end"], end)
            previous_text = "".join(turns[-1]["_words"])
            if (turns[-1]["_segment_index"] != item.get("_segment_index") and previous_text and text_part
                    and not previous_text[-1].isspace() and not text_part[0].isspace()):
                turns[-1]["_words"].append(" ")
            turns[-1]["_words"].append(text_part)
            turns[-1]["_segment_index"] = item.get("_segment_index")
        else:
            turns.append({"start": start, "end": end, "timingKnown": not untimed_unknown, "speakerIds": ids,
                          "_words": [text_part], "_segment_index": item.get("_segment_index")})

    result = []
    for turn in turns:
        text = "".join(turn.pop("_words")).strip()
        turn.pop("_segment_index", None)
        if restore_punctuation:
            from utils.text_postprocess import restore_russian_segment_text
            text = restore_russian_segment_text(text)
        result.append({"start": turn["start"], "end": turn["end"], "timingKnown": turn["timingKnown"], "text": text,
                       "speakerIds": turn["speakerIds"], "label": _label(turn["speakerIds"])})
    return result


def render_turns(turns) -> str:
    lines = []
    for turn in turns:
        if turn.get("timingKnown") is False or turn.get("start") is None:
            lines.append(f"{turn['label']}: {turn['text']}")
            continue
        seconds = max(0, int(float(turn["start"])))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        lines.append(f"[{hours:02d}:{minutes:02d}:{seconds:02d}] {turn['label']}: {turn['text']}")
    return "\n".join(lines)
