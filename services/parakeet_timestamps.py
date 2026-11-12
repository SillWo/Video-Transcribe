"""Native Parakeet TDT token durations and word timestamp conversion."""

from __future__ import annotations

import math
import re
from typing import Any, Callable


def recognize_with_native_durations(model: Any, recognize: Callable[[], Any]) -> tuple[Any, list[int]]:
    """Observe TDT decoder decisions during recognition without another inference."""
    asr = model.asr
    original = asr._decode
    had_instance_value = "_decode" in vars(asr)
    previous_instance_value = vars(asr).get("_decode")
    durations: list[int] = []

    def capture(prev_tokens: Any, prev_state: Any, encoder_out: Any):
        logits, step, state = original(prev_tokens, prev_state, encoder_out)
        try:
            if hasattr(logits, "argmax"):
                predicted = int(logits.argmax())
            else:
                predicted = max(range(len(logits)), key=logits.__getitem__)
            if predicted != int(asr._blank_idx):
                # TDT's step is the predicted duration in encoder frames.
                duration_value = float(step)
                durations.append(int(duration_value) if duration_value.is_integer() else -1)
        except (AttributeError, TypeError, ValueError, OverflowError):
            durations.append(-1)
        return logits, step, state

    asr._decode = capture
    try:
        return recognize(), durations
    finally:
        if had_instance_value:
            asr._decode = previous_instance_value
        else:
            delattr(asr, "_decode")


def _decoded_text_with_owners(tokens: list[str], pattern: re.Pattern[str]) -> tuple[str, list[int]]:
    raw = "".join(tokens)
    owners = [token_index for token_index, token in enumerate(tokens) for _char in token]
    out: list[str] = []
    mapped: list[int] = []
    cursor = 0
    for match in pattern.finditer(raw):
        out.append(raw[cursor:match.start()])
        mapped.extend(owners[cursor:match.start()])
        if match.group(1):
            group_index = match.start(1)
            out.append(" ")
            mapped.append(owners[group_index])
        cursor = match.end()
    out.append(raw[cursor:])
    mapped.extend(owners[cursor:])
    decoded = "".join(out)
    left = len(decoded) - len(decoded.lstrip())
    right = len(decoded.rstrip())
    return decoded[left:right], mapped[left:right]


def words_from_timestamped_result(
    recognized: Any,
    native_durations: list[int] | None,
    *,
    frame_seconds: float,
    chunk_start: float,
    chunk_duration: float,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return transcript-preserving words and unknown times on invalid metadata."""
    text = str(getattr(recognized, "text", "") or "").strip()
    if not text:
        return [], None

    fallback = [{"text": word.group(), "start": 0.0, "end": 0.0}
                for word in re.finditer(r"\s*\S+", text)]
    tokens = getattr(recognized, "tokens", None)
    timestamps = getattr(recognized, "timestamps", None)
    try:
        if not isinstance(tokens, (list, tuple)) or not isinstance(timestamps, (list, tuple)):
            raise ValueError("token or timestamp metadata is missing")
        if any(not isinstance(token, str) for token in tokens):
            raise ValueError("token metadata contains non-text values")
        if not tokens or len(timestamps) != len(tokens):
            raise ValueError("token and timestamp counts differ")
        if not isinstance(native_durations, (list, tuple)) or len(native_durations) != len(tokens):
            raise ValueError("native duration count does not match token count")
        frame_seconds = float(frame_seconds)
        chunk_start = float(chunk_start)
        chunk_duration = float(chunk_duration)
        if not math.isfinite(frame_seconds) or frame_seconds <= 0:
            raise ValueError("native frame duration is invalid")
        if not math.isfinite(chunk_start) or not math.isfinite(chunk_duration) or chunk_duration <= 0:
            raise ValueError("chunk bounds are invalid")

        starts: list[float] = []
        ends: list[float] = []
        last_start = -1.0
        for raw_start, raw_duration in zip(timestamps, native_durations, strict=True):
            start = float(raw_start)
            duration = float(raw_duration)
            if not math.isfinite(start) or start < 0 or start < last_start:
                raise ValueError("token timestamps are invalid or nonmonotonic")
            if not math.isfinite(duration) or duration < 0 or not duration.is_integer():
                raise ValueError("native token duration is invalid")
            end = start + duration * frame_seconds
            if not math.isfinite(end):
                raise ValueError("native token end is invalid")
            if end > chunk_duration:
                overshoot = end - chunk_duration
                if overshoot <= frame_seconds + 1e-9:
                    end = chunk_duration
                else:
                    raise ValueError("native token end exceeds chunk bounds")
            if start > chunk_duration:
                raise ValueError("native token start exceeds chunk bounds")
            starts.append(start)
            ends.append(end)
            last_start = start

        # This is the exact pattern used by onnx_asr._AsrWithDecoding.
        pattern = re.compile(r"\A\s|\s\B|(\s)\b")
        decoded, char_owners = _decoded_text_with_owners(tokens, pattern)
        if decoded != text:
            raise ValueError("decoded tokens do not match recognized text")

        words: list[dict[str, Any]] = []
        for match in re.finditer(r"\s*\S+", text):
            owned = {char_owners[index] for index in range(match.start(), match.end())
                     if not text[index].isspace()}
            if not owned:
                words.append({"text": match.group(), "start": 0.0, "end": 0.0})
                continue
            word_start = min(starts[index] for index in owned)
            word_end = max(ends[index] for index in owned)
            if word_end <= word_start:
                words.append({"text": match.group(), "start": 0.0, "end": 0.0})
            else:
                words.append({"text": match.group(), "start": chunk_start + word_start,
                              "end": chunk_start + word_end})
        if "".join(word["text"] for word in words) != text:
            raise ValueError("word conversion did not preserve transcript")
        return words, None
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:
        return fallback, str(exc)
