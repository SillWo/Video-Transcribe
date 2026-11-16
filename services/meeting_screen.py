"""Extract screen states from local recordings with native-size PyAV frames.

PyAV is loaded only when screen extraction starts; FFmpeg remains useful for
single-frame image helpers and media probing, while sampled frames never pass
through an intermediate JPEG stream.
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
import unicodedata
import uuid
from bisect import insort
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image


Progress = Callable[[Any], None]


def _exe(name: str) -> str | None:
    return shutil.which(name) or shutil.which(name + ".exe")


def _run(args: list[str], *, timeout: float = 60) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False, timeout=timeout, shell=False)


def probe_media(source: str | Path) -> dict[str, Any]:
    """Return media duration and audio/video stream metadata using ffprobe."""
    path = Path(source)
    ffprobe = _exe("ffprobe")
    if not ffprobe:
        return {"duration": None, "video": [], "audio": [], "error": "ffprobe unavailable"}
    try:
        result = _run([ffprobe, "-v", "error", "-show_streams", "-show_format",
                       "-of", "json", str(path)])
        if result.returncode:
            return {"duration": None, "video": [], "audio": [],
                    "error": result.stderr.decode("utf-8", "replace").strip()}
        data = json.loads(result.stdout.decode("utf-8", "replace"))
        streams = data.get("streams", [])
        duration = _number(data.get("format", {}).get("duration"))
        if duration is None:
            durations = [_number(s.get("duration")) for s in streams]
            duration = max((d for d in durations if d is not None), default=None)
        def brief(stream: dict[str, Any], index: int) -> dict[str, Any]:
            return {"index": index, "codec": stream.get("codec_name"),
                    "start_time": _number(stream.get("start_time")),
                    "duration": _number(stream.get("duration")),
                    "width": stream.get("width"), "height": stream.get("height"),
                    "sample_rate": stream.get("sample_rate"), "channels": stream.get("channels"),
                    "language": stream.get("tags", {}).get("language")}
        return {"duration": duration,
                "video": [brief(s, i) for i, s in enumerate(streams) if s.get("codec_type") == "video"],
                "audio": [brief(s, i) for i, s in enumerate(streams) if s.get("codec_type") == "audio"]}
    except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError) as exc:
        return {"duration": None, "video": [], "audio": [], "error": str(exc)}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _decode_frame(source: str | Path, second: float, *, max_width: int | None = None,
                  max_height: int = 1080) -> np.ndarray | None:
    ffmpeg = _exe("ffmpeg")
    if not ffmpeg:
        return None
    args = [ffmpeg, "-v", "error", "-ss", f"{max(0.0, second):.3f}", "-i", str(source),
            "-map", "0:v:0", "-frames:v", "1"]
    if max_width:
        args += ["-vf", (f"scale='min({int(max_width)},iw)':'min({int(max_height)},ih)':"
                          "force_original_aspect_ratio=decrease")]
    args += ["-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
    try:
        completed = _run(args, timeout=20)
        if completed.returncode or not completed.stdout:
            return None
        return cv2.imdecode(np.frombuffer(completed.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, subprocess.TimeoutExpired, cv2.error):
        return None


def _iter_video_frames(source: str | Path, sample_seconds: float):
    """Yield nearest native-size frames on a media-zero sampling grid.

    PyAV is imported only when this worker is consumed, keeping the public
    module import light.  The container start time is subtracted uniformly
    from video PTS so video stays on the same timeline as the other streams.
    """
    metadata: dict[str, Any] = {
        "decoder": "pyav-native", "returncode": None, "coverage_seconds": 0.0,
        "decode_error": False, "decode_complete": False,
        "timing_source": "pts", "sampling_interval_seconds": sample_seconds,
        "sampling_gaps": 0, "resize_count": 0, "first_frame_size": None,
        "last_frame_size": None, "first_video_time": None,
    }
    container = None
    last_emitted_time: float | None = None
    last_decoded_time: float | None = None
    emitted = 0
    normal_eof = False
    try:
        try:
            import av
        except ImportError as exc:
            metadata.update({"decoder_unavailable": True, "decode_error": True,
                             "error": f"PyAV unavailable: {exc}"})
            yield None, metadata
            return

        container = av.open(str(source))
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            metadata.update({"decode_error": True, "error": "No video stream"})
            yield None, metadata
            return

        container_origin = (float(container.start_time) / av.time_base
                            if container.start_time is not None else 0.0)
        rate_value = _number(stream.average_rate)
        if rate_value is None or rate_value <= 0:
            rate_value = _number(stream.base_rate)
        if rate_value is not None and rate_value > 0:
            fallback_step = 1.0 / rate_value
            metadata["average_rate_fallback_fps"] = rate_value
        else:
            # PTS alone are still sufficient when present. This interval is
            # used only to estimate terminal coverage; missing PTS below are
            # rejected because inventing their time would be misleading.
            fallback_step = sample_seconds
        metadata["container_start_seconds"] = container_origin

        # Keep only neighboring decoded AV frames.  Pixel conversion and the
        # allocation of a numpy image happen only for selected samples.
        previous = None
        previous_time: float | None = None
        previous_end: float | None = None
        next_target = 0.0
        decoded_index = 0
        saw_missing_pts = False
        previous_size: tuple[int, int] | None = None

        def emit(frame, timestamp: float, target: float):
            nonlocal last_emitted_time, emitted
            if last_emitted_time is not None and timestamp <= last_emitted_time:
                return None
            if abs(timestamp - target) > sample_seconds:
                metadata["sampling_gaps"] += 1
                return None
            image = frame.to_ndarray(format="bgr24")
            last_emitted_time = timestamp
            emitted += 1
            return timestamp, image

        for frame in container.decode(stream):
            size = (int(frame.width), int(frame.height))
            if metadata["first_frame_size"] is None:
                metadata["first_frame_size"] = list(size)
            if previous_size is not None and size != previous_size:
                metadata["resize_count"] += 1
            previous_size = size
            metadata["last_frame_size"] = list(size)
            if frame.pts is not None:
                tb = frame.time_base or stream.time_base
                timestamp = float(frame.pts * tb) - container_origin
            else:
                if rate_value is None or rate_value <= 0:
                    metadata["timing_source"] = "unknown"
                    raise RuntimeError("Video PTS missing and average frame rate is unknown")
                saw_missing_pts = True
                stream_origin = (float(stream.start_time * stream.time_base) - container_origin
                                 if stream.start_time is not None and stream.time_base else 0.0)
                timestamp = stream_origin + decoded_index * fallback_step
            decoded_index += 1
            if not math.isfinite(timestamp) or (last_decoded_time is not None and timestamp <= last_decoded_time):
                continue
            if timestamp < 0:
                continue
            last_decoded_time = timestamp
            if metadata["first_video_time"] is None:
                metadata["first_video_time"] = timestamp
            frame_duration = _number(getattr(frame, "duration", None))
            frame_time_base = frame.time_base or stream.time_base
            duration_step = (frame_duration * float(frame_time_base)
                             if frame_duration is not None and frame_duration > 0 and frame_time_base else fallback_step)
            metadata["coverage_seconds"] = max(0.0, timestamp + duration_step)

            if previous is None:
                previous, previous_time = frame, timestamp
                previous_end = timestamp + duration_step
                while next_target <= timestamp:
                    if timestamp - next_target > sample_seconds:
                        # There was no video image yet at this grid point. Do
                        # not count an audio prelude as missing video coverage.
                        next_target += sample_seconds
                        continue
                    item = emit(frame, timestamp, next_target)
                    if item is not None:
                        yield item
                    next_target += sample_seconds
                continue

            while next_target <= timestamp:
                assert previous_time is not None
                candidate, candidate_time = (previous, previous_time)
                # Prefer the prior image while its declared hold interval is
                # active. If duration metadata is shorter than the PTS gap,
                # fall back to whichever decoded frame is nearer.
                held = previous_end is not None and previous_time <= next_target < previous_end
                if not held and (next_target >= timestamp
                                 or abs(timestamp - next_target) < abs(previous_time - next_target)):
                    candidate, candidate_time = frame, timestamp
                item = emit(candidate, candidate_time, next_target)
                if item is not None:
                    yield item
                next_target += sample_seconds
            previous, previous_time = frame, timestamp
            previous_end = timestamp + duration_step

        # The final decoded state may not coincide with a regular sample
        # point (for example a short source ending at 0.8 s with a 1 s step).
        # Keep it as a terminal control sample, without extrapolating it past
        # its own PTS. Existing unchanged-screen handling avoids duplicate events.
        if previous is not None and previous_time is not None:
            item = emit(previous, previous_time, previous_time)
            if item is not None:
                yield item

        if saw_missing_pts:
            metadata["timing_source"] = "pts_with_average_rate_fallback"
            metadata["timing_limitation"] = "Frames without PTS use stream start plus decode order divided by average frame rate."
        first_video_time = metadata["first_video_time"]
        if first_video_time is not None and first_video_time > 0:
            offset_note = (f"Video begins at {first_video_time:.3f}s after the container timeline origin; "
                           "the earlier interval contains no video frames.")
            metadata["timing_limitation"] = "; ".join(filter(None, (metadata.get("timing_limitation"), offset_note)))
        normal_eof = True
    except GeneratorExit:
        raise
    except Exception as exc:
        metadata.update({"decode_error": True, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if container is not None:
            container.close()
    metadata["decode_complete"] = bool(normal_eof and not metadata.get("decode_error")
                                       and not metadata.get("sampling_gaps"))
    metadata["sampled_frames"] = emitted
    yield None, metadata


class _PrefetchedFrames:
    """Read a frame iterator on one daemon thread with bounded backpressure."""

    _END = object()

    def __init__(self, frames, capacity: int = 2):
        self._frames = frames
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._closed = False
        self._ended = False
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._produce, name="screen-frame-prefetch", daemon=True)
        self._thread.start()

    def __iter__(self):
        return self

    def _put(self, message: tuple[str, Any]) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(message, timeout=.05)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        error: BaseException | None = None
        try:
            for item in self._frames:
                if self._stop.is_set() or not self._put(("item", item)):
                    break
        except BaseException as exc:
            error = exc
        finally:
            close = getattr(self._frames, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    if error is None:
                        error = exc
            if not self._stop.is_set():
                if error is None:
                    self._put(("end", self._END))
                else:
                    self._put(("error", error))

    def __next__(self):
        with self._state_lock:
            if self._closed or self._ended:
                raise StopIteration
        while True:
            if self._stop.is_set():
                with self._state_lock:
                    self._ended = True
                raise StopIteration
            try:
                kind, value = self._queue.get(timeout=.05)
            except queue.Empty:
                continue
            with self._state_lock:
                if self._closed or self._stop.is_set():
                    self._ended = True
                    raise StopIteration
                if kind == "item":
                    return value
                self._ended = True
                break
        self._thread.join(timeout=2)
        if kind == "error":
            raise value
        raise StopIteration

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._ended = True
            self._stop.set()
        self._discard_queued_frames()
        self._thread.join(timeout=2)
        self._discard_queued_frames()

    def _discard_queued_frames(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return


def _load_ocr(device: str = "cpu") -> Callable[[np.ndarray], Any]:
    device = str(device).casefold()
    if device not in {"cpu", "directml", "gpu"}:
        raise ValueError("device must be cpu or directml")
    use_dml = device in {"directml", "gpu"}
    rec_precision = os.environ.get("VT_MEETING_OCR_REC_PRECISION", "").strip().casefold()
    rec_model_path = os.environ.get("VT_MEETING_OCR_REC_MODEL_PATH", "").strip()
    rec_model_sha = os.environ.get("VT_MEETING_OCR_REC_MODEL_SHA256", "").strip()
    if rec_precision not in {"", "fp32", "fp16"}:
        raise ValueError(f"Unsupported OCR recognizer precision: {rec_precision}")
    if rec_precision == "fp16":
        if not rec_model_path or not rec_model_sha:
            raise RuntimeError("FP16 OCR recognizer requires a model path and SHA-256")
        from services.meeting_model_variants import validate_fp16_variant

        if not validate_fp16_variant(rec_model_path, rec_model_sha):
            raise RuntimeError("FP16 OCR recognizer variant is missing or failed receipt validation")
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
    _install_rapidocr_ort_provider_adapter()
    # Keep model files in this project's runtime and avoid unrestricted ORT
    # thread pools on workstations with many logical CPUs.
    model_root = Path(__file__).resolve().parents[1] / ".runtime" / "models" / "ocr"
    model_root.mkdir(parents=True, exist_ok=True)
    params = {
        "Rec.lang_type": LangRec.CYRILLIC,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Global.model_root_dir": str(model_root),
        "EngineConfig.onnxruntime.intra_op_num_threads": 4,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_dml": use_dml,
    }
    if rec_precision == "fp16":
        # RapidOCR routes this path only to Rec.model_path. Detection and
        # classification retain their configured FP32 models.
        params["Rec.model_path"] = rec_model_path
    if use_dml:
        try:
            device_id = max(0, int(os.environ.get("VT_MEETING_DML_DEVICE_ID", "0")))
        except ValueError:
            device_id = 0
        params["EngineConfig.onnxruntime.dml_ep_cfg"] = {"device_id": device_id}
    engine = RapidOCR(params=params)
    if os.environ.get("VT_MEETING_OCR_BATCHING", "").strip().casefold() == "equal-width":
        def recognize_equal_width(img_list):
            from rapidocr.ch_ppocr_rec.typings import TextRecInput
            from services.meeting_ocr_batches import (
                DEFAULT_EQUAL_WIDTH_MAX_BATCH, recognize_equal_width_batches,
            )

            rec_model = engine._load_rec_model()
            return recognize_equal_width_batches(
                rec_model,
                TextRecInput(img=img_list, return_word_box=engine.return_word_box),
                max_batch=DEFAULT_EQUAL_WIDTH_MAX_BATCH,
            )

        # This is an instance-local wrapper. No RapidOCR package file or class
        # is patched, and the stock path remains the default without the env.
        engine.recognize_txt = recognize_equal_width
    return engine


def _as_native_config(value: Any) -> Any:
    """Convert OmegaConf provider options into the native Python types ORT expects."""
    if isinstance(value, dict):
        return {str(key): _as_native_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_native_config(item) for item in value]
    # DictConfig/ListConfig implement container-like protocols but are not
    # consistently registered as Mapping/Sequence across OmegaConf releases.
    if hasattr(value, "items") and callable(value.items):
        try:
            return {str(key): _as_native_config(item) for key, item in value.items()}
        except (TypeError, ValueError):
            pass
    return value


def _provider_pairs(providers: Any) -> Any:
    if providers is None:
        return providers
    normalized = []
    for provider in providers:
        if isinstance(provider, tuple) and len(provider) == 2:
            name, options = provider
            normalized.append((str(name), _as_native_config(options)))
        else:
            normalized.append(provider)
    return normalized


def _requested_directml(providers: Any) -> bool:
    if providers is None:
        return False
    for provider in providers:
        name = provider[0] if isinstance(provider, tuple) and provider else provider
        if name == "DmlExecutionProvider":
            return True
    return False


def _disable_ort_fallback(session: Any) -> None:
    """Guard one requested DirectML session against ORT's whole-session CPU retry."""
    getter = getattr(session, "get_providers", None)
    if not callable(getter) or "DmlExecutionProvider" not in getter():
        raise RuntimeError("DirectML was requested but the ONNX Runtime session is not using DirectML")
    disable_fallback = getattr(session, "disable_fallback", None)
    if not callable(disable_fallback):
        raise RuntimeError("DirectML was requested but ONNX Runtime cannot disable CPU fallback")
    disable_fallback()


def _install_rapidocr_ort_provider_adapter() -> None:
    """Normalize RapidOCR 3.x provider configs for ONNX Runtime 1.24+.

    RapidOCR 3.10 gives ONNX Runtime OmegaConf ``DictConfig`` values inside
    provider tuples; ORT 1.24.4 requires ordinary ``dict`` instances. The
    adapter is process-local and wraps the imported factory, so package files
    and other stages remain untouched. It composes with a profiling factory
    installed by the calibration worker.
    """
    import rapidocr.inference_engine.onnxruntime.main as ort_main
    factory = ort_main.InferenceSession
    if getattr(factory, "_vt_provider_options_normalized", False):
        return
    if isinstance(factory, type):
        class NativeProviderInferenceSession(factory):
            _vt_provider_options_normalized = True

            def __init__(self, *args, **kwargs):
                positional = len(args) > 2 and "providers" not in kwargs
                providers = args[2] if positional else kwargs.get("providers")
                normalized = _provider_pairs(providers)
                if positional:
                    args = (*args[:2], normalized, *args[3:])
                elif "providers" in kwargs:
                    kwargs["providers"] = normalized
                super().__init__(*args, **kwargs)
                if _requested_directml(normalized):
                    _disable_ort_fallback(self)

        NativeProviderInferenceSession.__name__ = factory.__name__
        NativeProviderInferenceSession.__module__ = factory.__module__
        ort_main.InferenceSession = NativeProviderInferenceSession
    else:
        # Calibration may instrument the factory rather than the class. Keep
        # that factory in the call chain while normalizing its provider args.
        def normalized_factory(*args, **kwargs):
            positional = len(args) > 2 and "providers" not in kwargs
            providers = args[2] if positional else kwargs.get("providers")
            normalized = _provider_pairs(providers)
            if positional:
                args = (*args[:2], normalized, *args[3:])
            elif "providers" in kwargs:
                kwargs["providers"] = normalized
            session = factory(*args, **kwargs)
            if _requested_directml(normalized):
                _disable_ort_fallback(session)
            return session

        normalized_factory._vt_provider_options_normalized = True
        ort_main.InferenceSession = normalized_factory


def _assert_ocr_device(ocr: Any, device: str) -> None:
    """Reject any OCR part that silently fell back from the selected DirectML path."""
    if str(device).casefold() not in {"gpu", "directml"}:
        return
    session_count = 0
    for part_name in ("text_det", "text_cls", "text_rec"):
        part = getattr(ocr, part_name, None)
        infer = getattr(part, "session", None)
        ort_session = getattr(infer, "session", None)
        if ort_session is None:
            continue
        session_count += 1
        getter = getattr(ort_session, "get_providers", None)
        if not callable(getter) or "DmlExecutionProvider" not in getter():
            raise RuntimeError(
                f"DirectML was selected but RapidOCR {part_name} did not create a DirectML session")
    if session_count == 0:
        raise RuntimeError("DirectML was selected but RapidOCR did not create any OCR sessions")


def _parse_ocr(image: np.ndarray, ocr: Callable[[np.ndarray], Any]) -> list[dict[str, Any]]:
    raw = ocr(image)
    result = raw[0] if isinstance(raw, tuple) else raw
    if hasattr(result, "boxes") and hasattr(result, "txts"):
        boxes, texts, scores = result.boxes, result.txts, result.scores
        result = list(zip(boxes, texts, scores)) if boxes is not None and texts is not None and scores is not None else []
    regions: list[dict[str, Any]] = []
    for item in result or []:
        try:
            points, text, confidence = item[0], str(item[1]).strip(), float(item[2])
            if not text:
                continue
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            left, top = max(0, int(round(min(xs)))), max(0, int(round(min(ys))))
            right, bottom = int(round(max(xs))), int(round(max(ys)))
            regions.append({"text": text, "bbox": [left, top, max(1, right-left), max(1, bottom-top)],
                            "confidence": max(0.0, min(1.0, confidence))})
        except (TypeError, ValueError, IndexError):
            continue
    return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))


def _same_screen_regions(left: list[dict[str, Any]], right: list[dict[str, Any]],
                         width: int, height: int) -> bool:
    """Compare literal OCR text and geometry to annotate equivalent visual changes."""
    if not left or len(left) != len(right):
        return False
    tolerance_x, tolerance_y = max(3, round(width * .006)), max(3, round(height * .006))
    for first, second in zip(left, right):
        normalized_a = "".join(unicodedata.normalize("NFKC", first["text"]).casefold().split())
        normalized_b = "".join(unicodedata.normalize("NFKC", second["text"]).casefold().split())
        if normalized_a != normalized_b:
            return False
        if any(abs(a - b) > tolerance for a, b, tolerance in
               zip(first["bbox"], second["bbox"],
                   (tolerance_x, tolerance_y, tolerance_x, tolerance_y))):
            return False
    return True


def _regions_to_tables(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group contiguous aligned text rows, preserving candidate header references.

    This is a geometry-only heuristic. It can mistake aligned prose for a
    table, so callers disclose that uncertainty in ``limitations``.
    """
    if len(regions) < 5:
        return []
    ordered = sorted(regions, key=lambda r: (r["bbox"][1] + r["bbox"][3] / 2, r["bbox"][0]))
    rows: list[list[dict[str, Any]]] = []
    row_stats: list[dict[str, Any]] = []
    for region in ordered:
        x, y, w, h = region["bbox"]
        cy = y + h / 2
        match_index = next((i for i, stats in enumerate(row_stats)
                            if abs(cy - stats["center_sum"] / stats["count"])
                            <= max(8, min(h, stats["median_height"]) * .65)), None)
        if match_index is None:
            rows.append([region])
            row_stats.append({"center_sum": cy, "count": 1,
                              "heights": [h], "median_height": float(h)})
        else:
            rows[match_index].append(region)
            stats = row_stats[match_index]
            stats["center_sum"] += cy
            stats["count"] += 1
            insort(stats["heights"], h)
            heights = stats["heights"]
            middle = len(heights) // 2
            stats["median_height"] = (float(heights[middle]) if len(heights) % 2
                                      else (heights[middle - 1] + heights[middle]) / 2)
    rows = [sorted(row, key=lambda r: r["bbox"][0]) for row in rows]
    rows.sort(key=lambda row: min(r["bbox"][1] for r in row))
    def aligned(block: list[list[dict[str, Any]]]) -> bool:
        count = len(block[0])
        if count < 2 or any(len(row) != count for row in block):
            return False
        tolerance = max(22, int(np.median([r["bbox"][2] for row in block for r in row]) * .8))
        if not all(max(row[col]["bbox"][0] for row in block)
                   - min(row[col]["bbox"][0] for row in block) <= tolerance
                   for col in range(count)):
            return False
        if len(block) >= 3:
            centers = [float(np.mean([r["bbox"][1] + r["bbox"][3] / 2 for r in row])) for row in block]
            gaps = np.diff(centers)
            median_gap = float(np.median(gaps))
            if median_gap <= 0 or any(gap < median_gap * .4 or gap > median_gap * 2.5 for gap in gaps):
                return False
        return True

    # Keep maximal contiguous runs of matching, aligned row shapes. This lets
    # an isolated title/footer row remain outside an otherwise valid table.
    blocks: list[list[list[dict[str, Any]]]] = []
    current: list[list[dict[str, Any]]] = []
    for row in rows:
        if not current:
            current = [row]
        elif len(row) == len(current[0]) and aligned(current + [row]):
            current.append(row)
        else:
            if len(current) >= 3 and aligned(current):
                blocks.append(current)
            current = [row]
    if len(current) >= 3 and aligned(current):
        blocks.append(current)

    tables = []
    for block in blocks:
        headers = block[0]
        all_regions = [r for row in block for r in row]
        left = min(r["bbox"][0] for r in all_regions)
        top = min(r["bbox"][1] for r in all_regions)
        right = max(r["bbox"][0] + r["bbox"][2] for r in all_regions)
        bottom = max(r["bbox"][1] + r["bbox"][3] for r in all_regions)
        header_items = [{"text": r["text"], "bbox": list(r["bbox"]),
                         "confidence": r.get("confidence")} for r in headers]
        cells = []
        for row in block[1:]:
            cells.append([{"text": r["text"], "bbox": list(r["bbox"]),
                           "confidence": r.get("confidence"),
                           "header": {"index": col, "text": headers[col]["text"]}}
                          for col, r in enumerate(row)])
        tables.append({"bbox": [left, top, right-left, bottom-top],
                       "headers": header_items, "cells": cells,
                       "header_is_candidate": True, "uncertain": True})

    # A header plus one body row is useful evidence when it has at least
    # three independently aligned columns. Keep this path deliberately
    # stricter than the multi-row heuristic: two-column SQL/property layouts
    # still need three body rows before being called a table.
    def row_parts(row: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Split a same-y OCR line at large horizontal gaps (neighboring tables)."""
        parts: list[list[dict[str, Any]]] = []
        part: list[dict[str, Any]] = []
        prior_right = None
        for item in row:
            x, _y, w, _h = item["bbox"]
            if part:
                widths = [entry["bbox"][2] for entry in part]
                # A generous floor handles wide labels; the cap prevents one
                # unusually long OCR box from joining distinct table regions.
                gap_limit = max(220, min(360, float(np.median(widths)) * 2.5))
                if x - prior_right > gap_limit:
                    parts.append(part)
                    part = []
            part.append(item)
            prior_right = x + w
        if part:
            parts.append(part)
        return parts

    def centers_and_bounds(headers: list[dict[str, Any]]):
        centers = [r["bbox"][0] + r["bbox"][2] / 2 for r in headers]
        bounds = [-math.inf]
        bounds.extend((a + b) / 2 for a, b in zip(centers, centers[1:]))
        bounds.append(math.inf)
        return centers, bounds

    def map_row(headers: list[dict[str, Any]], values: list[dict[str, Any]]):
        """Return unique header/value matches; boxes spanning columns are omitted."""
        centers, bounds = centers_and_bounds(headers)
        mapped = []
        ambiguous = []
        used = set()
        for value in values:
            x, _y, w, _h = value["bbox"]
            center = x + w / 2
            col = next((i for i in range(len(headers))
                        if bounds[i] <= center < bounds[i + 1]), None)
            if col is None or col in used:
                continue
            # A merged OCR box that crosses a neighboring column boundary is
            # not an exact cell. Preserve it as ambiguous, without a header.
            if x < bounds[col] or x + w > bounds[col + 1]:
                ambiguous.append(value)
                continue
            # Reject distant matches, including text from a neighboring table.
            local_width = headers[col]["bbox"][2]
            if abs(center - centers[col]) > max(65, local_width * 1.5):
                continue
            mapped.append((col, value))
            used.add(col)
        return mapped, ambiguous

    partial_keys = set()
    # Use a tighter vertical grouping for this path. In dense forms, the last
    # line of a neighboring table can sit beside the next table's header and
    # fall into the legacy row tolerance; separating by baseline keeps the
    # two geometries independent.
    partial_rows: list[list[dict[str, Any]]] = []
    partial_stats: list[dict[str, Any]] = []
    for region in ordered:
        _x, y, _w, h = region["bbox"]
        cy = y + h / 2
        match = next((i for i, stats in enumerate(partial_stats)
                      if abs(cy - stats["center_sum"] / stats["count"])
                      <= max(5, min(h, stats["median_height"]) * .45)), None)
        if match is None:
            partial_rows.append([region])
            partial_stats.append({"center_sum": cy, "count": 1,
                                  "heights": [h], "median_height": float(h)})
        else:
            row = partial_rows[match]
            row.append(region)
            stats = partial_stats[match]
            stats["center_sum"] += cy
            stats["count"] += 1
            insort(stats["heights"], h)
            heights = stats["heights"]
            middle = len(heights) // 2
            stats["median_height"] = (float(heights[middle]) if len(heights) % 2
                                      else (heights[middle - 1] + heights[middle]) / 2)
    partial_rows = [sorted(row, key=lambda r: r["bbox"][0]) for row in partial_rows]
    partial_rows.sort(key=lambda row: min(r["bbox"][1] for r in row))

    for row_index in range(len(partial_rows) - 1):
        upper_parts = row_parts(partial_rows[row_index])
        for headers in upper_parts:
            if len(headers) < 3:
                continue
            # Partial candidates have only one body row to support them, so
            # require at least three textual (non-numeric) header anchors.
            # This avoids mistaking a neighboring data row with three values
            # for a second table header.
            textual_headers = sum(
                any(char.isalpha() for char in item["text"])
                and not any(char.isdigit() for char in item["text"])
                for item in headers)
            if textual_headers < 3:
                continue
            hy = float(np.mean([r["bbox"][1] + r["bbox"][3] / 2 for r in headers]))
            hh = float(np.median([r["bbox"][3] for r in headers]))
            # Inspect only the next few geometric baselines. This tolerates a
            # neighboring table's intervening row while keeping inference
            # bounded on dense OCR frames.
            for value_index in range(row_index + 1, min(len(partial_rows), row_index + 4)):
                lower_parts = row_parts(partial_rows[value_index])
                if min(r["bbox"][1] for r in partial_rows[value_index]) - hy > 45:
                    break
                for values in lower_parts:
                    vy = float(np.mean([r["bbox"][1] + r["bbox"][3] / 2 for r in values]))
                    vh = float(np.median([r["bbox"][3] for r in values]))
                    max_center_gap = min(45, float(np.median([hh, vh])) * 2.5)
                    if vy <= hy or vy - hy > max_center_gap:
                        continue
                    mapped, ambiguous = map_row(headers, values)
                    if len(mapped) < 3:
                        continue
                    key = (tuple(id(r) for r in headers), tuple(id(r) for r in values))
                    if key in partial_keys:
                        continue
                    # Existing three-row tables already carry this evidence.
                    if any(block[0] == headers and block[1] == values for block in blocks):
                        continue
                    # Avoid reducing a longer, valid table to its first two rows.
                    continuation_found = False
                    for next_index in range(value_index + 1,
                                            min(len(partial_rows), value_index + 4)):
                        if min(r["bbox"][1] for r in partial_rows[next_index]) - vy > 45:
                            break
                        if any(len(map_row(headers, candidate)[0]) >= 3
                               for candidate in row_parts(partial_rows[next_index])):
                            continuation_found = True
                            break
                    if continuation_found:
                        continue

                    header_items = [{"text": r["text"], "bbox": list(r["bbox"]),
                                     "confidence": r.get("confidence")} for r in headers]
                    row_cells = [{"text": value["text"], "bbox": list(value["bbox"]),
                                  "confidence": value.get("confidence"),
                                  "header": {"index": col, "text": headers[col]["text"]}}
                                 for col, value in mapped]
                    row_cells.extend({"text": value["text"], "bbox": list(value["bbox"]),
                                      "confidence": value.get("confidence"),
                                      "ambiguous": True} for value in ambiguous)
                    row_cells.sort(key=lambda cell: cell["bbox"][0])
                    cells = [row_cells]
                    selected_regions = list(headers) + [value for _, value in mapped] + ambiguous
                    left = min(r["bbox"][0] for r in selected_regions)
                    top = min(r["bbox"][1] for r in selected_regions)
                    right = max(r["bbox"][0] + r["bbox"][2] for r in selected_regions)
                    bottom = max(r["bbox"][1] + r["bbox"][3] for r in selected_regions)
                    tables.append({"bbox": [left, top, right-left, bottom-top],
                                   "headers": header_items, "cells": cells,
                                   "header_is_candidate": True, "uncertain": True})
                    partial_keys.add(key)
    return tables


def _changed_tile_mask(previous: np.ndarray, current: np.ndarray, threshold: float = 0.03):
    """Return changed 16x16-native-pixel tiles and exact source-coordinate edges."""
    if previous.shape != current.shape:
        return None
    a = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    height, width = a.shape
    rows, cols = (height + 15) // 16, (width + 15) // 16
    padded = np.pad(cv2.absdiff(a, b) > 20,
                    ((0, rows * 16 - height), (0, cols * 16 - width)),
                    mode="constant", constant_values=False)
    counts = padded.reshape(rows, 16, cols, 16).sum(axis=(1, 3))
    tile_heights = np.minimum(16, height - np.arange(rows) * 16)
    tile_widths = np.minimum(16, width - np.arange(cols) * 16)
    tile_areas = tile_heights[:, None] * tile_widths[None, :]
    changed = (counts >= 12) & (counts >= tile_areas * float(threshold))
    x_starts = np.arange(cols, dtype=np.int32) * 16
    x_ends = np.minimum(x_starts + 16, width)
    y_starts = np.arange(rows, dtype=np.int32) * 16
    y_ends = np.minimum(y_starts + 16, height)
    return changed.astype(np.uint8), x_starts, x_ends, y_starts, y_ends


def _tile_changed(previous: np.ndarray, current: np.ndarray, threshold: float = 0.03) -> bool:
    """Detect a local screen edit with a shared native-resolution 16px grid."""
    tile_data = _changed_tile_mask(previous, current, threshold)
    return True if tile_data is None else bool(np.any(tile_data[0]))


def _changed_rois(previous: np.ndarray, current: np.ndarray, *, padding: int = 48) -> list[list[int]]:
    """Return merged padded rectangles for changed tiles, in source coordinates."""
    if previous.shape != current.shape:
        return [[0, 0, int(current.shape[1]), int(current.shape[0])]]
    tile_data = _changed_tile_mask(previous, current)
    assert tile_data is not None
    mask, x_starts, x_ends, y_starts, y_ends = tile_data
    height, width = current.shape[:2]
    if not np.any(mask):
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    rois: list[list[int]] = []
    for label in range(1, count):
        x, y, tile_w, tile_h, _area = stats[label]
        x0 = max(0, int(x_starts[x]) - padding)
        y0 = max(0, int(y_starts[y]) - padding)
        x1 = min(width, int(x_ends[x + tile_w - 1]) + padding)
        y1 = min(height, int(y_ends[y + tile_h - 1]) + padding)
        rois.append([x0, y0, x1 - x0, y1 - y0])
    # Components can become overlapping after padding; combine them to avoid
    # duplicate recognition and accidental duplicate region coordinates.
    merged: list[list[int]] = []
    for roi in sorted(rois, key=lambda r: (r[1], r[0])):
        x, y, w, h = roi
        right, bottom = x + w, y + h
        for old in merged:
            ox, oy, ow, oh = old
            if x <= ox + ow and ox <= right and y <= oy + oh and oy <= bottom:
                nx, ny = min(x, ox), min(y, oy)
                nr, nb = max(right, ox + ow), max(bottom, oy + oh)
                old[:] = [nx, ny, nr - nx, nb - ny]
                break
        else:
            merged.append(roi)
    return merged


def _intersects(region: dict[str, Any], roi: list[int]) -> bool:
    x, y, w, h = region.get("bbox", [0, 0, 0, 0])
    rx, ry, rw, rh = roi
    return x < rx + rw and rx < x + w and y < ry + rh and ry < y + h


def _merge_rectangles(rois: list[list[int]], width: int, height: int) -> list[list[int]]:
    """Clamp and transitively merge intersecting rectangles."""
    pending = []
    for x, y, w, h in rois:
        left, top = max(0, int(x)), max(0, int(y))
        right, bottom = min(width, int(x + w)), min(height, int(y + h))
        if right > left and bottom > top:
            pending.append([left, top, right-left, bottom-top])
    merged: list[list[int]] = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            x, y, w, h = current
            for index, other in enumerate(pending):
                ox, oy, ow, oh = other
                if x <= ox + ow and ox <= x + w and y <= oy + oh and oy <= y + h:
                    nx, ny = min(x, ox), min(y, oy)
                    nr, nb = max(x + w, ox + ow), max(y + h, oy + oh)
                    current = [nx, ny, nr-nx, nb-ny]
                    pending.pop(index)
                    changed = True
                    break
        merged.append(current)
    return sorted(merged, key=lambda r: (r[1], r[0]))


def _expand_rois_to_regions(rois: list[list[int]], previous_regions: list[dict[str, Any]],
                            width: int, height: int, *, padding: int = 24) -> list[list[int]]:
    """Pad changed areas once, then close over full OCR boxes without growing padding."""
    initial = _merge_rectangles(rois, width, height)
    region_boxes = [r.get("bbox", [0, 0, 0, 0]) for r in previous_regions]
    # Padding is deliberately applied only to the changed ROI and boxes it
    # directly intersects. Re-padding newly reached boxes causes dense text
    # rows to expand by 2*padding per iteration until the whole page is OCR'd.
    seeds: list[list[int]] = []
    for roi in initial:
        hits = [box for box in region_boxes if len(box) == 4 and _intersects({"bbox": box}, roi)]
        left = min([roi[0], *(int(box[0]) for box in hits)]) - padding
        top = min([roi[1], *(int(box[1]) for box in hits)]) - padding
        right = max([roi[0] + roi[2], *(int(box[0] + box[2]) for box in hits)]) + padding
        bottom = max([roi[1] + roi[3], *(int(box[1] + box[3]) for box in hits)]) + padding
        seeds.append([left, top, right - left, bottom - top])
    expanded = _merge_rectangles(seeds, width, height)

    # Close over any text box reached by the one-time padded seed. Newly
    # included boxes are added at their exact bounds, never with extra padding.
    while True:
        next_rois = list(expanded)
        for roi in expanded:
            hits = [box for box in region_boxes if len(box) == 4 and _intersects({"bbox": box}, roi)]
            if not hits:
                continue
            left = min([roi[0], *(int(box[0]) for box in hits)])
            top = min([roi[1], *(int(box[1]) for box in hits)])
            right = max([roi[0] + roi[2], *(int(box[0] + box[2]) for box in hits)])
            bottom = max([roi[1] + roi[3], *(int(box[1] + box[3]) for box in hits)])
            next_rois.append([left, top, right-left, bottom-top])
        combined = _merge_rectangles(next_rois, width, height)
        if combined == expanded:
            return combined
        expanded = combined


def _visually_blank_text_crop(crop: np.ndarray) -> bool:
    if crop.size == 0:
        return True
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    low, high = np.percentile(gray, (5, 95))
    return float(np.std(gray)) < 2.5 and float(high - low) < 12


def _try_anchored_regions(previous_image: np.ndarray, current_image: np.ndarray,
                          previous_regions: list[dict[str, Any]], ocr: Any,
                          recognized_at: float) -> tuple[list[dict[str, Any]], set[int]] | None:
    """Re-recognize changed pixels within confident prior text boxes, without detection.

    Images and region boxes are ROI-local. A None return means the conservative
    geometric/content guard failed and the caller should use detector OCR.
    Engine exceptions intentionally propagate to the existing stage handler.
    """
    recognize = getattr(ocr, "recognize_txt", None)
    if not callable(recognize) or previous_image.shape != current_image.shape:
        return None
    height, width = current_image.shape[:2]
    tile_data = _changed_tile_mask(previous_image, current_image)
    if tile_data is None:
        return None
    tile_mask = tile_data[0].astype(bool)
    if not np.any(tile_mask):
        return None
    diff_pixels = cv2.absdiff(
        cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY),
    ) > 20
    tile_pixels = np.repeat(np.repeat(tile_mask, 16, axis=0), 16, axis=1)[:height, :width]
    changed_pixels = diff_pixels & tile_pixels
    if not np.any(changed_pixels):
        return None

    coverage = np.zeros((height, width), dtype=np.uint8)
    touched: list[tuple[int, dict[str, Any], tuple[int, int, int, int]]] = []
    for index, region in enumerate(previous_regions):
        try:
            x, y, w, h = [int(round(float(v))) for v in region.get("bbox", [])]
            confidence = float(region.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if (confidence < .8 or w < h or h <= 0 or h > 64 or w <= 0):
            continue
        x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        box = (x0, y0, x1, y1)
        coverage[y0:y1, x0:x1] = 1
        if np.any(changed_pixels[y0:y1, x0:x1]):
            touched.append((index, region, box))
    if not touched or np.any(changed_pixels & (coverage == 0)):
        return None

    crops: list[np.ndarray] = []
    for _index, _region, (x0, y0, x1, y1) in touched:
        crop_x0, crop_y0 = max(0, x0 - 2), max(0, y0 - 2)
        crop_x1, crop_y1 = min(width, x1 + 2), min(height, y1 + 2)
        crops.append(current_image[crop_y0:crop_y1, crop_x0:crop_x1])
    result = recognize(crops)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if texts is None or scores is None:
        if all(_visually_blank_text_crop(crop) for crop in crops):
            return [], {index for index, _region, _box in touched}
        return None
    try:
        if len(texts) != len(touched) or len(scores) != len(touched):
            if len(texts) == 0 and len(scores) == 0 and all(
                    _visually_blank_text_crop(crop) for crop in crops):
                return [], {index for index, _region, _box in touched}
            return None
    except TypeError:
        return None

    updated: list[dict[str, Any]] = []
    touched_indices: set[int] = set()
    for crop_index, ((index, old_region, box), text_value, score_value) in enumerate(zip(touched, texts, scores)):
        text = str(text_value or "").strip()
        if _visually_blank_text_crop(crops[crop_index]):
            touched_indices.add(index)
            continue
        if text:
            try:
                confidence = float(score_value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(confidence) or confidence < .8:
                return None
        else:
            return None
        region = dict(old_region)
        region.update({"text": text, "bbox": list(old_region["bbox"]),
                       "confidence": max(0.0, min(1.0, confidence)),
                       "recognized_at": recognized_at,
                       "recognition_mode": "anchored_recognition",
                       "geometry_inherited": True,
                       "uncertainty": "text re-recognized within prior OCR geometry; bbox is a candidate"})
        updated.append(region)
        touched_indices.add(index)
    return updated, touched_indices


def _incremental_regions(image: np.ndarray, previous_regions: list[dict[str, Any]], rois: list[list[int]],
                         ocr: Callable[[np.ndarray], Any], recognized_at: float,
                         previous_image: np.ndarray | None = None,
                         stats: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    replaced_indices: set[int] = set()
    changed_regions: list[dict[str, Any]] = []
    anchored_region_count = 0
    for x, y, w, h in rois:
        region_indices = {i for i, region in enumerate(previous_regions)
                          if _intersects(region, [x, y, w, h])}
        anchored = None
        if previous_image is not None:
            # Keep local tile origins aligned to the detector's native grid so
            # a changed 4x4 glyph at a tile boundary is not split into four
            # sub-threshold tiles by the ROI crop itself.
            local_x, local_y = (x // 16) * 16, (y // 16) * 16
            local_right, local_bottom = min(image.shape[1], x + w), min(image.shape[0], y + h)
            local_w, local_h = local_right - local_x, local_bottom - local_y
            local_roi = [local_x, local_y, local_w, local_h]
            local_region_indices = {i for i, region in enumerate(previous_regions)
                                    if _intersects(region, local_roi)}
            region_indices |= local_region_indices
            local_previous = previous_image[local_y:local_bottom, local_x:local_right]
            local_current = image[local_y:local_bottom, local_x:local_right]
            local_regions = []
            local_to_global: list[int] = []
            for index in sorted(local_region_indices):
                region = dict(previous_regions[index])
                bx, by, bw, bh = region["bbox"]
                region["bbox"] = [bx - local_x, by - local_y, bw, bh]
                local_regions.append(region)
                local_to_global.append(index)
            if local_regions:
                anchored = _try_anchored_regions(local_previous, local_current, local_regions,
                                                 ocr, recognized_at)
            if anchored is not None:
                anchored_local, touched_local = anchored
                replaced_indices.update(local_to_global[i] for i in touched_local)
                anchored_region_count += len(touched_local)
                for region in anchored_local:
                    bx, by, bw, bh = region["bbox"]
                    region["bbox"] = [bx + local_x, by + local_y, bw, bh]
                    changed_regions.append(region)
                continue

        # Detector OCR is the fallback for this ROI only. All old regions that
        # intersect it are replaced as in the original incremental path.
        replaced_indices.update(region_indices)
        parsed = _parse_ocr(image[y:y+h, x:x+w], ocr)
        for region in parsed:
            bx, by, bw, bh = region["bbox"]
            region["bbox"] = [bx + x, by + y, bw, bh]
            region["recognized_at"] = recognized_at
            region["uncertainty"] = "localized OCR; neighboring context may be outside the padded crop"
            changed_regions.append(region)

    kept = []
    for index, region in enumerate(previous_regions):
        if index in replaced_indices:
            continue
        inherited = dict(region)
        inherited["recognized_at"] = inherited.get("recognized_at", recognized_at)
        inherited["uncertainty"] = "inherited from prior OCR; area was outside changed tiles"
        kept.append(inherited)
    if stats is not None and anchored_region_count:
        stats["anchored_rec_frames"] = stats.get("anchored_rec_frames", 0) + 1
        stats["anchored_rec_regions"] = stats.get("anchored_rec_regions", 0) + anchored_region_count
    return sorted(kept + changed_regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))


def extract_screen(source: str | Path, output_dir: str | Path, *, sample_seconds: float = 1.0,
                   checkpoint_seconds: float = 30.0, progress: Progress | None = None,
                   cancel: Callable[[], bool] | None = None, device: str = "cpu",
                   prefetch_frames: int = 0) -> dict[str, Any]:
    """Sample a video in one decode pass, OCR selected states, and save event JPEGs."""
    prefetch_frames = int(prefetch_frames)
    if prefetch_frames not in (0, 2):
        raise ValueError("prefetch_frames must be 0 or 2")
    started_at = time.perf_counter()
    out = Path(output_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sample_seconds = max(.1, float(sample_seconds))
    checkpoint_seconds = max(sample_seconds, float(checkpoint_seconds))
    media = probe_media(source)
    limitations: list[str] = []
    stats: dict[str, Any] = {"sampled_frames": 0, "ocr_frames": 0, "duration": media.get("duration"),
                             "decode_seconds": 0.0, "ocr_seconds": 0.0, "decoder": "pyav-native",
                             "duplicate_candidates": 0, "full_ocr_frames": 0,
                             "roi_ocr_frames": 0, "roi_count": 0, "skipped_ocr_candidates": 0,
                             "ocr_equivalent_events": 0, "anchored_rec_frames": 0,
                             "anchored_rec_regions": 0}
    if not media.get("video") or not _exe("ffprobe"):
        return {"events": [], "status": "unavailable", "limitations": ["ffprobe or a video stream is unavailable."], "stats": stats}
    duration = media.get("duration") or 0.0
    events: list[dict[str, Any]] = []
    reference: np.ndarray | None = None
    ocr: Callable[[np.ndarray], Any] | None = None
    ocr_unavailable = False
    last_full_ocr_time: float | None = None
    ocr_failed = False
    cancelled = False
    decoder_started = time.perf_counter()
    frames = _iter_video_frames(source, sample_seconds)
    if prefetch_frames == 2:
        frames = _PrefetchedFrames(frames, capacity=2)
    decode_info: dict[str, Any] = {}
    last_sample_time: float | None = None
    try:
        for second, image in frames:
            if second is None and isinstance(image, dict):
                decode_info = image
                break
            if cancel and cancel():
                cancelled = True
                break
            if second >= duration and duration > 0:
                break
            stats["sampled_frames"] += 1
            last_sample_time = second
            reason = "initial" if not events else None
            changed_rois: list[list[int]] = []
            if (reason is None and last_full_ocr_time is not None
                    and second - last_full_ocr_time >= checkpoint_seconds):
                reason = "checkpoint"
            elif reason is None and reference is not None:
                changed_rois = _changed_rois(reference, image)
                if changed_rois:
                    reason = "region_change"
            if reason:
                regions = []
                ocr_equivalent_to_previous = False
                full_ocr_completed = False
                if not ocr_unavailable:
                    try:
                        ocr_started = time.perf_counter()
                        if ocr is None:
                            ocr = _load_ocr(device)
                        if reason == "region_change" and events:
                            changed_rois = _expand_rois_to_regions(
                                changed_rois, events[-1]["regions"], image.shape[1], image.shape[0])
                            total_area = sum(w * h for _x, _y, w, h in changed_rois)
                            area_fraction = total_area / max(1, image.shape[0] * image.shape[1])
                            # Scrolling and broad redraws need a fresh page OCR;
                            # compact edits are OCRed with their neighboring context.
                            if area_fraction > .42 or len(changed_rois) > 8:
                                regions = _parse_ocr(image, ocr)
                                full_ocr_completed = True
                                stats["full_ocr_frames"] += 1
                                regions = [{**r, "recognized_at": second,
                                            "uncertainty": "full-frame OCR"} for r in regions]
                            else:
                                regions = _incremental_regions(image, events[-1]["regions"], changed_rois,
                                                               ocr, second, previous_image=reference,
                                                               stats=stats)
                                stats["roi_ocr_frames"] += 1
                                stats["roi_count"] += len(changed_rois)
                        else:
                            regions = _parse_ocr(image, ocr)
                            full_ocr_completed = True
                            stats["full_ocr_frames"] += 1
                            regions = [{**r, "recognized_at": second,
                                        "uncertainty": "full-frame OCR"} for r in regions]
                        _assert_ocr_device(ocr, device)
                        stats["ocr_frames"] += 1
                        if full_ocr_completed:
                            last_full_ocr_time = second
                    except Exception as exc:
                        if str(device).casefold() in {"gpu", "directml"}:
                            close = getattr(frames, "close", None)
                            if close:
                                close()
                            raise RuntimeError(
                                f"Не удалось обработать экран на видеокарте (GPU, DirectML), момент {second:.3f} с."
                            ) from exc
                        ocr_failed = True
                        ocr_unavailable = ocr is None
                        limitations.append(f"OCR failed on a selected frame ({type(exc).__name__}).")
                    finally:
                        if "ocr_started" in locals():
                            stats["ocr_seconds"] += time.perf_counter() - ocr_started
                            del ocr_started
                if (reason == "region_change" and events and
                        _same_screen_regions(regions, events[-1]["regions"], image.shape[1], image.shape[0])):
                    stats["duplicate_candidates"] += 1
                    stats["ocr_equivalent_events"] = stats.get("ocr_equivalent_events", 0) + 1
                    ocr_equivalent_to_previous = True
                    if "A visual frame change produced OCR-equivalent text; the image is retained as evidence." not in limitations:
                        limitations.append("A visual frame change produced OCR-equivalent text; the image is retained as evidence.")
                    # Retain the visual evidence, and compare the next sample to
                    # it so an identical following frame does not create another event.
                    reference = image
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok:
                    name = f"screen-{len(events)+1:04d}-{uuid.uuid4().hex[:8]}.jpg"
                    (frames_dir / name).write_bytes(encoded.tobytes())
                    event = {"id": f"screen-{len(events)+1:04d}", "start": round(second, 3),
                             "end": None, "frame": f"frames/{name}",
                             "text": "\n".join(r["text"] for r in regions), "regions": regions,
                             "tables": _regions_to_tables(regions), "reason": reason,
                             "width": int(image.shape[1]), "height": int(image.shape[0]),
                             "sample_interval_seconds": sample_seconds}
                    if ocr_equivalent_to_previous:
                        event["ocr_equivalent_to_previous"] = True
                    events.append(event)
                    reference = image
            if progress:
                progress({"stage": "screen", "time": round(second, 3), "duration": duration,
                          "sampled_frames": stats["sampled_frames"], "events": len(events)})
        if cancelled:
            close = getattr(frames, "close", None)
            if close:
                close()
    finally:
        if prefetch_frames == 2:
            frames.close()
    stats["decode_seconds"] = round(max(0.0, time.perf_counter() - decoder_started - stats["ocr_seconds"]), 3)
    sampled_coverage = max(0.0, (last_sample_time + sample_seconds) if last_sample_time is not None else 0.0)
    decoded_coverage = _number(decode_info.get("coverage_seconds"))
    coverage = decoded_coverage if decoded_coverage is not None else sampled_coverage
    coverage = min(duration, max(0.0, coverage)) if duration else max(0.0, coverage)
    decode_returncode = decode_info.get("returncode", 0 if not cancelled else None)
    decode_complete = (not cancelled and not decode_info.get("decode_error", False)
                       and decode_info.get("decode_complete", True) is not False
                       and decode_returncode in (0, None)
                       and (duration <= 0 or coverage >= duration - max(sample_seconds * 1.5, .25)))
    stats.update({"coverage_seconds": round(coverage, 3), "decode_returncode": decode_returncode,
                  "decode_complete": bool(decode_complete)})
    for key in ("decoder", "timing_source", "average_rate_fallback_fps", "resize_count",
                "first_frame_size", "last_frame_size", "first_video_time", "sampling_gaps"):
        if key in decode_info:
            stats[key] = decode_info[key]
    for i, event in enumerate(events):
        event["end"] = round(events[i+1]["start"] if i + 1 < len(events) else coverage, 3)
        event["time_uncertainty_seconds"] = sample_seconds
    if any(event["tables"] for event in events):
        limitations.append("Tables, row links, and header candidates are inferred from geometry; overlays can make these associations ambiguous.")
    limitations.append(f"Frame timestamps are sampling points with about {sample_seconds:g} s uncertainty; visibility boundaries are approximate.")
    if decode_info.get("timing_limitation"):
        limitations.append(str(decode_info["timing_limitation"]))
    if decode_info.get("sampling_gaps"):
        limitations.append(f"{decode_info['sampling_gaps']} sampling points had no decoded frame within the requested time tolerance.")
    if decode_info.get("decoder_unavailable"):
        limitations.append("PyAV is unavailable; native-size screen decoding could not be performed.")
    elif decode_info.get("error"):
        limitations.append(f"Video decoding stopped with an error ({decode_info['error']}).")
    if cancelled:
        limitations.append("Extraction was cancelled before the source ended.")
    if not decode_complete:
        limitations.append(f"Decoded coverage ended at {coverage:.2f}s of {duration:.2f}s; the screen extraction is incomplete.")
    if not events and not decode_info.get("decode_error") and not decode_info.get("decoder_unavailable"):
        status = "unavailable"
    elif cancelled or not decode_complete or ocr_failed or stats["ocr_frames"] == 0:
        status = "partial"
    else:
        status = "ready"
    stats["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
    return {"events": events, "status": status, "limitations": limitations, "stats": stats}


def _safe_event_frame(meeting_dir: str | Path, event: dict[str, Any]) -> Path:
    root = Path(meeting_dir).resolve()
    relative = Path(str(event.get("frame", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("frame path must stay inside the meeting directory")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("frame path must stay inside the meeting directory")
    return path


def frame_image(meeting_dir: str | Path, event: dict[str, Any],
                crop: list[int] | tuple[int, int, int, int] | None = None,
                max_width: int = 1920) -> bytes:
    """Read event JPEG bytes, optionally cropping/resizing without path traversal."""
    path = _safe_event_frame(meeting_dir, event)
    with Image.open(path) as source:
        image = source.convert("RGB")
    if crop is not None:
        if len(crop) != 4:
            raise ValueError("crop must be [x, y, width, height]")
        x, y, width, height = [int(v) for v in crop]
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image.width or y + height > image.height:
            raise ValueError("crop is outside the frame")
        image = image.crop((x, y, x + width, y + height))
    if max_width > 0 and image.width > max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def extract_episode_frames(source: str | Path, start: float, end: float, *,
                           step: float = 1.0, limit: int = 8) -> list[dict[str, Any]]:
    """Read a bounded set of JPEG frames from a time range without writing files."""
    start, end = max(0.0, float(start)), max(0.0, float(end))
    step, limit = max(.1, float(step)), max(0, min(8, int(limit)))
    if end <= start or limit == 0:
        return []
    results: list[dict[str, Any]] = []
    second = start
    while second < end and len(results) < limit:
        image = _decode_frame(source, second, max_width=1920)
        if image is None:
            if not results:
                break
            break
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            results.append({"time": round(second, 3), "image": encoded.tobytes(),
                            "width": int(image.shape[1]), "height": int(image.shape[0])})
        second += step
    return results


def compress_audio(source: str | Path, output_dir: str | Path, *,
                   progress: Progress | None = None) -> list[dict[str, Any]]:
    """Encode each audio stream independently to Opus while preserving timeline offsets."""
    media = probe_media(source)
    ffmpeg = _exe("ffmpeg")
    if not ffmpeg or media.get("error"):
        return []
    target = Path(output_dir) / "audio"
    target.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for ordinal, stream in enumerate(media.get("audio", [])):
        index = int(stream["index"])
        name = f"audio-{ordinal+1:02d}.opus"
        path = target / name
        args = [ffmpeg, "-v", "error", "-y", "-copyts", "-start_at_zero", "-i", str(source),
                "-map", f"0:{index}", "-vn", "-c:a", "libopus", "-b:a", "48k",
                "-avoid_negative_ts", "disabled", str(path)]
        result = _run(args, timeout=3600)
        if result.returncode:
            if path.exists():
                path.unlink()
            continue
        outputs.append({"stream_index": index, "path": f"audio/{name}",
                        "start_time": stream.get("start_time"), "duration": stream.get("duration")})
        if progress:
            progress({"stage": "audio_compress", "completed": len(outputs),
                      "total": len(media.get("audio", [])), "stream_index": index})
    return outputs

