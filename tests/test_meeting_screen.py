import tempfile
import subprocess
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction
import sys
import types
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from services import meeting_screen as screen
from services import meeting_hardware as hardware


def _fake_ort_module_tree(factory):
    rapidocr = types.ModuleType("rapidocr")
    rapidocr.__path__ = []
    inference = types.ModuleType("rapidocr.inference_engine")
    inference.__path__ = []
    onnxruntime = types.ModuleType("rapidocr.inference_engine.onnxruntime")
    onnxruntime.__path__ = []
    main = types.ModuleType("rapidocr.inference_engine.onnxruntime.main")
    main.InferenceSession = factory
    rapidocr.inference_engine = inference
    inference.onnxruntime = onnxruntime
    onnxruntime.main = main
    return {
        "rapidocr": rapidocr,
        "rapidocr.inference_engine": inference,
        "rapidocr.inference_engine.onnxruntime": onnxruntime,
        "rapidocr.inference_engine.onnxruntime.main": main,
    }, main


class MeetingScreenTests(unittest.TestCase):
    def test_production_frame_extraction_saves_visual_states_without_ocr(self):
        with tempfile.TemporaryDirectory() as temp:
            base = np.zeros((80, 120, 3), dtype=np.uint8)
            changed = base.copy()
            changed[10:50, 10:60] = 255
            samples = [(0.0, base), (1.0, base.copy()), (2.0, changed),
                       (3.0, changed.copy()), (4.0, changed.copy())]
            with (patch.object(screen, "probe_media", return_value={"duration": 5, "video": [{"index": 0}]}),
                  patch.object(screen, "_iter_video_frames", return_value=iter(samples)),
                  patch.object(screen, "_load_ocr", side_effect=AssertionError("OCR must not load"))):
                result = screen.extract_frames("video.mp4", temp, checkpoint_seconds=2)
            self.assertEqual(result["status"], "ready")
            self.assertEqual([e["reason"] for e in result["events"]], ["initial", "region_change", "checkpoint"])
            for event in result["events"]:
                self.assertEqual(set(event), {"id", "start", "end", "frame", "reason", "width", "height",
                                              "sample_interval_seconds", "time_uncertainty_seconds"})
                self.assertTrue((Path(temp) / event["frame"]).is_file())
            self.assertEqual(result["stats"]["sampled_frames"], 5)

    def test_frame_extraction_reports_no_frames_and_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(screen, "probe_media", return_value={"duration": 5, "video": [{"index": 0}]}), \
                    patch.object(screen, "_iter_video_frames", return_value=iter([])):
                no_frames = screen.extract_frames("video.mp4", temp)
            self.assertEqual(no_frames["status"], "unavailable")
            frame = np.zeros((32, 48, 3), dtype=np.uint8)
            with patch.object(screen, "probe_media", return_value={"duration": 5, "video": [{"index": 0}]}), \
                    patch.object(screen, "_iter_video_frames", return_value=iter([
                        (0.0, frame), (None, {"coverage_seconds": 1.0, "decode_error": True,
                                               "decode_complete": False, "error": "damaged"})])):
                partial = screen.extract_frames("video.mp4", temp)
            self.assertEqual(partial["status"], "partial")
            self.assertEqual(partial["events"][0]["start"], 0.0)
            self.assertEqual(partial["events"][0]["end"], 1.0)

    def test_manual_stock_plan_environment_reaches_fp32_loader_without_variant_validation(self):
        class FakeEngine:
            return_word_box = False

            def __init__(self):
                self.recognize_txt = lambda images: ("stock", images)

        fake_engine = FakeEngine()
        rapidocr_stub = types.ModuleType("rapidocr")
        rapidocr_stub.LangRec = SimpleNamespace(CYRILLIC="cyrillic")
        rapidocr_stub.ModelType = SimpleNamespace(MOBILE="mobile")
        rapidocr_stub.OCRVersion = SimpleNamespace(PPOCRV5="v5")
        captured_params = {}

        def create_ocr(*, params):
            captured_params.update(params)
            return fake_engine

        rapidocr_stub.RapidOCR = create_ocr
        with tempfile.TemporaryDirectory() as temp:
            python = Path(temp) / "python.exe"
            python.write_bytes(b"python")
            with patch.object(hardware, "GPU_PYTHON", python), \
                    patch.object(hardware, "_gpu_component_state", return_value=(True, [])), \
                    patch.object(hardware, "_identity", return_value={
                        "name": "Radeon", "vendor_id": 0x1002, "device_index": 0}), \
                    patch.object(hardware, "_receipt_matches", return_value=False):
                plan = {"device": "directml", "env": {
                    "VT_MEETING_OCR_REC_PRECISION": "fp32",
                    "VT_MEETING_OCR_BATCHING": "stock",
                    "VT_MEETING_OCR_REC_MODEL_PATH": "",
                    "VT_MEETING_OCR_REC_MODEL_SHA256": "",
                }}

        typings_stub = types.ModuleType("rapidocr.ch_ppocr_rec.typings")
        typings_stub.TextRecInput = lambda *, img, return_word_box: SimpleNamespace(
            img=img, return_word_box=return_word_box)
        with patch.dict(sys.modules, {
            "rapidocr": rapidocr_stub,
            "rapidocr.ch_ppocr_rec": types.ModuleType("rapidocr.ch_ppocr_rec"),
            "rapidocr.ch_ppocr_rec.typings": typings_stub,
        }), patch.object(screen, "_install_rapidocr_ort_provider_adapter"), \
                patch("services.meeting_model_variants.validate_fp16_variant") as validate, \
                patch.dict(os.environ, plan["env"], clear=False):
            engine = screen._load_ocr(plan["device"])

        self.assertEqual(plan["env"]["VT_MEETING_OCR_REC_PRECISION"], "fp32")
        self.assertEqual(plan["env"]["VT_MEETING_OCR_BATCHING"], "stock")
        self.assertEqual(plan["env"]["VT_MEETING_OCR_REC_MODEL_PATH"], "")
        self.assertNotIn("Rec.model_path", captured_params)
        self.assertTrue(callable(engine.recognize_txt))
        validate.assert_not_called()

    def test_fp16_route_validates_selected_model_and_wraps_only_recognition(self):
        class FakeEngine:
            return_word_box = False

            def __init__(self):
                self.stock_recognize = lambda images: ("stock", images)
                self.recognize_txt = self.stock_recognize
                self.rec_model = object()

            def _load_rec_model(self):
                return self.rec_model

        fake_engine = FakeEngine()
        rapidocr_stub = types.ModuleType("rapidocr")
        rapidocr_stub.LangRec = SimpleNamespace(CYRILLIC="cyrillic")
        rapidocr_stub.ModelType = SimpleNamespace(MOBILE="mobile")
        rapidocr_stub.OCRVersion = SimpleNamespace(PPOCRV5="v5")
        captured_params = {}
        def create_ocr(*, params):
            captured_params.update(params)
            return fake_engine
        rapidocr_stub.RapidOCR = create_ocr
        typings_stub = types.ModuleType("rapidocr.ch_ppocr_rec.typings")
        typings_stub.TextRecInput = lambda *, img, return_word_box: SimpleNamespace(
            img=img, return_word_box=return_word_box)
        rapidocr_rec_stub = types.ModuleType("rapidocr.ch_ppocr_rec")
        with patch.dict(sys.modules, {
            "rapidocr": rapidocr_stub,
            "rapidocr.ch_ppocr_rec": rapidocr_rec_stub,
            "rapidocr.ch_ppocr_rec.typings": typings_stub,
        }), patch.object(screen, "_install_rapidocr_ort_provider_adapter"), \
                patch("services.meeting_model_variants.validate_fp16_variant", return_value=True), \
                patch("services.meeting_ocr_batches.recognize_equal_width_batches",
                      return_value="batched") as batch:
            with patch.dict(os.environ, {
                "VT_MEETING_OCR_REC_PRECISION": "fp16",
                "VT_MEETING_OCR_REC_MODEL_PATH": "variant.onnx",
                "VT_MEETING_OCR_REC_MODEL_SHA256": "sha",
                "VT_MEETING_OCR_BATCHING": "equal-width",
            }, clear=False):
                engine = screen._load_ocr("directml")
                self.assertIsNot(engine.recognize_txt, fake_engine.stock_recognize)
                images = [np.zeros((12, 20, 3), dtype=np.uint8)]
                self.assertEqual(engine.recognize_txt(images), "batched")

        self.assertEqual(captured_params["Rec.model_path"], "variant.onnx")
        batch.assert_called_once()
        recognizer, rec_input = batch.call_args.args
        self.assertIs(recognizer, fake_engine.rec_model)
        self.assertIs(rec_input.img, images)
        self.assertFalse(rec_input.return_word_box)

    def test_invalid_fp16_route_fails_before_importing_rapidocr(self):
        with patch.dict(os.environ, {
            "VT_MEETING_OCR_REC_PRECISION": "fp16",
            "VT_MEETING_OCR_REC_MODEL_PATH": "invalid.onnx",
            "VT_MEETING_OCR_REC_MODEL_SHA256": "bad",
        }, clear=False), patch(
            "services.meeting_model_variants.validate_fp16_variant", return_value=False
        ) as validate, patch.object(screen, "_install_rapidocr_ort_provider_adapter") as install:
            with self.assertRaisesRegex(RuntimeError, "failed receipt validation"):
                screen._load_ocr("directml")
        validate.assert_called_once_with("invalid.onnx", "bad")
        install.assert_not_called()

    def test_pyav_sampling_preserves_native_sizes_and_container_timeline(self):
        class Frame:
            time_base = Fraction(1, 10)

            def __init__(self, pts, size):
                self.pts = pts
                self.size = size
                self.width, self.height = size

            def to_ndarray(self, *, format):
                self.assert_format = format
                width, height = self.size
                return np.full((height, width, 3), self.pts, dtype=np.uint8)

        frames = [Frame(50, (160, 120)), Frame(59, (160, 120)),
                  Frame(61, (320, 240)), Frame(79, (320, 240)),
                  Frame(81, (320, 240))]
        stream = SimpleNamespace(type="video", average_rate=Fraction(2, 1),
                                 base_rate=None, time_base=Fraction(1, 10), start_time=50)

        class Container:
            streams = [stream]
            start_time = 5_000_000

            def decode(self, _stream):
                yield from frames

            def close(self):
                self.closed = True

        container = Container()
        av_module = SimpleNamespace(time_base=1_000_000, open=lambda _path: container)
        with patch.dict(sys.modules, {"av": av_module}):
            result = list(screen._iter_video_frames(Path("Папка с пробелами") / "video.webm", 1))

        selected = [(time, image.shape[1], image.shape[0]) for time, image in result if time is not None]
        self.assertEqual([(round(time, 3), width, height) for time, width, height in selected],
                         [(0.0, 160, 120), (0.9, 160, 120), (1.1, 320, 240),
                          (2.9, 320, 240), (3.1, 320, 240)])
        metadata = result[-1][1]
        self.assertTrue(container.closed)
        self.assertEqual(metadata["timing_source"], "pts")
        self.assertEqual(metadata["decoder"], "pyav-native")
        self.assertEqual(metadata["resize_count"], 1)
        self.assertEqual(metadata["first_frame_size"], [160, 120])
        self.assertEqual(metadata["last_frame_size"], [320, 240])
        self.assertTrue(metadata["decode_complete"])
        self.assertEqual(metadata["container_start_seconds"], 5.0)

    def test_low_fps_frames_on_hold_are_not_lost_or_counted_as_decode_gaps(self):
        class Frame:
            time_base = Fraction(1, 1)
            width, height = 2160, 120
            duration = 2

            def __init__(self, pts):
                self.pts = pts

            def to_ndarray(self, *, format):
                return np.full((self.height, self.width, 3), self.pts, dtype=np.uint8)

        stream = SimpleNamespace(type="video", average_rate=Fraction(1, 2), base_rate=None,
                                 time_base=Fraction(1, 1), start_time=0)

        class Container:
            streams = [stream]
            start_time = 0

            def decode(self, _stream):
                yield from (Frame(0), Frame(2), Frame(4))

            def close(self):
                pass

        with patch.dict(sys.modules, {"av": SimpleNamespace(time_base=1_000_000,
                                                              open=lambda _path: Container())}):
            result = list(screen._iter_video_frames("slow-pts.webm", 1))
        self.assertEqual([time for time, _image in result[:-1]], [0.0, 2.0, 4.0])
        self.assertEqual(result[-1][1]["coverage_seconds"], 6.0)
        self.assertEqual(result[-1][1]["sampling_gaps"], 0)
        self.assertTrue(result[-1][1]["decode_complete"])

    def test_long_declared_frame_holds_cover_grid_points_without_decode_gaps(self):
        class Frame:
            time_base = Fraction(1, 1)
            width, height = 160, 90
            duration = 5

            def __init__(self, pts):
                self.pts = pts

            def to_ndarray(self, *, format):
                return np.full((self.height, self.width, 3), self.pts, dtype=np.uint8)

        stream = SimpleNamespace(type="video", average_rate=Fraction(1, 5), base_rate=None,
                                 time_base=Fraction(1, 1), start_time=0)

        class Container:
            streams = [stream]
            start_time = 0

            def decode(self, _stream):
                yield from (Frame(0), Frame(5), Frame(10))

            def close(self):
                pass

        with patch.dict(sys.modules, {"av": SimpleNamespace(time_base=1_000_000,
                                                              open=lambda _path: Container())}):
            result = list(screen._iter_video_frames("held-frame.webm", 1))
        self.assertEqual([time for time, _image in result[:-1]], [0.0, 5.0, 10.0])
        self.assertEqual(result[-1][1]["coverage_seconds"], 15.0)
        self.assertEqual(result[-1][1]["sampling_gaps"], 0)
        self.assertTrue(result[-1][1]["decode_complete"])

    def test_short_source_keeps_its_unaligned_final_frame_as_control_sample(self):
        class Frame:
            time_base = Fraction(1, 10)
            width, height = 24, 16
            duration = 2

            def __init__(self, pts):
                self.pts = pts

            def to_ndarray(self, *, format):
                return np.full((self.height, self.width, 3), self.pts, dtype=np.uint8)

        stream = SimpleNamespace(type="video", average_rate=Fraction(5, 2), base_rate=None,
                                 time_base=Fraction(1, 10), start_time=0)

        class Container:
            streams = [stream]
            start_time = 0

            def decode(self, _stream):
                yield from (Frame(0), Frame(4), Frame(8))

            def close(self):
                pass

        with patch.dict(sys.modules, {"av": SimpleNamespace(time_base=1_000_000,
                                                              open=lambda _path: Container())}):
            result = list(screen._iter_video_frames("short.webm", 1))
        self.assertEqual([time for time, _image in result[:-1]], [0.0, .8])
        self.assertEqual(result[-1][1]["coverage_seconds"], 1.0)
        self.assertTrue(result[-1][1]["decode_complete"])

    def test_video_start_offset_is_reported_without_marking_audio_prelude_incomplete(self):
        class Frame:
            time_base = Fraction(1, 10)
            width, height = 16, 12

            def __init__(self, pts):
                self.pts = pts

            def to_ndarray(self, *, format):
                return np.zeros((12, 16, 3), dtype=np.uint8)

        stream = SimpleNamespace(type="video", average_rate=Fraction(1, 1), base_rate=None,
                                 time_base=Fraction(1, 10), start_time=3)

        class Container:
            streams = [stream]
            start_time = 0

            def decode(self, _stream):
                yield Frame(3)
                yield Frame(13)

            def close(self):
                pass

        with patch.dict(sys.modules, {"av": SimpleNamespace(time_base=1_000_000,
                                                              open=lambda _path: Container())}):
            result = list(screen._iter_video_frames("offset.webm", 1))
        metadata = result[-1][1]
        self.assertAlmostEqual(metadata["first_video_time"], .3)
        self.assertTrue(metadata["decode_complete"])
        self.assertIn("earlier interval contains no video frames", metadata["timing_limitation"])

    def test_pyav_missing_pts_uses_known_average_rate_and_reports_fallback(self):
        class Frame:
            pts = None
            time_base = None
            width, height = 16, 12

            def to_ndarray(self, *, format):
                return np.zeros((12, 16, 3), dtype=np.uint8)

        stream = SimpleNamespace(type="video", average_rate=Fraction(2, 1), base_rate=None,
                                 time_base=Fraction(1, 10), start_time=50)

        class Container:
            streams = [stream]
            start_time = 5_000_000

            def decode(self, _stream):
                yield Frame()
                yield Frame()

            def close(self):
                self.closed = True

        container = Container()
        with patch.dict(sys.modules, {"av": SimpleNamespace(time_base=1_000_000,
                                                              open=lambda _path: container)}):
            result = list(screen._iter_video_frames("video.webm", .5))
        self.assertEqual([round(item[0], 3) for item in result[:-1]], [0.0, 0.5])
        self.assertEqual(result[-1][1]["timing_source"], "pts_with_average_rate_fallback")
        self.assertIn("average frame rate", result[-1][1]["timing_limitation"])
        self.assertTrue(result[-1][1]["decode_complete"])

    def test_pyav_unknown_rate_or_midstream_error_is_never_reported_complete(self):
        class Frame:
            pts = None
            time_base = None
            width, height = 16, 12

            def to_ndarray(self, *, format):
                return np.zeros((12, 16, 3), dtype=np.uint8)

        class Container:
            start_time = 0

            def __init__(self, rate, fail=False):
                self.streams = [SimpleNamespace(type="video", average_rate=rate, base_rate=None,
                                                time_base=Fraction(1, 10), start_time=None)]
                self.fail = fail

            def decode(self, _stream):
                if self.fail:
                    yield Frame()
                    yield Frame()
                    raise RuntimeError("broken packet")
                yield Frame()

            def close(self):
                self.closed = True

        unknown = Container(None)
        with patch.dict(sys.modules, {"av": SimpleNamespace(time_base=1_000_000,
                                                              open=lambda _path: unknown)}):
            result = list(screen._iter_video_frames("unknown.webm", 1))
        self.assertTrue(result[-1][1]["decode_error"])
        self.assertFalse(result[-1][1]["decode_complete"])
        self.assertIn("average frame rate is unknown", result[-1][1]["error"])
        self.assertTrue(unknown.closed)

        broken = Container(Fraction(2, 1), fail=True)
        with patch.dict(sys.modules, {"av": SimpleNamespace(time_base=1_000_000,
                                                              open=lambda _path: broken)}):
            result = list(screen._iter_video_frames("broken.webm", 1))
        self.assertEqual(len([item for item in result if item[0] is not None]), 1)
        self.assertTrue(result[-1][1]["decode_error"])
        self.assertFalse(result[-1][1]["decode_complete"])
        self.assertTrue(broken.closed)

    def test_closing_lazy_decoder_releases_container_without_reading_the_rest(self):
        class Frame:
            time_base = Fraction(1, 1)
            width, height = 16, 12

            def __init__(self, pts):
                self.pts = pts

            def to_ndarray(self, *, format):
                return np.zeros((12, 16, 3), dtype=np.uint8)

        class Container:
            start_time = 0
            streams = [SimpleNamespace(type="video", average_rate=Fraction(1, 1),
                                       base_rate=None, time_base=Fraction(1, 1), start_time=0)]

            def __init__(self):
                self.decoded = 0
                self.closed = False

            def decode(self, _stream):
                for pts in range(100):
                    self.decoded += 1
                    yield Frame(pts)

            def close(self):
                self.closed = True

        container = Container()
        with patch.dict(sys.modules, {"av": SimpleNamespace(time_base=1_000_000,
                                                              open=lambda _path: container)}):
            frames = screen._iter_video_frames("long.webm", 1)
            self.assertEqual(next(frames)[0], 0.0)
            frames.close()
        self.assertTrue(container.closed)
        self.assertEqual(container.decoded, 1)

    def test_real_variable_resolution_fixture_keeps_native_event_and_jpeg_geometry(self):
        ffmpeg = screen._exe("ffmpeg")
        if not ffmpeg:
            self.skipTest("FFmpeg is unavailable")
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest("PyAV is unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u043a\u0430\u0434\u0440\u043e\u0432 \u0441 \u043f\u0440\u043e\u0431\u0435\u043b\u0430\u043c\u0438"
            folder.mkdir()
            for name, color, size in (("part-a.webm", "black", "160x120"),
                                      ("part-b.webm", "white", "320x240")):
                command = [ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",
                           f"color=c={color}:s={size}:r=1:d=2", "-an", "-c:v", "libvpx-vp9",
                           "-deadline", "realtime", "-cpu-used", "8", "-pix_fmt", "yuv420p", name]
                completed = subprocess.run(command, cwd=folder, capture_output=True,
                                           timeout=30, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
            (folder / "concat.txt").write_text("file 'part-a.webm'\nfile 'part-b.webm'\n", encoding="ascii")
            fixture = folder / "\u043f\u0440\u0438\u043c\u0435\u0440 \u0437\u0430\u043f\u0438\u0441\u0438.webm"
            completed = subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt",
                 "-c", "copy", str(fixture)], cwd=folder, capture_output=True, timeout=30, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))

            image_sizes = []
            exe = lambda name: ffmpeg if name == "ffmpeg" else "ffprobe"
            with tempfile.TemporaryDirectory() as output, \
                    patch.object(screen, "probe_media", return_value={"duration": 4, "video": [{}]}), \
                    patch.object(screen, "_exe", side_effect=exe), \
                    patch.object(screen, "_load_ocr", return_value=lambda _image: []), \
                    patch.object(screen, "_parse_ocr", side_effect=lambda image, _ocr: image_sizes.append(
                        (image.shape[1], image.shape[0])) or []):
                result = screen.extract_screen(fixture, output, sample_seconds=1, checkpoint_seconds=30)
                self.assertEqual(image_sizes, [(160, 120), (320, 240)])
                size_events = [(event["width"], event["height"]) for event in result["events"]]
                self.assertEqual(size_events, [(160, 120), (320, 240)])
                saved_sizes = []
                for event in result["events"]:
                    saved = cv2.imread(str(Path(output) / event["frame"]))
                    saved_sizes.append((saved.shape[1], saved.shape[0]))
                self.assertEqual(saved_sizes, size_events)
                self.assertEqual(result["stats"]["decoder"], "pyav-native")
                self.assertEqual(result["stats"]["resize_count"], 1)
                self.assertEqual(result["status"], "ready")

    def test_provider_adapter_converts_omegaconf_like_options_to_native_dict(self):
        class ConfigNode:
            def items(self):
                return [("device_id", 2), ("nested", ConfigNodeValue())]

        class ConfigNodeValue:
            def items(self):
                return [("enabled", True)]

        providers = screen._provider_pairs([
            ("DmlExecutionProvider", ConfigNode()), "CPUExecutionProvider"])
        self.assertEqual(providers[0], ("DmlExecutionProvider", {"device_id": 2, "nested": {"enabled": True}}))
        self.assertIs(type(providers[0][1]), dict)
        self.assertEqual(providers[1], "CPUExecutionProvider")

    def test_directml_session_fallback_is_disabled_before_session_is_returned(self):
        class FakeSession:
            def __init__(self, *args, providers=None, **kwargs):
                self.providers = providers or []
                self.fallback_disabled = False

            def get_providers(self):
                return [provider[0] if isinstance(provider, tuple) else provider
                        for provider in self.providers]

            def disable_fallback(self):
                self.fallback_disabled = True

        modules, main = _fake_ort_module_tree(FakeSession)
        with patch.dict(sys.modules, modules):
            screen._install_rapidocr_ort_provider_adapter()
            adapter = main.InferenceSession
            directml = adapter("model.onnx", providers=[("DmlExecutionProvider", {"device_id": 0}),
                                                          "CPUExecutionProvider"])
            cpu = adapter("model.onnx", providers=["CPUExecutionProvider"])
        self.assertTrue(directml.fallback_disabled)
        self.assertFalse(cpu.fallback_disabled)

    def test_directml_factory_rejects_missing_disable_fallback_and_cpu_only_session(self):
        class MissingDisable:
            def __init__(self, *args, providers=None, **kwargs):
                self.providers = providers or []

            def get_providers(self):
                return [provider[0] if isinstance(provider, tuple) else provider
                        for provider in self.providers]

        modules, main = _fake_ort_module_tree(MissingDisable)
        with patch.dict(sys.modules, modules):
            screen._install_rapidocr_ort_provider_adapter()
            with self.assertRaisesRegex(RuntimeError, "cannot disable CPU fallback"):
                main.InferenceSession("model.onnx", providers=["DmlExecutionProvider"])

        class CpuOnly(MissingDisable):
            def get_providers(self):
                return ["CPUExecutionProvider"]

            def disable_fallback(self):
                raise AssertionError("CPU-only construction must fail before fallback configuration")

        modules, main = _fake_ort_module_tree(CpuOnly)
        with patch.dict(sys.modules, modules):
            screen._install_rapidocr_ort_provider_adapter()
            with self.assertRaisesRegex(RuntimeError, "session is not using DirectML"):
                main.InferenceSession("model.onnx", providers=["DmlExecutionProvider"])

    def test_profiled_function_factory_is_preserved_and_guarded(self):
        calls = []

        class FakeSession:
            def __init__(self, providers):
                self.providers = providers
                self.fallback_disabled = False

            def get_providers(self):
                return [provider[0] if isinstance(provider, tuple) else provider
                        for provider in self.providers]

            def disable_fallback(self):
                self.fallback_disabled = True

        def profiling_factory(*args, providers=None, **kwargs):
            calls.append((args, providers))
            return FakeSession(providers or [])

        profiling_factory._profiled_for_test = True
        modules, main = _fake_ort_module_tree(profiling_factory)
        with patch.dict(sys.modules, modules):
            screen._install_rapidocr_ort_provider_adapter()
            wrapped = main.InferenceSession
            session = wrapped("model.onnx", providers=["DmlExecutionProvider"])
        self.assertIsNot(wrapped, profiling_factory)
        self.assertEqual(len(calls), 1)
        self.assertTrue(session.fallback_disabled)

    def test_directml_assertion_rejects_any_cpu_part_and_requires_a_session(self):
        def part(providers):
            session = SimpleNamespace(get_providers=lambda: providers)
            return SimpleNamespace(session=SimpleNamespace(session=session))

        mixed = SimpleNamespace(text_det=part(["DmlExecutionProvider"]),
                                text_cls=None, text_rec=part(["CPUExecutionProvider"]))
        with self.assertRaisesRegex(RuntimeError, "text_rec did not create a DirectML session"):
            screen._assert_ocr_device(mixed, "directml")

        with self.assertRaisesRegex(RuntimeError, "did not create any OCR sessions"):
            screen._assert_ocr_device(SimpleNamespace(text_det=None, text_cls=None, text_rec=None),
                                      "directml")

        cpu = part(["CPUExecutionProvider"])
        screen._assert_ocr_device(SimpleNamespace(text_det=cpu), "cpu")

    def test_rapidocr_empty_region_result_is_not_an_ocr_failure(self):
        class EmptyRapidOCROutput:
            boxes = None
            txts = None
            scores = None

        image = np.full((96, 240, 3), 250, dtype=np.uint8)
        self.assertEqual(screen._parse_ocr(image, lambda _image: (EmptyRapidOCROutput(), None)), [])

        def actual_failure(_image):
            raise RuntimeError("model execution failed")
        with self.assertRaisesRegex(RuntimeError, "model execution failed"):
            screen._parse_ocr(image, actual_failure)

    def test_tile_diff_detects_single_local_cell_change(self):
        before = np.full((360, 640, 3), 25, dtype=np.uint8)
        after = before.copy()
        after[215:235, 405:430] = 240
        self.assertTrue(screen._tile_changed(before, after))

    def test_tile_diff_detects_small_character_edit_against_selected_frame(self):
        before = np.full((240, 480, 3), 245, dtype=np.uint8)
        after = before.copy()
        cv2.putText(before, "value=24", (260, 130), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1)
        cv2.putText(after, "value=25", (260, 130), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1)
        self.assertTrue(screen._tile_changed(before, after))

    def test_native_16px_grid_detects_tiny_cell_edit_at_native_and_1080p(self):
        for height, width in ((1800, 2880), (1080, 1920)):
            with self.subTest(width=width, height=height):
                before = np.zeros((height, width, 3), dtype=np.uint8)
                after = before.copy()
                x, y = width // 2 + 7, height // 2 + 5
                after[y:y + 4, x:x + 4] = 255  # 16 changed pixels in one native tile.
                self.assertTrue(screen._tile_changed(before, after))
                rois = screen._changed_rois(before, after, padding=0)
                self.assertTrue(any(rx <= x and ry <= y and rx + rw >= x + 4 and ry + rh >= y + 4
                                    for rx, ry, rw, rh in rois))

    def test_native_tile_ignores_single_pixel_noise_and_shape_change_is_full_frame(self):
        before = np.zeros((1800, 2880, 3), dtype=np.uint8)
        noise = before.copy()
        noise[500, 600] = 255
        self.assertFalse(screen._tile_changed(before, noise))
        self.assertEqual(screen._changed_rois(before, noise), [])

        resized = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertTrue(screen._tile_changed(before, resized))
        self.assertEqual(screen._changed_rois(before, resized), [[0, 0, 1920, 1080]])

    def test_screen_events_capture_region_changes_and_checkpoints(self):
        frames = []
        for value in (0, 0, 180, 180, 180, 180):
            frame = np.zeros((100, 160, 3), dtype=np.uint8)
            frame[60:75, 120:145] = value
            frames.append(frame)
        samples = list(zip(range(6), frames))
        with tempfile.TemporaryDirectory() as temp:
            with (patch.object(screen, "probe_media", return_value={"duration": 6, "video": [{}]}),
                  patch.object(screen, "_exe", return_value="ffmpeg"),
                  patch.object(screen, "_load_ocr", return_value=lambda _image: []),
                  patch.object(screen, "_iter_video_frames", return_value=iter(samples))):
                result = screen.extract_screen("video.mp4", temp, sample_seconds=1, checkpoint_seconds=3)
            self.assertEqual([e["reason"] for e in result["events"]], ["initial", "region_change", "checkpoint"])
            self.assertEqual([e["start"] for e in result["events"]], [0.0, 2.0, 5.0])
            self.assertEqual([e["end"] for e in result["events"]], [2.0, 5.0, 6.0])
            self.assertEqual(result["status"], "ready")

    def test_geometric_rows_keep_header_relationship(self):
        regions = []
        for y, texts in ((0, ("Отчет за квартал",)), (25, ("Имя", "Роль")),
                         (50, ("Анна", "Ведущая")), (75, ("Иван", "Разработчик")),
                         (110, ("Страница 1",))):
            for x, text in zip((10, 120), texts):
                regions.append({"text": text, "bbox": [x, y, 70, 12], "confidence": .9})
        table = screen._regions_to_tables(regions)[0]
        self.assertEqual([header["text"] for header in table["headers"]], ["Имя", "Роль"])
        self.assertEqual([[cell["text"] for cell in row] for row in table["cells"]],
                         [["Анна", "Ведущая"], ["Иван", "Разработчик"]])
        self.assertEqual(table["cells"][0][0]["header"], {"index": 0, "text": "Имя"})
        self.assertEqual(table["bbox"], [10, 25, 180, 62])
        self.assertTrue(table["header_is_candidate"])
        self.assertTrue(table["uncertain"])

    def test_partial_geometric_table_keeps_missing_and_ambiguous_cells(self):
        # Saved OCR excerpt from screen-0389, restricted to the target table
        # and a nearby left-hand table sharing its header/body y coordinates.
        regions = [
            {"text": "9283", "bbox": [250, 391, 131, 22], "confidence": .9},
            {"text": "7.0 Март", "bbox": [415, 395, 141, 16], "confidence": .9},
            {"text": "Март", "bbox": [670, 396, 56, 15], "confidence": .9},
            {"text": "9864", "bbox": [251, 410, 127, 23], "confidence": .9},
            {"text": "70.13 Апрель", "bbox": [409, 414, 159, 16], "confidence": .9},
            {"text": "Апрель", "bbox": [670, 414, 67, 17], "confidence": .9},
            {"text": "10583", "bbox": [253, 430, 123, 23], "confidence": .9},
            {"text": "5.5 Январь", "bbox": [414, 433, 154, 17], "confidence": .9},
            {"text": "Январь", "bbox": [670, 434, 67, 15], "confidence": .9},
            {"text": "Начало периода", "bbox": [803, 406, 87, 14], "confidence": .9},
            {"text": "Окончание периода", "bbox": [899, 406, 103, 14], "confidence": .9},
            {"text": "Заявка, приемниктрудозатрат", "bbox": [1012, 405, 155, 16], "confidence": .9},
            {"text": "Трудозатраты %", "bbox": [1409, 405, 88, 16], "confidence": .9},
            {"text": "Пернод создания строк", "bbox": [1521, 404, 130, 17], "confidence": .9},
            {"text": "Май 2026", "bbox": [803, 425, 52, 15], "confidence": .9},
            {"text": "Декабрь 2026", "bbox": [899, 425, 73, 16], "confidence": .9},
            {"text": "9864. 2026. Запись в детский сад Модернизация услуги", "bbox": [1012, 425, 379, 17], "confidence": .9},
            {"text": "100.0000000000", "bbox": [1426, 426, 80, 14], "confidence": .9},
            # Nearby toolbar/form labels. Their centers are 56 px apart, so
            # they must not be inferred as another partial table row.
            {"text": "Проекты", "bbox": [105, 105, 66, 24], "confidence": .9},
            {"text": "Код:", "bbox": [236, 107, 35, 22], "confidence": .9},
            {"text": "Прикрепленные файлы", "bbox": [832, 110, 122, 17], "confidence": .9},
            {"text": "Еще", "bbox": [1334, 107, 45, 22], "confidence": .9},
            {"text": "Заявки и кубики", "bbox": [106, 139, 111, 23], "confidence": .9},
            {"text": "SCR SCR#5904616", "bbox": [237, 136, 113, 20], "confidence": .9},
            {"text": "Система", "bbox": [845, 162, 57, 18], "confidence": .9},
            {"text": "Отменена", "bbox": [1151, 141, 57, 18], "confidence": .9},
        ]

        tables = screen._regions_to_tables(regions)
        table = next(item for item in tables if item["headers"][0]["text"] == "Начало периода")
        self.assertEqual([header["text"] for header in table["headers"]], [
            "Начало периода", "Окончание периода", "Заявка, приемниктрудозатрат",
            "Трудозатраты %", "Пернод создания строк",
        ])
        row = table["cells"][0]
        cells_by_text = {cell["text"]: cell for cell in row}
        self.assertEqual(cells_by_text["Май 2026"]["header"], {"index": 0, "text": "Начало периода"})
        self.assertEqual(cells_by_text["Декабрь 2026"]["header"], {"index": 1, "text": "Окончание периода"})
        self.assertEqual(cells_by_text["100.0000000000"]["header"], {"index": 3, "text": "Трудозатраты %"})
        self.assertTrue(cells_by_text["9864. 2026. Запись в детский сад Модернизация услуги"]["ambiguous"])
        self.assertNotIn("header", cells_by_text["9864. 2026. Запись в детский сад Модернизация услуги"])
        self.assertFalse(any("Апрель" in cell["text"] for cell in row))
        self.assertEqual(table["bbox"], [803, 404, 848, 38])
        self.assertTrue(table["header_is_candidate"])
        self.assertTrue(table["uncertain"])
        self.assertFalse(any(
            any(header["text"] == "Прикрепленные файлы" for header in candidate["headers"])
            and any(cell["text"] == "Система" for row in candidate["cells"] for cell in row)
            for candidate in tables))

    def test_single_sql_row_with_two_columns_remains_below_candidate_threshold(self):
        regions = [
            {"text": "Field", "bbox": [50, 50, 70, 18], "confidence": .99},
            {"text": "Query", "bbox": [300, 50, 70, 18], "confidence": .99},
            {"text": "Invoice", "bbox": [50, 100, 70, 18], "confidence": .99},
            {"text": "SELECT value FROM table", "bbox": [300, 100, 220, 18], "confidence": .99},
        ]
        self.assertEqual(screen._regions_to_tables(regions), [])

    def test_incremental_ocr_keeps_header_and_reads_only_changed_value_area(self):
        frames = []
        for value in (24, 25):
            frame = np.full((360, 640, 3), 245, dtype=np.uint8)
            cv2.putText(frame, "Limit", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, (10, 10, 10), 2)
            cv2.putText(frame, f"value={value}", (410, 230), cv2.FONT_HERSHEY_SIMPLEX, .65, (10, 10, 10), 2)
            frames.append(frame)
        samples = list(zip((0.0, 1.0), frames))
        calls = []

        def fake_parse(image, _ocr):
            calls.append(image.shape[1])
            if image.shape[1] == 640:
                return [
                    {"text": "Limit", "bbox": [30, 35, 70, 20], "confidence": .99},
                    {"text": "value=24", "bbox": [410, 215, 85, 20], "confidence": .99},
                ]
            return [{"text": "value=25", "bbox": [60, 60, 85, 20], "confidence": .98}]

        with tempfile.TemporaryDirectory() as temp, \
                patch.object(screen, "probe_media", return_value={"duration": 2, "video": [{}]}), \
                patch.object(screen, "_exe", return_value="ffmpeg"), \
                patch.object(screen, "_load_ocr", return_value=lambda _image: []), \
                patch.object(screen, "_parse_ocr", side_effect=fake_parse), \
                patch.object(screen, "_iter_video_frames", return_value=iter(samples)):
            result = screen.extract_screen("video.mp4", temp, sample_seconds=1, checkpoint_seconds=30)
        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(calls[0], 640)
        self.assertLess(calls[1], 640)
        latest = result["events"][1]["regions"]
        self.assertEqual([row["text"] for row in latest], ["Limit", "value=25"])
        self.assertEqual(latest[0]["recognized_at"], 0.0)
        self.assertIn("inherited", latest[0]["uncertainty"])
        self.assertEqual(latest[1]["recognized_at"], 1.0)
        self.assertEqual(result["stats"]["roi_ocr_frames"], 1)

    def test_incremental_long_sql_region_keeps_full_text_and_table_header(self):
        before = np.full((400, 1000, 3), 245, dtype=np.uint8)
        after = before.copy()
        after[105:118, 790:805] = 15
        after[105:118, 805:820] = 220
        long_sql = "SELECT customer_id, SUM(amount) FROM invoices WHERE period = 2026 AND amount > 24 GROUP BY customer_id"
        updated_sql = long_sql.replace("> 24", "> 25")
        old_regions = [
            {"text": "Field", "bbox": [50, 50, 70, 18], "confidence": .99},
            {"text": "Query", "bbox": [300, 50, 70, 18], "confidence": .99},
            {"text": "Invoice", "bbox": [50, 100, 70, 18], "confidence": .99},
            {"text": long_sql, "bbox": [300, 100, 650, 18], "confidence": .99},
            {"text": "Status", "bbox": [50, 150, 70, 18], "confidence": .99},
            {"text": "SELECT status FROM invoices", "bbox": [300, 150, 300, 18], "confidence": .99},
        ]
        initial_rois = screen._changed_rois(before, after)
        expanded = screen._expand_rois_to_regions(initial_rois, old_regions, 1000, 400)
        self.assertTrue(any(roi[2] >= 650 for roi in expanded))

        def fake_parse(image, _ocr):
            if image.shape[1] == 1000:
                return old_regions
            roi = expanded[0]
            output = []
            for region in old_regions:
                if screen._intersects(region, roi):
                    row = dict(region)
                    if region["text"] == long_sql:
                        row["text"] = updated_sql
                    x, y, w, h = region["bbox"]
                    row["bbox"] = [x - roi[0], y - roi[1], w, h]
                    output.append(row)
            return output

        with tempfile.TemporaryDirectory() as temp, \
                patch.object(screen, "probe_media", return_value={"duration": 2, "video": [{}]}), \
                patch.object(screen, "_exe", return_value="ffmpeg"), \
                patch.object(screen, "_load_ocr", return_value=lambda _image: []), \
                patch.object(screen, "_parse_ocr", side_effect=fake_parse), \
                patch.object(screen, "_iter_video_frames", return_value=iter([(0.0, before), (1.0, after)])):
            result = screen.extract_screen("video.mp4", temp, sample_seconds=1, checkpoint_seconds=30)
        self.assertEqual(len(result["events"]), 2)
        regions = result["events"][1]["regions"]
        self.assertIn(updated_sql, [item["text"] for item in regions])
        table = result["events"][1]["tables"][0]
        self.assertEqual([header["text"] for header in table["headers"]], ["Field", "Query"])
        self.assertEqual(table["cells"][0][1]["text"], updated_sql)
        self.assertEqual(table["cells"][0][1]["header"], {"index": 1, "text": "Query"})

    def test_anchored_recognition_updates_only_changed_confident_boxes(self):
        previous = np.full((220, 600, 3), 245, dtype=np.uint8)
        current = previous.copy()
        current[99:103, 205:209] = 20
        current[99:103, 355:359] = 20
        regions = [
            {"text": "Header", "bbox": [100, 90, 75, 18], "confidence": .98,
             "recognized_at": 0.0, "recognition_mode": "full"},
            {"text": "value=25", "bbox": [190, 90, 100, 18], "confidence": .97},
            {"text": "SELECT value FROM table WHERE id = 25", "bbox": [305, 90, 265, 18], "confidence": .99},
        ]
        rois = screen._expand_rois_to_regions(screen._changed_rois(previous, current), regions, 600, 220)

        class RecOnly:
            def __init__(self):
                self.recognized_crops = []
                self.detector_calls = 0

            def recognize_txt(self, crops):
                self.recognized_crops.append(crops)
                return SimpleNamespace(txts=("value=27", "SELECT value FROM table WHERE id = 27"),
                                       scores=(.96, .95))

            def __call__(self, _image):
                self.detector_calls += 1
                raise AssertionError("anchored path must not invoke detector OCR")

        engine = RecOnly()
        stats = {}
        updated = screen._incremental_regions(current, regions, rois, engine, 1.5,
                                              previous_image=previous, stats=stats)
        by_text = {region["text"]: region for region in updated}
        self.assertIn("Header", by_text)
        sql = by_text["SELECT value FROM table WHERE id = 27"]
        value = by_text["value=27"]
        self.assertEqual(value["bbox"], regions[1]["bbox"])
        self.assertEqual(value["recognized_at"], 1.5)
        self.assertEqual(value["recognition_mode"], "anchored_recognition")
        self.assertTrue(value["geometry_inherited"])
        self.assertEqual(value["confidence"], .96)
        self.assertEqual(sql["bbox"], regions[2]["bbox"])
        self.assertTrue(sql["geometry_inherited"])
        self.assertEqual(engine.detector_calls, 0)
        self.assertEqual(len(engine.recognized_crops), 1)
        crops = engine.recognized_crops[0]
        self.assertEqual(len(crops), 2)
        self.assertLessEqual(crops[0].shape[1], regions[1]["bbox"][2] + 4)
        self.assertLessEqual(crops[0].shape[0], regions[1]["bbox"][3] + 4)
        self.assertLessEqual(crops[1].shape[1], regions[2]["bbox"][2] + 4)
        self.assertEqual(stats, {"anchored_rec_frames": 1, "anchored_rec_regions": 2})

    def test_anchored_guard_rejects_new_menu_pixels_outside_old_text_boxes(self):
        previous = np.full((180, 400, 3), 245, dtype=np.uint8)
        current = previous.copy()
        # A changed text pixel and a menu/overlay pixel outside the old text box.
        current[80:84, 105:109] = 20
        current[88:92, 120:124] = 20
        region = {"text": "value=25", "bbox": [90, 75, 50, 15], "confidence": .99}

        class RecOnly:
            def recognize_txt(self, _crops):
                return SimpleNamespace(txts=("value=27",), scores=(.99,))

        self.assertIsNone(screen._try_anchored_regions(previous, current, [region], RecOnly(), 1.0))

    def test_menu_or_scroll_roi_falls_back_to_detector_ocr(self):
        previous = np.full((360, 640, 3), 245, dtype=np.uint8)
        current = previous.copy()
        current[80:84, 105:109] = 20  # Edited known value.
        current[84:88, 160:164] = 20  # Menu/scroll redraw outside its previous bbox.
        regions = [{"text": "value=25", "bbox": [90, 75, 55, 16], "confidence": .99}]

        class HybridOCR:
            def __init__(self):
                self.rec_only_calls = 0
                self.detector_calls = 0

            def recognize_txt(self, _crops):
                self.rec_only_calls += 1
                return SimpleNamespace(txts=("value=27",), scores=(.99,))

            def __call__(self, _image):
                self.detector_calls += 1
                return [([[20, 20], [110, 20], [110, 38], [20, 38]], "menu and value", .98)]

        engine = HybridOCR()
        stats = {}
        updated = screen._incremental_regions(current, regions, [[60, 45, 140, 90]],
                                              engine, 1.0, previous_image=previous, stats=stats)
        self.assertEqual(engine.rec_only_calls, 0)
        self.assertEqual(engine.detector_calls, 1)
        self.assertEqual(updated[0]["text"], "menu and value")
        self.assertEqual(stats, {})

    def test_anchored_empty_recognition_deletes_only_a_visually_blank_box(self):
        previous = np.full((120, 240, 3), 245, dtype=np.uint8)
        previous[49:55, 80:130] = 10
        current = np.full_like(previous, 245)
        region = {"text": "deleted value", "bbox": [78, 47, 54, 10], "confidence": .98}

        class RecOnly:
            def recognize_txt(self, _crops):
                return SimpleNamespace(txts=("",), scores=(0.0,))

        result = screen._try_anchored_regions(previous, current, [region], RecOnly(), 2.0)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], [])
        self.assertEqual(result[1], {0})

    def test_anchored_low_confidence_falls_back_and_engine_errors_propagate(self):
        previous = np.full((120, 240, 3), 245, dtype=np.uint8)
        current = previous.copy()
        current[49:53, 90:94] = 10
        region = {"text": "value=25", "bbox": [80, 45, 70, 15], "confidence": .98}

        class LowConfidence:
            def recognize_txt(self, _crops):
                return SimpleNamespace(txts=("value=27",), scores=(.4,))

        class Broken:
            def recognize_txt(self, _crops):
                raise RuntimeError("recognizer failed")

        self.assertIsNone(screen._try_anchored_regions(previous, current, [region], LowConfidence(), 1.0))
        with self.assertRaisesRegex(RuntimeError, "recognizer failed"):
            screen._try_anchored_regions(previous, current, [region], Broken(), 1.0)

    def test_roi_expansion_stays_local_for_dense_table_and_contains_intersected_boxes(self):
        width, height = 1000, 1600
        regions = []
        for row in range(24):
            y = 100 + row * 20  # Dense rows with only 8 px between OCR boxes.
            regions.extend((
                {"text": f"Field {row}", "bbox": [70, y, 90, 12], "confidence": .99},
                {"text": f"value={row}", "bbox": [600, y, 80, 12], "confidence": .99},
            ))

        changed_row = 12
        changed_roi = [[620, 100 + changed_row * 20 + 3, 5, 5]]
        expanded = screen._expand_rois_to_regions(changed_roi, regions, width, height, padding=24)
        area = sum(w * h for _x, _y, w, h in expanded)
        self.assertLess(area, width * height * .42)
        self.assertEqual(len(expanded), 1)

        for region in regions:
            if any(screen._intersects(region, roi) for roi in expanded):
                x, y, w, h = region["bbox"]
                self.assertTrue(any(rx <= x and ry <= y and rx + rw >= x + w and ry + rh >= y + h
                                    for rx, ry, rw, rh in expanded), region)

    def test_ocr_load_failure_keeps_frame_and_reports_partial(self):
        frame = np.full((40, 60, 3), 80, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(screen, "probe_media", return_value={"duration": 2, "video": [{}]}), \
                patch.object(screen, "_exe", return_value="ffmpeg"), \
                patch.object(screen, "_load_ocr", side_effect=RuntimeError("model download missing")), \
                patch.object(screen, "_iter_video_frames", return_value=iter([(0.0, frame)])):
            result = screen.extract_screen("video.mp4", temp)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(len(result["events"]), 1)
            self.assertTrue((Path(temp) / result["events"][0]["frame"]).is_file())

    def test_directml_ocr_load_and_parse_errors_abort_stage_with_cause(self):
        frame = np.full((40, 60, 3), 250, dtype=np.uint8)
        failures = (
            ("load", OSError("runtime missing")),
            ("parse", ValueError("inference failed")),
        )
        for phase, original_error in failures:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temp, \
                    patch.object(screen, "probe_media", return_value={"duration": 1, "video": [{}]}), \
                    patch.object(screen, "_exe", return_value="ffmpeg"), \
                    patch.object(screen, "_iter_video_frames", return_value=iter([(0.0, frame)])):
                load = patch.object(screen, "_load_ocr", side_effect=original_error) if phase == "load" else \
                    patch.object(screen, "_load_ocr", return_value=lambda _image: [])
                parse = patch.object(screen, "_parse_ocr", side_effect=original_error) if phase == "parse" else \
                    patch.object(screen, "_parse_ocr", return_value=[])
                with load, parse:
                    with self.assertRaisesRegex(RuntimeError, "Не удалось обработать экран.*GPU, DirectML") as caught:
                        screen.extract_screen("video.mp4", temp, device="directml")
                self.assertIs(caught.exception.__cause__, original_error)

    def test_directml_provider_assertion_failure_aborts_stage(self):
        frame = np.full((40, 60, 3), 250, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(screen, "probe_media", return_value={"duration": 1, "video": [{}]}), \
                patch.object(screen, "_exe", return_value="ffmpeg"), \
                patch.object(screen, "_load_ocr", return_value=lambda _image: []), \
                patch.object(screen, "_iter_video_frames", return_value=iter([(0.0, frame)])):
            with self.assertRaisesRegex(RuntimeError, "Не удалось обработать экран.*GPU, DirectML") as caught:
                screen.extract_screen("video.mp4", temp, device="directml")
        self.assertIn("did not create any OCR sessions", str(caught.exception.__cause__))

    def test_directml_empty_ocr_is_a_successful_blank_frame(self):
        frame = np.full((40, 60, 3), 250, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(screen, "probe_media", return_value={"duration": 1, "video": [{}]}), \
                patch.object(screen, "_exe", return_value="ffmpeg"), \
                patch.object(screen, "_load_ocr", return_value=lambda _image: []), \
                patch.object(screen, "_assert_ocr_device") as assert_device, \
                patch.object(screen, "_iter_video_frames", return_value=iter([(0.0, frame)])):
            result = screen.extract_screen("video.mp4", temp, device="directml")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["events"][0]["text"], "")
        assert_device.assert_called_once()

    def test_directml_anchored_recognition_error_aborts_without_detector_fallback(self):
        frame0 = np.full((360, 640, 3), 245, dtype=np.uint8)
        frame1 = frame0.copy()
        frame0[102:106, 205:209] = 20
        frame1[102:106, 205:209] = 80
        region = {"text": "value=25", "bbox": [190, 95, 100, 20], "confidence": .99}

        class BrokenGPUOCR:
            def recognize_txt(self, _crops):
                raise OSError("DirectML recognition failed")

        samples = iter([(0.0, frame0), (1.0, frame1)])
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(screen, "probe_media", return_value={"duration": 2, "video": [{}]}), \
                patch.object(screen, "_exe", return_value="ffmpeg"), \
                patch.object(screen, "_load_ocr", return_value=BrokenGPUOCR()), \
                patch.object(screen, "_parse_ocr", return_value=[region]), \
                patch.object(screen, "_assert_ocr_device"), \
                patch.object(screen, "_iter_video_frames", return_value=samples):
            with self.assertRaisesRegex(RuntimeError, "Не удалось обработать экран.*GPU, DirectML") as caught:
                screen.extract_screen("video.mp4", temp, device="directml")
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertIn("DirectML recognition failed", str(caught.exception.__cause__))

    def test_ocr_equivalent_visual_change_is_saved_once_and_checkpoint_remains(self):
        frames = []
        for changed in (False, True, True, True, True):
            frame = np.zeros((100, 160, 3), dtype=np.uint8)
            if changed:
                frame[60:75, 120:145] = 240
            frames.append(frame)
        regions = [{"text": "stable text", "bbox": [10, 10, 80, 12], "confidence": .99}]
        samples = list(zip(range(5), frames))
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(screen, "probe_media", return_value={"duration": 5, "video": [{}]}), \
                patch.object(screen, "_exe", return_value="ffmpeg"), \
                patch.object(screen, "_load_ocr", return_value=lambda _image: ([([[10, 10], [90, 10], [90, 22], [10, 22]], "stable text", .99)], None)), \
                patch.object(screen, "_iter_video_frames", return_value=iter(samples)):
            result = screen.extract_screen("video.mp4", temp, sample_seconds=1, checkpoint_seconds=3)
        self.assertEqual([event["reason"] for event in result["events"]],
                         ["initial", "region_change", "checkpoint"])
        self.assertEqual(result["stats"]["duplicate_candidates"], 1)
        self.assertTrue(result["events"][1]["ocr_equivalent_to_previous"])
        self.assertEqual(result["stats"]["ocr_equivalent_events"], 1)
        self.assertTrue(any("retained as evidence" in item for item in result["limitations"]))

    def test_native_short_cell_change_before_checkpoint_is_saved_with_neighboring_value(self):
        width, height = 2880, 1800
        frames = [np.full((height, width, 3), 245, dtype=np.uint8) for _ in range(5)]
        frames[2][968:972, 1420:1424] = 20
        frames[3] = frames[2].copy()
        frames[4] = frames[2].copy()
        old_regions = [
            {"text": "Checkbox: off", "bbox": [1408, 960, 100, 18], "confidence": .99},
            {"text": "Neighbor value: 42", "bbox": [1600, 960, 130, 18], "confidence": .99},
        ]
        changed_regions = [
            {"text": "Checkbox: on", "bbox": [1408, 960, 100, 18], "confidence": .99},
            old_regions[1],
        ]
        changed_rois = screen._changed_rois(frames[0], frames[2])
        expected_roi = screen._expand_rois_to_regions(changed_rois, old_regions, width, height)[0]
        full_calls = 0

        def fake_parse(image, _ocr):
            nonlocal full_calls
            if image.shape[1] == width:
                full_calls += 1
                return old_regions if full_calls == 1 else changed_regions
            return [{"text": "Checkbox: on",
                     "bbox": [1408 - expected_roi[0], 960 - expected_roi[1], 100, 18],
                     "confidence": .99}]

        samples = list(zip(range(5), frames))
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(screen, "probe_media", return_value={"duration": 5, "video": [{}]}), \
                patch.object(screen, "_exe", return_value="ffmpeg"), \
                patch.object(screen, "_load_ocr", return_value=lambda _image: []), \
                patch.object(screen, "_parse_ocr", side_effect=fake_parse), \
                patch.object(screen, "_iter_video_frames", return_value=iter(samples)):
            result = screen.extract_screen("video.mp4", temp, sample_seconds=1, checkpoint_seconds=3)

        self.assertEqual([event["reason"] for event in result["events"]],
                         ["initial", "region_change", "checkpoint"])
        changed_event = result["events"][1]
        self.assertIn("Checkbox: on", changed_event["text"])
        self.assertIn("Neighbor value: 42", changed_event["text"])
        self.assertEqual(changed_event["start"], 2.0)

    def test_continuous_roi_changes_do_not_defer_full_checkpoint_ocr(self):
        width, height = 640, 360
        frames = []
        for value in (20, 50, 80, 110, 140, 170, 200):
            frame = np.full((height, width, 3), 245, dtype=np.uint8)
            frame[77:81, 90:94] = value
            frames.append(frame)
        region = {"text": "value=25", "bbox": [70, 70, 100, 24], "confidence": .99}
        full_ocr_values = []

        def fake_parse(image, _ocr):
            if image.shape[1] == width:
                full_ocr_values.append(int(image[78, 91, 0]))
                return [region]
            return [{"text": "value=25", "bbox": [35, 35, 100, 24], "confidence": .99}]

        samples = list(zip(range(len(frames)), frames))
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(screen, "probe_media", return_value={"duration": 7, "video": [{}]}), \
                patch.object(screen, "_exe", return_value="ffmpeg"), \
                patch.object(screen, "_load_ocr", return_value=lambda _image: []), \
                patch.object(screen, "_parse_ocr", side_effect=fake_parse), \
                patch.object(screen, "_iter_video_frames", return_value=iter(samples)):
            result = screen.extract_screen("video.mp4", temp, sample_seconds=1, checkpoint_seconds=3)

        self.assertEqual(full_ocr_values, [20, 110, 200])
        self.assertEqual([event["reason"] for event in result["events"]],
                         ["initial", "region_change", "region_change", "checkpoint",
                          "region_change", "region_change", "checkpoint"])

    def test_early_decode_end_marks_partial_and_uses_covered_interval(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        samples = iter([(0.0, frame), (None, {"returncode": 0, "coverage_seconds": 1.0})])
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(screen, "probe_media", return_value={"duration": 10, "video": [{}]}), \
                patch.object(screen, "_exe", return_value="ffmpeg"), \
                patch.object(screen, "_load_ocr", return_value=lambda _image: []), \
                patch.object(screen, "_iter_video_frames", return_value=samples):
            result = screen.extract_screen("damaged.mp4", temp, sample_seconds=1)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["events"][0]["end"], 1.0)
        self.assertFalse(result["stats"]["decode_complete"])
        self.assertTrue(any("incomplete" in item for item in result["limitations"]))

    def test_missing_decoder_or_unusable_timestamps_cannot_report_ready(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(screen, "probe_media", return_value={"duration": 10, "video": [{}]}), \
                patch.object(screen, "_exe", return_value="ffmpeg"), \
                patch.object(screen, "_iter_video_frames", return_value=iter([
                    (None, {"decoder": "pyav-native", "decode_error": True,
                            "decoder_unavailable": True, "error": "PyAV unavailable",
                            "coverage_seconds": 0.0})])):
            result = screen.extract_screen("video.webm", temp)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["stats"]["decode_complete"])
        self.assertTrue(any("PyAV is unavailable" in item for item in result["limitations"]))

    def test_frame_crop_and_traversal_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frames" / "shot.jpg"
            path.parent.mkdir()
            Image.fromarray(np.full((80, 120, 3), 100, dtype=np.uint8)).save(path)
            payload = screen.frame_image(temp, {"frame": "frames/shot.jpg"}, crop=[10, 5, 40, 30])
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape[:2], (30, 40))
            with self.assertRaises(ValueError):
                screen.frame_image(temp, {"frame": "../outside.jpg"})

    @unittest.skipUnless(screen._exe("ffmpeg"), "FFmpeg is optional on developer machines")
    def test_unicode_path_ffmpeg_pipe_can_decode_synthetic_video(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "встреча"
            folder.mkdir()
            video = folder / "экран.mp4"
            frame = np.full((64, 96, 3), 120, dtype=np.uint8)
            encoded = cv2.imencode(".jpg", frame)[1].tobytes()
            # Supply the JPEG over stdin to avoid any dependency on a local fixture file.
            proc = subprocess.run(
                [screen._exe("ffmpeg"), "-v", "error", "-y", "-loop", "1", "-f", "image2pipe",
                 "-framerate", "1", "-i", "pipe:0", "-t", "4", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-r", "1", str(video)],
                input=encoded, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=20, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
            decoded = screen._decode_frame(video, 0)
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape[:2], (64, 96))
            episode = screen.extract_episode_frames(video, 1, 3, step=1, limit=2)
            self.assertEqual([frame["time"] for frame in episode], [1.0, 2.0])
            self.assertTrue(all(frame["image"].startswith(b"\xff\xd8") for frame in episode))
            self.assertTrue(all(frame["width"] <= 1920 and frame["height"] <= 1080 for frame in episode))


if __name__ == "__main__":
    unittest.main()
