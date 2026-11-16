from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import meeting_model_variants as variants


class MeetingModelVariantTests(unittest.TestCase):
    def test_status_is_read_only_and_does_not_import_conversion_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.onnx"
            source.write_bytes(b"original-model")
            requirements = root / "conversion.txt"
            requirements.write_text("onnx==1.20.1\nml_dtypes==0.5.4\n", encoding="utf-8")
            libs = root / "converter-libs"
            with patch.object(variants, "SOURCE_MODEL", source), \
                    patch.object(variants, "VARIANTS_ROOT", root / "variants"), \
                    patch.object(variants, "CONVERTER_LIBS", libs), \
                    patch.object(variants, "CONVERTER_REQUIREMENTS", requirements), \
                    patch.object(variants.subprocess, "run") as run:
                before = set(sys.modules)
                status = variants.conversion_status()
                loaded = set(sys.modules) - before

            self.assertFalse(status["ready"])
            self.assertEqual(status["estimated_download_bytes"], variants.CONVERTER_REQUIREMENTS_BYTES)
            self.assertFalse(libs.exists())
            run.assert_not_called()
            self.assertNotIn("onnx", loaded)
            self.assertNotIn("ml_dtypes", loaded)

    def test_conversion_failure_keeps_source_and_does_not_publish_partial_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.onnx"
            original = b"unchanged source model"
            source.write_bytes(original)
            requirements = root / "conversion.txt"
            requirements.write_text("pins", encoding="utf-8")
            model_root = root / "variants"
            libs = root / "converter-libs"
            failed = subprocess.CompletedProcess([], 1, "conversion error", "")
            calls = []
            def run(args, **_kwargs):
                calls.append(args)
                if len(calls) == 1:
                    for marker in ("onnx", "onnx-1.20.1.dist-info",
                                   "ml_dtypes", "ml_dtypes-0.5.4.dist-info"):
                        (libs / marker).mkdir(parents=True, exist_ok=True)
                    return subprocess.CompletedProcess([], 0, "installed", "")
                return failed
            with patch.object(variants, "SOURCE_MODEL", source), \
                    patch.object(variants, "VARIANTS_ROOT", model_root), \
                    patch.object(variants, "CONVERTER_LIBS", libs), \
                    patch.object(variants, "CONVERTER_REQUIREMENTS", requirements), \
                    patch.object(variants.subprocess, "run", side_effect=run):
                result = variants.prepare_fp16_variant(root / "python.exe")

            candidate, _receipt = variants.variant_paths(hashlib.sha256(original).hexdigest())
            self.assertFalse(result["ready"])
            self.assertEqual(result["reason"], "conversion_failed")
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(candidate.exists())
            install_args = calls[0]
            self.assertIn("--no-deps", install_args)
            self.assertEqual(install_args[install_args.index("--target") + 1], str(libs))

    def test_wrong_converter_versions_are_not_reported_prepared(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.onnx"
            source.write_bytes(b"model")
            requirements = root / "conversion.txt"
            requirements.write_text("pins", encoding="utf-8")
            libs = root / "converter-libs"
            for marker in ("onnx", "onnx-1.19.0.dist-info",
                           "ml_dtypes", "ml_dtypes-0.5.3.dist-info"):
                (libs / marker).mkdir(parents=True)
            with patch.object(variants, "SOURCE_MODEL", source), \
                    patch.object(variants, "VARIANTS_ROOT", root / "variants"), \
                    patch.object(variants, "CONVERTER_LIBS", libs), \
                    patch.object(variants, "CONVERTER_REQUIREMENTS", requirements):
                status = variants.conversion_status()
            self.assertFalse(status["converter_ready"])
            self.assertGreater(status["estimated_download_bytes"], 0)

    def test_prepare_reuses_exact_converter_versions_without_reinstalling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.onnx"
            source.write_bytes(b"source")
            requirements = root / "conversion.txt"
            requirements.write_text("pins", encoding="utf-8")
            libs = root / "converter-libs"
            for marker in ("onnx", "onnx-1.20.1.dist-info",
                           "ml_dtypes", "ml_dtypes-0.5.4.dist-info"):
                (libs / marker).mkdir(parents=True)

            def run(args, **_kwargs):
                destination = Path(args[-1])
                destination.write_bytes(b"converted fp16")
                return subprocess.CompletedProcess([], 0, "", "")

            with patch.object(variants, "SOURCE_MODEL", source), \
                    patch.object(variants, "VARIANTS_ROOT", root / "variants"), \
                    patch.object(variants, "CONVERTER_LIBS", libs), \
                    patch.object(variants, "CONVERTER_REQUIREMENTS", requirements), \
                    patch.object(variants.subprocess, "run", side_effect=run) as mocked_run:
                result = variants.prepare_fp16_variant(root / "python.exe")

            self.assertTrue(result["ready"])
            mocked_run.assert_called_once()
            self.assertEqual(mocked_run.call_args.args[0][1], "-c")

    def test_cold_model_preload_precedes_conversion_and_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "missing-rec.onnx"
            events = []

            def preload(_python):
                events.append("preload")
                source.write_bytes(b"downloaded official source")
                return {"ready": True}

            def convert(_python):
                events.append("convert")
                return {"ready": True, "model_path": "variant.onnx", "model_sha256": "sha"}

            def calibrate(*, progress, fp16_variant):
                events.append("calibrate")
                self.assertEqual(fp16_variant["model_sha256"], "sha")
                return {"verified": False}

            progress = []
            with patch.object(variants, "SOURCE_MODEL", source), \
                    patch.object(variants, "preload_rec_model", side_effect=preload), \
                    patch.object(variants, "prepare_fp16_variant", side_effect=convert):
                variant, calibration = variants.prepare_variants_and_calibrate(
                    "gpu-python", calibrate, progress.append)

        self.assertEqual(events, ["preload", "convert", "calibrate"])
        self.assertEqual(progress[0]["stage"], "models")
        self.assertEqual(progress[-1]["stage"], "calibration")
        self.assertTrue(variant["ready"])
        self.assertFalse(calibration["verified"])

    def test_variant_receipt_and_content_hash_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.onnx"
            source.write_bytes(b"FP32")
            requirements = root / "conversion.txt"
            requirements.write_text("pins-v1", encoding="utf-8")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            variant_root = root / "variants"
            model_path = variant_root / f"{source_hash}.onnx"
            receipt_path = variant_root / f"{source_hash}.json"
            model_path.parent.mkdir()
            model_path.write_bytes(b"FP16")
            receipt_path.write_text(json.dumps({
                "schema": 1,
                "source_sha256": source_hash,
                "model_sha256": hashlib.sha256(b"FP16").hexdigest(),
                "converter_requirements_sha256": hashlib.sha256(b"pins-v1").hexdigest(),
            }), encoding="utf-8")
            model_sha = hashlib.sha256(b"FP16").hexdigest()
            with patch.object(variants, "SOURCE_MODEL", source), \
                    patch.object(variants, "VARIANTS_ROOT", variant_root), \
                    patch.object(variants, "CONVERTER_REQUIREMENTS", requirements):
                self.assertTrue(variants.validate_fp16_variant(model_path, model_sha))
                model_path.write_bytes(b"tampered")
                self.assertFalse(variants.validate_fp16_variant(model_path, model_sha))


if __name__ == "__main__":
    unittest.main()
