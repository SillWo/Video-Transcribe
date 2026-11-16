import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import meeting_hardware as hardware
from services import meeting_model_variants as variants
from services import runtime_devices


class MeetingHardwareTests(unittest.TestCase):
    def test_dxgi_order_and_unique_wmi_identity_select_amd_directml_ordinal(self):
        dxgi = [
            {"name": "Intel Graphics", "vendorId": 0x8086, "deviceId": 10,
             "deviceIndex": 0, "luid": "intel-luid"},
            {"name": "AMD Radeon", "vendorId": 0x1002, "deviceId": 20,
             "deviceIndex": 1, "luid": "amd-luid"},
        ]
        wmi = [
            {"name": "AMD Radeon", "vendorId": 0x1002, "deviceId": 20,
             "driverVersion": "31.0", "pnpDeviceId": "PCI\\VEN_1002&DEV_0014"},
            {"name": "Intel Graphics", "vendorId": 0x8086, "deviceId": 10,
             "driverVersion": "32.0", "pnpDeviceId": "PCI\\VEN_8086&DEV_000A"},
        ]
        with patch.object(runtime_devices, "_dxgi_gpu_inventory", return_value=dxgi), \
                patch.object(runtime_devices, "_windows_gpu_inventory", return_value=wmi), \
                patch.object(runtime_devices, "enumerate_vulkan_gpus", return_value=[]):
            identity = hardware._identity()
            with tempfile.TemporaryDirectory() as temp:
                python = Path(temp) / "python.exe"
                python.write_bytes(b"python")
                with patch.object(hardware, "GPU_PYTHON", python), \
                        patch.object(hardware, "_gpu_component_state", return_value=(True, [])), \
                        patch.object(hardware, "_receipt_matches", return_value=False):
                    plan = hardware.plan_stage("screen", "manual", "gpu")
        self.assertEqual(identity["name"], "AMD Radeon")
        self.assertEqual(identity["device_index"], 1)
        self.assertTrue(identity["driver_identity_verified"])
        self.assertEqual(plan["env"]["VT_MEETING_DML_DEVICE_ID"], "1")

    def test_vulkan_ordinal_never_replaces_dxgi_directml_ordinal(self):
        with patch.object(runtime_devices, "_dxgi_gpu_inventory", return_value=[
            {"name": "AMD Radeon", "vendorId": 0x1002, "deviceId": 20,
             "deviceIndex": 1, "luid": "amd-luid"}]), \
                patch.object(runtime_devices, "_windows_gpu_inventory", return_value=[
                    {"name": "AMD Radeon", "vendorId": 0x1002, "deviceId": 20,
                     "driverVersion": "31.0", "pnpDeviceId": "PCI\\VEN_1002&DEV_0014"}]), \
                patch.object(runtime_devices, "enumerate_vulkan_gpus", return_value=[
                    {"name": "AMD Radeon", "vendorId": 0x1002, "deviceId": 20, "deviceIndex": 7}]):
            self.assertEqual(hardware._identity()["device_index"], 1)

    def test_gpu_without_dxgi_index_stays_cpu_and_manual_and_calibration_explain(self):
        with patch.object(runtime_devices, "_dxgi_gpu_inventory", return_value=[]), \
                patch.object(runtime_devices, "_windows_gpu_inventory", return_value=[
                    {"name": "AMD Radeon", "vendorId": 0x1002, "deviceId": 20,
                     "driverVersion": "31.0", "pnpDeviceId": "PCI\\VEN_1002&DEV_0014"}]), \
                patch.object(runtime_devices, "enumerate_vulkan_gpus", return_value=[
                    {"name": "AMD Radeon", "vendorId": 0x1002, "deviceId": 20, "deviceIndex": 7}]):
            auto = hardware.plan_stage("screen", "auto", "gpu")
            manual = hardware.plan_stage("screen", "manual", "gpu")
            with patch.object(hardware, "GPU_PYTHON", Path("prepared-python.exe")), \
                    patch.object(hardware, "write_screen_receipt") as write_receipt:
                calibration = hardware.calibrate_screen_runtime()
        self.assertEqual(auto["device"], "cpu")
        self.assertTrue(auto["verified"])
        self.assertEqual(manual["error"], "gpu_device_index_unavailable")
        self.assertIn("индекс", manual["reason"])
        self.assertFalse(calibration["verified"])
        write_receipt.assert_not_called()

    def test_ambiguous_wmi_driver_rows_do_not_verify_receipt_identity(self):
        dxgi = [{"name": "AMD Radeon", "vendorId": 0x1002, "deviceId": 20,
                 "deviceIndex": 0, "luid": "amd-luid"}]
        duplicate = {"name": "AMD Radeon", "vendorId": 0x1002, "deviceId": 20,
                     "driverVersion": "31.0", "pnpDeviceId": "PCI\\VEN_1002&DEV_0014"}
        with patch.object(runtime_devices, "_dxgi_gpu_inventory", return_value=dxgi), \
                patch.object(runtime_devices, "_windows_gpu_inventory", return_value=[duplicate, duplicate]), \
                patch.object(runtime_devices, "enumerate_vulkan_gpus", return_value=[]):
            identity = hardware._identity()
        self.assertEqual(identity["device_index"], 0)
        self.assertFalse(identity["driver_identity_verified"])
        self.assertFalse(hardware._receipt_matches(identity))

    def test_single_amd_dxgi_zero_remains_a_valid_target(self):
        with patch.object(runtime_devices, "_dxgi_gpu_inventory", return_value=[
            {"name": "AMD Radeon", "vendorId": 0x1002, "deviceId": 20,
             "deviceIndex": 0, "luid": "amd-luid"}]), \
                patch.object(runtime_devices, "_windows_gpu_inventory", return_value=[
                    {"name": "AMD Radeon", "vendorId": 0x1002, "deviceId": 20,
                     "driverVersion": "31.0", "pnpDeviceId": "PCI\\VEN_1002&DEV_0014"}]), \
                patch.object(runtime_devices, "enumerate_vulkan_gpus", return_value=[]):
            identity = hardware._identity()
        self.assertTrue(hardware._has_dml_target(identity))
        self.assertEqual(identity["device_index"], 0)

    def test_auto_stays_on_cpu_without_an_exact_verified_receipt(self):
        with patch.object(hardware, "_receipt_matches", return_value=False), \
                patch.object(hardware, "_identity", return_value={"name": "Radeon 780M"}):
            plan = hardware.plan_stage("screen", "auto", "gpu")
        self.assertEqual(plan["device"], "cpu")
        self.assertTrue(plan["verified"])
        self.assertEqual(plan["python"], hardware.sys.executable)
        self.assertEqual(plan["env"]["VT_MEETING_OCR_REC_PRECISION"], "fp32")
        self.assertEqual(plan["env"]["VT_MEETING_OCR_BATCHING"], "stock")
        self.assertEqual(plan["env"]["VT_MEETING_OCR_REC_MODEL_PATH"], "")
        self.assertEqual(plan["env"]["VT_MEETING_OCR_REC_MODEL_SHA256"], "")

    def test_manual_gpu_keeps_requested_device_and_reports_missing_runtime(self):
        with patch.object(hardware, "GPU_PYTHON", Path("missing-directml-python.exe")), \
                patch.object(hardware, "_identity", return_value={"name": "Radeon 780M", "vendor_id": 0x1002,
                                                                    "device_index": 0}):
            plan = hardware.plan_stage("screen", "manual", "gpu")
        self.assertEqual(plan["device"], "directml")
        self.assertEqual(plan["error"], "gpu_runtime_missing")

    def test_missing_av_is_incomplete_runtime_and_manual_gpu_reports_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
            (root / "Lib" / "site-packages").mkdir(parents=True)
            identity = {"name": "Radeon 780M", "vendor_id": 0x1002, "device_index": 0}
            with patch.object(hardware, "GPU_PYTHON", python), \
                    patch.object(hardware, "_identity", return_value=identity), \
                    patch.object(hardware, "_receipt_matches", return_value=False), \
                    patch.object(hardware, "_model_manifest", return_value=[["model.onnx", 1, 1]]), \
                    patch("services.meeting_model_variants.conversion_status", return_value={
                        "estimated_download_bytes": 123, "ready": False,
                    }), \
                    patch("services.meeting_settings.load_settings", return_value={
                        "execution_mode": "auto", "device": "cpu"}):
                status = hardware.get_hardware_status()
                manual = hardware.plan_stage("screen", "manual", "gpu")

        self.assertFalse(status["gpu_runtime"]["prepared"])
        self.assertEqual(status["gpu_runtime"]["estimated_download_bytes"], hardware.AV_WHEEL_BYTES + 123)
        self.assertFalse(status["stages"]["screen"]["gpu"]["prepared"])
        self.assertTrue(status["stages"]["screen"]["gpu"]["can_prepare"])
        self.assertEqual(status["stages"]["diarization"], {
            "auto_device": "cpu",
            "reason": "Нет сохранённого сопоставимого замера скорости и качества диаризации.",
        })
        self.assertEqual(manual["error"], "gpu_runtime_missing")
        self.assertEqual(manual["reason"],
                         "Не подготовлен компонент чтения видео. Подготовьте среду GPU в настройках.")

    def test_av_dist_info_is_sufficient_for_prepared_component_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
            site = root / "Lib" / "site-packages"
            site.mkdir(parents=True)
            for pattern, exact_name in hardware._GPU_COMPONENT_PATTERNS:
                dist = site / (exact_name or pattern.replace("*", "3.10.0"))
                dist.mkdir()
            with patch.object(hardware, "GPU_PYTHON", python):
                prepared, missing = hardware._gpu_component_state()

        self.assertTrue(prepared)
        self.assertEqual(missing, [])

    def test_manual_gpu_allows_prepared_runtime_without_an_auto_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            python = Path(temp) / "python.exe"
            python.write_bytes(b"python")
            with patch.object(hardware, "GPU_PYTHON", python), \
                    patch.object(hardware, "_gpu_component_state", return_value=(True, [])), \
                    patch.object(hardware, "_identity", return_value={"name": "Radeon 780M", "vendor_id": 0x1002,
                                                                        "device_index": 0}), \
                    patch.object(hardware, "_receipt_matches", return_value=False):
                plan = hardware.plan_stage("screen", "manual", "gpu")
        self.assertEqual(plan["device"], "directml")
        self.assertFalse(plan["verified"])
        self.assertIsNone(plan["error"])
        self.assertEqual(plan["env"]["VT_MEETING_OCR_REC_PRECISION"], "fp32")
        self.assertEqual(plan["env"]["VT_MEETING_OCR_BATCHING"], "stock")
        self.assertIn("экспериментальным", plan["reason"])

    def test_verified_fp32_receipt_passes_selected_variant_to_worker_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / "python.exe"
            python.write_bytes(b"python")
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps({
                "selected_variant": {
                    "precision": "fp32", "batching": "equal-width",
                }
            }), encoding="utf-8")
            with patch.object(hardware, "GPU_PYTHON", python), \
                    patch.object(hardware, "RECEIPT", receipt_path), \
                    patch.object(hardware, "_identity", return_value={
                        "name": "Radeon", "vendor_id": 0x1002, "device_index": 0}), \
                    patch.object(hardware, "_gpu_component_state", return_value=(True, [])), \
                    patch.object(hardware, "_receipt_matches", return_value=True):
                plan = hardware.plan_stage("screen", "auto", "gpu")

        self.assertEqual(plan["device"], "directml")
        self.assertEqual(plan["env"]["VT_MEETING_OCR_REC_PRECISION"], "fp32")
        self.assertEqual(plan["env"]["VT_MEETING_OCR_BATCHING"], "equal-width")
        self.assertEqual(plan["env"]["VT_MEETING_OCR_REC_MODEL_SHA256"], "")

    def test_fp16_hardware_receipt_is_not_auto_verified(self):
        import hashlib

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.onnx"
            source.write_bytes(b"fp32 model")
            requirements = root / "conversion.txt"
            requirements.write_text("pins", encoding="utf-8")
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            variant_root = root / "variants"
            model_path = variant_root / f"{source_sha}.onnx"
            receipt_artifact = variant_root / f"{source_sha}.json"
            model_path.parent.mkdir()
            model_path.write_bytes(b"fp16 model")
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            converter_sha = hashlib.sha256(b"pins").hexdigest()
            receipt_artifact.write_text(json.dumps({
                "schema": 1, "source_sha256": source_sha, "model_sha256": model_sha,
                "converter_requirements_sha256": converter_sha,
            }), encoding="utf-8")
            identity = {"name": "Radeon", "vendor_id": 0x1002, "device_id": 1,
                        "device_index": 0, "identity": "luid", "driver": "1.0",
                        "pnp_device_id": "PCI\\VEN_1002", "driver_identity_verified": True}
            python = root / "python.exe"
            python.write_bytes(b"python")
            requirements = root / "gpu-requirements.txt"
            requirements.write_text("gpu pins", encoding="utf-8")
            conversion_requirements = root / "conversion-requirements.txt"
            conversion_requirements.write_text("pins", encoding="utf-8")
            receipt_path = root / "screen-receipt.json"
            receipt_path.write_text(json.dumps({
                "schema": 2, "verified": True, "identity": identity,
                "python": str(python.resolve()),
                "requirements_hash": hashlib.sha256(b"gpu pins").hexdigest(),
                "model_manifest": [["model.onnx", 1, 2]], "model_hash": "model-hash",
                "runtime_signature": {"runtime": 1}, "workload": hardware.WORKLOAD,
                "selected_variant": {
                    "precision": "fp16", "batching": "stock",
                    "model_path": str(model_path), "model_sha256": model_sha,
                },
            }), encoding="utf-8")
            with patch.object(hardware, "RECEIPT", receipt_path), \
                    patch.object(hardware, "GPU_PYTHON", python), \
                    patch.object(hardware, "GPU_REQUIREMENTS", requirements), \
                    patch.object(hardware, "_identity", return_value=identity), \
                    patch.object(hardware, "_requirements_hash", return_value=hashlib.sha256(b"gpu pins").hexdigest()), \
                    patch.object(hardware, "_model_manifest", return_value=[["model.onnx", 1, 2]]), \
                    patch.object(hardware, "_model_hash", return_value="model-hash"), \
                    patch.object(hardware, "_runtime_signature", return_value={"runtime": 1}), \
                    patch.object(variants, "SOURCE_MODEL", source), \
                    patch.object(variants, "VARIANTS_ROOT", variant_root), \
                    patch.object(variants, "CONVERTER_REQUIREMENTS", conversion_requirements):
                self.assertFalse(hardware._receipt_matches(identity))
                model_path.write_bytes(b"tampered")
                self.assertFalse(hardware._receipt_matches(identity))

    def test_fp16_is_experimental_even_when_all_short_calibration_gates_pass(self):
        identity = {"name": "Radeon", "vendor_id": 0x1002, "device_id": 1,
                    "device_index": 0, "identity": "luid", "driver": "1.0",
                    "pnp_device_id": "PCI\\VEN_1002", "driver_identity_verified": True}
        regions = [{"text": "ARCH-427", "bbox": [1, 2, 80, 20]},
                   {"text": "Limit: 25", "bbox": [1, 32, 100, 20]}]
        control_regions = [{"text": "ARCH-427"}, {"text": "Limit: 25"}]
        control = {"regions": control_regions, "full_reference_regions": control_regions,
                   "anchored_rec_regions": 1}
        cpu = {"regions": regions, "elapsed_seconds": .2, "anchored_control": control}

        def candidate(precision):
            return {"regions": regions, "elapsed_seconds": .1,
                    "operator_profile": {"dml_operator_count": 10},
                    "variant": {"precision": precision, "batching": "stock"},
                    "anchored_control": control}

        fp16 = hardware._assess_screen_candidate(identity, cpu, candidate("fp16"))
        fp32 = hardware._assess_screen_candidate(identity, cpu, candidate("fp32"))

        self.assertTrue(fp16["experimental"])
        self.assertFalse(fp16["verified"])
        self.assertEqual(fp16["reason"],
                         "FP16 requires representative critical-value acceptance; real pilot found numeric regressions")
        self.assertTrue(fp32["verified"])
        self.assertFalse(fp32["experimental"])

    def test_calibration_selects_fastest_verified_candidate_not_faster_bad_candidate(self):
        identity = {"name": "Radeon", "vendor_id": 0x1002, "device_id": 1,
                    "device_index": 0, "identity": "luid", "driver": "1.0",
                    "pnp_device_id": "PCI\\VEN_1002", "driver_identity_verified": True}
        cpu_regions = [{"text": "ARCH-427", "bbox": [1, 2, 80, 20]},
                       {"text": "Limit: 25", "bbox": [1, 32, 100, 20]}]
        control = {"regions": [{"text": "ARCH-427"}, {"text": "Limit: 25"}],
                   "full_reference_regions": [{"text": "ARCH-427"}, {"text": "Limit: 25"}],
                   "anchored_rec_regions": 1}
        cpu = {"regions": cpu_regions, "elapsed_seconds": .2, "anchored_control": control}

        def candidate(precision, batching, elapsed, regions, anchored=None, merged=False):
            batch_size = 8 if merged else 6
            return {"regions": regions, "elapsed_seconds": elapsed,
                    "operator_profile": {"dml_operator_count": 20},
                    "variant": {"precision": precision, "batching": batching},
                    "batch_proof": {
                        "stock_rec_batch_num": 6,
                        "max_batch": 48 if merged else 6,
                        "session_batch_sizes": [batch_size],
                        "max_session_batch_size": batch_size,
                        "equal_width_merge_exercised": merged,
                    },
                    "anchored_control": anchored or control}

        candidates = [
            candidate("fp16", "equal-width", .04,
                      [{"text": "ARCH-999", "bbox": [1, 2, 80, 20]},
                       {"text": "Limit: 27", "bbox": [1, 32, 100, 20]}],
                      {"regions": [{"text": "ARCH-999"}, {"text": "Limit: 27"}],
                       "full_reference_regions": control["full_reference_regions"],
                       "anchored_rec_regions": 1}, merged=True),
            candidate("fp32", "equal-width", .03, cpu_regions),
            candidate("fp32", "stock", .11, cpu_regions),
            candidate("fp16", "stock", .001, cpu_regions),
            candidate("fp32", "equal-width", .08, cpu_regions, merged=True),
        ]
        payload = {"cpu": cpu, "candidates": candidates,
                   "fp16_unavailable": None}
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="MEETING_OCR_CALIBRATION=" + json.dumps(payload, ensure_ascii=False),
            stderr="")
        receipt = {"verified": True, "selected_variant": candidates[-1]["variant"],
                   "cpu_seconds": .2, "gpu_seconds": .08}
        with tempfile.TemporaryDirectory() as temp:
            python = Path(temp) / "python.exe"
            python.write_bytes(b"python")
            with patch.object(hardware, "GPU_PYTHON", python), \
                    patch.object(hardware, "_identity", return_value=identity), \
                    patch.object(hardware, "_model_hash", return_value="model-sha"), \
                    patch.object(hardware, "_runtime_signature", return_value={"packages": []}), \
                    patch.object(hardware, "write_screen_receipt", return_value=receipt) as write_receipt, \
                    patch.object(hardware.subprocess, "run", return_value=completed):
                result = hardware.calibrate_screen_runtime(
                    fp16_variant={"ready": True, "model_path": "variant.onnx", "model_sha256": "fp16-sha"})

        self.assertTrue(result["verified"])
        self.assertEqual(result["selected_variant"], candidates[-1]["variant"])
        self.assertEqual(write_receipt.call_args.kwargs["gpu_result"]["elapsed_seconds"], .08)
        self.assertEqual(write_receipt.call_args.kwargs["variant"], candidates[-1]["variant"])
        summaries = write_receipt.call_args.kwargs["candidates"]
        self.assertFalse(summaries[0]["assessment"]["verified"])
        self.assertFalse(summaries[1]["assessment"]["verified"])
        self.assertFalse(summaries[1]["assessment"]["equal_width_merge_exercised"])
        self.assertTrue(summaries[2]["assessment"]["verified"])
        self.assertFalse(summaries[3]["assessment"]["verified"])
        self.assertTrue(summaries[3]["assessment"]["experimental"])
        self.assertEqual(summaries[3]["assessment"]["reason"],
                         "FP16 requires representative critical-value acceptance; real pilot found numeric regressions")
        self.assertTrue(summaries[4]["assessment"]["verified"])
        self.assertTrue(summaries[4]["assessment"]["equal_width_merge_exercised"])
        self.assertEqual(summaries[4]["batch_proof"]["session_batch_sizes"], [8])
        self.assertEqual(summaries[4]["batch_proof"]["max_batch"], 48)

    def test_manual_opus_audio_stage_stays_cpu_with_explicit_reason(self):
        plan = hardware.plan_stage("audio", "manual", "gpu")
        self.assertEqual(plan["device"], "cpu")
        self.assertIsNone(plan["error"])
        self.assertIn("Opus", plan["reason"])

    def test_manual_directml_targets_the_discovered_adapter_id(self):
        with tempfile.TemporaryDirectory() as temp:
            python = Path(temp) / "python.exe"
            python.write_bytes(b"python")
            with patch.object(hardware, "GPU_PYTHON", python), \
                    patch.object(hardware, "_gpu_component_state", return_value=(True, [])), \
                    patch.object(hardware, "_identity", return_value={"name": "Radeon 780M", "vendor_id": 0x1002,
                                                                        "device_index": 2}), \
                    patch.object(hardware, "_receipt_matches", return_value=False):
                plan = hardware.plan_stage("screen", "manual", "gpu")
        self.assertEqual(plan["device"], "directml")
        self.assertEqual(plan["env"]["VT_MEETING_DML_DEVICE_ID"], "2")

    def test_hardware_api_import_does_not_load_numpy_or_onnxruntime(self):
        probe = subprocess.run(
            [hardware.sys.executable, "-c",
             "import sys; import services.meeting_hardware_api; "
             "print('numpy' in sys.modules, 'onnxruntime' in sys.modules, 'cv2' in sys.modules)"],
            cwd=hardware.APP_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=20, check=False)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(probe.stdout.strip(), "False False False")

    def test_auto_audio_and_index_are_cpu_without_quality_speed_receipts(self):
        for stage in ("audio", "index"):
            with self.subTest(stage=stage):
                plan = hardware.plan_stage(stage, "auto", "gpu")
                self.assertEqual(plan["device"], "cpu")

    def test_receipt_requires_exact_text_ids_dml_ops_and_speedup(self):
        cpu = {"regions": [{"text": "ARCH-427", "bbox": [1, 2, 80, 20]},
                            {"text": "Значение 24", "bbox": [1, 32, 100, 20]}],
               "elapsed_seconds": 2.0}
        gpu = {"regions": [{"text": "ARCH-427", "bbox": [1, 2, 80, 20]},
                            {"text": "Значение 24", "bbox": [1, 32, 100, 20]}],
               "elapsed_seconds": 1.0,
               "operator_profile": {"dml_operator_count": 50}}
        control_reference = [{"text": "ARCH-427"}, {"text": "Лимит повторов: 27"}]
        cpu["anchored_control"] = {"regions": control_reference,
                                    "full_reference_regions": control_reference,
                                    "anchored_rec_regions": 1}
        gpu["anchored_control"] = {"regions": control_reference,
                                    "full_reference_regions": control_reference,
                                    "anchored_rec_regions": 1}
        with tempfile.TemporaryDirectory() as temp:
            receipt_path = Path(temp) / "receipt.json"
            with patch.object(hardware, "RECEIPT", receipt_path), \
                    patch.object(hardware, "GPU_PYTHON", Path(temp) / "py.exe"), \
                    patch.object(hardware, "GPU_REQUIREMENTS", Path(temp) / "req.txt"), \
                    patch.object(hardware, "_runtime_signature", return_value={"signature": 1}), \
                    patch.object(hardware, "_model_manifest", return_value=[["m.onnx", 10, 1]]), \
                    patch.object(hardware, "_model_hash", return_value="abc"):
                Path(temp, "req.txt").write_text("pins", encoding="utf-8")
                identity = {"name": "Radeon", "vendor_id": 0x1002, "device_id": 1,
                            "device_index": 0, "identity": "luid", "driver": "1.0",
                            "pnp_device_id": "PCI\\VEN_1002", "driver_identity_verified": True}
                accepted = hardware.write_screen_receipt(identity=identity, model_hash="abc",
                                                         runtime="ort", workload=hardware.WORKLOAD,
                                                         cpu_result=cpu, gpu_result=gpu)
                self.assertTrue(accepted["verified"])
                self.assertEqual(json.loads(receipt_path.read_text(encoding="utf-8"))["model_hash"], "abc")
                rejected = hardware.write_screen_receipt(
                    identity=identity, model_hash="abc", runtime="ort", workload=hardware.WORKLOAD,
                    cpu_result={**cpu, "elapsed_seconds": .5}, gpu_result=gpu,
                    output_path=Path(temp) / "rejected.json")
                self.assertFalse(rejected["verified"])
                no_control = {key: value for key, value in cpu.items() if key != "anchored_control"}
                rejected_control = hardware.write_screen_receipt(
                    identity=identity, model_hash="abc", runtime="ort", workload=hardware.WORKLOAD,
                    cpu_result=no_control, gpu_result=gpu,
                    output_path=Path(temp) / "missing-control.json")
                self.assertFalse(rejected_control["verified"])
                self.assertFalse(rejected_control["anchored_control_passed"])
                wrong_value = {**gpu, "anchored_control": {
                    **gpu["anchored_control"],
                    "regions": [{"text": "ARCH-427"}, {"text": "Лимит повторов: 26"}]}}
                rejected_value = hardware.write_screen_receipt(
                    identity=identity, model_hash="abc", runtime="ort", workload=hardware.WORKLOAD,
                    cpu_result=cpu, gpu_result=wrong_value,
                    output_path=Path(temp) / "wrong-control-value.json")
                self.assertFalse(rejected_value["verified"])
                self.assertFalse(rejected_value["anchored_control_passed"])

    def test_persisted_json_receipt_matches_after_list_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / "python.exe"
            python.write_bytes(b"python")
            requirements = root / "requirements.txt"
            requirements.write_text("pins", encoding="utf-8")
            receipt_path = root / "receipt.json"
            identity = {"name": "Radeon", "vendor_id": 0x1002, "device_id": 1,
                        "device_index": 0, "identity": "luid", "driver": "1.0",
                        "pnp_device_id": "PCI\\VEN_1002", "driver_identity_verified": True}
            cpu = {"regions": [{"text": "ARCH-427", "bbox": [1, 2, 80, 20]}],
                   "elapsed_seconds": 2.0}
            gpu = {"regions": [{"text": "ARCH-427", "bbox": [1, 2, 80, 20]}],
                   "elapsed_seconds": 1.0,
                   "operator_profile": {"dml_operator_count": 12}}
            control_reference = [{"text": "ARCH-427"}, {"text": "Лимит повторов: 27"}]
            cpu["anchored_control"] = {"regions": control_reference,
                                        "full_reference_regions": control_reference,
                                        "anchored_rec_regions": 1}
            gpu["anchored_control"] = {"regions": control_reference,
                                        "full_reference_regions": control_reference,
                                        "anchored_rec_regions": 1}
            signature = {"packages": [["rapidocr-3.10.0.dist-info", 20, 3, "hash"]]}
            manifest = [["model.onnx", 40, 2]]
            with patch.object(hardware, "RECEIPT", receipt_path), \
                    patch.object(hardware, "GPU_PYTHON", python), \
                    patch.object(hardware, "GPU_REQUIREMENTS", requirements), \
                    patch.object(hardware, "_runtime_signature", return_value=signature), \
                    patch.object(hardware, "_model_manifest", return_value=manifest), \
                    patch.object(hardware, "_model_hash", return_value="model-sha"):
                receipt = hardware.write_screen_receipt(
                    identity=identity, model_hash="model-sha", runtime="ort", workload=hardware.WORKLOAD,
                    cpu_result=cpu, gpu_result=gpu)
                self.assertTrue(receipt["verified"])
                self.assertTrue(hardware._receipt_matches(identity))
                old_workload = json.loads(receipt_path.read_text(encoding="utf-8"))
                old_workload["workload"] = "rapidocr-ppocrv5-mobile-cyrillic;screen-roi-v4-native16"
                receipt_path.write_text(json.dumps(old_workload), encoding="utf-8")
                self.assertFalse(hardware._receipt_matches(identity))
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                self.assertTrue(hardware._receipt_matches(identity))

    def test_runtime_signature_tracks_each_package_and_missing_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / "directml" / ("Scripts/python.exe" if hardware.os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
            site = (python.parent.parent / "Lib" / "site-packages" if hardware.os.name == "nt"
                    else python.parent.parent / "lib")
            site.mkdir(parents=True)
            for name in ("rapidocr-3.10.0.dist-info", "onnxruntime_directml-1.24.4.dist-info",
                         "numpy-2.4.4.dist-info", "opencv_python_headless-4.14.dist-info",
                         "pillow-12.3.dist-info", "av-18.1.0.dist-info"):
                dist = site / name
                dist.mkdir()
                (dist / "METADATA").write_text("Version: one", encoding="utf-8")
            with patch.object(hardware, "GPU_PYTHON", python):
                first = hardware._runtime_signature()
                self.assertEqual(len(first["packages"]), 7)
                self.assertEqual(next(item for item in first["packages"]
                                      if item[0] == "av-18.1.0.dist-info")[1], 12)
                self.assertEqual(first["packages"][-1], ["tokenizers-*.dist-info", None])
                metadata = site / "rapidocr-3.10.0.dist-info" / "METADATA"
                metadata.write_text("Version: two", encoding="utf-8")
                second = hardware._runtime_signature()
            self.assertNotEqual(first["packages"], second["packages"])

    def test_calibration_subprocess_uses_utf8_and_preserves_cyrillic_json(self):
        with tempfile.TemporaryDirectory() as temp:
            python = Path(temp) / "python.exe"
            python.write_bytes(b"python")
            payload = {
                "cpu": {"regions": [{"text": "Значение 24", "bbox": [1, 2, 80, 20]}],
                        "elapsed_seconds": .2,
                        "anchored_control": {
                            "regions": [{"text": "Значение 24"}],
                            "full_reference_regions": [{"text": "Значение 24"}],
                            "anchored_rec_regions": 1}},
                "candidates": [{
                    "regions": [{"text": "Значение 24", "bbox": [1, 2, 80, 20]}],
                    "elapsed_seconds": .1,
                    "operator_profile": {"dml_operator_count": 12},
                    "variant": {"precision": "fp32", "batching": "stock"},
                    "anchored_control": {
                        "regions": [{"text": "Значение 24"}],
                        "full_reference_regions": [{"text": "Значение 24"}],
                        "anchored_rec_regions": 1}},
                ],
                "fp16_unavailable": "conversion_unavailable",
            }
            completed = subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout="diagnostic\nMEETING_OCR_CALIBRATION=" + json.dumps(payload, ensure_ascii=False),
                stderr="")
            receipt = {"verified": True, "cpu_seconds": .2, "gpu_seconds": .1}
            with patch.object(hardware, "GPU_PYTHON", python), \
                    patch.object(hardware, "_identity", return_value={
                        "vendor_id": 0x1002, "device_index": 3, "identity": "luid",
                        "driver": "1.0", "pnp_device_id": "PCI\\VEN_1002",
                        "driver_identity_verified": True}), \
                    patch.object(hardware, "_model_hash", return_value="model"), \
                    patch.object(hardware, "_runtime_signature", return_value={"packages": []}), \
                    patch.object(hardware, "write_screen_receipt", return_value=receipt) as write_receipt, \
                    patch.object(hardware.subprocess, "run", return_value=completed) as run:
                result = hardware.calibrate_screen_runtime()

        self.assertTrue(result["verified"])
        kwargs = run.call_args.kwargs
        calibration_script = run.call_args.args[0][2]
        compile(calibration_script, "<meeting-ocr-calibration>", "exec")
        self.assertIn('Image.new("RGB", (1440, 1100)', calibration_script)
        self.assertIn("ImageFont.truetype(candidate, 32)", calibration_script)
        self.assertIn("for row, text in enumerate(canvas_lines)", calibration_script)
        self.assertIn('"Q2026Q4"', calibration_script)
        self.assertIn("class SessionBatchRecorder", calibration_script)
        self.assertIn("session_batch_sizes.clear()", calibration_script)
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")
        self.assertEqual(kwargs["env"]["PYTHONUTF8"], "1")
        self.assertEqual(kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(kwargs["env"]["VT_MEETING_DML_DEVICE_ID"], "3")
        self.assertEqual(write_receipt.call_args.kwargs["cpu_result"]["regions"][0]["text"], "Значение 24")

    def test_calibration_missing_stdout_returns_unverified_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            python = Path(temp) / "python.exe"
            python.write_bytes(b"python")
            completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=None, stderr=None)
            with patch.object(hardware, "GPU_PYTHON", python), \
                    patch.object(hardware, "_identity", return_value={
                        "vendor_id": 0x1002, "device_index": 3, "identity": "luid",
                        "driver": "1.0", "pnp_device_id": "PCI\\VEN_1002",
                        "driver_identity_verified": True}), \
                    patch.object(hardware.subprocess, "run", return_value=completed), \
                    patch.object(hardware, "write_screen_receipt") as write_receipt:
                result = hardware.calibrate_screen_runtime()

        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "Calibration subprocess failed.")
        self.assertIn("no JSON output", result["error"])
        write_receipt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
