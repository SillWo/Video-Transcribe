import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from services import diarization_runtime as diarization
from services.task_errors import TaskFailure


def device(identifier, name, backend, index, available=True, reason=None):
    return {"id": identifier, "name": name, "backend": backend, "deviceIndex": index,
            "available": available, "reason": reason}


class DiarizationDeviceResolutionTests(unittest.TestCase):
    def setUp(self):
        self.model = Path("model.gguf")
        self.exe = {"cpu": "cpu.exe", "cuda": "cuda.exe", "vulkan": "vulkan.exe"}

    def resolve_mocks(self, devices, inventory):
        return (
            patch.object(diarization, "resolve_diarization_model", return_value=self.model),
            patch.object(diarization, "_inventory", return_value=inventory),
            patch.object(diarization, "_prepared", return_value=(devices, {})),
            patch.object(diarization, "find_diarization_executable", side_effect=lambda b: self.exe.get(b)),
        )

    def test_no_gpu_auto_selects_cpu(self):
        devices = [device("cpu", "CPU", "cpu", 0)]
        with ExitStack() as stack:
            for item in self.resolve_mocks(devices, []):
                stack.enter_context(item)
            result = diarization.resolve_diarization_device()
        self.assertEqual((result["id"], result["backend"], result["deviceIndex"]), ("cpu", "cpu", 0))

    def test_nvidia_uses_cuda_runtime_ordinal(self):
        gpu = {"id": "gpu-nv", "name": "RTX", "vendorId": 0x10DE, "cudaIndex": 1}
        devices = [device("gpu-nv", "RTX", "cuda", 1)]
        with self.resolve_mocks(devices, [gpu])[0], self.resolve_mocks(devices, [gpu])[1], \
                self.resolve_mocks(devices, [gpu])[2], self.resolve_mocks(devices, [gpu])[3]:
            result = diarization.resolve_diarization_device("gpu-nv")
        self.assertEqual(result["backend"], "cuda")
        self.assertEqual(result["deviceIndex"], 1)
        self.assertEqual(result["executable"], "cuda.exe")

    def test_amd_and_integrated_intel_use_vulkan_and_auto_prefers_discrete(self):
        gpus = [
            {"id": "intel", "name": "Intel UHD", "vendorId": 0x8086, "vulkanIndex": 2, "integrated": True},
            {"id": "amd", "name": "Radeon RX", "vendorId": 0x1002, "vulkanIndex": 1, "integrated": False},
        ]
        devices = [device("amd", "Radeon RX", "vulkan", 1), device("intel", "Intel UHD", "vulkan", 2)]
        with patch.object(diarization, "resolve_diarization_model", return_value=self.model), \
                patch.object(diarization, "_inventory", return_value=gpus), \
                patch.object(diarization, "_prepared", return_value=(devices, {})), \
                patch.object(diarization, "find_diarization_executable", side_effect=lambda b: self.exe[b]):
            self.assertEqual(diarization.resolve_diarization_device()["id"], "amd")
            self.assertEqual(diarization.resolve_diarization_device("intel")["deviceIndex"], 2)

    def test_auto_priority_nvidia_then_discrete_then_integrated(self):
        gpus = [
            {"id": "i", "name": "Intel", "vendorId": 0x8086, "integrated": True},
            {"id": "amd-d", "name": "AMD discrete", "vendorId": 0x1002, "integrated": False},
            {"id": "nv", "name": "NVIDIA", "vendorId": 0x10DE},
        ]
        devices = [device("nv", "NVIDIA", "cuda", 0), device("amd-d", "AMD discrete", "vulkan", 1), device("i", "Intel", "vulkan", 2)]
        with patch.object(diarization, "resolve_diarization_model", return_value=self.model), \
                patch.object(diarization, "_inventory", return_value=gpus), \
                patch.object(diarization, "_prepared", return_value=(devices, {})), \
                patch.object(diarization, "find_diarization_executable", side_effect=lambda b: self.exe[b]):
            self.assertEqual(diarization.resolve_diarization_device()["id"], "nv")
            devices[0]["available"] = False
            with self.assertRaises(TaskFailure) as unavailable:
                diarization.resolve_diarization_device()
        self.assertEqual(unavailable.exception.code, "device_unavailable")

    def test_doctor_backend_name_drives_index_and_matches_gpu_description_and_type(self):
        gpu = {"id": "radeon", "name": "Radeon RX", "vendorId": 0x1002, "vulkanIndex": 0}
        doctor = {
            "version": "0.3.0", "features": {"diarization": True, "backend_vulkan": True},
            "accelerator_available": True, "driver_runtime_compatible": True,
            "devices": [
                {"name": "CPU", "index": 0, "type": "cpu", "description": "Host CPU"},
                {"name": "Vulkan0", "index": 1, "type": "integrated-gpu", "description": "Radeon RX"},
            ],
        }
        result = diarization._ordered_devices([gpu], doctor, {"vulkan"})[0]
        self.assertTrue(result["available"])
        self.assertEqual(result["deviceIndex"], 0)
        doctor["devices"][1]["type"] = "cpu"
        self.assertFalse(diarization._ordered_devices([gpu], doctor, {"vulkan"})[0]["available"])

    def test_duplicate_identical_adapters_have_distinct_stable_uuid_ids_after_reordering(self):
        first = [
            {"name": "Radeon RX", "vendorId": 0x1002, "deviceUUID": "uuid-card-a", "deviceIndex": 0},
            {"name": "Radeon RX", "vendorId": 0x1002, "deviceUUID": "uuid-card-b", "deviceIndex": 1},
        ]
        with patch.object(diarization, "_windows_inventory", return_value=[]), \
                patch.object(diarization, "_vulkan_inventory", return_value=first), \
                patch.object(diarization, "_cuda_inventory", return_value=[]):
            before = {x["deviceUUID"]: x["id"] for x in diarization._inventory()}
        reordered = [dict(first[1], deviceIndex=0), dict(first[0], deviceIndex=1)]
        with patch.object(diarization, "_windows_inventory", return_value=[]), \
                patch.object(diarization, "_vulkan_inventory", return_value=reordered), \
                patch.object(diarization, "_cuda_inventory", return_value=[]):
            after = {x["deviceUUID"]: x["id"] for x in diarization._inventory()}
        self.assertEqual(before, after)
        self.assertEqual(len(set(before.values())), 2)
        self.assertNotEqual(before["uuid-card-a"], before["uuid-card-b"])

    def test_reordered_same_name_vulkan_cards_keep_selected_uuid_and_refresh_index(self):
        first = [
            {"name": "Radeon RX", "vendorId": 0x1002, "deviceUUID": "uuid-card-a", "deviceIndex": 0},
            {"name": "Radeon RX", "vendorId": 0x1002, "deviceUUID": "uuid-card-b", "deviceIndex": 1},
        ]
        with patch.object(diarization, "_windows_inventory", return_value=[]), \
                patch.object(diarization, "_cuda_inventory", return_value=[]):
            with patch.object(diarization, "_vulkan_inventory", return_value=first):
                selected_id = next(g["id"] for g in diarization._inventory() if g["deviceUUID"] == "uuid-card-a")
            reordered = [dict(first[1], deviceIndex=0), dict(first[0], deviceIndex=1)]
            current = {g["id"]: g for g in [
                {**g, "id": diarization._stable_id(g), "vulkanIndex": g["deviceIndex"]} for g in reordered
            ]}
            status = [device(gpu_id, "Radeon RX", "vulkan", row["vulkanIndex"]) for gpu_id, row in current.items()]
            with patch.object(diarization, "resolve_diarization_model", return_value=self.model), \
                    patch.object(diarization, "_inventory", return_value=list(current.values())), \
                    patch.object(diarization, "_prepared", return_value=(status, {})), \
                    patch.object(diarization, "find_diarization_executable", side_effect=lambda b: self.exe[b]):
                result = diarization.resolve_diarization_device(selected_id)
        self.assertEqual(result["id"], selected_id)
        self.assertEqual(result["deviceIndex"], 1)

    def test_missing_id_and_unready_gpu_fail_without_cpu_fallback(self):
        gpu = {"id": "bad-gpu", "name": "GPU", "vendorId": 0x1002}
        devices = [device("bad-gpu", "GPU", "vulkan", 0, False, "driver missing")]
        with patch.object(diarization, "resolve_diarization_model", return_value=self.model), \
                patch.object(diarization, "_inventory", return_value=[gpu]), \
                patch.object(diarization, "_prepared", return_value=(devices, {})), \
                patch.object(diarization, "find_diarization_executable", return_value="cpu.exe"):
            with self.assertRaises(TaskFailure) as missing:
                diarization.resolve_diarization_device("gone")
            with self.assertRaises(TaskFailure) as unavailable:
                diarization.resolve_diarization_device()
        self.assertEqual(missing.exception.code, "device_unavailable")
        self.assertEqual(unavailable.exception.code, "device_unavailable")
        self.assertEqual(unavailable.exception.stage, "diarization")


class DiarizationPreparationTests(unittest.TestCase):
    def test_model_checksum_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "diarization" / diarization.MODEL_FILENAME
            target.parent.mkdir()
            target.write_bytes(b"corrupt")
            with patch.object(diarization, "MODEL_STORAGE_ROOT", root), \
                    patch.object(diarization, "MODEL_SIZE", 7), \
                    patch.object(diarization, "MODEL_SHA256", "0" * 64):
                with self.assertRaises(TaskFailure) as caught:
                    diarization.resolve_diarization_model()
        self.assertEqual(caught.exception.code, "model_unavailable")
        self.assertEqual(caught.exception.stage, "diarization")

    def test_check_only_does_not_download_or_mutate_missing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(diarization, "MODEL_STORAGE_ROOT", root / "models"), \
                    patch.object(diarization, "APP_ROOT", root), \
                    patch.object(diarization, "_inventory", return_value=[]), \
                    patch.object(diarization, "_install_runtime") as install_runtime, \
                    patch.object(diarization, "_install_model") as install_model, \
                    patch.object(diarization, "_download") as download:
                result = diarization.prepare_diarization(check_only=True)
            install_runtime.assert_not_called()
            install_model.assert_not_called()
            download.assert_not_called()
            self.assertFalse(result["ready"])
            self.assertFalse((root / "models").exists())


if __name__ == "__main__":
    unittest.main()
