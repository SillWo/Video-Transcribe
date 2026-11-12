import array
import dataclasses
import importlib.util
import json
import sys
import types
import wave
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_langid_native.py"
SPEC = importlib.util.spec_from_file_location("probe_langid_native", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def no_runtime_environment_side_effects(monkeypatch):
    monkeypatch.setattr(probe, "configure_runtime", lambda: None)


@dataclasses.dataclass
class Device:
    name: str
    description: str
    kind: str
    device_type: str
    device_id: str | None
    memory_total: int = 4096
    memory_free: int = 2048
    index: int | None = 0


CPU = Device("CPU", "Test CPU", "cpu", "cpu", None)
VULKAN_GPU = Device("Test Vulkan", "Test GPU", "vulkan", "igpu", None)
CUDA_GPU = Device("Test CUDA", "Test GPU", "cuda", "gpu", "cuda-test")


def _write_wav(path: Path):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(array.array("h", [100, -100, 200, -200]).tobytes())


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded


def _write_f32_gguf(path: Path, tensor_types=(0,)):
    import struct

    data = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensor_types), 0))
    for index, tensor_type in enumerate(tensor_types):
        data.extend(_gguf_string(f"weight-{index}"))
        data.extend(struct.pack("<I", 1))
        data.extend(struct.pack("<Q", 2))
        data.extend(struct.pack("<IQ", tensor_type, 0))
    path.write_bytes(data)


@dataclasses.dataclass(frozen=True)
class Candidate:
    index: int
    code: str
    name: str
    p: float
    logit: float


@dataclasses.dataclass(frozen=True)
class Timings:
    load_ms: float
    mel_ms: float
    encode_ms: float
    decode_ms: float


@dataclasses.dataclass(frozen=True)
class Result:
    candidates: tuple[Candidate, ...] = (Candidate(42, "ru", "Russian", 0.91, 2.4),)
    allowed_mass: float = 1.0

    @property
    def code(self):
        return self.candidates[0].code if self.candidates else None


class FakeSession:
    def __init__(self, failure=None):
        self.failure = failure
        self.timings = Timings(3.0, 2.0, 4.0, 0.0)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, pcm):
        assert len(pcm) == 4
        if self.failure:
            raise self.failure
        return Result()


class FakeModel:
    selected_devices = []
    thread_values = []
    failure = None

    def __init__(self, path, *, device):
        self.path = path
        self.device = device
        self.selected_devices.append(device)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def langid_session(self, *, n_threads):
        self.thread_values.append(n_threads)
        return FakeSession(self.failure)


def _fake_binding(devices, model_factory=FakeModel):
    module = types.ModuleType("transcribe_cpp")
    module.__version__ = "0.4.1"
    module.backends = lambda: devices
    module.Model = model_factory
    module.native_version = lambda: "0.4.1-native"
    return module


def _args(tmp_path, *, device="cpu"):
    audio = tmp_path / "sample.wav"
    model = tmp_path / "langid-f32.gguf"
    _write_wav(audio)
    _write_f32_gguf(model)
    return types.SimpleNamespace(
        audio=str(audio),
        model=str(model),
        device=device,
        threads=8,
        output=str(tmp_path / "result.json"),
    )


def test_select_device_never_falls_back_and_rejects_ambiguous_devices():
    assert probe.select_device([CPU, VULKAN_GPU], "vulkan") is VULKAN_GPU
    with pytest.raises(probe.ProbeError, match="fallback is disabled"):
        probe.select_device([CPU], "vulkan")
    with pytest.raises(probe.ProbeError, match="fallback is disabled"):
        probe.select_device([CUDA_GPU], "vulkan")
    with pytest.raises(probe.ProbeError, match="ambiguous"):
        probe.select_device([VULKAN_GPU, dataclasses.replace(VULKAN_GPU, name="Second Vulkan")], "vulkan")


def test_f32_gguf_requires_only_type_zero_and_expected_identity(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    _write_f32_gguf(model, tensor_types=(0, 0))
    monkeypatch.setattr(probe, "EXPECTED_MODEL_SIZE", model.stat().st_size)
    monkeypatch.setattr(probe, "EXPECTED_MODEL_SHA256", probe._sha256(model))

    info = probe.validate_f32_gguf(model)

    assert info["size_bytes"] == model.stat().st_size
    assert info["sha256"] == probe._sha256(model)
    assert info["tensor_type_ids"] == [0]


@pytest.mark.parametrize("tensor_types", [(1,), (0, 2), (0, 8)])
def test_non_f32_or_quantized_gguf_is_rejected(tmp_path, tensor_types):
    model = tmp_path / "bad.gguf"
    _write_f32_gguf(model, tensor_types=tensor_types)

    with pytest.raises(probe.ProbeError, match="only F32 tensors"):
        probe.validate_f32_gguf(model, verify_identity=False)


def test_cpu_language_candidates_and_timings_are_serialized(tmp_path, monkeypatch):
    FakeModel.selected_devices.clear()
    FakeModel.thread_values.clear()
    fake = _fake_binding([CPU])
    monkeypatch.setitem(sys.modules, "transcribe_cpp", fake)
    args = _args(tmp_path)
    monkeypatch.setattr(probe, "EXPECTED_MODEL_SIZE", Path(args.model).stat().st_size)
    monkeypatch.setattr(probe, "EXPECTED_MODEL_SHA256", probe._sha256(Path(args.model)))

    payload = probe.run_probe(args)
    probe._write_json(Path(args.output), payload)
    saved = json.loads(Path(args.output).read_text(encoding="utf-8"))

    assert saved["ok"] is True
    assert saved["code"] == "ru"
    assert saved["candidates"] == [{
        "index": 42, "code": "ru", "name": "Russian", "p": 0.91, "logit": 2.4,
    }]
    assert saved["allowed_mass"] == 1.0
    assert saved["native_timings"]["encode_ms"] == 4.0
    assert saved["model_device"]["device_type"] == "cpu"
    assert saved["audio"]["sha256"]
    assert saved["model"]["sha256"] == probe._sha256(Path(args.model))
    assert FakeModel.selected_devices == [CPU]
    assert FakeModel.thread_values == [8]


def test_gpu_uses_exact_selected_device_without_fallback(tmp_path, monkeypatch):
    FakeModel.selected_devices.clear()
    fake = _fake_binding([CPU, VULKAN_GPU])
    monkeypatch.setitem(sys.modules, "transcribe_cpp", fake)
    args = _args(tmp_path, device="vulkan")
    monkeypatch.setattr(probe, "EXPECTED_MODEL_SIZE", Path(args.model).stat().st_size)
    monkeypatch.setattr(probe, "EXPECTED_MODEL_SHA256", probe._sha256(Path(args.model)))

    payload = probe.run_probe(args)

    assert payload["ok"] is True
    assert payload["requested_device"] == "vulkan"
    assert payload["selected_device"]["name"] == "Test Vulkan"
    assert FakeModel.selected_devices == [VULKAN_GPU]


def test_runtime_input_too_short_becomes_error_json_without_language(tmp_path, monkeypatch):
    class InputTooShort(Exception):
        pass

    class TooShortModel(FakeModel):
        failure = InputTooShort("audio is shorter than 250 ms")

    fake = _fake_binding([CPU], TooShortModel)
    monkeypatch.setitem(sys.modules, "transcribe_cpp", fake)
    args = _args(tmp_path)
    monkeypatch.setattr(probe, "EXPECTED_MODEL_SIZE", Path(args.model).stat().st_size)
    monkeypatch.setattr(probe, "EXPECTED_MODEL_SHA256", probe._sha256(Path(args.model)))

    exit_code = probe.main([
        "--audio", args.audio,
        "--model", args.model,
        "--device", "cpu",
        "--output", args.output,
    ])
    saved = json.loads(Path(args.output).read_text(encoding="utf-8"))

    assert exit_code == 1
    assert saved["ok"] is False
    assert "code" not in saved
    assert saved["error"] == {
        "type": "InputTooShort",
        "message": "audio is shorter than 250 ms",
    }
