import array
import dataclasses
import importlib.util
import json
import struct
import sys
import types
import wave
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_qwen_native.py"
SPEC = importlib.util.spec_from_file_location("probe_qwen_native", SCRIPT)
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


CPU = Device("CPU", "Test CPU", "cpu", "cpu", "cpu-test")
VULKAN_GPU = Device("Test Vulkan", "Test GPU", "vulkan", "gpu", "gpu-test")
CUDA_GPU = Device("Test CUDA", "Test GPU", "cuda", "gpu", "cuda-test")


def _write_wav(path: Path):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(array.array("h", [100, -100, 200, -200]).tobytes())


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _write_f16_gguf(path: Path, tensor_types=(1,)):
    # Minimal GGUF v3 header with one rank-1 tensor per requested GGML type.
    data = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensor_types), 0))
    for index, tensor_type in enumerate(tensor_types):
        data.extend(_gguf_string(f"weight-{index}"))
        data.extend(struct.pack("<I", 1))
        data.extend(struct.pack("<Q", 2))
        data.extend(struct.pack("<IQ", tensor_type, 0))
    path.write_bytes(data)


class FakeSession:
    def __init__(self, failure=None):
        self.failure = failure

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, pcm):
        assert len(pcm) == 4
        if self.failure:
            raise self.failure
        return FakeResult()


@dataclasses.dataclass(frozen=True)
class FakeTimings:
    load_ms: float
    mel_ms: float
    encode_ms: float
    decode_ms: float


@dataclasses.dataclass(frozen=True)
class FakeSegment:
    text: str
    t0_ms: int
    t1_ms: int


@dataclasses.dataclass(frozen=True)
class FakeResult:
    text: str = "hello"
    language: str = "en"
    timings: FakeTimings = FakeTimings(5.0, 2.0, 3.0, 4.0)
    segments: tuple = (FakeSegment("hello", 0, 500),)
    words: tuple = ()
    tokens: tuple = ()


class FakeModel:
    def __init__(self, path, device):
        self.path = path
        self.device = device

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def session(self, n_threads):
        assert n_threads == 8
        return FakeSession()


def _fake_binding(devices, model_factory=FakeModel):
    callbacks = []
    module = types.ModuleType("transcribe_cpp")
    module.__version__ = "0.4.1"
    module.backends = lambda: devices
    module.Model = model_factory
    module.native_version = lambda: "0.4.1-native"
    module.set_log_callback = lambda callback: callbacks.append(callback)
    module._callbacks = callbacks
    return module


def _args(tmp_path, *, device="cpu", device_id=None):
    audio = tmp_path / "sample.wav"
    model = tmp_path / "model-f16.gguf"
    _write_wav(audio)
    _write_f16_gguf(model)
    return types.SimpleNamespace(
        audio=str(audio),
        model=str(model),
        device=device,
        device_id=device_id,
        threads=8,
        output=str(tmp_path / "result.json"),
        log=str(tmp_path / "native.log"),
    )


def test_select_device_requires_backend_and_never_falls_back():
    assert probe.select_device([CPU, VULKAN_GPU], "vulkan", "gpu-test") is VULKAN_GPU
    with pytest.raises(probe.ProbeError, match="fallback is disabled"):
        probe.select_device([CPU], "vulkan")
    with pytest.raises(probe.ProbeError, match="fallback is disabled"):
        probe.select_device([CUDA_GPU], "vulkan")
    with pytest.raises(probe.ProbeError, match="fallback is disabled"):
        probe.select_device([VULKAN_GPU], "vulkan", "different-id")


def test_cpu_probe_serializes_result_dataclasses(tmp_path, monkeypatch):
    fake = _fake_binding([CPU])
    monkeypatch.setitem(sys.modules, "transcribe_cpp", fake)
    args = _args(tmp_path)

    payload = probe.run_probe(args)
    probe._write_json(Path(args.output), payload)
    saved = json.loads(Path(args.output).read_text(encoding="utf-8"))

    assert saved["ok"] is True
    assert saved["status"] == "verified"
    assert saved["text"] == "hello"
    assert saved["language"] == "en"
    assert saved["result_timings"]["encode_ms"] == 3.0
    assert saved["segments"][0]["text"] == "hello"
    assert saved["model_device"]["device_id"] == "cpu-test"
    assert len(saved["model"]["sha256"]) == 64
    assert fake._callbacks[-1] is None


def test_gpu_probe_stays_unverified_without_encoder_and_decoder_evidence(tmp_path, monkeypatch):
    fake = _fake_binding([VULKAN_GPU])
    monkeypatch.setitem(sys.modules, "transcribe_cpp", fake)
    args = _args(tmp_path, device="vulkan", device_id="gpu-test")

    payload = probe.run_probe(args)

    assert payload["ok"] is True
    assert payload["status"] == "unverified"
    assert payload["model_device"]["device_id"] == "gpu-test"


def test_output_truncated_is_non_success_json_error(tmp_path, monkeypatch, capsys):
    class OutputTruncated(Exception):
        pass

    class TruncatedModel(FakeModel):
        def session(self, n_threads):
            return FakeSession(OutputTruncated("decoder output limit reached"))

    fake = _fake_binding([CPU], TruncatedModel)
    fake.OutputTruncated = OutputTruncated
    monkeypatch.setitem(sys.modules, "transcribe_cpp", fake)
    args = _args(tmp_path)

    exit_code = probe.main([
        "--audio", args.audio,
        "--model", args.model,
        "--device", "cpu",
        "--output", args.output,
        "--log", args.log,
    ])
    saved = json.loads(Path(args.output).read_text(encoding="utf-8"))

    assert exit_code == 1
    assert saved["ok"] is False
    assert saved["error"] == {
        "type": "OutputTruncated",
        "message": "decoder output limit reached",
    }
    assert "Probe failed" in capsys.readouterr().err


def test_quantized_gguf_is_rejected(tmp_path):
    model = tmp_path / "q8.gguf"
    _write_f16_gguf(model, tensor_types=(8,))

    with pytest.raises(probe.ProbeError, match="quantized tensors"):
        probe._validate_f16_gguf(model)


def test_mixed_f16_f32_gguf_is_accepted(tmp_path):
    model = tmp_path / "mixed-f16-f32.gguf"
    _write_f16_gguf(model, tensor_types=(1, 0, 1, 0))

    probe._validate_f16_gguf(model)


def test_only_f32_gguf_is_rejected(tmp_path):
    model = tmp_path / "f32.gguf"
    _write_f16_gguf(model, tensor_types=(0,))

    with pytest.raises(probe.ProbeError, match="at least one F16 tensor"):
        probe._validate_f16_gguf(model)


def test_scheduler_diagnostics_ignore_load_summaries_and_do_not_verify(tmp_path, monkeypatch):
    log = probe.NativeLog(tmp_path / "native.log")
    log(4, "encoder tensor backend Vulkan")
    log(4, "offloaded 24/24 layers to Vulkan")
    log(4, "node #  1 (     MUL_MAT): encoder-layer.weight (  2K) [Vulka          ] use=1,c=1:")
    summary = log.summary()
    log.close()

    assert summary["scheduler_compute_node_assignments"] == {
        "gpu_by_backend": {"Vulka": 1},
        "cpu": 0,
    }
    assert summary["line_count"] == 3

    # Scheduler planning records also appear before a compute graph executes,
    # so the status path must never promote a GPU transcript to verified.
    fake = _fake_binding([VULKAN_GPU])
    monkeypatch.setitem(sys.modules, "transcribe_cpp", fake)
    args = _args(tmp_path, device="vulkan", device_id="gpu-test")
    payload = probe.run_probe(args)
    assert payload["status"] == "unverified"
