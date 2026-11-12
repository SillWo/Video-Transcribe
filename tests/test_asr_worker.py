import array
import dataclasses
import json
import struct
import sys
import types
import wave
from pathlib import Path

import pytest

from services import asr_worker as worker


@dataclasses.dataclass
class Device:
    name: str
    kind: str
    device_type: str
    device_id: str | None = None


CPU = Device("CPU", "cpu", "cpu")
GPU = Device("Radeon", "vulkan", "igpu")
CUDA = Device("NVIDIA", "cuda", "gpu", "0")


def write_wav(path: Path, seconds: float = 1.0, *, silence: bool = False) -> None:
    count = int(16000 * seconds)
    values = [0 if silence else (100 if index % 2 else -100) for index in range(count)]
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(array.array("h", values).tobytes())


def fake_native(monkeypatch, *, selected=CPU, actual=None, run_result=None, run_error=None):
    actual = actual or selected

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def run(self, _pcm, **_kwargs):
            if run_error:
                raise run_error
            return run_result or types.SimpleNamespace(
                text="hello", language="en", timings={"total_ms": 32}, segments=[]
            )

    class Model:
        device = actual

        def __init__(self, _path, *, device):
            self.device = actual
            assert device is selected

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def session(self, *, n_threads):
            assert n_threads == 8
            return Session()

    cpp = types.SimpleNamespace(
        __version__="0.4.1", OutputTruncated=(type(run_error) if run_error else type("OutputTruncated", (RuntimeError,), {})),
        OutputRepetition=type("OutputRepetition", (RuntimeError,), {}),
        backends=lambda: [selected], Model=Model, native_version=lambda: "0.4.1-native",
        native_commit=lambda: "test", library_path=lambda: "native.dll",
        set_log_callback=lambda _callback: None,
    )
    monkeypatch.setitem(sys.modules, "transcribe_cpp", cpp)
    monkeypatch.setattr(worker, "configure_runtime", lambda *_args, **_kwargs: None)
    return cpp


def test_device_selection_is_exact_and_does_not_fall_back_to_cpu():
    with pytest.raises(worker.WorkerError, match="unavailable"):
        worker._select_native_device(types.SimpleNamespace(backends=lambda: [CPU]), "vulkan", 0)


def test_native_success_serializes_result_and_timings(tmp_path, monkeypatch):
    fake_native(monkeypatch)
    wav = tmp_path / "audio.wav"
    write_wav(wav)

    result = worker._native_transcribe(tmp_path / "model.gguf", wav, "cpu", 0, 8, None, False)

    assert result["ok"] is True
    assert result["text"] == "hello"
    assert result["language"] == "en"
    assert result["diagnostics"]["model_device"]["name"] == "CPU"
    assert result["diagnostics"]["native_timings"] == {"total_ms": 32}
    assert result["timing_seconds"]["load"] >= 0
    assert result["timing_seconds"]["inference"] >= 0
    json.dumps(result, allow_nan=False)


def test_gpu_selection_that_resolves_to_cpu_is_rejected(tmp_path, monkeypatch):
    fake_native(monkeypatch, selected=GPU, actual=CPU)
    wav = tmp_path / "audio.wav"
    write_wav(wav)

    with pytest.raises(worker.WorkerError, match="no fallback"):
        worker._native_transcribe(tmp_path / "model.gguf", wav, "vulkan", 0, 8, None, True)


def test_native_cuda_uses_enumerated_device_instead_of_fixed_rejection(tmp_path, monkeypatch):
    fake_native(monkeypatch, selected=CUDA)
    wav = tmp_path / "audio.wav"
    write_wav(wav)

    result = worker._native_transcribe(tmp_path / "model.gguf", wav, "cuda", 0, 8, None, False)

    assert result["ok"] is True
    assert result["effective_device"] == "cuda"


def test_output_truncation_below_retry_minimum_is_non_success(tmp_path, monkeypatch):
    truncation = type("OutputTruncated", (RuntimeError,), {})("truncated")
    fake_native(monkeypatch, run_error=truncation)
    wav = tmp_path / "short.wav"
    write_wav(wav, 2.9)

    with pytest.raises(worker.WorkerError, match="minimum retry chunk"):
        worker._native_transcribe(tmp_path / "model.gguf", wav, "cpu", 0, 8, None, False)


def test_main_writes_json_error_and_returns_nonzero(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    monkeypatch.setattr(worker, "run", lambda _args: (_ for _ in ()).throw(worker.WorkerError("bad input")))

    status = worker.main([
        "--mode", "transcribe", "--model", worker.NATIVE_NAME, "--device", "cpu",
        "--audio", str(tmp_path / "missing.wav"), "--output", str(output),
    ])

    assert status != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["error"] == {"type": "WorkerError", "message": "bad input"}


def test_chunk_iterator_has_contiguous_sample_coverage(tmp_path):
    wav = tmp_path / "long.wav"
    write_wav(wav, 31.25)

    chunks = list(worker.iter_audio_chunks(wav))

    assert chunks[0].start_frame == 0
    assert chunks[-1].end_frame == int(31.25 * 16000)
    assert all(left.end_frame == right.start_frame for left, right in zip(chunks, chunks[1:]))
    assert all(chunk.duration <= worker.MAX_CHUNK_SECONDS for chunk in chunks)


def test_chunk_iterator_rejects_wav_ending_before_declared_sample_count(tmp_path):
    wav = tmp_path / "truncated.wav"
    write_wav(wav, 1)
    content = bytearray(wav.read_bytes())
    data_bytes = len(content) - 44
    struct.pack_into("<I", content, 4, len(content) - 8 + 16000 * 2)
    struct.pack_into("<I", content, 40, data_bytes + 16000 * 2)
    wav.write_bytes(content)

    with pytest.raises(worker.WorkerError, match="before its declared"):
        list(worker.iter_audio_chunks(wav))


def test_alignment_raw_units_keep_compounds_as_exact_source_tokens():
    class Processor:
        def split_words_for_alignment(self, raw, _language):
            token = "".join(char for char in raw if char.isalpha() or char == "-")
            return token.replace("-", " ").split()

    source = "  из-за well-known!"
    units, normalized = worker._raw_alignment_units(source, Processor(), "English")

    assert "".join(unit["text"] for unit in units) == source
    assert [unit["text"] for unit in units] == ["  из-за", " well-known!"]
    assert [unit["alignment_texts"] for unit in units] == [["из", "за"], ["well", "known"]]
    assert normalized == ["из", "за", "well", "known"]


def test_onnx_verification_ignores_unprofiled_resampler_and_keeps_graph_profiles(tmp_path, monkeypatch):
    class Session:
        def __init__(self, profile, provider):
            self.profile = profile
            self.provider = provider
            self.fallback_disabled = False

        def get_providers(self):
            return [self.provider, "CPUExecutionProvider"]

        def get_provider_options(self):
            return {}

        def end_profiling(self):
            return self.profile

        def disable_fallback(self):
            self.fallback_disabled = True

    gpu_provider = "DmlExecutionProvider"
    encoder_profile = tmp_path / "encoder-profile.json"
    decoder_profile = tmp_path / "decoder-profile.json"
    event = lambda name, op: {"cat": "Node", "name": name,
                              "args": {"provider": gpu_provider, "op_name": op}}
    encoder_profile.write_text(json.dumps([event("encoder.Conv_0", "Conv")]), encoding="utf-8")
    decoder_profile.write_text(json.dumps([event("decoder.MatMul_0", "MatMul")]), encoding="utf-8")
    auxiliary = Session("", "CPUExecutionProvider")
    encoder = Session(str(encoder_profile), gpu_provider)
    decoder = Session(str(decoder_profile), gpu_provider)
    model = types.SimpleNamespace(
        sessions=[auxiliary, encoder, decoder],
        _encoder=encoder, _decoder_joint=decoder,
        asr=types.SimpleNamespace(_blank_idx=0, window_step=0.01, _subsampling_factor=8,
                                  _decode=lambda *_args: ([0, 1], 1, None)),
        recognize=lambda _waveform, sample_rate: types.SimpleNamespace(text="hello"),
    )
    ort = types.SimpleNamespace(get_available_providers=lambda: [gpu_provider, "CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnx_asr", types.ModuleType("onnx_asr"))
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setattr(worker, "configure_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "_onnx_create_model",
                        lambda *_args, **_kwargs: (model, ort, 0.01, tmp_path))
    monkeypatch.setattr(worker, "_configure_onnx_preprocessor",
                        lambda *_args, **_kwargs: {"device": "directml", "backend": gpu_provider,
                            "fallback_reason": None,
                            "profile_counts": {(gpu_provider, "Conv", "preprocessor.Conv_0"): 1}})
    wav = tmp_path / "audio.wav"
    write_wav(wav)

    result = worker._onnx_transcribe(tmp_path / "model", wav, "directml", 0, 8, True)

    assert result["ok"] is True
    assert result["diagnostics"]["gpu_operator_count"] == 2
    assert result["diagnostics"]["actual_operator_counts_by_provider"][gpu_provider] == {
        "Conv": 1, "MatMul": 1,
    }
    assert result["diagnostics"]["gpu_verified"] is True
    assert result["diagnostics"]["profiling_errors"] == []
    assert result["diagnostics"]["preprocessor_operator_counts_by_provider"][gpu_provider] == {"Conv": 1}
    json.dumps(result, allow_nan=False)
    output = tmp_path / "result.json"
    worker._write_json(output, result)
    assert json.loads(output.read_text(encoding="utf-8"))["diagnostics"]["preprocessing"]["device"] == "directml"
    assert not encoder_profile.exists()
    assert not decoder_profile.exists()


def test_result_segments_uses_full_recognition_text_when_native_segments_are_partial():
    chunk = worker.AudioChunk(0, 16000, array.array("h", [0] * 16000), array.array("f", [0.0] * 16000))
    partial = types.SimpleNamespace(text="Hello, world!", segments=[
        types.SimpleNamespace(t0_ms=0, t1_ms=400, text="Hello,")])
    assert worker._result_segments(partial, chunk) == [
        {"start": 0.0, "end": 1.0, "text": "Hello, world!"}]


def test_gpu_preprocessor_exception_falls_back_for_same_chunk_without_asr_retry(tmp_path, monkeypatch):
    calls = {"gpu": 0, "cpu_init": 0, "cpu": 0, "asr": 0}
    class Session:
        def get_providers(self): return ["DmlExecutionProvider"]
        def end_profiling(self): return ""
        def disable_fallback(self): calls["disabled"] = calls.get("disabled", 0) + 1
    class GpuPreprocessor:
        _preprocessor = Session()
        def __init__(self, name, options):
            calls["name"] = name
            calls["provider_options"] = options
        def __call__(self, *args, **kwargs):
            calls["gpu"] += 1
            raise RuntimeError("preprocess failed")
    class CpuPreprocessor:
        def __init__(self, name): calls["cpu_init"] += 1
        def __call__(self, *args, **kwargs): calls["cpu"] += 1; return "features"
    monkeypatch.setattr(worker, "_onnx_preprocessor_classes", lambda: (GpuPreprocessor, CpuPreprocessor))
    class Options:
        pass
    ort = types.SimpleNamespace(SessionOptions=Options,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential", ORT_PARALLEL="parallel"))
    owner = types.SimpleNamespace(_preprocessor=None)
    state = worker._configure_onnx_preprocessor(owner, ort, "directml", 0, 4, tmp_path)
    assert state["device"] == "directml"
    assert calls["name"] == "nemo128_conv" and calls["disabled"] == 1
    assert owner._preprocessor("wave", 1) == "features"
    assert calls["gpu"] == calls["cpu"] == calls["cpu_init"] == 1
    assert state["device"] == "cpu" and "preprocess failed" in state["fallback_reason"]


def test_gpu_preprocessor_constructor_provider_mismatch_uses_cpu_and_explicit_cpu_stays_numpy(tmp_path, monkeypatch):
    calls = []
    class Session:
        def get_providers(self): return ["CPUExecutionProvider"]
        def end_profiling(self): return ""
    class GpuPreprocessor:
        _preprocessor = Session()
        def __init__(self, *_args): pass
        def __call__(self, *_args): raise AssertionError("invalid GPU preprocessor must not run")
    class CpuPreprocessor:
        def __init__(self, name): calls.append(name)
        def __call__(self, *args): return "cpu"
    monkeypatch.setattr(worker, "_onnx_preprocessor_classes", lambda: (GpuPreprocessor, CpuPreprocessor))
    ort = types.SimpleNamespace(SessionOptions=lambda: types.SimpleNamespace(),
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=1, ORT_PARALLEL=2))
    original = object()
    owner = types.SimpleNamespace(_preprocessor=original)
    state = worker._configure_onnx_preprocessor(owner, ort, "directml", 0, 4, tmp_path)
    assert state["device"] == "cpu" and "did not select" in state["fallback_reason"]
    assert owner._preprocessor is original
    cpu_owner = types.SimpleNamespace(_preprocessor="kept")
    cpu_state = worker._configure_onnx_preprocessor(cpu_owner, ort, "cpu", 0, 4, tmp_path)
    assert cpu_state["backend"] == "numpy" and cpu_owner._preprocessor == "kept"


def test_preprocessor_profile_is_collected_and_removed_without_affecting_asr_counts(tmp_path):
    profile = tmp_path / "preprocessor.json"
    profile.write_text(json.dumps([{"cat": "Node", "name": "Conv_0",
        "args": {"provider": "DmlExecutionProvider", "op_name": "Conv"}}]), encoding="utf-8")
    class Session:
        def end_profiling(self): return str(profile)
    state = {"session": Session(), "profile_counts": {}}
    worker._finish_preprocessor_profile(state, verify_gpu=True)
    assert state["profile_counts"] == {("DmlExecutionProvider", "Conv", "Conv_0"): 1}
    assert not profile.exists()


def test_qwen_scheduler_parses_callback_fragments_between_node_headers():
    diagnostics = worker.QwenPlacementDiagnostics("vulkan")
    fragments = [
        "qwen3_asr: encoder tensor backend Vulkan\n",
        "node #  0 (    IM2COL):               node_0 ( 618K) [Vulka         ] use=1,c=1:\n",
        "    enc.conv.0.weight (  16K) [Vulka         ]\n",
        "node #  3 (   MUL_MAT):               node_3 (  64M) [Vulka         ] use=1,c=1:\n",
        "    enc.conv.0.weight (r (  16K) [Vulka         ]\n",
        "node # 59 (FLASH_ATTN):              node_59 (   8K) [Vulka         ] use=1,c=1:\n",
        "    dec.blocks.1.attn.q.weight (  8M) [Vulka         ]\n",
    ]
    for fragment in fragments:
        # Simulate native callback boundaries that split text at arbitrary bytes.
        split = max(1, len(fragment) // 3)
        diagnostics(4, fragment[:split])
        diagnostics(4, fragment[split:])

    result = diagnostics.finish()

    assert result["gpu_assignments"] == 3
    assert result["gpu_assignments_by_stage"] == {"encoder": 2, "decoder": 1}
    assert result["gpu_heavy_assignments_by_stage"] == {"encoder": 1, "decoder": 1}
    assert result["gpu_verified"] is True


def test_qwen_load_messages_without_scheduler_nodes_never_verify_gpu():
    diagnostics = worker.QwenPlacementDiagnostics("vulkan")
    diagnostics(1, "encoder tensor backend Vulkan; decoder backend Vulkan")

    result = diagnostics.finish()

    assert result["gpu_verified"] is False
    assert result["gpu_assignments"] == 0


def test_exact_silence_is_detected_without_language_observation(tmp_path):
    wav = tmp_path / "silence.wav"
    write_wav(wav, 1, silence=True)

    assert all(chunk.is_silent for chunk in worker.iter_audio_chunks(wav))
