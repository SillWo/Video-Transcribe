import array
import hashlib
import importlib.util
import json
import os
import runpy
import shutil
import sys
import types
import wave
from collections import Counter
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_parakeet_onnx.py"
SPEC = importlib.util.spec_from_file_location("probe_parakeet_onnx", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _write_wav(path: Path, samples=(100, -100), *, channels=1, rate=16000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(array.array("h", samples).tobytes())


def _write_model_dir(path: Path):
    path.mkdir()
    for name in probe.REQUIRED_MODEL_FILES:
        (path / name).write_bytes(("artifact:" + name).encode())
    return path


def _args(tmp_path: Path, *, device="cpu", index=0):
    audio = tmp_path / "sample.wav"
    _write_wav(audio)
    return types.SimpleNamespace(
        model_dir=_write_model_dir(tmp_path / "model"),
        audio=audio,
        output=tmp_path / "result.json",
        device=device,
        device_index=index,
        threads=8,
    )


def test_import_does_not_change_environment_or_create_runtime_directories(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    isolated_script = scripts / "probe_parakeet_onnx.py"
    shutil.copyfile(SCRIPT, isolated_script)
    env_names = (
        "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
        "TORCH_HOME", "XDG_CACHE_HOME", "TMP", "TEMP", "TMPDIR",
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    )
    before = {name: os.environ.get(name) for name in env_names}

    runpy.run_path(str(isolated_script), run_name="isolated_probe_import")

    runtime = tmp_path / ".runtime" / "asr-evaluation" / "parakeet-onnx-probe"
    assert {name: os.environ.get(name) for name in env_names} == before
    assert not runtime.exists()


def test_missing_selected_gpu_provider_fails_before_load_without_cpu_retry(tmp_path, monkeypatch):
    args = _args(tmp_path, device="cuda", index=2)
    calls = []
    onnx_asr = types.ModuleType("onnx_asr")
    onnx_asr.load_model = lambda *a, **kw: calls.append((a, kw))
    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: [probe.PROVIDERS["cpu"]]
    monkeypatch.setitem(sys.modules, "onnx_asr", onnx_asr)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    with pytest.raises(probe.ProbeError, match="Automatic provider fallback and CPU retry are disabled"):
        probe.run_probe(args)

    assert calls == []


def test_audio_validation_rejects_empty_and_incorrect_wav_shape(tmp_path):
    empty = tmp_path / "empty.wav"
    _write_wav(empty, samples=())
    with pytest.raises(probe.ProbeError, match="non-empty"):
        probe.read_wav_info(empty)

    # A 2-channel WAV with one sample is not frame aligned and cannot be a
    # complete audio frame shape.
    bad_shape = tmp_path / "bad-shape.wav"
    _write_wav(bad_shape, samples=(1,), channels=2)
    with pytest.raises(probe.ProbeError, match="frame|shape|channel"):
        probe.read_wav_info(bad_shape)


def test_gpu_verification_uses_actual_operator_placement_not_provider_availability():
    assert probe.verify_actual_placement("directml", Counter(), 0)["status"] == "unsupported"
    assert probe.verify_actual_placement(
        "cuda", Counter({(probe.PROVIDERS["cuda"], "Conv"): 3}), 2
    )["status"] == "verified"

    partial = probe.verify_actual_placement(
        "directml",
        Counter({(probe.PROVIDERS["directml"], "Conv"): 3, (probe.PROVIDERS["cpu"], "MatMul"): 1}),
        1,
    )
    assert partial["status"] == "partial"
    assert partial["cpu_heavy_neural_operator_counts"] == {"MatMul": 1}
    assert partial["device_index"] == 1


def test_fp32_model_artifact_provenance_contains_hashes_and_sizes(tmp_path):
    model_dir = _write_model_dir(tmp_path / "model")

    artifacts = probe.artifact_provenance(model_dir)

    assert set(artifacts) == set(probe.REQUIRED_MODEL_FILES)
    for name, record in artifacts.items():
        content = (model_dir / name).read_bytes()
        assert record["size_bytes"] == len(content)
        assert record["sha256"] == hashlib.sha256(content).hexdigest()
        assert len(record["sha256"]) == 64


def test_incomplete_model_is_rejected_without_download(tmp_path):
    model_dir = tmp_path / "partial"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(probe.ProbeError, match="no download was attempted"):
        probe.artifact_provenance(model_dir)
