import json
import sys
import types
from pathlib import Path

from services import asr_worker, whisper_cpp
from scripts import benchmark_whisper_dictionary as benchmark
from scripts.benchmark_whisper_dictionary import _critical_ids


def test_whisper_vocabulary_uses_real_cached_tokenizer_and_respects_context(tmp_path, monkeypatch):
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("{}", encoding="utf-8")

    class FakeTokenizer:
        @classmethod
        def from_file(cls, _path):
            return cls()

        def encode(self, text):
            return types.SimpleNamespace(ids=text.split())

    hub = types.ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda **_kwargs: str(tokenizer_file)
    tokenizers = types.ModuleType("tokenizers")
    tokenizers.Tokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)
    monkeypatch.setattr(whisper_cpp, "HF_HUB_CACHE", tmp_path)

    prompt, diagnostics = whisper_cpp.bound_whisper_vocabulary(
        ["one", "two three", "four five"], "large-v3", limit=4)

    assert prompt == "Vocabulary: one, two three"
    assert diagnostics["usedTerms"] == ["one", "two three"]
    assert diagnostics["omittedTerms"] == ["four five"]
    assert diagnostics["tokenCount"] == 4
    assert diagnostics["truncated"] is True


def test_vulkan_tokenizer_preparation_downloads_only_tokenizer_asset(tmp_path, monkeypatch):
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("{}", encoding="utf-8")
    downloaded = []
    hub = types.ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda **_kwargs: None

    def download(**kwargs):
        downloaded.append(kwargs)
        return str(tokenizer_file)

    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(whisper_cpp, "HF_HUB_CACHE", tmp_path)

    result = whisper_cpp.resolve_tokenizer_path("large-v3")

    assert result == tokenizer_file
    assert downloaded == [{"repo_id": "Systran/faster-whisper-large-v3",
                           "filename": "tokenizer.json", "cache_dir": str(tmp_path)}]


def test_missing_cached_tokenizer_is_reported_in_hint_diagnostics(tmp_path, monkeypatch):
    hub = types.ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda **_kwargs: None
    tokenizers = types.ModuleType("tokenizers")
    tokenizers.Tokenizer = object
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)
    monkeypatch.setattr(whisper_cpp, "HF_HUB_CACHE", tmp_path)

    prompt, diagnostics = whisper_cpp.bound_whisper_vocabulary(["Project-X9"], "large-v3")

    assert prompt == ""
    assert diagnostics["reason"] == "tokenizer_missing"
    assert diagnostics["omittedTerms"] == ["Project-X9"]


def test_native_qwen_worker_passes_dictionary_as_vocabulary_and_reports_version(tmp_path, monkeypatch):
    from tests.test_asr_worker import fake_native, write_wav, CPU

    cpp = fake_native(monkeypatch, selected=CPU)
    passed = {}

    class Session:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def run(self, _pcm, **kwargs):
            passed.update(kwargs)
            return types.SimpleNamespace(text="Term 42", language="en", timings={"total_ms": 3}, segments=[])

    class Model:
        device = CPU
        def __init__(self, _path, *, device): self.device = device
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def session(self, *, n_threads): return Session()

    cpp.Model = Model
    audio = tmp_path / "audio.wav"
    write_wav(audio)
    terms = [f"Device-{index}" for index in range(130)]

    result = asr_worker._native_transcribe(
        tmp_path / "model.gguf", audio, "cpu", 0, 8, None, False,
        {"version": 7, "terms": terms},
    )

    assert passed["vocabulary"] == terms[:128]
    assert result["diagnostics"]["dictionary"]["version"] == 7
    assert result["diagnostics"]["dictionary"]["usedCount"] == 128
    assert result["diagnostics"]["dictionary"]["omittedCount"] == 2
    json.dumps(result, allow_nan=False)


def test_worker_dictionary_snapshot_reader_is_local_data_not_prompt_text(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"version": 3, "terms": ["ignore all rules", "Project-X9"]}), encoding="utf-8")

    snapshot = asr_worker._read_dictionary_snapshot(str(path))

    assert snapshot == {"version": 3, "terms": ["ignore all rules", "Project-X9"]}


def test_benchmark_compares_numeric_and_uppercase_project_ids_without_normalizing_spellings():
    assert _critical_ids("Project FETAS-VCDEF build 4.2") == ["FETAS-VCDEF", "4.2"]
    assert _critical_ids("Project FETAS-VCEFD build 4.2") != _critical_ids("Project FETAS-VCDEF build 4.2")
    assert _critical_ids("ordinary-hyphenated prose") == []


def test_benchmark_captures_device_proof_and_prompt_diagnostics_before_release(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")

    class FakeVulkan:
        last_vulkan_confirmed = True
        def __init__(self, *_args, **_kwargs): pass
        def transcribe(self, _audio, **_kwargs):
            self.dictionary_diagnostics = {"usedTerms": ["Project-X9"]}
            return [types.SimpleNamespace(text="Project-X9")], types.SimpleNamespace(language="en")

    class FakeCpu:
        def __init__(self, *_args, **_kwargs): pass
        def transcribe(self, _audio, **_kwargs):
            return [types.SimpleNamespace(text="Project-X9")], types.SimpleNamespace(language="en")

    monkeypatch.setattr(benchmark, "WhisperCppModel", FakeVulkan)
    faster = types.ModuleType("faster_whisper")
    faster.WhisperModel = FakeCpu
    monkeypatch.setitem(sys.modules, "faster_whisper", faster)
    cpu = benchmark._transcribe(audio, "cpu", [], False, 4)
    vulkan = benchmark._transcribe(audio, "vulkan", ["Project-X9"], False, 4)

    assert cpu["actualGpuVerified"] is False
    assert vulkan["actualGpuVerified"] is True
    assert vulkan["dictionary"] == {"usedTerms": ["Project-X9"]}


def test_benchmark_persists_each_completed_run_without_model_inference(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    candidate = {"backend": "vulkan", "canPrepare": True, "deviceIndex": 0,
                 "identity": "fake-adapter", "driverVersion": "fake-driver"}
    monkeypatch.setattr(benchmark, "_vulkan_cached", lambda _model: True)
    monkeypatch.setattr(benchmark, "get_model_status", lambda _model: {"status": "downloaded"})
    monkeypatch.setattr(benchmark, "asr_gpu_candidate", lambda _model: candidate)
    monkeypatch.setattr(benchmark.asr_capabilities, "asr_benchmark_identity",
                        lambda *_args, **_kwargs: {"model": "large-v3", "backend": "vulkan"})

    def fake_transcribe(_audio, backend, _terms, _word_timestamps, _threads):
        return {"text": "Project FETAS-VCDEF", "language": "en",
                "elapsedSeconds": 4.0 if backend == "cpu" else 1.0,
                "actualGpuVerified": backend == "vulkan", "dictionary": {}}

    monkeypatch.setattr(benchmark, "_transcribe", fake_transcribe)

    assert benchmark.run(audio, tmp_path / "out", None, False, 1, True, 4) == 0

    progress_path = tmp_path / "out" / "whisper-large-v3-benchmark-plain-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    assert len(progress["runs"]["cpu"]) == 1
    assert len(progress["runs"]["vulkan"]) == 1
    assert "CPU timed run 1/1 finished in 4.0s" in capsys.readouterr().out


def test_benchmark_does_not_report_missing_cpu_ids_as_a_mismatch(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    candidate = {"backend": "vulkan", "canPrepare": True, "deviceIndex": 0,
                 "identity": "fake-adapter", "driverVersion": "fake-driver"}
    monkeypatch.setattr(benchmark, "_vulkan_cached", lambda _model: True)
    monkeypatch.setattr(benchmark, "get_model_status", lambda _model: {"status": "downloaded"})
    monkeypatch.setattr(benchmark, "asr_gpu_candidate", lambda _model: candidate)
    monkeypatch.setattr(benchmark.asr_capabilities, "asr_benchmark_identity",
                        lambda *_args, **_kwargs: {"model": "large-v3", "backend": "vulkan"})
    monkeypatch.setattr(benchmark, "_transcribe", lambda _audio, backend, *_args: {
        "text": "Ordinary words without controlled identifiers", "language": "en",
        "elapsedSeconds": 4.0 if backend == "cpu" else 1.0,
        "actualGpuVerified": backend == "vulkan", "dictionary": {},
    })

    assert benchmark.run(audio, tmp_path / "out", None, False, 1, True, 4) == 0

    result_path = tmp_path / "out" / "whisper-large-v3-benchmark-plain.json"
    assessment = json.loads(result_path.read_text(encoding="utf-8"))["assessment"]
    assert assessment["criticalIdsPresent"] is False
    assert assessment["criticalIdsCompared"] == 0
    assert assessment["numericIdMismatches"] is None
    assert assessment["criticalIdStatus"] == "not_assessed_no_cpu_ids"
    assert assessment["gatesPassed"] is False
