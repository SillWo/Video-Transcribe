import json
from pathlib import Path

from services import asr_capabilities, simple_profiles


def _candidate(backend="directml", index=0, *, name="Test GPU", identity="gpu-luid", driver="1.2"):
    return {
        "available": False,
        "canPrepare": True,
        "backend": backend,
        "deviceIndex": index,
        "name": name,
        "identity": identity,
        "driverVersion": driver,
        "reason": "Not smoke verified",
    }


def _successful_smoke():
    return {
        "ok": True,
        "text": "Hello there.",
        "language": "en",
        "diagnostics": {"gpu_verified": True, "verification_basis": "actual model ops on GPU"},
    }


def test_quality_balanced_economy_map_to_new_models_and_small():
    assert simple_profiles.PROFILES["quality"][1] == "qwen3-asr-1.7b"
    assert simple_profiles.PROFILES["balanced"][1] == "parakeet-tdt-0.6b-v3"
    assert simple_profiles.PROFILES["economy"][1] == "small"


def test_gpu_verification_cache_is_invalidated_by_resource_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(asr_capabilities, "CAPABILITY_PATH", tmp_path / "capabilities.json")
    fingerprint = {"value": "resources-v1"}
    monkeypatch.setattr(asr_capabilities, "resource_fingerprint", lambda *_args, **_kwargs: fingerprint["value"])
    monkeypatch.setattr(asr_capabilities, "_resources_ready", lambda *_args: (True, None))
    candidate = _candidate()

    recorded = asr_capabilities.record_gpu_probe("parakeet-tdt-0.6b-v3", candidate, _successful_smoke())
    assert recorded["verified"] is True
    assert asr_capabilities.gpu_capability("parakeet-tdt-0.6b-v3", candidate)["state"] == "ready"

    fingerprint["value"] = "resources-v2"
    stale = asr_capabilities.gpu_capability("parakeet-tdt-0.6b-v3", candidate)
    assert stale["available"] is False
    assert stale["state"] == "needs_preparation"
    assert stale["canPrepare"] is True


def test_partial_or_failed_gpu_smoke_is_unavailable_with_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(asr_capabilities, "CAPABILITY_PATH", tmp_path / "capabilities.json")
    monkeypatch.setattr(asr_capabilities, "resource_fingerprint", lambda *_args, **_kwargs: "same")
    candidate = _candidate()
    failed = _successful_smoke()
    failed["diagnostics"] = {"gpu_verified": False, "reason": "Only CPU operators were recorded."}

    record = asr_capabilities.record_gpu_probe("parakeet-tdt-0.6b-v3", candidate, failed)
    capability = asr_capabilities.gpu_capability("parakeet-tdt-0.6b-v3", candidate)

    assert record["verified"] is False
    assert capability["available"] is False
    assert capability["state"] == "unavailable"
    assert capability["canPrepare"] is True
    assert "Only CPU operators" in capability["reason"]


def test_active_gpu_retry_overrides_cached_failure_and_restores_final_state(tmp_path, monkeypatch):
    monkeypatch.setattr(asr_capabilities, "CAPABILITY_PATH", tmp_path / "capabilities.json")
    monkeypatch.setattr(asr_capabilities, "resource_fingerprint", lambda *_args, **_kwargs: "same")
    monkeypatch.setattr(asr_capabilities, "_resources_ready", lambda *_args: (True, None))
    candidate = _candidate()
    failed = _successful_smoke()
    failed["diagnostics"] = {"gpu_verified": False, "reason": "Previous proof failed."}
    asr_capabilities.record_gpu_probe("parakeet-tdt-0.6b-v3", candidate, failed)

    asr_capabilities.start_gpu_check("parakeet-tdt-0.6b-v3", candidate)
    during_check = asr_capabilities.gpu_capability("parakeet-tdt-0.6b-v3", candidate)
    assert during_check["state"] == "checking"
    assert during_check["available"] is False
    assert during_check["canPrepare"] is True
    assert during_check["backend"] == candidate["backend"]
    assert during_check["deviceIndex"] == candidate["deviceIndex"]
    assert "проверка gpu" in during_check["reason"].casefold()

    asr_capabilities.finish_gpu_check("parakeet-tdt-0.6b-v3", candidate)
    restored_failure = asr_capabilities.gpu_capability("parakeet-tdt-0.6b-v3", candidate)
    assert restored_failure["state"] == "unavailable"
    assert restored_failure["canPrepare"] is True

    asr_capabilities.start_gpu_check("parakeet-tdt-0.6b-v3", candidate)
    asr_capabilities.record_gpu_probe("parakeet-tdt-0.6b-v3", candidate, _successful_smoke())
    asr_capabilities.finish_gpu_check("parakeet-tdt-0.6b-v3", candidate)
    assert asr_capabilities.gpu_capability("parakeet-tdt-0.6b-v3", candidate)["state"] == "ready"


def test_old_gpu_probe_record_without_verification_version_is_rechecked(tmp_path, monkeypatch):
    path = tmp_path / "capabilities.json"
    monkeypatch.setattr(asr_capabilities, "CAPABILITY_PATH", path)
    monkeypatch.setattr(asr_capabilities, "resource_fingerprint", lambda *_args, **_kwargs: "same")
    candidate = _candidate()
    old_record = asr_capabilities._identity("parakeet-tdt-0.6b-v3", candidate)
    old_record.pop("verificationVersion")
    old_record.update({"verified": True, "textNonempty": True, "language": "en"})
    path.write_text(json.dumps({"schemaVersion": 1, "records": [old_record]}), encoding="utf-8")

    capability = asr_capabilities.gpu_capability("parakeet-tdt-0.6b-v3", candidate)

    assert capability["available"] is False
    assert capability["state"] == "needs_preparation"
    assert capability["canPrepare"] is True


def test_verified_capability_is_hidden_when_resources_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(asr_capabilities, "CAPABILITY_PATH", tmp_path / "capabilities.json")
    monkeypatch.setattr(asr_capabilities, "resource_fingerprint", lambda *_args, **_kwargs: "same")
    monkeypatch.setattr(asr_capabilities, "_resources_ready", lambda *_args: (False, "model weight removed"))
    candidate = _candidate()
    asr_capabilities.record_gpu_probe("parakeet-tdt-0.6b-v3", candidate, _successful_smoke())

    capability = asr_capabilities.gpu_capability("parakeet-tdt-0.6b-v3", candidate)

    assert capability["available"] is False
    assert capability["state"] == "needs_preparation"
    assert capability["canPrepare"] is True
    assert capability["reason"] == "model weight removed"


def test_options_report_profile_specific_backend_and_default(tmp_path, monkeypatch):
    monkeypatch.setattr(asr_capabilities, "CAPABILITY_PATH", tmp_path / "capabilities.json")
    monkeypatch.setattr(asr_capabilities, "get_device_details", lambda: [{"id": "cpu", "name": "Test CPU"}])
    candidates = {
        "qwen3-asr-1.7b": _candidate("vulkan", 1, name="Arc"),
        "parakeet-tdt-0.6b-v3": _candidate("directml", 2, name="Radeon"),
        "small": _candidate("vulkan", 3, name="Arc"),
    }
    monkeypatch.setattr(asr_capabilities, "asr_gpu_candidate", lambda model: candidates[model])

    result = simple_profiles.options(16)

    by_id = {row["id"]: row for row in result["profiles"]}
    assert by_id["quality"]["devices"]["gpu"]["backend"] == "vulkan"
    assert by_id["balanced"]["devices"]["gpu"]["backend"] == "directml"
    assert by_id["economy"]["devices"]["gpu"]["backend"] == "vulkan"
    assert all(row["defaultDevice"] == "gpu" for row in by_id.values())
    assert result["devices"] == by_id["balanced"]["devices"]


def test_probe_cache_records_runtime_identity_precision_and_actual_diagnostics(tmp_path, monkeypatch):
    path = tmp_path / "caps.json"
    monkeypatch.setattr(asr_capabilities, "CAPABILITY_PATH", path)
    monkeypatch.setattr(asr_capabilities, "resource_fingerprint", lambda *_args, **_kwargs: "fp")
    candidate = _candidate("directml", 0, name="AMD 780M", identity="00001282", driver="32.0.32015.7029")

    row = asr_capabilities.record_gpu_probe("parakeet-tdt-0.6b-v3", candidate, _successful_smoke())
    saved = json.loads(path.read_text(encoding="utf-8"))["records"][0]

    assert row["precision"] == "FP32"
    assert saved["driverVersion"] == "32.0.32015.7029"
    assert saved["resourceFingerprint"] == "fp"
    assert saved["diagnostics"]["gpu_verified"] is True
