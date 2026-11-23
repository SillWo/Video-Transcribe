"""Paired offline CPU/Vulkan Whisper benchmark and automatic-device receipt writer.

Run against an existing representative 60-second WAV fixture. The script never
prepares or downloads models; it fails closed if either local cache is missing.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from services.model_paths import HF_HUB_CACHE
from services.prepare_models import _vulkan_cached
from services.model_management import get_model_status
from services.runtime_devices import asr_gpu_candidate
from services.whisper_cpp import WhisperCppModel
from services import asr_capabilities


def _text_and_language(rows, stats):
    text = " ".join(str(getattr(row, "text", "") or "").strip() for row in rows).strip()
    return text, getattr(stats, "language", None)


def _transcribe(audio: Path, backend: str, terms: list[str], word_timestamps: bool,
                threads: int) -> dict:
    started = time.perf_counter()
    if backend == "cpu":
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=threads,
                             num_workers=1, download_root=str(HF_HUB_CACHE), local_files_only=True)
    else:
        model = WhisperCppModel("large-v3", threads=threads)
    loaded = time.perf_counter()
    kwargs = {"language": None, "word_timestamps": word_timestamps}
    if backend == "cpu" and terms:
        from services.whisper_cpp import bound_whisper_vocabulary
        hotwords, prompt_diagnostics = bound_whisper_vocabulary(terms, "large-v3")
        if hotwords:
            kwargs["hotwords"] = hotwords
    elif backend == "vulkan":
        kwargs["vocabulary"] = terms
        prompt_diagnostics = {}
    else:
        prompt_diagnostics = {"usedTerms": [], "omittedTerms": terms,
                              "usedCount": 0, "omittedCount": len(terms), "truncated": bool(terms)}
    segments, stats = model.transcribe(str(audio), **kwargs)
    segments = list(segments)
    finished = time.perf_counter()
    text, language = _text_and_language(segments, stats)
    actual_gpu_verified = bool(backend == "vulkan" and getattr(model, "last_vulkan_confirmed", False))
    if backend == "vulkan":
        prompt_diagnostics = getattr(model, "dictionary_diagnostics", prompt_diagnostics)
    del model
    return {"text": text, "language": language, "elapsedSeconds": finished - started,
            "loadSeconds": loaded - started, "inferenceSeconds": finished - loaded,
            "segmentCount": len(segments), "dictionary": prompt_diagnostics,
            "actualGpuVerified": actual_gpu_verified}


def _critical_ids(text: str) -> list[str]:
    # Compare digit-bearing tokens and uppercase hyphen/slash codes exactly in order.
    tokens = re.findall(r"(?<!\w)[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*(?!\w)", text)
    return [token for token in tokens if any(char.isdigit() for char in token)
            or (("-" in token or "/" in token) and token.isupper()
                and any(char.isalpha() for char in token))]


def _agreement(cpu_text: str, gpu_text: str) -> float:
    left = re.findall(r"\w+|[^\w\s]", cpu_text.casefold(), flags=re.UNICODE)
    right = re.findall(r"\w+|[^\w\s]", gpu_text.casefold(), flags=re.UNICODE)
    if not left:
        return 0.0
    return difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio()


def run(audio: Path, output_dir: Path, dictionary_file: Path | None,
        word_timestamps: bool, runs: int, quick: bool, threads: int) -> int:
    if not audio.is_file():
        raise FileNotFoundError(f"Audio fixture not found: {audio}")
    if not _vulkan_cached("large-v3"):
        raise RuntimeError("Cached whisper.cpp ggml-large-v3.bin is required; offline benchmark will not download it.")
    if get_model_status("large-v3").get("status") != "downloaded":
        raise RuntimeError("Cached faster-whisper large-v3 weights are required; offline benchmark will not download them.")
    candidate = asr_gpu_candidate("large-v3")
    if candidate.get("backend") != "vulkan" or not candidate.get("canPrepare"):
        raise RuntimeError("No preparable Vulkan GPU candidate is available for this benchmark.")
    snapshot = {"version": 0, "terms": []}
    if dictionary_file:
        snapshot = json.loads(dictionary_file.read_text(encoding="utf-8"))
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("terms"), list):
            raise ValueError("Dictionary file must contain {version, terms:[string,...]}.")
    terms = [item for item in snapshot["terms"] if isinstance(item, str) and item.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    all_results = {"cpu": [], "vulkan": []}
    warmups = {}
    mode_suffix = "words" if word_timestamps else "plain"
    progress_file = output_dir / f"whisper-large-v3-benchmark-{mode_suffix}-progress.json"

    def persist_progress(status: str, current_backend: str | None = None,
                         current_run: int | None = None) -> None:
        progress = {"schemaVersion": 1, "status": status,
                    "audio": str(audio.resolve()), "wordTimestamps": bool(word_timestamps),
                    "dictionaryVersion": int(snapshot.get("version", 0) or 0),
                    "timedRunsPerBackend": runs, "warmupRunsPerBackend": 1,
                    "currentBackend": current_backend, "currentRun": current_run,
                    "warmups": warmups, "runs": all_results,
                    "updatedAt": time.time()}
        temporary_progress = progress_file.with_suffix(progress_file.suffix + ".tmp")
        temporary_progress.write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        os.replace(temporary_progress, progress_file)

    persist_progress("starting")
    for backend in ("cpu", "vulkan"):
        print(f"{backend.upper()} warmup (model load included)", flush=True)
        warmups[backend] = _transcribe(audio, backend, terms, word_timestamps, threads)
        print(f"{backend.upper()} warmup finished in {warmups[backend]['elapsedSeconds']:.1f}s",
              flush=True)
        persist_progress("warmup-finished", backend)
        for run_index in range(runs):
            print(f"{backend.upper()} timed run {run_index + 1}/{runs}", flush=True)
            result = _transcribe(audio, backend, terms, word_timestamps, threads)
            all_results[backend].append(result)
            print(f"{backend.upper()} timed run {run_index + 1}/{runs} finished in "
                  f"{result['elapsedSeconds']:.1f}s", flush=True)
            persist_progress("timed-run-finished", backend, run_index + 1)

    cpu_text = all_results["cpu"][0]["text"]
    cpu_ids = _critical_ids(cpu_text)
    comparisons = []
    for item in all_results["vulkan"]:
        ids = _critical_ids(item["text"])
        comparisons.append({"qualityAgreement": _agreement(cpu_text, item["text"]),
                            "cpuCriticalIds": cpu_ids, "gpuCriticalIds": ids,
                            "criticalIdsEqual": bool(cpu_ids) and ids == cpu_ids})
    quality = min((item["qualityAgreement"] for item in comparisons), default=0.0)
    critical_ids_present = bool(cpu_ids)
    critical_equal = critical_ids_present and all(item["criticalIdsEqual"] for item in comparisons)
    numeric_id_mismatches = (0 if critical_equal else 1 if critical_ids_present else None)
    critical_id_status = ("matched" if critical_equal else "mismatch" if critical_ids_present
                         else "not_assessed_no_cpu_ids")
    cpu_seconds = statistics.median(item["elapsedSeconds"] for item in all_results["cpu"])
    gpu_seconds = statistics.median(item["elapsedSeconds"] for item in all_results["vulkan"])
    speedup = cpu_seconds / gpu_seconds if gpu_seconds > 0 else 0.0
    actual_gpu_verified = all(item.get("actualGpuVerified") is True for item in all_results["vulkan"])
    verified = (not quick and runs == 3 and quality >= .95 and critical_equal and speedup >= 1.5
                and actual_gpu_verified)
    identity = asr_capabilities.asr_benchmark_identity("large-v3", candidate,
                                                       word_timestamps=word_timestamps)
    assessment = {"speedup": speedup, "cpuMedianSeconds": cpu_seconds,
                  "gpuMedianSeconds": gpu_seconds, "qualityAgreement": quality,
                  "numericIdMismatches": numeric_id_mismatches,
                  "criticalIdsCompared": len(cpu_ids), "criticalIdsPresent": bool(cpu_ids),
                  "criticalIdStatus": critical_id_status,
                  "gatesPassed": verified,
                  "actualGpuVerified": actual_gpu_verified,
                  "quickRun": bool(quick), "timedRunsPerBackend": runs,
                  "warmupRunsPerBackend": 1}
    payload = {"schemaVersion": 1, "verified": bool(verified), "identity": identity,
               "audio": str(audio.resolve()), "dictionaryVersion": int(snapshot.get("version", 0) or 0),
               "wordTimestamps": bool(word_timestamps), "assessment": assessment,
               "warmups": warmups, "runs": all_results, "comparisons": comparisons,
               "createdAt": time.time()}
    result_file = output_dir / f"whisper-large-v3-benchmark-{mode_suffix}.json"
    result_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if verified:
        receipt = output_dir / f"whisper-large-v3-auto-receipt-{mode_suffix}.json"
        receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        canonical = (asr_capabilities.ASR_BENCHMARK_WORDS_RECEIPT if word_timestamps
                     else asr_capabilities.ASR_BENCHMARK_RECEIPT)
        canonical.parent.mkdir(parents=True, exist_ok=True)
        temp = canonical.with_suffix(canonical.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, canonical)
    persist_progress("complete")
    print(json.dumps({"verified": verified, "speedup": round(speedup, 3),
                      "qualityAgreement": round(quality, 4), "criticalIdsCompared": len(cpu_ids),
                      "criticalIdsPresent": bool(cpu_ids),
                      "actualGpuVerified": actual_gpu_verified,
                      "wordTimestamps": bool(word_timestamps), "output": str(result_file)}, ensure_ascii=False))
    return 0 if verified or quick or runs != 3 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path, help="Existing representative ~60-second WAV fixture")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / ".runtime" / "asr" / "benchmark")
    parser.add_argument("--dictionary-file", type=Path)
    parser.add_argument("--word-timestamps", choices=("on", "off"), default="off")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="One timed run; never issues an accepted receipt")
    args = parser.parse_args()
    runs = 1 if args.quick else args.runs
    if runs < 1:
        parser.error("--runs must be positive")
    try:
        return run(args.audio.resolve(), args.output_dir.resolve(), args.dictionary_file,
                   args.word_timestamps == "on", runs, args.quick, max(1, args.threads))
    except Exception as exc:
        print(f"Benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
