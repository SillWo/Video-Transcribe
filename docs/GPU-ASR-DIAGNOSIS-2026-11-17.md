# GPU output-length diagnosis — 2026-11-17

## Finding

The apparent loss of information on GPU was not reproduced on the 60-second control clip. The large output-length difference came from a hallucinated continuation in the faster-whisper CPU result. This is a diagnosis, not a claim that GPU is universally more accurate or that the CPU defect is fixed.

A controlled device comparison ran the **same whisper.cpp executable, F16 model, audio, beam settings, language selection, and vocabulary**, changing only `--no-gpu`. CPU and Vulkan returned the same 99-word sequence after case/punctuation normalization. Their punctuation and formatting of `Клон++` differ. Both terminate at 60.0 seconds.

| Current diagnostic run | Word count | Last output timestamp | Actual audio duration |
|---|---:|---:|---:|
| whisper.cpp, Vulkan | 99 | 60.0 s | 60.0 s |
| whisper.cpp, CPU | 99 | 60.0 s | 60.0 s |
| faster-whisper CPU, INT8, fixed diagnostic seed 0 | 138 | **87.0 s** | 60.0 s |

The 60-second clip is the interval 210–270 s of the original Jira tutorial. CPU and GPU receive exactly the same 960,000 PCM samples at 16 kHz; hashes of the input PCM and the GPU-converted PCM are identical. The raw Vulkan JSON has three rows, and the adapter preserves all three. Input conversion and JSON parsing did not truncate the result.

## Reproduced failure mechanism

The current faster-whisper 1.2.1 / CTranslate2 4.8.5 run processed windows starting at **0, 29, and 59 seconds**. After recognizing the main speech, it handled the last **one-second** remainder as another model window. The model pads its input to the standard window length. It also carries previous text into the decoder; the final prompt contained 206 tokens.

That last window produced repetition or low-confidence output at every fallback temperature:

- 0.0: compression ratio 5.5185, above the 2.4 limit;
- 0.2: 11.55, above 2.4;
- 0.4: 9.95, above 2.4;
- 0.6: 4.7135, above 2.4;
- 0.8 and 1.0: average log probability below the -1.0 threshold.

`generate_with_fallback` nevertheless returns its best failed candidate when all attempts fail. Its returned temperature field is 1.0 after exhausting the loop; this does not establish that the selected candidate itself was generated at 1.0. The selected average log probability was **-1.9510**. Its no-speech probability was 0.2194, so the library's no-speech skip did not reject it.

The timestamp splitter maps generated timestamp tokens onto the padded window without bounding the resulting segment times to the actual one-second remainder. The result contains **12 segments extending beyond the real recording**, up to 87 seconds. Examples:

| Claimed time | Emitted CPU text |
|---|---|
| 59–62 s | `Нажимаем «Продолжить»,落изем.` |
| 64–67 s | `В готовом тексте появится контакт на файл.` |
| 81–83 s | `Хорошо, сейчас я вставлю тестирование.` |
| 85–87 s | `И это то, что мне надо.` |

There is no audio from 60 to 87 seconds in this input. Those rows are not additional recognized material. The application currently accepts that engine text/timing output; this is a CPU-output validation defect to address separately.

The final generation took **141.990 s of 182.965 s** inside the transcribe call: approximately **77.6%** of CPU inference elapsed time was spent on that last one-second remainder. This explains both the spurious extra text and much of the apparent speed advantage in the earlier benchmark. The old 8.3× figure measures the observed runs, but is not a clean comparison of useful speech-recognition throughput.

## Cross-checks

- A fresh complete Vulkan run covered 0–60 s and ended with `клон клон плюс плюс`.
- The same native engine on CPU returned the same word sequence, with punctuation differences only.
- Isolated Vulkan recognition of 45–60 s, with and without the Jira hint, returned the final cloning/menu phrase and terminated at 15 s.
- Historical Qwen and Nemotron results for this control clip also end with the cloning/menu phrase. They are supporting machine cross-checks, not a manually annotated reference.
- Replaying the *same encoded last window* without previous text produced another hallucination (`Редактор субтитров А.Семкин Корректор А.Егорова`). Restricting temperature to zero produced repetition. Neither simple switch is demonstrated to be a safe production fix.
- An early parallel CPU/GPU diagnostic exceeded available allocation capacity (GGML/MKL allocation failures). Its partial CPU output is explicitly excluded from the conclusion. The incomplete CPU attempt is not used as a completed run. The native CPU device-control run completed while that earlier diagnostic was also running; its latency is not used for a hardware speed comparison. The subsequent complete faster-whisper reproduction and isolated GPU rerun ran sequentially. The application queue itself was not used for these experiments.

## Interpretation and next repair scope

The earlier CPU/GPU correspondence percentages cannot establish that the GPU loses 20–37% of spoken information. They compare two implementations, and the CPU reference contains hallucinated text. The active automatic-device qualification is unchanged and no passing receipt was issued. The separate identifier-formatting regression with a vocabulary hint is still unresolved.

A repair should validate CPU output against real input duration, identify failed/low-confidence padded-tail decoding, and re-run the comparison against a checked reference. Merely clipping timestamp numbers would leave hallucinated text intact; disabling context or temperature fallback alone was not shown to solve it. Production ASR settings have not been changed during this diagnosis.

## Evidence

All diagnostic data is retained under `.runtime/gpu-diagnosis/`:

- `diagnostic-summary.json`: verified invariants and counts;
- `input-audit.json`: exact PCM identity and duration;
- `coverage-reproduction.json`, `gpu-full-raw.json`, `gpu-full-log.txt`: live original GPU reproduction;
- `cpp-cpu-full-hint-summary.json`: controlled same-engine CPU execution;
- `ct2-complete-full.json`, `ct2-complete-generations.json`, `ct2-generation-attempts.log`: complete CPU failure and fallback attempts;
- `ct2-last-window-controls.json`: same-encoder context/temperature controls;
- `isolated-gpu-tail-with-hint.json`, `gpu-tail-no-hint-summary.json`: final 15-second checks.
