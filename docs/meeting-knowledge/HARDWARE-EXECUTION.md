# Meeting hardware execution

<!-- CURRENT-HARDWARE -->
## Действующая граница ускорения, 16 ноября 2026

Согласованы два режима: Auto выбирает самый быстрый **проверенный** путь каждого этапа; в ручном пользователь выбирает CPU или GPU, а неподдерживаемое выполнение завершается понятной ошибкой. Подготовка показывает объём компонентов, проверяет короткий образец и кеширует результат до изменения модели, драйвера, устройства или среды. Драйверы не меняются. Созвон и запись имеют приоритет над обработкой. После GPU-сбоя автоматического повтора на CPU нет: ошибка, краткая рекомендация, только «ОК», очередь приостановлена до обычного «Продолжить».

| Этап | Auto сейчас | GPU-путь и ограничение |
|---|---|---|
| Речь Parakeet / Qwen / Whisper | Проверенный CPU | Существующие CUDA/DirectML/Vulkan-адаптеры сохраняются. Автоматическое ускорение речи требует отдельного сопоставимого замера. RTX/CUDA и Intel Arc не объявляются проверенными на этом ноутбуке. |
| Определение языка | Устройство этапа речи / разрешённый совместимый путь | Для CPU-распознавания применяется CPU. Если при выбранном GPU нет совместимого GPU-определения языка, возвращается ошибка с предложением выбрать CPU; сбой уже запущенного GPU также не вызывает повтор на CPU. Фаза отдельно обозначается «Определение языка речи». |
| Диаризация встреч | CPU до сравнения качества и скорости | Существующий native Vulkan/CUDA доступен для явно выбранного ручного GPU. Прежний `auto` диаризации обычных файлов оставлен совместимым; настройки новых встреч имеют собственную строгую политику. |
| Экранный OCR | DirectML FP32 stock на проверенной Radeon780M | CPU0,493499с / GPU0,415406с на последнем контроле. На пятиминутном реальном фрагменте GPU257,931с, тексты/время/bbox совпали с CPU. |
| FP16 recognizer | Не выбирается автоматически | Несмотря на ускорение1,147×, реальный фрагмент выявил числовые расхождения. Режим экспериментальный; старые FP16-допуски также отклоняются. |
| Индекс и запросы E5 | CPU | Короткий запрос ранее показал преимущество CPU; для пакетного GPU-индекса нет сопоставимого допуска. |
| Сжатие Opus | CPU | Для этого кодека GPU-путь не используется. |
| Чтение кадров / архив | Лёгкое чтение, без ASR/OCR | Получение кадров использует сохранённые изображения или ограниченное извлечение из оригинала, без фоновой подготовки всего архива. |

Качество должно быть не ниже95% оценки CPU; новые ошибки в проверяемых ID, числах и значениях недопустимы. Успешная загрузка провайдера сама по себе не является допуском. Для OCR дополнительно проверены реальные GPU-операции и малое изменение25→27. Поддержка 780M подтверждена только для испытанных путей; Arc и RTX требуют собственного устройства и испытания.

Планирование постобработки сосредоточено в `meeting_hardware.plan_stage`: вход — этап, режим, выбранное устройство и профиль; выход — устройство, отдельный интерпретатор, окружение, допуск, причина или ошибка. Новые адаптеры должны предоставлять CPU-путь и проверяемый GPU-путь с отпечатком модели/драйвера/среды и измерением качества/скорости. ASR и native-диаризация пока используют сохранённые собственные адаптеры; это граница расширения, а не обещание универсального GPU для каждой операции.

Текущий длинный прогон был поставлен до изменения политики диаризации и использует прежний `auto` → Vulkan для неё. Его время относится к этой явно записанной конфигурации; оно не выдаётся за замер нового Auto с CPU-диаризацией. Остальные части: ParakeetCPU, OCRDirectMLFP32, OpusCPU, E5CPU. Старые секции ниже сохранены как история испытаний, их слово `current` относится к дате соответствующей записи.
<!-- /CURRENT-HARDWARE -->

This document describes the stage runtime contract for the meeting worker. CPU
is the reliable baseline. A GPU is selected automatically only after the exact
machine, runtime, model and workload have a current quality and speed receipt.

## Modes and stage choices

- **Auto** uses CPU unless a verified measurement shows that a compatible GPU
  is both faster and meets the quality gate. The current implementation only
  has a screen OCR calibration path. Audio and index therefore remain on CPU in
  Auto until they have their own paired receipts.
- **Manual** preserves the requested device. A missing runtime, failed provider
  check or failed receipt returns an explicit error; it never retries on CPU.
- CPU execution uses the main project interpreter. DirectML screen OCR uses
  `.runtime/meeting-ml/directml`, isolated from the ASR environments.
- NVIDIA ASR CUDA and existing model-specific adapters remain controlled by
  their current strict capability checks. DirectML registration does not imply
  that every model or stage is GPU-verified. Intel Arc is listed as hardware
  where discovered, but is not considered verified merely because it exists.
- E5 query embedding stays on CPU for the current workload; the recorded local
  comparison found its query path faster there.

## Screen OCR preparation and calibration

`GET /api/meetings/hardware` reports installed runtimes and the estimated
download size. `POST /api/meetings/hardware/prepare` starts a background setup;
`GET /api/meetings/hardware/prepare` returns its state and recent progress.
Preparation creates the dedicated Python environment and installs only
`requirements-meetings-gpu.txt`. It does not change GPU drivers.

The preparation job then performs one warm-up and three timed OCR passes on a
fixed bilingual screen containing Russian text, English text, an identifier,
and numeric values. CPU and DirectML use the same RapidOCR model. Calibration
records literal OCR regions, timings, model/runtime fingerprints and ONNX
operator execution evidence. The receipt is accepted only when:

1. CPU/GPU OCR region agreement is at least 95%.
2. Identifiers and numeric-bearing regions match exactly.
3. The DirectML ONNX profile contains actual `DmlExecutionProvider` operators.
4. Median GPU OCR time is lower than the CPU reference time.

The receipt is invalidated when the adapter/driver identity, model files,
requirements, isolated interpreter/site-packages, or OCR workload changes.
Having the DirectML provider installed is not by itself proof that OCR ran on
the GPU. If calibration cannot verify all gates, Auto remains on CPU.

## Screen extraction workload

Screen extraction decodes the video in a single PyAV pass backed by FFmpeg libraries, preserving each frame's native dimensions and presentation timestamp. FFmpeg CLI remains responsible for audio compression and bounded on-demand original frame extraction. It runs full-frame
OCR for the initial state and regular checkpoints. For small edits it compares
local tiles, pads and merges affected tiles, OCRs those crops, translates their
boxes back into frame coordinates, and keeps prior OCR regions outside the
changed crops with an explicit inherited-recognition timestamp and uncertainty.
Scrolling or broad redraws trigger a full-frame pass. OCR-equivalent visual changes retain an image event and an explicit limitation. Every saved event still contains the full frame
so the original visual context is available.

Sampling timestamps are approximate: the event time is the first sampled frame
where a state was detected, with uncertainty equal to the configured sample
interval. Geometry-based table and header associations remain candidates, not
semantic assertions.

## Remaining validation

Automated tests cover routing, fallback prohibition, receipt gates, local OCR
updates and inherited regions. They do not substitute for the controlled
five-minute Jira benchmark or the two-hour composite recording run on the
target Ryzen 7 8845H/Radeon 780M. Those measurements must record elapsed time,
OCR counts, output coverage and quality before changing the Auto policy.

The local synthetic calibration on AMD Radeon 780M (driver `32.0.32015.7029`)
passed: 1.0 literal-region agreement, exact identifier/number agreement, and
2,440 ONNX operators attributed to DirectML. Median OCR time was 0.288 s on
CPU and 0.210 s on DirectML (about 1.37× faster on this probe). This confirms
the provider path and enables Auto screen OCR on the measured machine; it is
not the controlled Jira or two-hour workload benchmark.


## Validation update — 15 November 2026

The screen workload is now `screen-roi-v3-one-padding`. ROI padding is applied once; subsequent closure includes complete intersected OCR boxes without repeatedly extending the margin across a dense page. Older workload receipts are rejected. Fresh calibration on the same Radeon 780M passed (CPU 0.30048 s, DirectML 0.20048 s).

A paired five-minute real Jira fragment processed by this same version took 363.061 s on CPU and 236.082 s on DirectML (1.538×). Both decoded the whole fragment and produced identical event times, OCR text and bounding boxes: 301 sampled frames, 272 OCR calls, including 122 full-frame and 150 ROI passes. This measures screen extraction, not the entire meeting workflow. Machine-readable result: `.runtime/meeting-pilot/benchmark-5min/roi-v3/comparison.json`.

An explicitly selected GPU OCR failure now terminates the stage instead of continuing as partial or falling back to CPU. A meeting job failure pauses the shared queue before the next item, retaining the specific failure code and explanation. Cancellation and managed shutdown remain separate. A real HTTP integration run confirmed one failed unsupported manual GPU job and one queued job, with no CPU retry.

The two-hour end-to-end measurement is separate. RTX/CUDA and Intel Arc remain unverified on actual hardware. Short-query E5 CPU timing is not evidence about bulk embedding throughput.


## Current implementation correction — 15 November 2026

Current workload is `screen-roi-v5-anchored`: native 16-pixel change tiles retain visual changes even when OCR text is equivalent, full OCR checkpoints remain independent of intervening changes, and conservative recognition of changed known text boxes preserves context. Earlier benchmark numbers retain their explicit version labels.

The screen adapter now obtains DirectML device ordinals only from DXGI. WMI contributes a driver fingerprint only on an unambiguous match. Vulkan indices never identify DirectML devices. Auto rejects old receipts without the exact DXGI ordinal, LUID, PnP identity and driver fingerprint; Manual reports an unavailable target rather than guessing device zero. A fresh real v5 calibration on Radeon 780M/DXGI 0 passed (CPU 0.72644 s, DirectML 0.31748 s); this is a short control sample, not the two-hour preparation result.

Active recording and processing hold a temporary system-awake request. It releases after the last active operation; display power and user power policies are unchanged. This addresses the observed 85-minute Modern Standby gap inside the interrupted throughput test.


## Native decoder v6 — current workload

The current workload is `rapidocr-ppocrv5-mobile-cyrillic;screen-roi-v6-native-decoder`. PyAV 18.1.0 in the isolated DirectML environment preserves per-frame native dimensions and PTS; older receipts are invalid. A fresh Radeon 780M calibration passed (CPU 0.73026 s, DirectML 0.41483 s; critical values identical).

The same 301.378-second Jira fragment completed on CPU in 860.584 s and DirectML in 380.404 s, with identical screen event times, OCR text and boxes (301 samples, 280 OCR calls, 228 ROIs). CPU and GPU measurements came from separate runs with uncontrolled background activity. These are indicative timings and output agreement, not a controlled speed claim or a full two-hour acceptance. Earlier v3 timings used different geometry and must not be treated as current throughput.


## Recognition variants v7 — implementation awaiting full hardware acceptance

The current source workload is `rapidocr-ppocrv5-mobile-cyrillic;screen-roi-v7-native-decoder-rec-variants`. It can compare stock FP32, equal-width FP32, FP16 recognition with FP32 detection/classification, and FP16 recognition with equal-width batches. Equal-width batching preserves each original stock batch's normalized input tensors and decoder geometry; its acceptance additionally requires an observed recognition batch larger than the stock size on the control material.

Conversion is explicit preparation only: pinned ONNX1.20.1 and ml_dtypes0.5.4 are isolated in a separate converter target (about16.6MB before download), outside the inference environment. The original FP32 model remains intact; the converted recognizer has a source/content/requirements receipt. Status and ordinary reads never install or convert anything. Automatic choice requires the existing >=95% CPU agreement, unchanged critical IDs/numeric texts, real DML operator execution, anchored25→27 control, valid device/model/driver/runtime fingerprints, and measured speedup. Only the fastest passing variant is selected; an unavailable optional FP16 candidate does not imply a CPU retry of a failed meeting job.

A preliminary three-real-frame experiment found 98–100% line agreement for recognition-only FP16; one amount gained a thousands-grouping space without changing its digits/value. All-FP16 detection was rejected because it shifted geometry and fell below95% agreement on one frame. These are indicative probes under concurrent load, not a full v7 acceptance or an unconditional deployment decision. The paired hardware calibration and full-workflow timing remain required. See `.runtime/meeting-pilot/fp16-candidate/` and the [official ONNX Runtime float16 guidance](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).


## 16 ноября: реальный пятиминутный тест отменяет допуск FP16 в «Авто»

Последовательный тест одного Jira-фрагмента301,378с: DirectML FP32 stock257,931с, recognizerFP16/equal-width224,845с (ускорение1,147×). В обоих280событий/301выборка. Среди19620строк при тех же координатах точно совпали18657 (95,0917%). Однако есть расхождения в цифрах и идентификаторах: например дата `10/09/26 2031` в CPU/FP32 стала `10/09/25 2031` в FP16 на15,968с фрагмента. Это расхождение с контрольным FP32, а не независимо размеченный общий эталон OCR. Ранее полученный CPU FP32 и текущий DirectML FP32 дают полностью одинаковые тексты, времена и bbox на этом фрагменте.

Порог общего совпадения95% сам по себе недостаточен: критичные значения должны сохраняться. Поэтому сохранённый короткий допуск FP16 аннулирован; FP16 не принят для автоматического рабочего режима. Чистый крупный шрифт контрольного полотна оказался недостаточно представительным. Реальное ускорениеFP16 не объявляется допустимым ускорением продукта. Машинный отчёт `.runtime/meeting-pilot/benchmark-5min/native-v7/comparison.json`, исходная короткая квитанция сохранена как `rejected-short-calibration-receipt.json`. Для следующего полного прогона остаётся FP32, дальнейшая проверка FP16 требует более представительного набора критичных значений.
