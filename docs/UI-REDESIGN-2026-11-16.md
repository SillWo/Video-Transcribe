# UI Redesign and Unified Archive — 2026-11-16

## Implemented

Five screens share the approved dark palette, Segoe UI typography, a 64 px navigation rail with 44 px hit areas, and a persistent compact status bar. Recording, dictionary, transcription, settings, and archive retain their state while navigating. Archive panes resize; compact windows disclose video/frame materials without squeezing the transcript. Settings keep explicit saving. Real images, timelines, and source playback provide the visual content; decorative gradients, nested cards, redundant explanations, and inconsistent legacy control colors were removed. The transcription icon now describes captions rather than magic/AI.

The archive catalog now combines meetings with ordinary transcriptions. Ordinary jobs are assigned to an archive root when queued; legacy history is claimed once when catalog sync runs. Portable items live at `transcriptions/<job_id>/` with `item.json`, compact `catalog.json`, and `transcript.txt` / `transcript.md`. Original media is not copied for ordinary audio transcriptions. Manually submitted video files now follow the meeting pipeline described below. History hooks publish job status, transcript edits, revisions, and deletions. Catalog listing reads SQL-projected history summaries and compact catalog rows, avoiding full diagnostic metadata reads for unchanged items. A tombstone ledger prevents deleted items from returning during later syncs.

Ordinary transcript search uses the existing FTS and E5 index. Each transcript has a `transcription:<job_id>` source key and stable chunk evidence IDs. With the configured tokenizer, content is split into 384-token chunks with 48-token overlap; titles are limited to 64 tokens, and character offsets remain available for citations/snippets. If no tokenizer is present, bounded character chunks retain full text coverage and search remains text-only. Completion and edits queue indexing through one coalescing daemon worker, outside history transactions and long archive locks. Worker shutdown stops new work, drops pending items, and guards index publication. Explicit reindex queues ordinary transcripts and reports `queued_transcriptions` alongside meeting `jobs`.

Legacy completions receive portable text on sync but are not bulk-embedded during catalog reads. Until explicit reindex builds their vectors, hybrid search falls back to lexical results with an unavailable/stale semantic status; semantic mode does not claim lexical hits are semantic matches. Read-only MCP search never encodes transcript passages. MCP transcript result snippets include source kind, job/evidence IDs, character offsets, and no fabricated timestamps.

The app-wide personal dictionary persists at `%LOCALAPPDATA%/VideoTranscribe/personal-dictionary.json` (`VT_DICTIONARY_PATH` supports isolated runs). Writes are atomic and process-locked. Terms use NFC, casefold, and whitespace normalization; entries carry stable IDs, definitions, provenance, tombstones, and reversible tokens. Automatic learning defaults on, but skips numeric/ID, URL/path, SQL, sentence-wide rewrites, and candidates over three words or 64 characters. Undo removes only vocabulary learned by that edit; it does not revert transcript text. Opening the dictionary imports legacy meeting glossary entries once per source item, retaining definition and meeting/root provenance. MCP glossary without a meeting filter reads the global dictionary; a meeting filter preserves legacy glossary reads.

Backups include a hash-checked global dictionary snapshot in the manifest. Restore merges missing terms, preserves current terms, and reports conflicts. Restore-test validates the snapshot without changing the live profile. Archive and dictionary APIs are available at `/api/archive` and `/api/dictionary`; transcription edits use optimistic `baseRevision` and return HTTP 409 on conflict. Active transcription deletion remains protected.

## API and MCP contracts

- `GET /api/archive?kind=all|meeting|transcription&query=&limit=&offset=` returns `{items,total,next_offset,archive_root}`. Rows have stable `id`, `kind`, `title`, `created_at`, `status`, applicable source/job/meeting IDs, and the canonical active `archive_root`.
- `GET /api/archive/{kind}/{id}` returns `{item,record,text,revision}`; `item.archive_root` identifies the root actually read, independently of any historical root stored inside a restored record. `PUT /api/archive/transcription/{id}/text` accepts `{text,baseRevision,expectedArchiveRoot?}` and returns the updated record, text, revision, and `remembered_terms`. When provided, `expectedArchiveRoot` is checked against the current canonical root and a mismatch returns HTTP 409 before editing, protecting delayed saves when identical IDs exist in different archives. Download supports `txt` and `md`; delete refuses active history records.
- `GET /api/dictionary?query=` returns `{items,version,auto_learn}`. Create/update/undo return `{item}`, delete returns `{undo_id}`, and settings use `{auto_learn}`.
- MCP universal tools are `find_archive_items`, `read_archive_item`, and `search_archive`. Search modes are `text`, `hybrid`, and `semantic`; both ordinary and meeting transcript reads paginate at 24,000 characters. Meeting reads include the current effective edited text alongside metadata; existing timeline and frame tools remain available. Existing meeting timeline, glossary, source-status, evidence, and actual JPEG frame behavior remains available.

## Validation completed

- Final backend run: **693 passed, 2 skipped, 40 subtests** in 82.29 s. Final frontend run: **72 tests passed**. Lint, production build, Electron syntax checks, and **4 desktop tests passed**. The final UI-only wrapping, selected-tab visibility, and icon corrections also passed lint/build and browser review.
- Regression coverage includes migration idempotence and root assignment, meeting-job exclusion, standalone restore/edit/revision conflict, identical IDs in different roots, delayed-save rejection, active-delete safety, concurrent edits, cross-process dictionary mutation, corruption preservation, conservative learning/undo, backup merge/restore-test isolation, universal search, identifier boundaries, pagination, transcript tail chunking, and index-worker responsiveness.
- A **fresh MCP SDK stdio session**, rather than the already connected older plugin session, exposed all **11 tools**. Universal find/search returned both types; read/glossary worked; the frame tool returned a real JPEG. A second fresh session read a meeting and an ordinary transcription from the real archive and returned a real JPEG there too. Existing Codex connections need to reconnect to discover the three added universal tools.
- The evidence viewer uses the application palette, Russian labels, compact responsive spacing, and dark native media controls. Viewer discovery records include a UI version: old workers remain available for previously issued links; new links use a current worker.
- The 15-view matrix covered all five screens at **1024×600, 1366×768, and 1920×1080**, without horizontal document overflow. Task-state checks covered queued, running, failed, cancelled, and completed views, including long titles, focus rings, empty dictionary search, unavailable GPU controls, and resizable archive panes.
- Native Electron opened the real archive and displayed a saved transcript. The real catalog contains **17 meetings and 43 ordinary transcriptions, 60 unique items**. All **204 protected legacy result/editorial files** (129 ordinary-result files and 75 meeting editorial files) retained identical hashes. All **43 legacy ordinary metadata records** remained unchanged except their new archive-root binding. The existing queue remains **paused with one waiting job and no active processing**.
- A real OBS manual scenario in an isolated archive passed: prepare/configure, preview, start, navigate to Dictionary while recording, stop from the global bar, and find the recording in Archive. The resulting **13.461-second MKV** contains a real H.264 video stream and three actual AAC audio tracks. This was a recording-flow check, not an ASR accuracy claim.
- A live backup restored a meeting, an ordinary transcript, and dictionary. Restore-test left the dictionary hash unchanged; actual restore merged without conflicts. Cross-root restored copies retained independent text; stale-root saves returned 409.
- Live editor outage/recovery passed: an unavailable server left the draft on screen and blocked navigation with a visible retry action; reconnecting saved that draft and completed navigation. Hidden preserved workspaces no longer open invisible native dialogs.
- Dictionary add/edit forms, search, and undo survive section navigation. Automatic-learning undo immediately updates the visible dictionary while preserving the transcript correction. Settings drafts and ordinary transcript drafts also survive navigation.
- Motion uses restrained 120/160/200 ms timings for purposeful disclosures, toasts, dialogs, and pointer-opened queue details. Keyboard queue opening is immediate; Escape restores focus. Reduced-motion is supported. Route changes, transcript text, resizers, and audio levels are not animated.

## CPU/GPU and dictionary qualification

> Follow-up diagnosis (2026-11-17): the large output-length difference was traced to faster-whisper CPU hallucinating past the 60-second input, with output timestamps reaching 87 seconds. The same whisper.cpp engine on CPU and Vulkan returned the same normalized 99-word sequence. The historical measurements below are retained, but their correspondence percentage is not evidence of GPU information loss, and the speed ratio is affected by CPU fallback decoding of the final remainder. See [GPU output-length diagnosis](GPU-ASR-DIAGNOSIS-2026-11-17.md).

The existing 60-second sample was measured with exact Whisper large-v3, one warm-up and three timed runs per backend and timestamp mode, including model loading:

| Mode | CPU median | Vulkan GPU median | Speedup | CPU/GPU text agreement |
|---|---:|---:|---:|---:|
| Plain | 218.0 s | 26.2 s | 8.319× | 63.4% |
| Word timestamps | 224.814 s | 27.453 s | 8.189× | 80.0% |

Both speed results exceed 1.5×, but **neither satisfies the 95% agreement gate**. Agreement compares the outputs; it is not an independently annotated accuracy score. The control sample contained no usable critical numeric/ID tokens, so ID preservation is explicitly **not assessed**, rather than passed. No verified GPU receipt was issued; Auto correctly remains on CPU. Manual Vulkan selection remains available. Decoder experiments were not adopted and production defaults were not changed to improve this particular sample.

Additional short dictionary controls covered Russian and English speech, control numbers/IDs, and silence. Existing correct Russian terms remained correct with and without hints. The English values `25` and `30` were retained in all four sampled CPU/GPU runs with and without hints; one GPU wording difference and silence hallucinations were observed. These few runs do not establish that dictionary prompts improve recognition or prevent insertions. The user can disable automatic learning and independently edit/remove prompt terms.

## Test isolation incident and recovery

An early test fixture isolated the personal dictionary and embedding model but did not isolate the selected archive. Tests consequently published synthetic ordinary-job artifacts into the real archive. This was diagnosed before acceptance. The fixture now also isolates `VT_MEETING_ROOT` and `VT_MEETING_SETTINGS`, and a regression asserts that API publication stays inside its temporary archive.

Recovery was reversible: **489 verified synthetic folders and 23 foreign tombstones** were moved into `.runtime/redesign-qa/test-leak-quarantine/`; the affected index was backed up there before removing 15 verified foreign transcription sources. Real meeting evidence, FTS, and embedding tables each retained **7,893 rows with identical content hashes**. No user media or transcripts were deleted. Subsequent full-suite runs left the real archive at its 43 portable ordinary items. Detailed inventories and preservation proofs remain in `.runtime/redesign-qa/`.

## Manual video addition

The user chose manually submitted videos to appear under **Meetings**, with one canonical editor in Archive. Transcription retains launch, progress, read-only preview, copy/export, and an action to open the meeting. Ordinary audio retains its existing editor and portable transcription storage. Already completed video history is preserved without automatic copying or reprocessing; newly submitted videos and new attempts use the full meeting path.

Implementation and end-to-end validation are complete:

- Local files are classified by actual ffprobe streams in the worker, outside the global job lock. Attached-picture cover art does not make an audio file a video. Missing/unreadable probing fails explicitly.
- Video import fixes the archive root to the job, copies and verifies the original atomically, checks cancellation during hashing/copying, and uses content identity for deduplication. Publication happens after required files initialize. Concurrent import/cancellation tests protect existing meetings and their edits. Reimporting an available source can repair a missing managed original.
- The same existing speech → screen → audio → index pipeline handles manual video and app recordings. Real stream availability is retained; no extra audio tracks, timestamps, or images are fabricated. Index and audio compression use their CPU implementation even when ASR/OCR use manually selected GPU. Genuine GPU errors still pause the queue.
- One actual **13.064-second video with one audio stream** completed in **45.393 s** using exact large-v3 on CPU. All four stages were ready; output contained raw ASR, segmented/editorial transcript, speaker sidecars, 14 retained JPEGs, 14 screen events, one Opus file, Markdown exports, manifest, and indexed evidence. It appeared once as a meeting, with no ordinary transcription stub.
- A real MP3 with attached cover art completed with Whisper small as an ordinary transcription. It had portable text, no meeting ID, and no copy in managed originals.
- The UI keeps a read-only preview for linked jobs. The archive is the canonical editor; flat simple-text PUT returns 409 for linked jobs. Returning to Transcription refreshes that preview with selection/root guards, without showing stale text while loading. Copy and TXT/MD downloads used the actual edited text, verified through browser clipboard and saved files. Opening a meeting is available even for a partial/failed linked task.
- Fresh MCP reads, search, and an actual JPEG returned the new meeting and its current edit. Backup restore-test preserved the live dictionary; actual restore verified all 63 snapshot files, including 14 frames, one audio file, the edited meeting transcript, an ordinary transcript, and merged dictionary without conflicts.
- The final result header, actions, selected task tab, and editor were inspected at 1024×600, 1366×768, and 1920×1080. No document overflow occurred; all four result actions remained accessible. The selected restored tab stays visible without moving focus.

## Additional recognition controls and limits

A diagnostic run on that short video exposed a dictionary failure: hints `1С 8.3` and `DBeaver` replaced the spoken content with those unrelated terms. The same CPU model without hints produced recognizable speech. This is a failed insertion/number-control case; it is not presented as an improvement. The raw pre-edit result and no-hints control are retained separately from the UI editorial test. Disabling automatic learning stops new remembered terms; existing terms remain recognition hints until edited or removed.

A separate local synthetic English speech control explicitly spoke `ABC-123`, `42`, and `8080`. CPU preserved all three with and without `Jira`; Vulkan without hints also preserved their literal form. Vulkan with the hint rendered `ABC - 1 2 3`, changing the identifier's spacing. This does **not** pass strict identifier preservation. These diagnostic samples do not replace the paired 60-second qualification and do not issue a GPU receipt.

## Acceptance evidence

Reproducible local evidence lives under `.runtime/redesign-qa/` (runtime artifacts are not source files):

| Requirement | Evidence |
|---|---|
| Five screens and sizes | `ui-layout-matrix.json`, 15 screen images in `screenshots/`, task-state images, `manual-video-layouts.json` |
| Readability and focus | `contrast-proof.json`, keyboard/focus/empty/error manual checks, UI tests; primary text 17.44:1, secondary 6.62:1, primary-button text 8.09:1 on their defined backgrounds |
| Motion and reduced motion | shared CSS timing tokens, measured disclosures, retained toast exits, native dialog transitions, keyboard queue checks, `ui/tests/motion.test.ts` |
| Archive, dictionary, preservation | `final-preservation-proof.json`, `backup-live-proof.json`, `cross-root-live-proof.json`, server regression suite |
| Recording and manual video | `obs-live-proof.json`, `media-classification-live-proof.json`, `manual-video-job-result.json`, `audio-cover-live-proof.json` |
| Canonical editing and exports | browser checks, `export-manual-video.txt`, `export-manual-video.md`, `screenshots/final-interface-1366.jpg` |
| Codex and restore | `mcp-real-fresh-proof.json`, `manual-video-mcp-proof.json`, `manual-video-backup-proof.json` |
| Model/quality checks | `benchmark-plain/`, `benchmark-words/`, `dictionary-regression.json`, `manual-video-no-hints-control.json`, `numeric-id-control.json` |
| Final automated checks | `final-server-tests.txt`, `final-ui-tests.txt`, lint/build outputs; Electron check/test output |

The application was reloaded against the final backend. The real archive remains at **60 unique items**, all 204 protected files and 43 ordinary metadata rows passed preservation checks, and its pre-existing queue remains paused with one waiting job. No historical results were reprocessed to perform this acceptance.


The public selected-resource preparation entry and its CLI reject retired Parakeet requests with `retired_model` before any adapter/download call. Direct dormant adapter implementations and their regression tests remain. The isolated QA server and idle QA OBS copy were stopped after acceptance; the real application remains running.
