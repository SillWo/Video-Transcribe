---
name: video-transcribe-frames
description: Inspect video meetings in the Video-Transcribe archive when a question needs retained screen frames, visual context, or evidence linking frames to transcript excerpts.
---

# Inspect video frames

Find the video with `find_archive_items(kind="video")` or `search_archive`. Read relevant speech and screen-event context with `read_timeline`, following `next_offset` until the needed range is covered. Request retained JPEGs with `get_frames(meeting_id, source="stored")`; follow its `next_offset` for later frames.

## Match metadata to images

The tool response starts with a JSON text block followed by zero or more JPEG `ImageContent` blocks. Parse the JSON and match each item's zero-based `image_index` to that index among `ImageContent` blocks only; do not count the JSON block. Items with `image_error` have no JPEG and no `image_index`. Indices for actual JPEGs are contiguous within each response page, including after missing files. Use the item's `frame` relative path, `start`/`end`, and available sampling and timing uncertainty fields as provenance; do not open arbitrary filesystem paths.

Frame times are seconds from media zero. `start` is the frame capture time; `end` is an approximate boundary for the observed window, not an exact event end. Describe `end` as approximate and include `time_uncertainty_seconds` when available.

Base claims about what is visible on an image. Base claims about what was said on transcript evidence. OCR and pixels do not establish speech, and transcript text does not establish visual details. If the relevant image or text is unavailable, state that evidence gap without filling it by inference.

Cite each visual claim with the video meeting, actual frame timestamp, source type, and an `open_evidence` link opened at the event's `start`. Keep the link's timestamp aligned with that event.

## Legacy and frames-only metadata

Read both legacy screen-event records and v2 records with top-level `schema_version: 2` and `mode: "frames_only"`. A v2 event can include `id`, relative `frame`, `start`, approximate `end`, `width`, `height`, `sample_interval_seconds`, `time_uncertainty_seconds`, and `reason`; fields may be absent in legacy records. Treat all frame paths as provenance supplied by the archive, not as permission to inspect arbitrary filesystem paths.
