import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from services.meeting_archive import MeetingArchive
from services.meeting_index import MeetingIndex


def _new_archive(tmp_path, name="archive"):
    root = tmp_path / name
    source = tmp_path / f"{name}.wav"
    source.write_bytes(name.encode())
    archive = MeetingArchive(root)
    return root, archive, source


def _text_index(root):
    return MeetingIndex(root, embedding_model_path=root / "missing-e5-model")


class FakeE5:
    available = True
    model_hash = "test-model-sha256"

    def __init__(self):
        self.queries = []
        self.passage_batches = []

    def encode_queries(self, values):
        self.queries.extend(values)
        return np.asarray([[1.0, 0.0] for _ in values], dtype=np.float32)

    def encode_passages(self, values):
        self.passage_batches.append(list(values))
        return np.asarray([[1.0, 0.0] if "nightly build failed" in value else [0.0, 1.0]
                           for value in values], dtype=np.float32)


class ReadOnlySpyE5(FakeE5):
    def encode_passages(self, values):
        raise AssertionError(f"read-only search attempted passage encoding for {len(values)} items")


class BlockingE5(FakeE5):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def encode_passages(self, values):
        self.started.set()
        assert self.release.wait(5), "test did not release blocked embedder"
        return super().encode_passages(values)


def test_rebuild_indexes_meetings_and_discovers_new_imports_from_folder_overlay(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source, title="Обсуждение бюджета", started_at="2026-01-02") ["meeting_id"]
    archive.write_transcript(mid, [{"id": "evidence:1.2", "start": 1, "end": 2,
                                    "text": "Бюджет на следующий квартал"}])
    index = _text_index(root)
    assert index.rebuild()["status"] == "ready"
    result = index.search("следующий квартал", mode="text")
    row = next(row for row in result["results"] if row["evidence_id"] == "evidence:1.2")
    assert row["meeting_id"] == mid
    assert row["meeting_title"] == "Обсуждение бюджета"
    assert row["meeting_date"] == "2026-01-02"
    assert row["date_confirmed"] is True
    assert row["date_provenance"] == "explicit_started_at"
    assert row["imported_at"] == archive.get_meeting(mid)["created_at"]
    assert row["source"] == source.name
    assert {"original_text", "speaker_id", "confirmed_name", "kind", "start", "end", "match_kind", "score"} <= row.keys()

    source2 = tmp_path / "two.wav"
    source2.write_bytes(b"2")
    mid2 = archive.import_file(source2, title="План ресурсов") ["meeting_id"]
    archive.write_transcript(mid2, [{"id": "new-id", "start": 0, "end": 1, "text": "Новый проект"}])
    hits = index.search("проект", mode="text")["results"]
    assert any(row["meeting_id"] == mid2 for row in hits)
    assert not index.db_path.stat().st_size == 0


def test_search_date_metadata_never_uses_import_date_and_preserves_hints(tmp_path):
    root, archive, _ = _new_archive(tmp_path)

    def add_meeting(name, text, *, started_at=None, metadata=None):
        source = tmp_path / f"{name}.wav"
        source.write_bytes(name.encode())
        mid = archive.import_file(source, started_at=started_at)["meeting_id"]
        if metadata:
            archive.update_meeting(mid, **metadata)
        archive.write_transcript(mid, [{"id": f"{name}-segment", "start": 0, "end": 1, "text": text}])
        return mid

    daily_id = add_meeting("daily", "daily date evidence", metadata={
        "meeting_date": "2026-11-12",
        "date_provenance": "filename, not independently confirmed",
    })
    unknown_id = add_meeting("unknown", "unknown date evidence")
    conflict_candidates = [
        {"date": "2026-10-12", "source": "filename"},
        {"date": "2026-11-09", "source": "source_modified_at"},
    ]
    conflict_id = add_meeting("conflict", "conflict date evidence", metadata={
        "meeting_date": None,
        "date_provenance": "conflicting filename and source modification dates",
        "date_candidates": conflict_candidates,
    })
    explicit_id = add_meeting("explicit", "explicit date evidence", started_at="2026-11-10")

    index = _text_index(root)

    def result_for(query):
        return index.search(query, mode="text")["results"][0]

    daily = result_for("daily date evidence")
    assert daily["meeting_id"] == daily_id
    assert daily["meeting_date"] == "2026-11-12"
    assert daily["date_confirmed"] is False
    assert daily["date_provenance"] == "filename, not independently confirmed"
    assert daily["imported_at"] == archive.get_meeting(daily_id)["created_at"]

    unknown = result_for("unknown date evidence")
    assert unknown["meeting_id"] == unknown_id
    assert unknown["meeting_date"] is None
    assert unknown["date_confirmed"] is False
    assert unknown["imported_at"] == archive.get_meeting(unknown_id)["created_at"]
    assert unknown["meeting_date"] != unknown["imported_at"]

    conflict = result_for("conflict date evidence")
    assert conflict["meeting_id"] == conflict_id
    assert conflict["meeting_date"] is None
    assert conflict["date_confirmed"] is False
    assert conflict["date_candidates"] == conflict_candidates

    explicit = result_for("explicit date evidence")
    assert explicit["meeting_id"] == explicit_id
    assert explicit["meeting_date"] == "2026-11-10"
    assert explicit["date_confirmed"] is True
    assert explicit["date_provenance"] == "explicit_started_at"


def test_exact_identifiers_obey_boundaries_and_confusables_without_rewriting_literals(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [
        {"id": "segment-1", "start": 0, "end": 1, "text": "Fix ARCH-427 now"},
        {"id": "segment-2", "start": 2, "end": 3, "text": "Review ARCH-4270 later"},
    ])
    archive.write_screen_events(mid, [
        {"id": "screen-1", "start": 4, "end": 5, "text": "Ticket ARCН-427"},
        {"id": "screen-2", "start": 6, "end": 7, "text": "Ticket ARCН-4270"},
    ])
    result = _text_index(root).search("ARCH-427", mode="hybrid")
    rows = result["results"]
    assert [row["evidence_id"] for row in rows] == ["segment-1", "screen-1"]
    assert [row["match_kind"] for row in rows] == ["identifier_exact", "identifier_normalized"]
    assert rows[1]["original_text"] == "Ticket ARCН-427"
    assert "ARCН-427" in rows[1]["text"]
    assert all("4270" not in row["text"] for row in rows)


def test_exact_evidence_id_punctuation_is_case_insensitive(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "Turn:01/A.2", "start": 0, "end": 1, "text": "ordinary phrase"}])
    result = _text_index(root).search("turn:01/a.2", mode="text")
    assert result["results"][0]["evidence_id"] == "Turn:01/A.2"
    assert result["results"][0]["match_kind"] == "identifier_exact"


@pytest.mark.parametrize("mode", ["hybrid", "semantic"])
def test_mixed_year_query_uses_semantics_and_ranks_relevant_exact_year_hits(tmp_path, mode):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [
        {"id": "budget", "start": 0, "end": 1,
         "text": "Бюджет проекта согласовали на 2027 год"},
        {"id": "seating", "start": 2, "end": 3,
         "text": "Рассадку в переговорной запланировали на 2027 год"},
    ])

    class ContextE5(FakeE5):
        def encode_queries(self, values):
            self.queries.extend(values)
            return np.asarray([[1.0, 0.0] for _ in values], dtype=np.float32)

        def encode_passages(self, values):
            self.passage_batches.append(list(values))
            return np.asarray([[1.0, 0.0] if "Бюджет проекта" in value else [0.0, 1.0]
                               for value in values], dtype=np.float32)

    query = "бюджет проекта на 2027"
    encoder = ContextE5()
    index = MeetingIndex(root, embedding_model_path=root / "fake-model", embedder=encoder)
    result = index.search(query, mode=mode, meeting_id=mid)
    assert encoder.queries == [query]
    assert result["results"][0]["evidence_id"] == "budget"
    assert next(row for row in result["results"] if row["evidence_id"] == "budget")["score"] > \
        next(row for row in result["results"] if row["evidence_id"] == "seating")["score"]


def test_read_only_mixed_year_query_reports_missing_cache_without_passage_encoding(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "budget", "start": 0, "end": 1,
                                    "text": "Бюджет проекта на 2027 год"}])
    encoder = ReadOnlySpyE5()
    index = MeetingIndex(root, read_only=True, embedding_model_path=root / "fake-model", embedder=encoder)
    query = "бюджет проекта на 2027"
    result = index.search(query, mode="semantic", meeting_id=mid)
    assert encoder.queries == [query]
    assert encoder.passage_batches == []
    assert result["status"] == "semantic_reindex_required"
    assert result["semantic"]["status"] == "reindex_required"
    assert result["semantic"]["missing_items"] == 2


@pytest.mark.parametrize("query", ["ARCH-427", "«ARCH-427»?", "ARCH-427,"])
def test_standalone_identifier_fast_path_skips_query_embedding(tmp_path, query):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [
        {"id": "segment", "start": 0, "end": 1, "text": "Review ARCH-427 before release"},
        {"id": "longer", "start": 2, "end": 3, "text": "Review ARCH-4270 before release"},
    ])
    encoder = FakeE5()
    index = MeetingIndex(root, embedding_model_path=root / "fake-model", embedder=encoder)
    result = index.search(query, mode="semantic", meeting_id=mid)
    assert result["total"] == 1
    assert result["results"][0]["evidence_id"] == "segment"
    assert result["semantic"]["status"] == "skipped_identifier_query"
    assert encoder.queries == []


def test_mixed_question_with_id_keeps_original_query_for_semantics(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "segment", "start": 0, "end": 1,
                                    "text": "Review ARCH-427 before release"}])
    encoder = FakeE5()
    index = MeetingIndex(root, embedding_model_path=root / "fake-model", embedder=encoder)
    query = "что решили по ARCH-427?"
    index.search(query, mode="semantic", meeting_id=mid)
    assert encoder.queries == [query]


def test_indexed_edits_replace_stale_rows_but_preserve_orphan_edit(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "seg", "start": 1, "end": 2, "speaker": "speaker_1",
                                    "text": "raw original phrase"}])
    archive.confirm_speaker(mid, "speaker_1", "Алиса")
    archive.edit_segment(mid, "seg", "corrected meeting phrase")
    index = _text_index(root)
    index.index_meeting(mid)
    row = index.search("corrected meeting", mode="text")["results"][0]
    assert row["text"] == "corrected meeting phrase"
    assert row["original_text"] == "raw original phrase"
    assert row["speaker_id"] == "speaker_1"
    assert row["confirmed_name"] == "Алиса"
    assert index.search("алиcа", mode="text")["results"] == []  # Latin c is kept literal.
    assert index.search("алиса", mode="text")["results"][0]["confirmed_name"] == "Алиса"

    transcript_path = archive.meeting_dir(mid) / "transcript.json"
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    data["raw_segments"][0]["start"] = 10
    data["raw_segments"][0]["end"] = 11
    transcript_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    index.index_meeting(mid)
    rows = index.search("corrected meeting", mode="text")["results"]
    orphan = next(row for row in rows if row["kind"] == "orphan_edit")
    active = next(row for row in index.search("raw original", mode="text")["results"] if row["kind"] == "transcript")
    assert orphan["start"] == 1 and orphan["end"] == 2
    assert orphan["original_text"] == "raw original phrase"
    assert active["text"] == "raw original phrase"
    assert not any(row["kind"] == "transcript" for row in index.search("corrected meeting", mode="text")["results"])


def test_new_meeting_is_visible_even_when_index_already_has_other_hits(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    first = archive.import_file(source)["meeting_id"]
    archive.write_transcript(first, [{"id": "old", "start": 0, "end": 1, "text": "shared phrase"}])
    index = _text_index(root)
    index.rebuild()
    newer = tmp_path / "new.wav"
    newer.write_bytes(b"new")
    second = archive.import_file(newer)["meeting_id"]
    archive.write_transcript(second, [{"id": "new", "start": 0, "end": 1, "text": "shared phrase"}])
    result = index.search("shared phrase", mode="text")
    assert {row["meeting_id"] for row in result["results"]} == {first, second}


def test_rebuild_from_archive_copy_recreates_search_index(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source, title="Archive copy") ["meeting_id"]
    archive.write_transcript(mid, [{"id": "seg", "start": 0, "end": 1, "text": "copyable searchable phrase"}])
    _text_index(root).rebuild()
    copied_root = tmp_path / "copied-archive"
    shutil.copytree(root, copied_root)
    copied = _text_index(copied_root)
    assert copied.rebuild()["indexed_meetings"] == 1
    assert copied.search("searchable phrase", mode="text")["results"][0]["meeting_id"] == mid


def test_semantic_missing_model_is_reported_and_hybrid_stays_text_only(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "seg", "start": 0, "end": 1, "text": "a unique lexical hit"}])
    index = _text_index(root)
    semantic = index.search("unique lexical hit", mode="semantic")
    assert semantic["status"].startswith("semantic_unavailable")
    assert semantic["results"] == []
    hybrid = index.search("unique lexical hit", mode="hybrid")
    assert hybrid["status"].startswith("hybrid_text_only")
    assert hybrid["results"][0]["meeting_id"] == mid


def test_semantic_paraphrase_uses_live_overlay_then_persisted_cache(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [
        {"id": "match", "start": 0, "end": 1, "text": "Nightly build failed, so launch was postponed"},
        {"id": "other", "start": 2, "end": 3, "text": "A person reviewed the office seating plan"},
    ])
    encoder = FakeE5()
    index = MeetingIndex(root, embedding_model_path=root / "fake-model", embedder=encoder)
    # No shared content words with the matching passage; semantic ranking supplies the hit.
    query = "software deployment was delayed by an unreliable compilation"
    live = index.search(query, mode="hybrid")
    assert live["results"][0]["evidence_id"] == "match"
    assert live["results"][0]["match_kind"] == "hybrid"
    assert live["status"] == "partial_live_overlay"
    assert live["semantic"]["live_encoded"] == 3  # two segments plus meeting title
    assert encoder.queries == [query]
    assert not index.db_path.exists()  # live overlay search must not persist its vectors

    index.index_meeting(mid)
    cached = index.search(query, mode="hybrid")
    assert cached["results"][0]["evidence_id"] == "match"
    assert cached["semantic"]["cache_coverage"] == 1.0
    assert cached["semantic"]["live_encoded"] == 0
    with index._connect(read_only=True) as db:
        item = db.execute("SELECT vector,content_hash,model_hash FROM embeddings LIMIT 1").fetchone()
    assert len(item[0]) == 8  # two little-endian float32 values
    assert item[1] and item[2] == encoder.model_hash


@pytest.mark.parametrize("operation", ["index", "build"])
def test_embedding_inference_does_not_hold_archive_lock_and_revalidates_rows(tmp_path, operation):
    root, archive, source = _new_archive(tmp_path, f"archive-{operation}")
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [
        {"id": "changed", "start": 0, "end": 1, "text": "nightly build failed before edit"},
        {"id": "stable", "start": 2, "end": 3, "text": "a stable searchable passage"},
    ])
    encoder = BlockingE5()
    index = MeetingIndex(root, embedding_model_path=root / "fake-model", embedder=encoder)
    errors = []
    outcomes = []

    def build():
        try:
            if operation == "index":
                outcomes.append(index.index_meeting(mid))
            else:
                outcomes.append(index.build_embeddings(mid))
        except Exception as exc:  # surfaced in the main test thread
            errors.append(exc)

    worker = threading.Thread(target=build)
    worker.start()
    assert encoder.started.wait(5), "encoding did not start"
    edited = threading.Event()

    def edit():
        archive.edit_segment(mid, "changed", "literal replacement after snapshot")
        edited.set()

    editor = threading.Thread(target=edit)
    editor.start()
    try:
        assert edited.wait(2), "archive edit waited for passage inference to finish"
    finally:
        encoder.release.set()
        editor.join(5)
        worker.join(5)

    assert not editor.is_alive()
    assert not worker.is_alive()
    assert errors == []
    if operation == "build":
        assert outcomes[0]["status"] == "partial_reindex_required"
        assert outcomes[0]["indexed_items"] == 2  # unchanged passage and meeting title
        assert outcomes[0]["changed_items"] == 1

    literal = index.search("literal replacement after snapshot", mode="text", meeting_id=mid)
    changed_row = next(row for row in literal["results"] if row["evidence_id"] == "changed")
    assert changed_row["text"] == "literal replacement after snapshot"
    assert changed_row["match_kind"] == "literal"
    with index._connect(read_only=True) as db:
        cached = {row[0] for row in db.execute(
            "SELECT evidence_id FROM embeddings WHERE meeting_id=?", (mid,))}
    assert "transcript\x1fstable" in cached
    assert "transcript\x1fchanged" not in cached


def test_build_embeddings_rejects_incomplete_embedder_batch(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [
        {"id": "one", "start": 0, "end": 1, "text": "first passage"},
        {"id": "two", "start": 2, "end": 3, "text": "second passage"},
    ])

    class ShortE5(FakeE5):
        def encode_passages(self, values):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    index = MeetingIndex(root, embedding_model_path=root / "fake-model", embedder=ShortE5())
    result = index.build_embeddings(mid)
    assert result["status"] == "unavailable_embedding_failed"
    assert result["indexed_items"] == 0
    assert "incomplete" in result["reason"]
    assert not index.db_path.exists()


def test_read_only_semantic_search_uses_only_valid_cached_passages(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    indexed_id = archive.import_file(source, title="Cached meeting")["meeting_id"]
    archive.write_transcript(indexed_id, [{"id": "release", "start": 0, "end": 1,
                                           "text": "nightly build failed, so launch was postponed"}])
    writer_embedder = FakeE5()
    writer = MeetingIndex(root, embedding_model_path=root / "fake-model", embedder=writer_embedder)
    writer.index_meeting(indexed_id)

    archive.write_transcript(indexed_id, [{"id": "release", "start": 0, "end": 1,
                                           "text": "nightly build failed after a fresh restart"}])
    readonly_embedder = ReadOnlySpyE5()
    readonly = MeetingIndex(root, read_only=True, embedding_model_path=root / "fake-model",
                            embedder=readonly_embedder)
    changed = readonly.search("nightly build failed after a fresh restart", mode="hybrid",
                              meeting_id=indexed_id)
    assert any(row["evidence_id"] == "release" and
               row["text"] == "nightly build failed after a fresh restart"
               for row in changed["results"])
    assert changed["semantic"]["status"] == "partial_cached_index"
    assert changed["semantic"]["cache_coverage"] == .5
    assert changed["semantic"]["missing_items"] == 1
    assert changed["semantic"]["live_encoded"] == 0
    assert readonly_embedder.passage_batches == []

    new_source = tmp_path / "new-after-index.wav"
    new_source.write_bytes(b"new archive item")
    new_id = archive.import_file(new_source, title="New meeting")["meeting_id"]
    archive.write_transcript(new_id, [{"id": "new-release", "start": 0, "end": 1,
                                       "text": "nightly build failed during deployment"}])
    missing = readonly.search("software delivery was blocked by a broken compilation",
                              mode="semantic", meeting_id=new_id)
    assert missing["results"] == []
    assert missing["status"] == "semantic_reindex_required"
    assert missing["semantic"]["status"] == "reindex_required"
    assert missing["semantic"]["missing_items"] == 2
    assert missing["semantic"]["cache_coverage"] == 0.0
    assert missing["semantic"]["reason"]
    assert readonly_embedder.passage_batches == []

    lexical = readonly.search("nightly build failed during deployment", mode="hybrid", meeting_id=new_id)
    assert lexical["results"][0]["meeting_id"] == new_id
    assert lexical["results"][0]["kind"] == "transcript"
    assert lexical["semantic"]["status"] == "reindex_required"
    assert readonly_embedder.passage_batches == []

    writer.index_meeting(new_id)
    semantic = readonly.search("software delivery was blocked by a broken compilation",
                               mode="semantic", meeting_id=new_id)
    new_semantic_hit = next(row for row in semantic["results"] if row["evidence_id"] == "new-release")
    assert new_semantic_hit["score"] > 0
    assert semantic["semantic"]["status"] == "ready"
    assert semantic["semantic"]["cache_coverage"] == 1.0
    assert readonly_embedder.passage_batches == []


def test_read_only_search_does_not_create_or_modify_index(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "seg", "start": 0, "end": 1, "text": "read only phrase"}])
    before = {p.relative_to(root): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
              for p in root.rglob("*")}
    index = MeetingIndex(root, read_only=True, embedding_model_path=root / "missing-model")
    assert index.search("phrase", mode="text")["results"]
    after = {p.relative_to(root): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
             for p in root.rglob("*")}
    assert before == after
    assert not index.db_path.exists()


def test_read_only_search_leaves_existing_sqlite_index_unchanged(tmp_path):
    root, archive, source = _new_archive(tmp_path)
    mid = archive.import_file(source)["meeting_id"]
    archive.write_transcript(mid, [{"id": "seg", "start": 0, "end": 1, "text": "cached phrase"}])
    _text_index(root).rebuild()
    before = {path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
              for path in root.rglob("*")}
    readonly = MeetingIndex(root, read_only=True, embedding_model_path=root / "missing-model")
    assert readonly.search("cached phrase", mode="text")["results"]
    after = {path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
             for path in root.rglob("*")}
    assert before == after
