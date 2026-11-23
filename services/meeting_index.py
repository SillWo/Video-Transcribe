"""Rebuildable text and local-vector index over meeting archive folders.

Meeting folders remain authoritative. Search overlays their current content on
the cached indexes and never repairs or writes anything.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import sys
import unicodedata
from contextlib import contextmanager, nullcontext
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

from .meeting_archive import MeetingArchive, _archive_lock
from .meeting_embeddings import MAX_TOKENS, MODEL_NAME, MultilingualE5, content_hash
from . import meeting_index_fast


_IDENTIFIER_RE = re.compile(r"[\w]+(?:[-./:#][\w]+)*", re.UNICODE)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_CLOCK_TOKEN_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)")
_CONFUSABLES = str.maketrans({
    "а": "a", "в": "b", "с": "c", "е": "e", "н": "h", "к": "k",
    "м": "m", "о": "o", "р": "p", "т": "t", "х": "x", "у": "y",
    "і": "i", "ј": "j", "ѕ": "s", "ӏ": "l",
})


class LegacyWalIndexError(sqlite3.OperationalError):
    """Raised when a read-only worker sees an index using WAL journaling."""


class MeetingIndex:
    """SQLite FTS5/vector cache with archive-folder overlays for fresh reads."""

    DB_NAME = "meeting-index.sqlite3"
    FAST_SEARCH_ENABLED = False
    FAST_SEARCH_MIN_DOCUMENTS = 5000

    def __init__(self, root: str | Path, read_only: bool = False,
                 embedding_model_path: str | Path | None = None, embedder: Any = None):
        self.root = Path(root).expanduser().resolve()
        self.read_only = read_only
        self.archive = MeetingArchive(self.root, read_only=True)
        self.db_path = self.root / self.DB_NAME
        configured = embedding_model_path or os.environ.get("MEETING_EMBEDDING_MODEL")
        self.embedding_model_path = Path(configured).expanduser() if configured else None
        self._embedder = embedder
        self._chunk_tokenizer = None
        self._chunk_tokenizer_loaded = False

    @property
    def capabilities(self) -> dict:
        model_path = self._model_path()
        if self._embedder is not None:
            available = bool(getattr(self._embedder, "available", False))
        else:
            available = (model_path / "model.onnx").is_file() and (model_path / "tokenizer.json").is_file()
        return {
            "text_search": ("archive_fallback_needs_reindex" if self._database_uses_wal()
                            else "fts5" if self._has_fts5() else "archive_fallback"),
            "exact_id_search": True,
            "embeddings": {
                "available": available,
                "status": "local_model_available_lazy" if available else "unavailable_local_model_missing",
                "model": MODEL_NAME if available else None,
                "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3" if available else None,
            },
            "database_exists": self.db_path.is_file(),
            "index_state": "needs_reindex" if self._database_uses_wal() else "available",
        }

    def _database_uses_wal(self) -> bool:
        try:
            with self.db_path.open("rb") as handle:
                header = handle.read(20)
            return len(header) >= 20 and header[:16] == b"SQLite format 3\x00" and header[18:20] == b"\x02\x02"
        except OSError:
            return False

    def _model_path(self) -> Path:
        if self.embedding_model_path is not None:
            return self.embedding_model_path
        return MultilingualE5().model_path

    @staticmethod
    def _has_fts5() -> bool:
        try:
            with sqlite3.connect(":memory:") as db:
                db.execute("CREATE VIRTUAL TABLE ftscheck USING fts5(text)")
            return True
        except sqlite3.OperationalError:
            return False

    def _connect(self, *, read_only: bool | None = None) -> sqlite3.Connection:
        ro = self.read_only if read_only is None else read_only
        if ro:
            if self._database_uses_wal():
                raise LegacyWalIndexError("SQLite index uses legacy WAL mode; read from archive folders and reindex")
            uri = self.db_path.resolve().as_uri() + "?mode=ro"
            db = sqlite3.connect(uri, uri=True, timeout=.5)
            db.execute("PRAGMA query_only=ON")
            return db
        self.root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path, timeout=30)
        if self._database_uses_wal():
            checkpoint = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint and int(checkpoint[0]) != 0:
                db.close()
                raise sqlite3.OperationalError("Could not checkpoint legacy WAL index")
        db.execute("PRAGMA journal_mode=DELETE")
        return db

    @contextmanager
    def _db(self, *, read_only: bool | None = None):
        ro = self.read_only if read_only is None else read_only
        db = self._connect(read_only=ro)
        try:
            yield db
            if not ro:
                db.commit()
        except Exception:
            if not ro:
                db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return default

    @staticmethod
    def _interval_matches(raw: dict, override: dict) -> bool:
        return (override.get("original_start") == raw.get("start")
                and override.get("original_end") == raw.get("end"))

    @classmethod
    def _doc_rows(cls, archive: MeetingArchive, meeting_id: str) -> list[dict]:
        folder = archive.meeting_dir(meeting_id)
        meta = archive.get_meeting(meeting_id)
        transcript = archive._read_transcript(meeting_id)
        speakers = cls._read_json(folder / "speakers.json", {"confirmed": {}}).get("confirmed", {})
        overrides = transcript.get("overrides", {})
        segments = transcript.get("raw_segments", [])
        present: set[str] = set()
        rows: list[dict] = []
        started_at = meta.get("started_at")
        meeting_date = started_at or meta.get("meeting_date")
        date_provenance = meta.get("date_provenance")
        if started_at and not date_provenance:
            date_provenance = "explicit_started_at"

        def row(evidence_id: str, kind: str, start: Any, end: Any, text: Any,
                original_text: Any, *, speaker_id: Any = None, confirmed_name: Any = None) -> dict:
            return {
                "meeting_id": meeting_id, "evidence_id": evidence_id, "kind": kind,
                "start": start, "end": end, "text": str(text or ""),
                "original_text": str(original_text or ""), "speaker_id": speaker_id,
                "confirmed_name": confirmed_name, "meeting_title": str(meta.get("title") or ""),
                "meeting_date": meeting_date, "date_confirmed": bool(started_at),
                "date_provenance": date_provenance,
                "date_candidates": meta.get("date_candidates"),
                "imported_at": meta.get("created_at"),
                "source": meta.get("source_filename") or meta.get("source_type"),
            }

        for segment in segments:
            evidence_id = str(segment.get("id", ""))
            present.add(evidence_id)
            raw_text = str(segment.get("text", ""))
            edit = overrides.get(evidence_id)
            effective_text = raw_text
            if isinstance(edit, dict) and cls._interval_matches(segment, edit):
                effective_text = str(edit.get("text", raw_text))
            elif edit is not None and not isinstance(edit, dict):
                # Historical archives kept only a string, with no source interval.
                effective_text = str(edit)
            speaker = str(segment.get("speaker")) if segment.get("speaker") is not None else None
            confirmed = speakers.get(speaker, {}) if speaker else {}
            name = confirmed.get("name") if isinstance(confirmed, dict) else None
            rows.append(row(evidence_id, "transcript", segment.get("start"), segment.get("end"),
                            effective_text, raw_text, speaker_id=speaker, confirmed_name=name))

        raw_by_id = {str(seg.get("id", "")): seg for seg in segments}
        for evidence_id, edit in overrides.items():
            key = str(evidence_id)
            raw = raw_by_id.get(key)
            if raw and isinstance(edit, dict) and cls._interval_matches(raw, edit):
                continue
            if isinstance(edit, dict):
                text = edit.get("text", "")
                original = edit.get("original_text", raw.get("text", "") if raw else "")
                start, end = edit.get("original_start"), edit.get("original_end")
            else:
                text, original = edit, ""
                start = end = None
            rows.append(row(key, "orphan_edit", start, end, text, original))

        screen = cls._read_json(folder / "screen-events.json", {"events": []})
        # Version 2 screen events are visual references only. Keep legacy
        # archives searchable while excluding OCR from newly captured rows.
        screen_events = (screen.get("events", [])
                        if not (screen.get("schema_version") == 2
                                and screen.get("mode") == "frames_only") else [])
        for event in screen_events:
            literal = str(event.get("text", ""))
            region_text = " ".join(str(item.get("text", "")) for item in event.get("regions", []) if item.get("text"))
            rows.append(row(str(event.get("id", "")), "screen", event.get("start"), event.get("end"),
                            " ".join(part for part in (literal, region_text) if part), literal))
        rows.append(row(meeting_id, "meeting", None, None, meta.get("title", ""), meta.get("title", "")))
        return rows

    def _ensure_schema(self, db: sqlite3.Connection) -> None:
        db.execute("""CREATE TABLE IF NOT EXISTS evidence (
            meeting_id TEXT NOT NULL, evidence_id TEXT NOT NULL, kind TEXT NOT NULL,
            start REAL, end REAL, text TEXT NOT NULL, original_text TEXT NOT NULL,
            PRIMARY KEY(meeting_id,evidence_id,kind))""")
        if self._has_fts5():
            db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
                meeting_id UNINDEXED,evidence_id UNINDEXED,kind UNINDEXED,text,original_text,
                tokenize='unicode61')""")
        db.execute("""CREATE TABLE IF NOT EXISTS embeddings (
            meeting_id TEXT NOT NULL,evidence_id TEXT NOT NULL,model TEXT NOT NULL,
            vector BLOB NOT NULL,content_hash TEXT NOT NULL DEFAULT '',
            model_hash TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(meeting_id,evidence_id,model))""")
        db.execute("""CREATE TABLE IF NOT EXISTS document_snapshots (
            meeting_id TEXT PRIMARY KEY, source_signature TEXT NOT NULL, rows_json TEXT NOT NULL)""")
        if self.FAST_SEARCH_ENABLED:
            meeting_index_fast.ensure_schema(db, fts5_available=self._has_fts5())
        columns = {row[1] for row in db.execute("PRAGMA table_info(embeddings)")}
        if "content_hash" not in columns:
            db.execute("ALTER TABLE embeddings ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")
        if "model_hash" not in columns:
            db.execute("ALTER TABLE embeddings ADD COLUMN model_hash TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def _embedding_key(row: dict) -> str:
        return f"{row['kind']}\x1f{row['evidence_id']}"

    @staticmethod
    def _passage_text(row: dict) -> str:
        parts = [row.get("text", "")]
        if row.get("confirmed_name"):
            parts.append(f"Speaker: {row['confirmed_name']}")
        if row.get("meeting_title"):
            parts.append(f"Meeting: {row['meeting_title']}")
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _snapshot_source_signature(archive: MeetingArchive, meeting_id: str, meta: dict) -> str:
        folder = archive.meeting_dir(meeting_id)
        file_stats = {}
        for name in ("transcript.json", "speakers.json", "screen-events.json"):
            try:
                stat = (folder / name).stat()
                file_stats[name] = [stat.st_mtime_ns, stat.st_size, getattr(stat, "st_ino", None)]
            except OSError:
                file_stats[name] = None
        metadata = {key: meta.get(key) for key in (
            "title", "started_at", "meeting_date", "date_provenance", "date_candidates",
            "created_at", "source_filename", "source_type")}
        return json.dumps({"files": file_stats, "metadata": metadata}, ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"))

    def _active_embedder(self):
        if self._embedder is None:
            path = self._model_path()
            self._embedder = MultilingualE5(path)
        return self._embedder

    def _embeddings_available(self) -> bool:
        try:
            return bool(self._active_embedder().available)
        except Exception:
            return False

    def _write_rows(self, db: sqlite3.Connection, meeting_id: str, rows: list[dict]) -> None:
        self._ensure_schema(db)
        db.execute("DELETE FROM evidence WHERE meeting_id=?", (meeting_id,))
        db.execute("DELETE FROM embeddings WHERE meeting_id=?", (meeting_id,))
        try:
            db.execute("DELETE FROM evidence_fts WHERE meeting_id=?", (meeting_id,))
        except sqlite3.OperationalError:
            pass
        db.executemany("INSERT INTO evidence VALUES(?,?,?,?,?,?,?)", [
            (r["meeting_id"], r["evidence_id"], r["kind"], r["start"], r["end"], r["text"], r["original_text"])
            for r in rows])
        try:
            db.executemany("INSERT INTO evidence_fts(meeting_id,evidence_id,kind,text,original_text) VALUES(?,?,?,?,?)", [
                (r["meeting_id"], r["evidence_id"], r["kind"], r["text"], r["original_text"])
                for r in rows])
        except sqlite3.OperationalError:
            pass
        meta = self.archive.get_meeting(meeting_id)
        signature = self._snapshot_source_signature(self.archive, meeting_id, meta)
        rows_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        db.execute("""INSERT OR REPLACE INTO document_snapshots
            (meeting_id,source_signature,rows_json) VALUES(?,?,?)""",
                   (meeting_id, signature, rows_json))
        if self.FAST_SEARCH_ENABLED:
            meeting_index_fast.write_rows(
                db, meeting_id, rows, passage_text=self._passage_text, passage_hash=content_hash,
                identifier_tokens=self._identifier_tokens, fold_identifier=self._fold_identifier,
                fts5_available=self._has_fts5())

    @staticmethod
    def _transcription_source(job_id: str) -> str:
        return f"transcription:{job_id}"

    def _load_chunk_tokenizer(self):
        if self._chunk_tokenizer_loaded:
            return self._chunk_tokenizer
        self._chunk_tokenizer_loaded = True
        try:
            from tokenizers import Tokenizer

            tokenizer_path = self._model_path() / "tokenizer.json"
            if tokenizer_path.is_file():
                self._chunk_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception:
            self._chunk_tokenizer = None
        return self._chunk_tokenizer

    @staticmethod
    def _chunk_text(text: str, tokenizer=None) -> list[tuple[int, int, str]]:
        if not text:
            return []
        if tokenizer is not None:
            try:
                encoding = tokenizer.encode(text, add_special_tokens=False)
                offsets = [
                    (int(start), int(end))
                    for start, end in encoding.offsets
                    if int(end) > int(start)
                ]
                if offsets:
                    chunks = []
                    start_token = 0
                    content_token_limit = min(384, MAX_TOKENS - 128)
                    while start_token < len(offsets):
                        end_token = min(start_token + content_token_limit, len(offsets))
                        start_char = offsets[start_token][0]
                        end_char = offsets[end_token - 1][1]
                        chunks.append((start_char, end_char, text[start_char:end_char]))
                        if end_token == len(offsets):
                            break
                        start_token = end_token - 48
                    return chunks
            except Exception:
                pass
        chunk_chars = 1800
        overlap_chars = 240
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunks.append((start, end, text[start:end]))
            if end == len(text):
                break
            start = max(start + 1, end - overlap_chars)
        return chunks

    @staticmethod
    def _bounded_transcription_title(record: dict, tokenizer=None) -> str:
        raw_title = record.get("title") or record.get("source") or "Transcription"
        title = Path(str(raw_title)).name if record.get("title") is None else str(raw_title)
        if tokenizer is not None:
            try:
                encoding = tokenizer.encode(title, add_special_tokens=False)
                if len(encoding.ids) > 64 and encoding.offsets:
                    title = title[: encoding.offsets[63][1]]
            except Exception:
                title = title[:512]
        else:
            title = title[:512]
        return title

    def _transcription_rows_for_record(self, record: dict, text: str) -> list[dict]:
        job_id = str(record.get("id") or "")
        source_id = self._transcription_source(job_id)
        tokenizer = self._load_chunk_tokenizer()
        title = self._bounded_transcription_title(record, tokenizer)
        chunks = self._chunk_text(text, tokenizer)
        rows = []
        for chunk_index, (char_start, char_end, chunk_text) in enumerate(chunks):
            evidence_id = f"{job_id}:chunk:{chunk_index:06d}"
            rows.append(
                {
                    "meeting_id": source_id,
                    "evidence_id": evidence_id,
                    "kind": "transcription",
                    "start": None,
                    "end": None,
                    "text": chunk_text,
                    "original_text": "",
                    "speaker_id": None,
                    "confirmed_name": None,
                    "meeting_title": title,
                    "meeting_date": None,
                    "date_confirmed": False,
                    "date_provenance": None,
                    "date_candidates": None,
                    "imported_at": record.get("createdAt"),
                    "source": "transcription",
                    "job_id": job_id,
                    "title": title,
                    "chunk_index": chunk_index,
                    "char_start": char_start,
                    "char_end": char_end,
                    "full_text_length": len(text),
                }
            )
        return rows

    def _transcription_rows(self, job_id: str | None = None) -> list[dict]:
        from services.archive_catalog import list_items, read_text

        rows = []
        for record in list_items(self.root):
            current_id = str(record.get("id") or "")
            if job_id is not None and current_id != str(job_id):
                continue
            text = read_text(self.root, current_id)
            if text is None:
                continue
            rows.extend(self._transcription_rows_for_record(record, text))
        return rows

    def _write_transcription_rows(self, db: sqlite3.Connection, rows: list[dict]) -> None:
        if not rows:
            return
        source_id = rows[0]["meeting_id"]
        self._ensure_schema(db)
        db.execute("DELETE FROM evidence WHERE meeting_id=?", (source_id,))
        db.execute("DELETE FROM embeddings WHERE meeting_id=?", (source_id,))
        try:
            db.execute("DELETE FROM evidence_fts WHERE meeting_id=?", (source_id,))
        except sqlite3.OperationalError:
            pass
        db.executemany(
            "INSERT INTO evidence VALUES(?,?,?,?,?,?,?)",
            [
                (
                    row["meeting_id"],
                    row["evidence_id"],
                    row["kind"],
                    row["start"],
                    row["end"],
                    row["text"],
                    row["original_text"],
                )
                for row in rows
            ],
        )
        try:
            db.executemany(
                "INSERT INTO evidence_fts(meeting_id,evidence_id,kind,text,original_text) VALUES(?,?,?,?,?)",
                [
                    (
                        row["meeting_id"],
                        row["evidence_id"],
                        row["kind"],
                        row["text"],
                        row["original_text"],
                    )
                    for row in rows
                ],
            )
        except sqlite3.OperationalError:
            pass
        snapshot = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        signature = content_hash(
            "\n".join(
                f"{row['evidence_id']}:{content_hash(self._passage_text(row))}"
                for row in rows
            )
        )
        db.execute(
            "INSERT OR REPLACE INTO document_snapshots(meeting_id,source_signature,rows_json) VALUES(?,?,?)",
            (source_id, signature, snapshot),
        )
        if self.FAST_SEARCH_ENABLED:
            meeting_index_fast.write_rows(
                db,
                source_id,
                rows,
                passage_text=self._passage_text,
                passage_hash=content_hash,
                identifier_tokens=self._identifier_tokens,
                fold_identifier=self._fold_identifier,
                fts5_available=self._has_fts5(),
            )

    def index_transcription(
        self,
        record: dict,
        text: str | None = None,
        *,
        cancellation_check=None,
        publish_guard=None,
    ) -> dict:
        """Publish one portable transcript and its cached E5 vector outside history locks."""
        if self.read_only:
            raise PermissionError("MeetingIndex is read-only")
        job_id = str(record.get("id") or "")
        if not job_id:
            raise ValueError("A transcription job id is required")
        if text is None:
            from services.archive_catalog import read_text

            text = read_text(self.root, job_id)
        if text is None:
            return {"status": "missing_transcript", "indexed_items": 0}
        if cancellation_check and cancellation_check():
            return {"status": "cancelled", "indexed_items": 0}
        rows = self._transcription_rows_for_record(record, text)
        if not rows:
            return {"status": "empty_transcript", "indexed_items": 0}
        source_id = rows[0]["meeting_id"]
        signature = content_hash(
            "\n".join(
                f"{row['evidence_id']}:{content_hash(self._passage_text(row))}"
                for row in rows
            )
        )
        with _archive_lock(self.root):
            if cancellation_check and cancellation_check():
                return {"status": "cancelled", "indexed_items": 0}
            existing_signature = None
            if self.db_path.is_file() and not self._database_uses_wal():
                try:
                    with self._db(read_only=True) as db:
                        found = db.execute(
                            "SELECT source_signature FROM document_snapshots WHERE meeting_id=?",
                            (source_id,),
                        ).fetchone()
                        existing_signature = found[0] if found else None
                except sqlite3.Error:
                    existing_signature = None
            if existing_signature != signature:
                guard = publish_guard() if publish_guard else nullcontext(True)
                with guard as allowed:
                    if not allowed:
                        return {"status": "cancelled", "indexed_items": 0}
                    with self._db(read_only=False) as db:
                        self._write_transcription_rows(db, rows)
        if not self._embeddings_available():
            return {
                "status": "text_indexed_embeddings_unavailable",
                "indexed_items": len(rows),
            }
        embedder = self._active_embedder()
        model_hash = embedder.model_hash
        cached = False
        try:
            with self._db(read_only=True) as db:
                entries = {
                    str(key): str(content)
                    for key, content in db.execute(
                        "SELECT evidence_id,content_hash FROM embeddings WHERE meeting_id=? AND model=? AND model_hash=?",
                        (source_id, MODEL_NAME, model_hash),
                    )
                }
                cached = all(
                    entries.get(self._embedding_key(row))
                    == content_hash(self._passage_text(row))
                    for row in rows
                )
        except sqlite3.Error:
            cached = False
        if cached:
            return {"status": "ready", "indexed_items": len(rows), "cached": True}
        result = self._build_embeddings(
            source_id,
            rows,
            cancellation_check=cancellation_check,
            publish_guard=publish_guard,
        )
        return {**result, "cached": False}

    def index_meeting(self, meeting_id: str) -> int:
        if self.read_only:
            raise PermissionError("MeetingIndex is read-only")
        meeting_id = MeetingArchive._valid_id(meeting_id)
        with _archive_lock(self.root):
            rows = self._doc_rows(self.archive, meeting_id)
            with self._db(read_only=False) as db:
                self._write_rows(db, meeting_id, rows)
        # Model loading and inference can take minutes. In particular, do not
        # retain the cross-process archive lock while doing either.
        if self._embeddings_available():
            self._build_embeddings(meeting_id, rows)
        return len(rows)

    def rebuild(self) -> dict:
        if self.read_only:
            raise PermissionError("MeetingIndex is read-only")
        if not self._has_fts5():
            return {"indexed_meetings": 0, "indexed_items": 0, "status": "fts5_unavailable"}
        with _archive_lock(self.root):
            with self._db(read_only=False) as db:
                self._ensure_schema(db)
                db.execute("DELETE FROM evidence")
                db.execute("DELETE FROM evidence_fts")
                db.execute("DELETE FROM embeddings")
                db.execute("DELETE FROM document_snapshots")
                for table in ("search_documents", "search_identifiers", "search_fts"):
                    try:
                        db.execute(f"DELETE FROM {table}")
                    except sqlite3.OperationalError:
                        pass
        count = items = 0
        for meta in self.archive.list_meetings(limit=10**9):
            items += self.index_meeting(meta["meeting_id"])
            count += 1
        from services.archive_catalog import list_items, read_text

        transcriptions = 0
        for record in list_items(self.root):
            text = read_text(self.root, str(record.get("id") or ""))
            if text is None:
                continue
            indexed = self.index_transcription(record, text)
            transcriptions += 1
            items += int(indexed.get("indexed_items", 0))
        return {
            "indexed_meetings": count,
            "indexed_transcriptions": transcriptions,
            "indexed_items": items,
            "status": "ready",
        }

    def _build_embeddings(
        self,
        meeting_id: str,
        rows: list[dict],
        cancellation_check=None,
        publish_guard=None,
    ) -> dict:
        embedder = self._active_embedder()
        if not embedder.available:
            return {"status": "unavailable_local_model_missing", "indexed_items": 0}
        try:
            texts = [self._passage_text(row) for row in rows]
            vectors = embedder.encode_passages(texts)
            model_hash = embedder.model_hash
            vectors = np.asarray(vectors, dtype=np.float32)
            if (vectors.ndim != 2 or vectors.shape[0] != len(rows)
                    or vectors.shape[1] == 0 or not np.isfinite(vectors).all()):
                raise ValueError("embedder returned an incomplete or invalid vector batch")
        except Exception as exc:
            return {"status": "unavailable_embedding_failed", "reason": str(exc), "indexed_items": 0}
        if cancellation_check and cancellation_check():
            return {"status": "cancelled", "indexed_items": 0}
        model = MODEL_NAME
        # Revalidate under the archive lock. An edit/import may have completed
        # while inference was in progress, so only publish vectors whose key
        # and passage content still match the authoritative folder snapshot.
        with _archive_lock(self.root):
            if cancellation_check and cancellation_check():
                return {"status": "cancelled", "indexed_items": 0}
            if meeting_id.startswith("transcription:"):
                job_id = meeting_id.removeprefix("transcription:")
                current_rows = self._transcription_rows(job_id)
            else:
                current_rows = self._doc_rows(self.archive, meeting_id)
            current_by_key = {self._embedding_key(row): row for row in current_rows}
            snapshot_hashes = {self._embedding_key(row): content_hash(self._passage_text(row))
                               for row in rows}
            current_hashes = {key: content_hash(self._passage_text(row))
                              for key, row in current_by_key.items()}
            changed = sum(snapshot_hashes.get(key) != current_hashes.get(key)
                          for key in snapshot_hashes.keys() | current_hashes.keys())
            valid = []
            for row, text, vector in zip(rows, texts, vectors):
                key = self._embedding_key(row)
                current = current_by_key.get(key)
                if current is not None and content_hash(self._passage_text(current)) == content_hash(text):
                    valid.append((meeting_id, key, model, np.asarray(vector, dtype="<f4").tobytes(),
                                  content_hash(text), model_hash))
            guard = publish_guard() if publish_guard else nullcontext(True)
            with guard as allowed:
                if not allowed:
                    return {"status": "cancelled", "indexed_items": 0}
                with self._db(read_only=False) as db:
                    if meeting_id.startswith("transcription:"):
                        self._write_transcription_rows(db, current_rows)
                    else:
                        self._write_rows(db, meeting_id, current_rows)
                    self._ensure_schema(db)
                    db.executemany("""INSERT INTO embeddings
                        (meeting_id,evidence_id,model,vector,content_hash,model_hash) VALUES(?,?,?,?,?,?)""", valid)
        result = {"status": "ready" if changed == 0 else "partial_reindex_required",
                  "indexed_items": len(valid), "requested_items": len(rows),
                  "changed_items": changed, "model": model, "model_hash": model_hash}
        if changed:
            result["reindex_required"] = True
        return result

    def build_embeddings(self, meeting_id: str) -> dict:
        """Explicitly build/update cached passage vectors. This never downloads a model."""
        if self.read_only:
            raise PermissionError("MeetingIndex is read-only")
        meeting_id = MeetingArchive._valid_id(meeting_id)
        with _archive_lock(self.root):
            rows = self._doc_rows(self.archive, meeting_id)
        if not self._embeddings_available():
            return {"status": "unavailable_local_model_missing", "indexed_items": 0}
        return self._build_embeddings(meeting_id, rows)

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = [term for term in query.split() if term]
        return " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)

    def _all_folder_rows(self, meeting_id: str | None = None) -> list[dict]:
        metas = [self.archive.get_meeting(meeting_id)] if meeting_id else self.archive.list_meetings(limit=10**9)
        current_signatures = {
            str(meta["meeting_id"]): self._snapshot_source_signature(self.archive, str(meta["meeting_id"]), meta)
            for meta in metas if meta.get("meeting_id")
        }
        cached = self._read_document_snapshots(current_signatures)
        rows: list[dict] = []
        for meta in metas:
            current_id = str(meta.get("meeting_id", ""))
            try:
                snapshot_rows = cached.get(current_id)
                rows.extend(snapshot_rows if snapshot_rows is not None
                            else self._doc_rows(self.archive, current_id))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        if meeting_id is None:
            rows.extend(self._transcription_rows())
        return rows

    def _read_document_snapshots(self, signatures: dict[str, str]) -> dict[str, list[dict]]:
        """Load only current archive snapshots through one read-only DB connection."""
        if not signatures or not self.db_path.is_file() or self._database_uses_wal():
            return {}
        ids = list(signatures)
        found: dict[str, str] = {}
        try:
            with self._db(read_only=True) as db:
                for offset in range(0, len(ids), 500):
                    batch = ids[offset:offset + 500]
                    placeholders = ",".join("?" for _ in batch)
                    for meeting_id, signature in db.execute(
                            f"SELECT meeting_id,source_signature FROM document_snapshots "
                            f"WHERE meeting_id IN ({placeholders})", batch):
                        found[str(meeting_id)] = str(signature)
                matching = [meeting_id for meeting_id, signature in found.items()
                            if signatures.get(meeting_id) == signature]
                payloads: dict[str, str] = {}
                for offset in range(0, len(matching), 500):
                    batch = matching[offset:offset + 500]
                    placeholders = ",".join("?" for _ in batch)
                    payloads.update({str(meeting_id): str(rows_json) for meeting_id, rows_json in db.execute(
                        f"SELECT meeting_id,rows_json FROM document_snapshots "
                        f"WHERE meeting_id IN ({placeholders})", batch)})
        except sqlite3.Error:
            # A legacy index or interrupted schema upgrade remains searchable
            # from the authoritative archive folders.
            return {}

        snapshots: dict[str, list[dict]] = {}
        for meeting_id in matching:
            try:
                decoded = json.loads(payloads[meeting_id])
                if not self._valid_document_snapshot(meeting_id, decoded):
                    continue
                snapshots[meeting_id] = decoded
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return snapshots

    @staticmethod
    def _valid_document_snapshot(meeting_id: str, rows: Any) -> bool:
        if not isinstance(rows, list) or not rows:
            return False
        for row in rows:
            if not MeetingIndex._valid_document_row(meeting_id, row):
                return False
        return any(row["kind"] == "meeting" and row["evidence_id"] == meeting_id for row in rows)

    @staticmethod
    def _valid_document_row(meeting_id: str, row: Any) -> bool:
        required = {
            "meeting_id", "evidence_id", "kind", "start", "end", "text", "original_text",
            "speaker_id", "confirmed_name", "meeting_title", "meeting_date", "date_confirmed",
            "date_provenance", "date_candidates", "imported_at", "source",
        }
        string_fields = ("meeting_id", "evidence_id", "kind", "text", "original_text", "meeting_title")
        optional_string_fields = ("speaker_id", "confirmed_name", "date_provenance", "imported_at", "source")
        kinds = {"transcript", "orphan_edit", "screen", "meeting"}
        if not isinstance(row, dict) or not required <= row.keys():
            return False
        if any(not isinstance(row.get(key), str) for key in string_fields):
            return False
        if row["meeting_id"] != meeting_id or row["kind"] not in kinds:
            return False
        if any(row.get(key) is not None and not isinstance(row.get(key), str)
               for key in optional_string_fields):
            return False
        if row.get("meeting_date") is not None and not isinstance(row["meeting_date"], str):
            return False
        if not isinstance(row.get("date_confirmed"), bool):
            return False
        if row.get("date_candidates") is not None and not isinstance(row["date_candidates"], (list, dict)):
            return False
        for key in ("start", "end"):
            value = row.get(key)
            if value is None:
                continue
            try:
                if not math.isfinite(float(value)):
                    return False
            except (TypeError, ValueError, OverflowError):
                return False
        return True

    def _fts_ranks(self, query: str, meeting_id: str | None) -> dict[tuple[str, str, str], int]:
        if not self.db_path.is_file() or self._database_uses_wal() or not self._has_fts5():
            return {}
        try:
            with self._db(read_only=True) as db:
                sql = ("SELECT meeting_id,evidence_id,kind FROM evidence_fts "
                       "WHERE evidence_fts MATCH ? ORDER BY bm25(evidence_fts)")
                params: list[Any] = [self._fts_query(query)]
                if meeting_id:
                    sql = ("SELECT meeting_id,evidence_id,kind FROM evidence_fts "
                           "WHERE evidence_fts MATCH ? AND meeting_id=? ORDER BY bm25(evidence_fts)")
                    params.append(meeting_id)
                return {(str(mid), str(eid), str(kind)): rank
                        for rank, (mid, eid, kind) in enumerate(db.execute(sql, params), 1)}
        except sqlite3.Error:
            return {}

    @staticmethod
    def _identifier_tokens(value: str) -> list[str]:
        return [token for token in _IDENTIFIER_RE.findall(value)
                if any(char.isdigit() for char in token) or any(mark in token for mark in "-./:#")]

    @staticmethod
    def _identifier_only_query(value: str) -> bool:
        tokens = value.strip().split()
        def unwrapped(token: str) -> str:
            start, end = 0, len(token)
            while start < end and token[start] != "_" and unicodedata.category(token[start]).startswith("P"):
                start += 1
            while end > start and token[end - 1] != "_" and unicodedata.category(token[end - 1]).startswith("P"):
                end -= 1
            return token[start:end]

        tokens = [unwrapped(token) for token in tokens]
        return bool(tokens) and all(
            _IDENTIFIER_RE.fullmatch(token)
            and (any(char.isdigit() for char in token) or any(mark in token for mark in "-./:#"))
            for token in tokens
        )

    @staticmethod
    def _fold_identifier(value: str) -> str:
        return value.casefold().translate(_CONFUSABLES)

    @staticmethod
    def _lexical_score(query: str, row: dict) -> tuple[float, str | None]:
        haystack = " ".join(str(row.get(key) or "") for key in
                            ("text", "original_text", "evidence_id", "meeting_id", "meeting_title", "confirmed_name", "speaker_id", "job_id"))
        folded_query = query.casefold()
        folded_haystack = haystack.casefold()
        if folded_query and folded_query in folded_haystack:
            return 1.0, "literal"
        terms = list(dict.fromkeys(_TOKEN_RE.findall(folded_query)))
        haystack_terms = set(_TOKEN_RE.findall(folded_haystack))
        if not terms:
            return 0.0, None
        matched = sum(term in haystack_terms for term in terms)
        if not matched:
            return 0.0, None
        return matched / len(terms), "lexical"

    @staticmethod
    def _screen_duplicate_signature(text: str) -> tuple[frozenset[str], tuple[str, ...], bool]:
        """Return bounded similarity tokens and full-text ordered numeric tokens."""
        value = str(text or "")
        # Clock values are volatile OCR details. Other numeric and ID tokens
        # must match in both value and order, even beyond the similarity cap.
        without_clocks = _CLOCK_TOKEN_RE.sub(" ", value)
        critical_tokens = tuple(
            token.casefold() for token in _IDENTIFIER_RE.findall(without_clocks)
            if any(char.isdigit() for char in token)
        )

        # Refuse to group long OCR passages rather than allowing a changed tail
        # to disappear behind a truncated similarity comparison.
        if len(value) > 4096:
            return frozenset(), critical_tokens, True
        tokens = _TOKEN_RE.findall(without_clocks.casefold())
        if len(tokens) > 1024:
            return frozenset(), critical_tokens, True
        return frozenset(tokens), critical_tokens, False

    @classmethod
    def _collapse_near_duplicate_screens(cls, ranked: list[dict]) -> list[dict]:
        """Collapse near-identical screen evidence within a meeting and 30 seconds."""
        representatives: list[dict] = []
        # Each bucket spans 30 seconds; only the same and adjacent buckets can
        # contain a representative within the allowed time distance.
        buckets: dict[tuple[str, int], list[tuple[dict, frozenset[str], frozenset[str]]]] = {}
        for original in ranked:
            if original.get("kind") != "screen" or original.get("start") is None:
                representatives.append(original)
                continue
            try:
                start = float(original["start"])
            except (TypeError, ValueError):
                representatives.append(original)
                continue
            if not math.isfinite(start):
                representatives.append(original)
                continue
            bucket = int(start // 30)
            tokens, numbers, over_limit = cls._screen_duplicate_signature(original.get("text", ""))
            if over_limit:
                representatives.append(original)
                continue
            duplicate = None
            for nearby_bucket in (bucket - 1, bucket, bucket + 1):
                for representative, rep_tokens, rep_numbers in buckets.get(
                        (str(original.get("meeting_id", "")), nearby_bucket), ())[:64]:
                    rep_start = float(representative["start"])
                    if abs(start - rep_start) > 30 or not tokens or not rep_tokens or numbers != rep_numbers:
                        continue
                    similarity = len(tokens & rep_tokens) / len(tokens | rep_tokens)
                    if similarity >= .92:
                        duplicate = representative
                        break
                if duplicate is not None:
                    break
            if duplicate is None:
                row = dict(original)
                row["near_duplicate_count"] = 0
                row["related_start"] = original.get("start")
                row["related_end"] = original.get("end")
                representatives.append(row)
                buckets.setdefault((str(row.get("meeting_id", "")), bucket), []).append((row, tokens, numbers))
                continue

            duplicate["near_duplicate_count"] += 1
            starts = [value for value in (duplicate.get("related_start"), start) if value is not None]
            ends = [value for value in (duplicate.get("related_end"), original.get("end")) if value is not None]
            duplicate["related_start"] = min(starts, key=float)
            duplicate["related_end"] = max(ends, key=float)
        return representatives

    def _semantic_scores(self, query: str, rows: list[dict]) -> tuple[dict[tuple[str, str, str], float] | None, dict]:
        if not self._embeddings_available():
            return None, {"status": "unavailable_local_model_missing", "cache_coverage": 0.0, "live_encoded": 0}
        embedder = self._active_embedder()
        try:
            model_hash = embedder.model_hash
            query_vector = np.asarray(embedder.encode_queries([query])[0], dtype=np.float32)
        except Exception as exc:
            return None, {"status": "unavailable_embedding_failed", "reason": str(exc), "cache_coverage": 0.0, "live_encoded": 0}
        cached: dict[tuple[str, str], tuple[np.ndarray, str]] = {}
        try:
            if self.db_path.is_file() and not self._database_uses_wal():
                with self._db(read_only=True) as db:
                    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    if "embeddings" in tables:
                        for mid, key, blob, cached_content, cached_model_hash in db.execute(
                                "SELECT meeting_id,evidence_id,vector,content_hash,model_hash FROM embeddings WHERE model=? AND model_hash=?",
                                (MODEL_NAME, model_hash)):
                            cached[(str(mid), str(key))] = (np.frombuffer(blob, dtype="<f4").copy(), str(cached_content or ""))
        except sqlite3.Error:
            cached = {}
        vectors: dict[tuple[str, str, str], np.ndarray] = {}
        missing_rows, missing_texts = [], []
        cache_hits = 0
        for row in rows:
            text = self._passage_text(row)
            key = self._embedding_key(row)
            cache_item = cached.get((row["meeting_id"], key))
            if cache_item is not None:
                vector, stored_hash = cache_item
                if vector.size and len(vector) == len(query_vector) and stored_hash == content_hash(text):
                    vectors[(row["meeting_id"], row["evidence_id"], row["kind"])] = vector
                    cache_hits += 1
                    continue
            missing_rows.append(row)
            missing_texts.append(text)
        if self.read_only and missing_rows:
            missing_ids = [f"{row['meeting_id']}:{row['kind']}:{row['evidence_id']}"
                           for row in missing_rows[:50]]
            missing_count = len(missing_rows)
            coverage = cache_hits / max(1, len(rows))
            reason = "Read-only search uses cached passage vectors only; explicitly reindex to refresh missing or stale vectors."
            if not vectors:
                return None, {"status": "reindex_required", "reason": reason,
                              "cache_coverage": coverage, "missing_items": missing_count,
                              "missing_item_ids": missing_ids, "missing_items_truncated": missing_count > len(missing_ids),
                              "live_encoded": 0, "indexed_items": len(rows)}
            scores = {key: float(np.dot(vector, query_vector)) for key, vector in vectors.items()}
            return scores, {"status": "partial_cached_index", "reason": reason,
                            "cache_coverage": coverage, "missing_items": missing_count,
                            "missing_item_ids": missing_ids, "missing_items_truncated": missing_count > len(missing_ids),
                            "live_encoded": 0, "indexed_items": len(rows)}
        try:
            live = embedder.encode_passages(missing_texts) if missing_texts else np.empty((0, len(query_vector)), dtype=np.float32)
        except Exception as exc:
            return None, {"status": "unavailable_embedding_failed", "reason": str(exc),
                          "cache_coverage": cache_hits / max(1, len(rows)), "live_encoded": 0}
        for row, vector in zip(missing_rows, live):
            vectors[(row["meeting_id"], row["evidence_id"], row["kind"])] = np.asarray(vector, dtype=np.float32)
        scores = {key: float(np.dot(vector, query_vector)) for key, vector in vectors.items()}
        return scores, {"status": "ready" if not missing_rows else "partial_live_overlay",
                        "cache_coverage": cache_hits / max(1, len(rows)), "live_encoded": len(missing_rows),
                        "indexed_items": len(rows)}

    def search(self, query: str, limit: int = 20, offset: int = 0,
               meeting_id: str | None = None, mode: str = "hybrid") -> dict:
        limit, offset = max(0, int(limit)), max(0, int(offset))
        if mode not in {"hybrid", "text", "semantic"}:
            raise ValueError("mode must be 'hybrid', 'text', or 'semantic'")
        if meeting_id is not None:
            meeting_id = MeetingArchive._valid_id(meeting_id)
        needle = (query or "").strip()
        capabilities = self.capabilities
        if not needle:
            return {"results": [], "total": 0, "status": "empty_query", "mode": mode,
                    "capabilities": capabilities, "index_state": capabilities["index_state"], "next_offset": None}
        index_state = "needs_reindex" if self._database_uses_wal() else "available" if self.db_path.is_file() else "not_built"
        try:
            exact_meeting = MeetingArchive._valid_id(needle)
        except ValueError:
            exact_meeting = None
        identifier_query = self._identifier_tokens(needle)
        identifier_only = self._identifier_only_query(needle)
        if self.FAST_SEARCH_ENABLED:
            fast_result = meeting_index_fast.search(
                self, needle, limit=limit, offset=offset, meeting_id=meeting_id, mode=mode,
                identifier_only=identifier_only, exact_meeting=exact_meeting,
                capabilities=capabilities, index_state=index_state)
            if fast_result is not None:
                return fast_result
        rows = self._all_folder_rows(meeting_id)
        if exact_meeting:
            rows = [
                row
                for row in rows
                if row["meeting_id"].casefold() == exact_meeting.casefold()
                or (
                    row.get("kind") == "transcription"
                    and row["evidence_id"].casefold() == needle.casefold()
                )
            ]

        lexical_scores: dict[tuple[str, str, str], float] = {}
        result_rows: dict[tuple[str, str, str], dict] = {}
        exact_ids: set[tuple[str, str, str]] = set()
        normalized_ids: set[tuple[str, str, str]] = set()
        query_identifiers = set(identifier_query)
        folded_queries = {self._fold_identifier(value) for value in identifier_query}
        for row in rows:
            key = (row["meeting_id"], row["evidence_id"], row["kind"])
            haystack = " ".join(str(row.get(name) or "") for name in
                                ("text", "original_text", "evidence_id", "meeting_id", "meeting_title", "confirmed_name", "speaker_id", "job_id"))
            if exact_meeting or row["evidence_id"].casefold() == needle.casefold() or row["meeting_id"].casefold() == needle.casefold():
                exact_ids.add(key)
            if identifier_query:
                candidates = set(self._identifier_tokens(haystack))
                identifier_match = bool(query_identifiers & candidates)
                if identifier_match:
                    exact_ids.add(key)
                elif folded_queries & {self._fold_identifier(value) for value in candidates}:
                    normalized_ids.add(key)
                    identifier_match = True
            if identifier_only:
                # Standalone IDs stay bounded, so ARCH-427 never matches
                # ARCH-4270 by substring.
                score, _kind = (0.0, None) if not (key in exact_ids or key in normalized_ids) else (1.0, "identifier")
            else:
                score, _kind = self._lexical_score(needle, row)
                if query_identifiers and not identifier_match:
                    score = 0.0
            if score:
                lexical_scores[key] = score
            result_rows[key] = row

        if identifier_only:
            exact_keys = exact_ids | normalized_ids
            ranked_exact = []
            for key in exact_keys:
                row = dict(result_rows[key])
                row["score"] = 2.0 if key in exact_ids else 1.5
                row["match_kind"] = "identifier_exact" if key in exact_ids else "identifier_normalized"
                ranked_exact.append(row)
            ranked_exact.sort(key=lambda row: (0 if row["match_kind"] == "identifier_exact" else 1,
                                               row["meeting_date"] or "", float(row["start"] or 0), row["evidence_id"]))
            total = len(ranked_exact)
            page = ranked_exact[max(0, offset):max(0, offset) + max(0, limit)]
            return {"results": page, "total": total,
                    "next_offset": offset + len(page) if offset + len(page) < total else None,
                    "status": ("archive_fallback_legacy_wal_needs_reindex" if index_state == "needs_reindex"
                               else "ready" if self.db_path.is_file() else "archive_fallback_index_missing"),
                    "index_state": index_state,
                    "mode": mode, "semantic": {"status": "skipped_identifier_query"},
                    "capabilities": capabilities}

        fts_ranks = self._fts_ranks(needle, meeting_id) if not identifier_only else {}
        if not identifier_only:
            for key, rank in fts_ranks.items():
                # Cached FTS entries are hints only; a current folder row must
                # still match so an old edited version cannot leak into search.
                if key in result_rows and lexical_scores.get(key, 0.0) > 0:
                    lexical_scores[key] = max(lexical_scores.get(key, 0.0), 1.0 / (1.0 + rank * .05))

        semantic_scores = None
        semantic_info = {"status": "not_requested", "cache_coverage": None, "live_encoded": 0}
        if mode in {"hybrid", "semantic"}:
            semantic_scores, semantic_info = self._semantic_scores(needle, rows)
        if mode == "semantic" and semantic_scores is None:
            return {"results": [], "total": 0, "status": "semantic_" + semantic_info["status"],
                    "mode": mode, "semantic": semantic_info, "capabilities": capabilities,
                    "index_state": index_state, "next_offset": None}

        lexical_order = sorted(lexical_scores, key=lambda key: (-lexical_scores[key], key))
        lexical_rank = {key: pos for pos, key in enumerate(lexical_order, 1)}
        semantic_order = sorted(semantic_scores or {}, key=lambda key: (-semantic_scores[key], key))
        semantic_rank = {key: pos for pos, key in enumerate(semantic_order, 1)}
        found_keys = set(lexical_order) | set(semantic_order) | exact_ids | normalized_ids
        ranked = []
        for key in found_keys:
            row = dict(result_rows[key])
            if mode == "text" or semantic_scores is None:
                identifier_relevance = lexical_scores.get(key, 0.0)
            elif mode == "semantic":
                identifier_relevance = semantic_scores.get(key, 0.0)
            else:
                identifier_relevance = 2 * (
                    ((1 / (60 + lexical_rank[key])) if key in lexical_rank else 0.0)
                    + ((1 / (60 + semantic_rank[key])) if key in semantic_rank else 0.0))
            if key in exact_ids:
                match_kind, score = "identifier_exact", 2.0 + identifier_relevance
            elif key in normalized_ids:
                match_kind, score = "identifier_normalized", 1.5 + identifier_relevance
            elif mode == "text":
                match_kind, score = "literal" if lexical_scores.get(key) == 1 else "lexical", lexical_scores.get(key, 0.0)
            elif mode == "semantic":
                match_kind, score = "semantic", semantic_scores.get(key, 0.0) if semantic_scores else 0.0
            elif semantic_scores is not None:
                rrf = ((1 / (60 + lexical_rank[key])) if key in lexical_rank else 0.0) + ((1 / (60 + semantic_rank[key])) if key in semantic_rank else 0.0)
                score = 2 * rrf
                match_kind = "hybrid"
            else:
                match_kind, score = "literal" if lexical_scores.get(key) == 1 else "lexical", lexical_scores.get(key, 0.0)
            row.update({"score": float(score), "match_kind": match_kind})
            ranked.append(row)
        ranked.sort(key=lambda row: (
            0 if row["match_kind"] == "identifier_exact" else 1 if row["match_kind"] == "identifier_normalized" else 2,
            -row["score"], row["meeting_date"] or "", float(row["start"] or 0), row["evidence_id"]))
        ranked = self._collapse_near_duplicate_screens(ranked)
        total = len(ranked)
        page = ranked[max(0, offset):max(0, offset) + max(0, limit)]
        if mode == "semantic":
            status = "semantic_" + semantic_info["status"]
        elif mode == "hybrid" and semantic_scores is None:
            status = "hybrid_text_only_" + semantic_info["status"]
        elif semantic_info.get("status") == "partial_live_overlay":
            status = "partial_live_overlay"
        elif semantic_info.get("status") == "partial_cached_index":
            status = "partial_cached_index"
        elif index_state == "needs_reindex" and mode == "text":
            status = "archive_fallback_legacy_wal_needs_reindex"
        else:
            status = "ready" if self.db_path.is_file() else "archive_fallback_index_missing"
        return {"results": page, "total": total,
                "next_offset": offset + len(page) if offset + len(page) < total else None,
                "status": status, "index_state": index_state, "mode": mode, "semantic": semantic_info,
                "capabilities": capabilities}


def _search_stdio() -> int:
    """Serve one read-only JSON search request over stdin/stdout."""
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict) or not request.get("root"):
            raise ValueError("request must be an object with a root path")
        with redirect_stdout(sys.stderr):
            index = MeetingIndex(request["root"], read_only=True)
            result = index.search(query=str(request.get("query", "")),
                                  limit=int(request.get("limit", 20)),
                                  offset=int(request.get("offset", 0)),
                                  meeting_id=request.get("meeting_id"),
                                  mode=str(request.get("mode", "hybrid")))
        query = str(request.get("query", ""))
        for row in result.get("results", []):
            if row.get("kind") == "transcription" and isinstance(row.get("text"), str):
                full_text = row["text"]
                row["full_text_length"] = len(full_text)
                snippet, start, end = _context_snippet(full_text, query)
                row["snippet_char_start"] = row.get("char_start", 0) + start
                row["snippet_char_end"] = row.get("char_start", 0) + end
                row["text"] = snippet
        sys.stdout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()
        return 0
    except Exception as exc:
        error = {
            "results": [],
            "total": 0,
            "next_offset": None,
            "status": "worker_error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        sys.stdout.write(json.dumps(error, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return 1


def _context_snippet(text: str, query: str, limit: int = 1200) -> tuple[str, int, int]:
    if len(text) <= limit:
        return text, 0, len(text)
    folded_text = text.casefold()
    folded_query = query.casefold().strip()
    position = folded_text.find(folded_query) if folded_query else -1
    if position < 0:
        identifier_queries = MeetingIndex._identifier_tokens(query)
        for match in _IDENTIFIER_RE.finditer(text):
            candidate = match.group(0)
            if any(
                MeetingIndex._fold_identifier(candidate)
                == MeetingIndex._fold_identifier(needle)
                for needle in identifier_queries
            ):
                position = match.start()
                break
    if position < 0:
        for token in _TOKEN_RE.findall(folded_query):
            position = folded_text.find(token)
            if position >= 0:
                break
    if position < 0:
        position = 0
    start = max(0, position - limit // 3)
    end = min(len(text), start + limit)
    if end - start < limit:
        start = max(0, end - limit)
    prefix = "…" if start else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end] + suffix, start, end


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--search-stdio"]:
        return _search_stdio()
    sys.stderr.write("Usage: python -m services.meeting_index --search-stdio\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

