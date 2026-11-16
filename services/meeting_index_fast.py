"""Bounded SQLite candidate search helpers for large meeting indexes."""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Callable


SEARCH_DOCUMENTS = "search_documents"
SEARCH_IDENTIFIERS = "search_identifiers"
SEARCH_FTS = "search_fts"
FORMAT_VERSION = 2
CANDIDATE_LIMIT = 2000


def ensure_schema(db: sqlite3.Connection, *, fts5_available: bool) -> None:
    db.execute(f"""CREATE TABLE IF NOT EXISTS {SEARCH_DOCUMENTS} (
        meeting_id TEXT NOT NULL, evidence_id TEXT NOT NULL, kind TEXT NOT NULL,
        start REAL, end REAL, meeting_date TEXT, payload_json TEXT NOT NULL,
        passage_hash TEXT NOT NULL,
        PRIMARY KEY(meeting_id,evidence_id,kind))""")
    db.execute(f"""CREATE INDEX IF NOT EXISTS idx_search_documents_page
        ON {SEARCH_DOCUMENTS}(meeting_id,meeting_date,start,evidence_id)""")
    db.execute(f"""CREATE TABLE IF NOT EXISTS {SEARCH_IDENTIFIERS} (
        meeting_id TEXT NOT NULL, evidence_id TEXT NOT NULL, kind TEXT NOT NULL,
        token TEXT NOT NULL, folded_token TEXT NOT NULL,
        PRIMARY KEY(meeting_id,evidence_id,kind,token))""")
    db.execute(f"""CREATE INDEX IF NOT EXISTS idx_search_identifiers_token
        ON {SEARCH_IDENTIFIERS}(token,meeting_id,evidence_id,kind)""")
    db.execute(f"""CREATE INDEX IF NOT EXISTS idx_search_identifiers_folded
        ON {SEARCH_IDENTIFIERS}(folded_token,meeting_id,evidence_id,kind)""")
    if fts5_available:
        db.execute(f"""CREATE VIRTUAL TABLE IF NOT EXISTS {SEARCH_FTS} USING fts5(
            meeting_id UNINDEXED,evidence_id UNINDEXED,kind UNINDEXED,indexed_text,
            tokenize='unicode61')""")


def write_rows(db: sqlite3.Connection, meeting_id: str, rows: list[dict], *,
               passage_text: Callable[[dict], str], passage_hash: Callable[[str], str],
               identifier_tokens: Callable[[str], list[str]],
               fold_identifier: Callable[[str], str], fts5_available: bool) -> None:
    db.execute(f"DELETE FROM {SEARCH_DOCUMENTS} WHERE meeting_id=?", (meeting_id,))
    db.execute(f"DELETE FROM {SEARCH_IDENTIFIERS} WHERE meeting_id=?", (meeting_id,))
    if fts5_available:
        db.execute(f"DELETE FROM {SEARCH_FTS} WHERE meeting_id=?", (meeting_id,))
    documents, identifiers, fts_rows = [], [], []
    for row in rows:
        mid, evidence_id, kind = row["meeting_id"], row["evidence_id"], row["kind"]
        documents.append((mid, evidence_id, kind, row.get("start"), row.get("end"),
                          row.get("meeting_date"), json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                          passage_hash(passage_text(row))))
        haystack = " ".join(str(row.get(name) or "") for name in
                            ("text", "original_text", "evidence_id", "meeting_id", "meeting_title",
                             "confirmed_name", "speaker_id"))
        for token in set(identifier_tokens(haystack)):
            identifiers.append((mid, evidence_id, kind, token, fold_identifier(token)))
        fts_rows.append((mid, evidence_id, kind, haystack))
    db.executemany(f"INSERT INTO {SEARCH_DOCUMENTS} VALUES(?,?,?,?,?,?,?,?)", documents)
    db.executemany(f"INSERT INTO {SEARCH_IDENTIFIERS} VALUES(?,?,?,?,?)", identifiers)
    if fts5_available:
        db.executemany(f"INSERT INTO {SEARCH_FTS}(meeting_id,evidence_id,kind,indexed_text) VALUES(?,?,?,?)",
                       fts_rows)


def table_names(db: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}


def batches(values: list[str], size: int = 500):
    for offset in range(0, len(values), size):
        yield values[offset:offset + size]


def _fetch_identifiers(db: sqlite3.Connection, meeting_ids: list[str], tokens: list[str],
                       folded_tokens: list[str], *, cap: int | None) -> tuple[dict[tuple[str, str, str], str], bool]:
    matches: dict[tuple[str, str, str], str] = {}
    truncated = False
    if not tokens and not folded_tokens:
        return matches, truncated
    token_batches = list(batches(tokens, 150)) or [[]]
    folded_batches = list(batches(folded_tokens, 150)) or [[]]
    for mids in batches(meeting_ids, 400):
        if not mids:
            continue
        mid_params = ",".join("?" for _ in mids)
        for token_offset in range(max(len(token_batches), len(folded_batches))):
            exact = token_batches[token_offset] if token_offset < len(token_batches) else []
            folded = folded_batches[token_offset] if token_offset < len(folded_batches) else []
            clauses, params = [], list(mids)
            if exact:
                ph = ",".join("?" for _ in exact)
                clauses.append(f"i.token IN ({ph})")
                params.extend(exact)
            if folded:
                ph = ",".join("?" for _ in folded)
                clauses.append(f"i.folded_token IN ({ph})")
                params.extend(folded)
            if not clauses:
                continue
            exact_order = f"CASE WHEN i.token IN ({','.join('?' for _ in exact)}) THEN 0 ELSE 1 END" if exact else "1"
            if exact:
                params.extend(exact)
            sql = f"""SELECT d.meeting_id,d.evidence_id,d.kind,d.payload_json
                FROM {SEARCH_IDENTIFIERS} AS i JOIN {SEARCH_DOCUMENTS} AS d
                  ON d.meeting_id=i.meeting_id AND d.evidence_id=i.evidence_id AND d.kind=i.kind
                WHERE d.meeting_id IN ({mid_params}) AND ({' OR '.join(clauses)})
                ORDER BY {exact_order},d.meeting_id,d.meeting_date,d.start,d.evidence_id,d.kind"""
            if cap is not None:
                sql += f" LIMIT {cap + 1}"
            for mid, evidence_id, kind, payload in db.execute(sql, params):
                matches[(str(mid), str(evidence_id), str(kind))] = str(payload)
                if cap is not None and len(matches) > cap:
                    truncated = True
                    return dict(list(matches.items())[:cap]), truncated
    return matches, truncated


def _current_meetings(index: Any, meeting_id: str | None, exact_meeting: str | None) -> list[dict]:
    selected = meeting_id or exact_meeting
    if selected:
        return [index.archive.get_meeting(selected)]
    return index.archive.list_meetings(limit=10**9)


def search(index: Any, query: str, *, limit: int, offset: int, meeting_id: str | None,
           mode: str, identifier_only: bool, exact_meeting: str | None,
           capabilities: dict, index_state: str) -> dict | None:
    """Run bounded text search or complete identifier search on derived tables."""
    if exact_meeting and meeting_id and exact_meeting.casefold() != meeting_id.casefold():
        return None
    if not identifier_only and mode != "text":
        return None  # Semantic top-K is not implemented in this bounded path yet.
    if not index.db_path.is_file() or index._database_uses_wal():
        return None
    try:
        metas = _current_meetings(index, meeting_id, exact_meeting)
        current_signatures = {
            str(meta["meeting_id"]): index._snapshot_source_signature(
                index.archive, str(meta["meeting_id"]), meta)
            for meta in metas if meta.get("meeting_id")
        }
        current_ids = list(current_signatures)
        if not current_ids:
            return None
        with index._db(read_only=True) as db:
            names = table_names(db)
            required = {"document_snapshots", SEARCH_DOCUMENTS, SEARCH_IDENTIFIERS}
            if not required <= names or (not identifier_only and SEARCH_FTS not in names):
                return None
            source_signatures: dict[str, str] = {}
            indexed_count = 0
            for batch in batches(current_ids):
                placeholders = ",".join("?" for _ in batch)
                source_signatures.update({str(mid): str(signature) for mid, signature in db.execute(
                    f"SELECT meeting_id,source_signature FROM document_snapshots "
                    f"WHERE meeting_id IN ({placeholders})", batch)})
                indexed_count += int(db.execute(
                    f"SELECT COUNT(*) FROM {SEARCH_DOCUMENTS} WHERE meeting_id IN ({placeholders})",
                    batch).fetchone()[0])
            if indexed_count <= int(index.FAST_SEARCH_MIN_DOCUMENTS):
                return None
            unchanged_ids = [mid for mid in current_ids
                             if source_signatures.get(mid) == current_signatures.get(mid)]
            changed_ids = [mid for mid in current_ids if mid not in set(unchanged_ids)]
            id_tokens = list(dict.fromkeys(index._identifier_tokens(query)))
            folded_tokens = list(dict.fromkeys(index._fold_identifier(token) for token in id_tokens))
            fts_candidates: dict[tuple[str, str, str], tuple[str, float]] = {}
            fts_truncated = False
            if identifier_only:
                id_candidates, id_truncated = _fetch_identifiers(
                    db, unchanged_ids, id_tokens, folded_tokens, cap=None)
            else:
                id_candidates, id_truncated = _fetch_identifiers(
                    db, unchanged_ids, id_tokens, folded_tokens, cap=CANDIDATE_LIMIT)
                terms = list(dict.fromkeys(re.findall(r"\w+", query.casefold(), re.UNICODE)))
                if not terms:
                    return None
                fts_query = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
                if fts_query:
                    for mids in batches(unchanged_ids):
                        placeholders = ",".join("?" for _ in mids)
                        sql = f"""SELECT d.meeting_id,d.evidence_id,d.kind,d.payload_json,
                            bm25({SEARCH_FTS}) AS rank
                            FROM {SEARCH_FTS} JOIN {SEARCH_DOCUMENTS} AS d
                              ON d.meeting_id={SEARCH_FTS}.meeting_id
                             AND d.evidence_id={SEARCH_FTS}.evidence_id
                             AND d.kind={SEARCH_FTS}.kind
                            WHERE {SEARCH_FTS} MATCH ? AND {SEARCH_FTS}.meeting_id IN ({placeholders})
                            ORDER BY rank,d.meeting_id,d.evidence_id LIMIT {CANDIDATE_LIMIT + 1}"""
                        rows = list(db.execute(sql, [fts_query, *mids]))
                        if len(rows) > CANDIDATE_LIMIT:
                            fts_truncated = True
                        for mid, evidence_id, kind, payload, rank in rows:
                            key = (str(mid), str(evidence_id), str(kind))
                            fts_candidates[key] = (str(payload), float(rank))
                if len(fts_candidates) > CANDIDATE_LIMIT:
                    fts_truncated = True
                    best = sorted(fts_candidates.items(), key=lambda item: (item[1][1], item[0]))[:CANDIDATE_LIMIT]
                    fts_candidates = dict(best)
            payloads = dict(id_candidates)
            if not identifier_only:
                for key, (payload, _bm25) in sorted(fts_candidates.items(),
                                                     key=lambda item: (item[1][1], item[0])):
                    payloads.setdefault(key, payload)
    except (sqlite3.Error, OSError, ValueError, KeyError):
        return None

    live_rows: list[dict] = []
    for live_mid in changed_ids:
        try:
            live_rows.extend(index._doc_rows(index.archive, live_mid))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue

    candidate_rows: dict[tuple[str, str, str], dict] = {}
    corrupt_meetings: set[str] = set()
    for (mid, evidence_id, kind), raw in payloads.items():
        try:
            row = json.loads(raw)
            if not index._valid_document_row(mid, row) or row["evidence_id"] != evidence_id or row["kind"] != kind:
                corrupt_meetings.add(mid)
                continue
            candidate_rows[(mid, evidence_id, kind)] = row
        except (TypeError, ValueError, json.JSONDecodeError):
            corrupt_meetings.add(mid)
    if corrupt_meetings:
        for key in [key for key in candidate_rows if key[0] in corrupt_meetings]:
            del candidate_rows[key]
        live_ids = set(changed_ids)
        for corrupt_id in corrupt_meetings - live_ids:
            try:
                live_rows.extend(index._doc_rows(index.archive, corrupt_id))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
    for row in live_rows:
        key = (row["meeting_id"], row["evidence_id"], row["kind"])
        candidate_rows[key] = row

    needle = query.strip()
    exact_query_tokens = set(id_tokens)
    folded_query_tokens = set(folded_tokens)
    exact_ids: set[tuple[str, str, str]] = set()
    normalized_ids: set[tuple[str, str, str]] = set()
    lexical_scores: dict[tuple[str, str, str], float] = {}
    for key, row in candidate_rows.items():
        haystack = " ".join(str(row.get(name) or "") for name in
                            ("text", "original_text", "evidence_id", "meeting_id", "meeting_title",
                             "confirmed_name", "speaker_id"))
        if exact_meeting or row["evidence_id"].casefold() == needle.casefold() or row["meeting_id"].casefold() == needle.casefold():
            exact_ids.add(key)
        if exact_query_tokens:
            candidate_tokens = set(index._identifier_tokens(haystack))
            if exact_query_tokens & candidate_tokens:
                exact_ids.add(key)
            elif folded_query_tokens & {index._fold_identifier(token) for token in candidate_tokens}:
                normalized_ids.add(key)
        if not identifier_only:
            score, _kind = index._lexical_score(needle, row)
            if score:
                lexical_scores[key] = score
    if identifier_only:
        ranked = []
        for key in exact_ids | normalized_ids:
            row = dict(candidate_rows[key])
            row["score"] = 2.0 if key in exact_ids else 1.5
            row["match_kind"] = "identifier_exact" if key in exact_ids else "identifier_normalized"
            ranked.append(row)
        ranked.sort(key=lambda row: (0 if row["match_kind"] == "identifier_exact" else 1,
                                     row["meeting_date"] or "", float(row["start"] or 0), row["evidence_id"]))
        total = len(ranked)
        page = ranked[offset:offset + limit]
        return {"results": page, "total": total,
                "next_offset": offset + len(page) if offset + len(page) < total else None,
                "status": ("archive_fallback_legacy_wal_needs_reindex" if index_state == "needs_reindex"
                           else "ready" if index.db_path.is_file() else "archive_fallback_index_missing"),
                "index_state": index_state, "mode": mode,
                "semantic": {"status": "skipped_identifier_query"}, "capabilities": capabilities}

    lexical_order = sorted(lexical_scores, key=lambda key: (-lexical_scores[key], key))
    lexical_rank = {key: pos for pos, key in enumerate(lexical_order, 1)}
    found_keys = set(lexical_order) | exact_ids | normalized_ids
    ranked = []
    for key in found_keys:
        row = dict(candidate_rows[key])
        if key in exact_ids:
            match_kind, score = "identifier_exact", 2.0 + lexical_scores.get(key, 0.0)
        elif key in normalized_ids:
            match_kind, score = "identifier_normalized", 1.5 + lexical_scores.get(key, 0.0)
        else:
            match_kind = "literal" if lexical_scores.get(key) == 1 else "lexical"
            score = lexical_scores.get(key, 0.0)
        row.update({"score": float(score), "match_kind": match_kind})
        ranked.append(row)
    ranked.sort(key=lambda row: (
        0 if row["match_kind"] == "identifier_exact" else 1 if row["match_kind"] == "identifier_normalized" else 2,
        -row["score"], row["meeting_date"] or "", float(row["start"] or 0), row["evidence_id"]))
    ranked = index._collapse_near_duplicate_screens(ranked)
    total = len(ranked)
    page = ranked[offset:offset + limit]
    candidate_limited = bool(id_truncated or fts_truncated)
    result = {"results": page, "total": total,
              "next_offset": offset + len(page) if offset + len(page) < total else None,
              "status": "candidate_limited" if candidate_limited else "ready",
              "index_state": index_state, "mode": mode,
              "semantic": {"status": "not_requested"}, "capabilities": capabilities,
              "candidate_limited": candidate_limited, "truncated": candidate_limited,
              "candidate_limit_reason": (
                  f"Search candidates were limited to {CANDIDATE_LIMIT} FTS and identifier matches."
                  if candidate_limited else None)}
    return result
