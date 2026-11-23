"""Application-wide personal vocabulary with atomic, reversible edits."""

from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
import uuid
from pathlib import Path

_LOCK = threading.RLock()


def dictionary_path(project_root=None) -> Path:
    override = os.environ.get("VT_DICTIONARY_PATH")
    if override:
        return Path(override).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return (Path(local) / "VideoTranscribe" / "personal-dictionary.json").resolve()
    return (Path.home() / ".video-transcribe" / "personal-dictionary.json").resolve()


def _norm(term: str) -> str:
    return " ".join(unicodedata.normalize("NFC", term).split()).casefold()


def _default():
    return {
        "version": 0,
        "entries": [],
        "tombstones": [],
        "migrations": [],
        "undo": {},
        "settings": {"auto_learn": True},
    }


class PersonalDictionary:
    def __init__(self, path=None, project_root=None):
        self.path = (
            Path(path).expanduser().resolve() if path else dictionary_path(project_root)
        )

    def _load(self):
        if not self.path.exists():
            return _default()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Personal dictionary is unreadable: {self.path}") from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"Personal dictionary has an invalid structure: {self.path}"
            )
        base = _default()
        base.update(data)
        if not isinstance(base.get("entries"), list):
            raise ValueError(f"Personal dictionary entries are invalid: {self.path}")
        return base

    def _save(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8", newline="") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def _mutate(self, fn):
        with _LOCK, _process_lock(self.path):
            d = self._load()
            result = fn(d)
            self._save(d)
            return result

    def snapshot(self):
        d = self._load()
        entries = sorted(d["entries"], key=lambda e: (_norm(e["term"]), e["id"]))
        return {
            "version": int(d.get("version", 0)),
            "terms": [e["term"] for e in entries],
        }

    def export_state(self):
        with _LOCK, _process_lock(self.path):
            return self._load()

    def merge_snapshot(self, snapshot):
        incoming = snapshot.get("entries", []) if isinstance(snapshot, dict) else []

        def work(d):
            present = {_norm(e["term"]): e for e in d["entries"]}
            tomb = {entry.get("key") for entry in d["tombstones"]}
            added = []
            conflicts = []
            for entry in incoming:
                if not isinstance(entry, dict) or not isinstance(
                    entry.get("term"), str
                ):
                    continue
                key = _norm(entry["term"])
                if key in present:
                    if present[key] != entry:
                        conflicts.append(
                            {
                                "term": entry["term"],
                                "current": present[key],
                                "backup": entry,
                            }
                        )
                    continue
                if key in tomb:
                    conflicts.append(
                        {
                            "term": entry["term"],
                            "current": None,
                            "backup": entry,
                            "reason": "deleted_locally",
                        }
                    )
                    continue
                value = {
                    "id": uuid.uuid4().hex,
                    "term": _clean_term(entry["term"]),
                    "definition": _clean_definition(entry.get("definition")),
                    "provenance": list(entry.get("provenance") or []),
                }
                d["entries"].append(value)
                present[key] = value
                added.append(value["term"])
            if added:
                d["version"] += 1
            return {"added": added, "conflicts": conflicts}

        return self._mutate(work)

    def list(self, query=""):
        d = self._load()
        q = _norm(query)
        return sorted(
            [
                e
                for e in d["entries"]
                if not q or q in _norm(e["term"]) or q in _norm(e.get("definition", ""))
            ],
            key=lambda e: (_norm(e["term"]), e["id"]),
        )

    def add(self, term, definition=None, provenance=None):
        term = _clean_term(term)
        definition = _clean_definition(definition)

        def work(d):
            key = _norm(term)
            prior = next((e for e in d["entries"] if _norm(e["term"]) == key), None)
            if prior:
                if definition is not None:
                    prior["definition"] = definition
                if provenance:
                    prior.setdefault("provenance", []).append(dict(provenance))
                d["version"] += 1
                return dict(prior)
            item = {
                "id": uuid.uuid4().hex,
                "term": term,
                "definition": definition,
                "provenance": [dict(provenance)] if provenance else [],
            }
            d["entries"].append(item)
            d["version"] += 1
            return dict(item)

        return self._mutate(work)

    def update(self, entry_id, term, definition=None):
        term = _clean_term(term)
        definition = _clean_definition(definition)

        def work(d):
            item = next((e for e in d["entries"] if e["id"] == entry_id), None)
            if not item:
                raise KeyError(entry_id)
            dup = next(
                (
                    e
                    for e in d["entries"]
                    if e["id"] != entry_id and _norm(e["term"]) == _norm(term)
                ),
                None,
            )
            if dup:
                raise ValueError("A matching term already exists")
            item.update(term=term, definition=definition)
            d["version"] += 1
            return dict(item)

        return self._mutate(work)

    def delete(self, entry_id):
        token = uuid.uuid4().hex

        def work(d):
            item = next((e for e in d["entries"] if e["id"] == entry_id), None)
            if not item:
                raise KeyError(entry_id)
            d["entries"].remove(item)
            d["tombstones"].append(
                {"id": entry_id, "key": _norm(item["term"]), "term": item["term"]}
            )
            d["undo"][token] = item
            d["version"] += 1
            return token

        self._mutate(work)
        return token

    def undo(self, token):
        def work(d):
            item = d["undo"].pop(token, None)
            if item is None:
                raise KeyError(token)
            if isinstance(item, dict) and item.get("action") == "remove_learned":
                target = next(
                    (e for e in d["entries"] if e["id"] == item["item_id"]), None
                )
                if target:
                    d["entries"].remove(target)
                    d["tombstones"].append(
                        {
                            "id": target["id"],
                            "key": _norm(target["term"]),
                            "term": target["term"],
                        }
                    )
                d["version"] += 1
                return {"id": item["item_id"], "removed": target is not None}
            if not any(_norm(e["term"]) == _norm(item["term"]) for e in d["entries"]):
                d["entries"].append(item)
            d["tombstones"] = [t for t in d["tombstones"] if t.get("id") != item["id"]]
            d["version"] += 1
            return dict(item)

        return self._mutate(work)

    def settings(self):
        return dict(self._load().get("settings", {}))

    def set_auto_learn(self, value):
        if not isinstance(value, bool):
            raise ValueError("auto_learn must be boolean")

        def work(d):
            d.setdefault("settings", {})["auto_learn"] = value
            d["version"] += 1
            return d["settings"]

        return self._mutate(work)

    def remember_corrections(self, before, after, provenance=None):
        d = self._load()
        if not d.get("settings", {}).get("auto_learn", True):
            return []
        if re.search(
            r"https?://|www\.|\b[A-Za-z]:[\\/]|(?:^|\s)/[^\s]+",
            str(before) + " " + str(after),
            re.I,
        ):
            return []
        old_sentences = _sentence_parts(before)
        new_sentences = _sentence_parts(after)
        if len(old_sentences) != len(new_sentences):
            return []
        if any(
            _sentence_ending(old) != _sentence_ending(new)
            for old, new in zip(old_sentences, new_sentences)
        ):
            return []

        import difflib

        changes = []
        for old_sentence, new_sentence in zip(old_sentences, new_sentences):
            old_tokens = _tokens(old_sentence)
            new_tokens = _tokens(new_sentence)
            matcher = difflib.SequenceMatcher(
                a=[_norm(token) for token in old_tokens],
                b=[_norm(token) for token in new_tokens],
                autojunk=False,
            )
            opcodes = matcher.get_opcodes()
            changed_old = sum(i2 - i1 for tag, i1, i2, j1, j2 in opcodes if tag != "equal")
            changed_new = sum(j2 - j1 for tag, i1, i2, j1, j2 in opcodes if tag != "equal")
            if changed_old > 3 or changed_new > 3:
                continue
            for opcode_index, (tag, i1, i2, j1, j2) in enumerate(opcodes):
                if tag not in {"replace", "insert"}:
                    continue
                left_context = 0
                right_context = 0
                if opcode_index > 0 and opcodes[opcode_index - 1][0] == "equal":
                    left_context = opcodes[opcode_index - 1][2] - opcodes[opcode_index - 1][1]
                if (
                    opcode_index + 1 < len(opcodes)
                    and opcodes[opcode_index + 1][0] == "equal"
                ):
                    right_context = (
                        opcodes[opcode_index + 1][2] - opcodes[opcode_index + 1][1]
                    )
                if max(left_context, right_context) < 2:
                    continue
                candidate = " ".join(new_tokens[j1:j2]).strip()
                if _valid_candidate(candidate):
                    changes.append((candidate, " ".join(old_tokens[i1:i2])))

        learned = []
        for candidate, old_text in changes:
            if any(_norm(entry["term"]) == _norm(candidate) for entry in learned):
                continue
            if any(_norm(entry["term"]) == _norm(candidate) for entry in d["entries"]):
                continue
            entry_id = uuid.uuid4().hex
            token = uuid.uuid4().hex
            item = {
                "id": entry_id,
                "term": candidate,
                "definition": None,
                "provenance": [
                    {
                        **(provenance or {}),
                        "kind": "transcript_correction",
                        "before": old_text,
                    }
                ],
            }

            def save_learning(data):
                if any(
                    _norm(entry["term"]) == _norm(candidate)
                    for entry in data["entries"]
                ):
                    return False
                data["entries"].append(item)
                data["undo"][token] = {
                    "action": "remove_learned",
                    "item_id": entry_id,
                }
                data["version"] += 1
                return True

            if self._mutate(save_learning):
                learned.append({**item, "undo_id": token})
        return learned

    def migrate(self, origin, terms, provenance=None):
        origin = str(origin)

        def work(d):
            if origin in d["migrations"]:
                return 0
            d["migrations"].append(origin)
            tomb = {e.get("key") for e in d["tombstones"]}
            present = {_norm(e["term"]) for e in d["entries"]}
            count = 0
            for raw in terms:
                term = raw.get("term") if isinstance(raw, dict) else raw
                try:
                    clean = _clean_term(term)
                except ValueError:
                    continue
                key = _norm(clean)
                if key not in present and key not in tomb:
                    definition = (
                        raw.get("definition") if isinstance(raw, dict) else None
                    )
                    source_id = raw.get("id") if isinstance(raw, dict) else None
                    d["entries"].append(
                        {
                            "id": uuid.uuid4().hex,
                            "term": clean,
                            "definition": definition or None,
                            "provenance": [
                                {
                                    **(provenance or {}),
                                    "kind": "legacy_migration",
                                    "origin": origin,
                                    "source_id": source_id,
                                }
                            ],
                        }
                    )
                    present.add(key)
                    count += 1
            if count:
                d["version"] += 1
            return count

        return self._mutate(work)


def migrate_legacy_meetings(root):
    """Import selected archive glossary entries once without modifying the archive."""
    from services.meeting_archive import MeetingArchive

    archive = MeetingArchive(root, read_only=True)
    dictionary = get_dictionary()
    count = 0
    for meeting in archive.list_meetings(limit=10**9):
        meeting_id = meeting.get("meeting_id") or meeting.get("id")
        if not meeting_id:
            continue
        for entry in archive.glossary(str(meeting_id)):
            identity = entry.get("id") or _norm(str(entry.get("term", "")))
            origin = f"legacy-meeting:{Path(root).expanduser().resolve()}:{meeting_id}:{identity}"
            count += dictionary.migrate(
                origin,
                [entry],
                provenance={
                    "meeting_id": meeting_id,
                    "meeting_title": meeting.get("title"),
                    "archive_root": str(Path(root).expanduser().resolve()),
                },
            )
    return count


def _clean_term(term):
    if not isinstance(term, str):
        raise ValueError("term must be text")
    term = " ".join(unicodedata.normalize("NFC", term).split())
    if (
        not term
        or len(term) > 128
        or len(term.split()) > 12
        or any(ord(c) < 32 for c in term)
    ):
        raise ValueError("Invalid term")
    return term


def _clean_definition(value):
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 1000:
        raise ValueError("Invalid definition")
    return value.strip() or None


def _tokens(value):
    return re.findall(r"[^\W_]+(?:[-.][^\W_]+)*", str(value or ""), flags=re.UNICODE)


def _sentence_parts(value):
    parts = re.split(r"(?<=[.!?])\s+|\r?\n+", str(value or ""))
    return [part.strip() for part in parts if part.strip()]


def _sentence_ending(value):
    match = re.search(r"[.!?]+[\"')\]]*$", value.rstrip())
    return match.group(0) if match else ""


def _valid_candidate(value):
    return bool(
        value
        and len(value) <= 64
        and len(value.split()) <= 3
        and not re.search(
            r"\d|https?://|www\.|[/\\:]|\b[\w.-]+\.(?:com|org|net|ru|io|edu|gov)\b|\b(select|insert|update|delete|drop|join|where)\b",
            value,
            re.I,
        )
    )


from contextlib import contextmanager


@contextmanager
def _process_lock(path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(stream.fileno()).st_size == 0:
                stream.seek(0)
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            acquired = True
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            acquired = True
        yield
    finally:
        if acquired and os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        elif acquired:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def get_dictionary(project_root=None, path=None):
    return PersonalDictionary(path=path, project_root=project_root)


def get_dictionary_snapshot(project_root=None):
    return get_dictionary(project_root).snapshot()


def remember_corrections(before, after, provenance, project_root=None):
    return get_dictionary(project_root).remember_corrections(before, after, provenance)
