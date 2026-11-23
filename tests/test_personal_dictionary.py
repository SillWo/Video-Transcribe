import multiprocessing

from services.personal_dictionary import PersonalDictionary
from services import meeting_backup
from services.meeting_archive import MeetingArchive
from services.personal_dictionary import migrate_legacy_meetings


def _add_dictionary_term(path, term, barrier):
    dictionary = PersonalDictionary(path=path)
    barrier.wait(timeout=10)
    dictionary.add(term)


def test_dictionary_crud_normalization_delete_undo_and_snapshot(tmp_path):
    dictionary = PersonalDictionary(path=tmp_path / "profile.json")
    first = dictionary.add("  Cafe\u0301  ", "coffee")
    duplicate = dictionary.add("Café", "coffee")
    assert duplicate["id"] == first["id"]
    updated = dictionary.update(first["id"], "Café", "updated")
    assert updated["definition"] == "updated"
    undo_id = dictionary.delete(first["id"])
    assert dictionary.snapshot()["terms"] == []
    restored = dictionary.undo(undo_id)
    assert restored["term"] == "Café"
    assert dictionary.snapshot()["version"] >= 4


def test_cross_process_dictionary_mutations_preserve_all_entries(tmp_path):
    profile = tmp_path / "profile.json"
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    processes = [
        context.Process(
            target=_add_dictionary_term, args=(profile, f"Term {index}", barrier)
        )
        for index in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
    assert all(not process.is_alive() for process in processes)
    assert [process.exitcode for process in processes] == [0, 0, 0, 0]
    assert set(PersonalDictionary(path=profile).snapshot()["terms"]) == {
        "Term 0",
        "Term 1",
        "Term 2",
        "Term 3",
    }


def test_corrupt_dictionary_is_not_overwritten_by_mutation(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text("{broken", encoding="utf-8")
    dictionary = PersonalDictionary(path=profile)
    try:
        dictionary.add("Should not overwrite")
    except ValueError as exc:
        assert "unreadable" in str(exc)
    else:
        raise AssertionError("corrupt profile data must not be replaced silently")
    assert profile.read_text(encoding="utf-8") == "{broken"


def test_learning_is_contextual_short_and_reversible_without_transcript_undo(tmp_path):
    dictionary = PersonalDictionary(path=tmp_path / "profile.json")
    learned = dictionary.remember_corrections(
        "Use Typo phrase in context", "Use Fixed phrase in context", {"job_id": "abc"}
    )
    assert [item["term"] for item in learned] == ["Fixed"]
    assert dictionary.undo(learned[0]["undo_id"])["removed"]
    assert dictionary.snapshot()["terms"] == []
    assert not dictionary.remember_corrections("one two", "three four", {})
    assert not dictionary.remember_corrections(
        "set Q7 in context", "set Q8 in context", {}
    )
    assert not dictionary.remember_corrections(
        "see https://example.com here", "see https://example.org here", {}
    )
    assert not dictionary.remember_corrections(
        "run select query now", "run drop query now", {}
    )


def test_learning_rejects_sentence_rewrites_and_keeps_changes_sentence_local(tmp_path):
    dictionary = PersonalDictionary(path=tmp_path / "profile.json")
    before = (
        "Independent restored archive copy. Keep this separate from the original history."
    )
    after = (
        "Independent restored archive copy. This edit belongs only to the restored folder."
    )
    assert not dictionary.remember_corrections(before, after, {})

    assert not dictionary.remember_corrections(
        "The first note is stable. The second note is ready.",
        "The first note is stable. Replace the whole second sentence.",
        {},
    )
    assert not dictionary.remember_corrections(
        "Use old label.", "Pick new label.", {}
    )

    learned = dictionary.remember_corrections(
        "For the review, DBeaver was the preferred database client.",
        "For the review, DataGrip was the preferred database client.",
        {},
    )
    assert [item["term"] for item in learned] == ["DataGrip"]

    assert not dictionary.remember_corrections(
        "The heading is stable. A separate note is unchanged. Final details are present.",
        "The heading is stable. Rewrite the entire unrelated middle sentence. Final details are present.",
        {},
    )


def test_manual_terms_allow_numbers_and_deleted_migrations_stay_deleted(tmp_path):
    dictionary = PersonalDictionary(path=tmp_path / "profile.json")
    manual = dictionary.add("1C 8.3")
    assert manual["term"] == "1C 8.3"
    assert dictionary.migrate("legacy-meeting", ["Acme", "1C 8.3"]) == 1
    migrated = next(item for item in dictionary.list() if item["term"] == "Acme")
    dictionary.delete(migrated["id"])
    assert dictionary.migrate("legacy-meeting", ["Acme"]) == 0
    assert all(item["term"] != "Acme" for item in dictionary.list())
    assert dictionary.set_auto_learn(False)["auto_learn"] is False
    assert not dictionary.remember_corrections(
        "we have typo here", "we have fixed here", {}
    )


def test_backup_dictionary_snapshot_restore_test_isolation_and_non_destructive_merge(
    tmp_path, monkeypatch
):
    profile = tmp_path / "profile.json"
    monkeypatch.setenv("VT_DICTIONARY_PATH", str(profile))
    dictionary = PersonalDictionary(path=profile)
    dictionary.add("Backup term", "snapshot definition")
    archive = tmp_path / "archive"
    archive.mkdir()
    backup_root = tmp_path / "backups"
    created = meeting_backup.create_backup(archive, backup_root)
    current_profile = tmp_path / "current-profile.json"
    monkeypatch.setenv("VT_DICTIONARY_PATH", str(current_profile))
    dictionary = PersonalDictionary(path=current_profile)
    dictionary.add("Current term", "keep current value")
    before_test = current_profile.read_bytes()

    tested = meeting_backup.restore_test(archive, backup_root, created["snapshot_id"])
    assert tested["passed"]
    assert current_profile.read_bytes() == before_test

    restored = meeting_backup.restore_backup(
        archive, backup_root, created["snapshot_id"], tmp_path / "restored"
    )
    assert "Backup term" in dictionary.snapshot()["terms"]
    assert "Current term" in dictionary.snapshot()["terms"]
    assert restored["dictionary_merge"]["added"] == ["Backup term"]


def test_legacy_glossary_migration_preserves_definition_and_tracks_new_terms_by_root(
    tmp_path, monkeypatch
):
    profile = tmp_path / "profile.json"
    monkeypatch.setenv("VT_DICTIONARY_PATH", str(profile))
    dictionary = PersonalDictionary(path=profile)
    media = tmp_path / "source.mp4"
    media.write_bytes(b"test media")
    first_root = tmp_path / "archive-one"
    first_archive = MeetingArchive(first_root)
    first = first_archive.import_file(media)
    first_archive.add_glossary_entry(
        first["meeting_id"], "AtlasDB", "Internal database"
    )

    assert migrate_legacy_meetings(first_root) == 1
    assert migrate_legacy_meetings(first_root) == 0
    entry = dictionary.list()[0]
    assert entry["definition"] == "Internal database"
    assert entry["provenance"][0]["kind"] == "legacy_migration"
    assert entry["provenance"][0]["meeting_id"] == first["meeting_id"]
    assert entry["provenance"][0]["meeting_title"] == first["title"]
    assert entry["provenance"][0]["archive_root"] == str(first_root.resolve())
    dictionary.delete(entry["id"])
    assert migrate_legacy_meetings(first_root) == 0

    first_archive.add_glossary_entry(first["meeting_id"], "NewTerm", "Added later")
    second_root = tmp_path / "archive-two"
    second_archive = MeetingArchive(second_root)
    second = second_archive.import_file(media)
    second_archive.add_glossary_entry(
        second["meeting_id"], "SecondRootTerm", "From another archive"
    )
    assert migrate_legacy_meetings(first_root) == 1
    assert migrate_legacy_meetings(second_root) == 1
    assert set(dictionary.snapshot()["terms"]) == {"NewTerm", "SecondRootTerm"}
