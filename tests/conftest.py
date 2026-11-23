"""Isolate profile, archive, and model paths during the test suite."""

import pytest


@pytest.fixture(autouse=True)
def isolated_personal_dictionary(tmp_path, monkeypatch):
    monkeypatch.setenv("VT_DICTIONARY_PATH", str(tmp_path / "personal-dictionary.json"))
    monkeypatch.setenv("VT_MEETING_ROOT", str(tmp_path / "meeting-archive"))
    monkeypatch.setenv("VT_MEETING_SETTINGS", str(tmp_path / "meeting-settings.json"))
    monkeypatch.setenv("MEETING_EMBEDDING_MODEL", str(tmp_path / "missing-e5-model"))
    from services.archive_catalog import (
        reset_archive_indexing_for_tests,
        shutdown_archive_indexing,
        wait_archive_indexing_idle,
    )

    reset_archive_indexing_for_tests()
    yield
    assert wait_archive_indexing_idle(timeout=20), "portable index worker did not drain"
    shutdown_archive_indexing(timeout=2)
