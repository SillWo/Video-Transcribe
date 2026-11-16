import pytest

from services.meeting_archive import MeetingArchive
from services.meeting_index import MeetingIndex


def _meeting(tmp_path, name):
    root = tmp_path / name
    source = tmp_path / f"{name}.wav"
    source.write_bytes(name.encode())
    archive = MeetingArchive(root)
    meeting_id = archive.import_file(source)["meeting_id"]
    return root, archive, meeting_id


def _index(root):
    return MeetingIndex(root, embedding_model_path=root / "missing-e5-model")


def test_near_duplicate_screens_collapse_before_pagination_and_keep_speech(tmp_path):
    root, archive, meeting_id = _meeting(tmp_path, "same-meeting")
    screen_text = "Закрывать задачи в 1С отчетность трудозатраты, часы и курсор"
    archive.write_screen_events(meeting_id, [
        {"id": f"screen-{n}", "start": 390 + n, "end": 390.5 + n,
         "text": screen_text if n % 2 else screen_text.upper().replace(" ", "  ")}
        for n in range(10)
    ])
    archive.write_transcript(meeting_id, [{
        "id": "speech", "start": 405, "end": 406,
        "text": "Закрывать задачи в 1С отчетность трудозатраты обсуждали отдельно",
    }])

    index = _index(root)
    first = index.search("закрывать задачи в 1С отчетность трудозатраты", mode="text", limit=1)
    second = index.search("закрывать задачи в 1С отчетность трудозатраты", mode="text", limit=1, offset=1)

    assert first["total"] == 2
    assert first["results"][0]["kind"] == "screen"
    assert first["results"][0]["near_duplicate_count"] == 9
    assert first["results"][0]["related_start"] == 390
    assert first["results"][0]["related_end"] == 399.5
    assert second["results"][0]["evidence_id"] == "speech"
    assert second["next_offset"] is None


def test_changed_numbers_and_different_meetings_are_not_collapsed(tmp_path):
    root, archive, first_id = _meeting(tmp_path, "first")
    archive.write_screen_events(first_id, [
        {"id": "screen-25", "start": 390, "end": 391,
         "text": "Закрывать задачи в 1С отчетность трудозатраты 25"},
        {"id": "screen-27", "start": 391, "end": 392,
         "text": "Закрывать задачи в 1С отчетность трудозатраты 27"},
    ])

    other_source = tmp_path / "other.wav"
    other_source.write_bytes(b"other")
    second_id = archive.import_file(other_source)["meeting_id"]
    archive.write_screen_events(second_id, [{
        "id": "other-screen", "start": 390, "end": 391,
        "text": "Закрывать задачи в 1С отчетность трудозатраты 25",
    }])

    result = _index(root).search("закрывать задачи в 1С отчетность трудозатраты", mode="text")
    screens = [row for row in result["results"] if row["kind"] == "screen"]
    assert {(row["meeting_id"], row["evidence_id"]) for row in screens} == {
        (first_id, "screen-25"), (first_id, "screen-27"), (second_id, "other-screen")
    }


def test_standalone_identifier_fast_path_keeps_every_occurrence_and_offsets(tmp_path):
    root, archive, meeting_id = _meeting(tmp_path, "identifier-meeting")
    archive.write_screen_events(meeting_id, [
        {"id": f"screen-{n}", "start": 390 + n, "end": 391 + n,
         "text": "Ticket ARCH-427"}
        for n in range(10)
    ])

    index = _index(root)
    first = index.search("ARCH-427", mode="hybrid", limit=4)
    next_page = index.search("ARCH-427", mode="hybrid", limit=4, offset=4)
    final_page = index.search("ARCH-427", mode="hybrid", limit=4, offset=8)
    ids = [row["evidence_id"] for page in (first, next_page, final_page) for row in page["results"]]

    assert first["total"] == 10
    assert len(ids) == len(set(ids)) == 10
    assert final_page["next_offset"] is None


def _screen(start, text, evidence_id="screen"):
    return {"meeting_id": "meeting", "evidence_id": evidence_id, "kind": "screen",
            "start": start, "end": start + 1, "text": text}


@pytest.mark.parametrize("suffixes", [("value 25", "value 27"),
                                      ("ticket ARCH-427", "ticket ARCH-428")])
def test_long_screen_text_with_changed_numeric_or_id_tail_is_not_collapsed(suffixes):
    prefix = "shared phrase " * 400
    rows = [_screen(390, prefix + suffixes[0], "first"),
            _screen(391, prefix + suffixes[1], "second")]

    collapsed = MeetingIndex._collapse_near_duplicate_screens(rows)

    assert [row["evidence_id"] for row in collapsed] == ["first", "second"]


def test_screen_comparison_ignores_clock_values_but_preserves_number_order():
    rows = [_screen(390, "same phrase 09:30:15"),
            _screen(391, "same phrase 19:25:30"),
            _screen(392, "same phrase 25 27"),
            _screen(393, "same phrase 27 25")]

    collapsed = MeetingIndex._collapse_near_duplicate_screens(rows)

    assert [row["near_duplicate_count"] for row in collapsed] == [1, 0, 0]


def test_realistic_long_screen_with_different_clocks_still_collapses():
    screen_text = ("Закрывать задачи в отчетность трудозатраты " * 70).strip()
    assert len(screen_text.split()) == 350
    rows = [_screen(390, screen_text + " 09:30:15", "first"),
            _screen(391, screen_text + " 19:25:30", "second")]

    collapsed = MeetingIndex._collapse_near_duplicate_screens(rows)

    assert len(collapsed) == 1
    assert collapsed[0]["near_duplicate_count"] == 1


@pytest.mark.parametrize("start", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_screen_start_is_left_ungrouped(start):
    rows = [_screen(start, "same phrase"), _screen(390, "same phrase")]

    collapsed = MeetingIndex._collapse_near_duplicate_screens(rows)

    assert len(collapsed) == 2
