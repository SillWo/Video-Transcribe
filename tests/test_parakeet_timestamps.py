import math
from types import SimpleNamespace

import pytest

from services.parakeet_timestamps import words_from_timestamped_result, recognize_with_native_durations


def result(tokens, timestamps, durations, text=None):
    return SimpleNamespace(tokens=tokens, timestamps=timestamps,
                           text=text if text is not None else "".join(tokens))


def test_native_duration_and_token_mapping_preserves_punctuation_spaces_and_compounds():
    recognized = result(["Hello", ",", "  ", "well", "-", "known", "!"],
                        [0.0, 0.2, 0.22, 0.4, 0.52, 0.58, 0.8],
                        [2, 1, 0, 2, 1, 2, 1], "Hello, well-known!")

    words, warning = words_from_timestamped_result(recognized, [2, 1, 0, 2, 1, 2, 1],
                                                   frame_seconds=0.1, chunk_start=3.0,
                                                   chunk_duration=1.0)

    assert warning is None
    assert "".join(word["text"] for word in words) == recognized.text.strip()
    assert [word["text"] for word in words] == ["Hello,", " well-known!"]
    assert words[0] == {"text": "Hello,", "start": 3.0, "end": 3.3}
    assert words[1]["start"] == 3.4
    assert words[1]["end"] == 3.9


@pytest.mark.parametrize("bad", [None, [0.0], [0.0, math.nan], [0.2, 0.1]])
def test_invalid_or_mismatched_metadata_keeps_words_with_unknown_times(bad):
    recognized = result(["one", " ", "two"], bad, [1, 0, 2], "one two")
    words, warning = words_from_timestamped_result(recognized, [1, 0, 2], frame_seconds=0.08,
                                                   chunk_start=0.0, chunk_duration=2.0)
    assert warning
    assert [word["text"] for word in words] == ["one", " two"]
    assert all(word["start"] == word["end"] == 0.0 for word in words)


def test_duration_count_mismatch_and_nonfinite_duration_yield_unknown_times():
    for durations in (None, [1, 2], [1, math.nan, 2]):
        recognized = result(["hello", " ", "there"], [0.0, 0.3, 0.5], durations, "hello there")
        words, warning = words_from_timestamped_result(recognized, durations, frame_seconds=0.08,
                                                       chunk_start=0.0, chunk_duration=1.0)
        assert warning
        assert all(word["start"] == word["end"] == 0.0 for word in words)


def test_last_native_token_ends_at_its_duration_not_chunk_end():
    recognized = result(["last"], [0.1], [2], "last")
    words, _warning = words_from_timestamped_result(recognized, [2], frame_seconds=0.08,
                                                    chunk_start=4.0, chunk_duration=3.0)
    assert words == [{"text": "last", "start": 4.1, "end": 4.26}]


def test_small_frame_overshoot_is_clamped_to_chunk_boundary():
    recognized = result(["word"], [0.96], [1], "word")
    words, _warning = words_from_timestamped_result(recognized, [1], frame_seconds=0.08,
                                                    chunk_start=5.0, chunk_duration=1.0)
    assert words == [{"text": "word", "start": 5.96, "end": 6.0}]


def test_decode_wrapper_records_nonblank_durations_including_zero_and_restores():
    calls = []
    class Asr:
        _blank_idx = 0
        window_step = 0.01
        _subsampling_factor = 8
        def _decode(self, prev_tokens, prev_state, encoder_out):
            i = len(calls)
            calls.append(i)
            logits = [10, 0] if i == 2 else [0, 10]
            return logits, [3, 0, 2][i], prev_state
    class Model:
        asr = Asr()
    model = Model()
    original = model.asr._decode
    def recognize():
        for _ in range(3):
            model.asr._decode([], None, None)
        return "recognized"
    recognized, durations = recognize_with_native_durations(
        model, recognize,
    )
    assert recognized == "recognized"
    assert durations == [3, 0]
    assert model.asr._decode == original


def test_decode_wrapper_restores_original_method_after_recognition_error():
    class Asr:
        _blank_idx = 0
        def _decode(self, *_args):
            return [0, 1], 1, None
    class Model:
        asr = Asr()
    model = Model()
    original = model.asr._decode
    with pytest.raises(RuntimeError, match="recognition failed"):
        recognize_with_native_durations(model, lambda: (_ for _ in ()).throw(RuntimeError("recognition failed")))
    assert model.asr._decode == original
