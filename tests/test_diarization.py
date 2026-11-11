import unittest

from services.diarization import assign_speakers, render_turns, validate_intervals


class SpeakerAssignmentTests(unittest.TestCase):
    def test_native_interval_validation_rejects_bad_speakers_and_clamps_small_overshoot(self):
        normalized = validate_intervals([{"start": 0.2, "end": 1.01, "speaker": 2}], 1.0)
        self.assertEqual(normalized, [{"start": 0.2, "end": 1.0, "speaker": 2}])
        for interval in (
            {"start": 0, "end": 0.5, "speaker": 0},
            {"start": 0, "end": 0.5, "speaker": 9},
            {"start": 0, "end": float("nan"), "speaker": 1},
            {"start": 0, "end": 1.1, "speaker": 1},
        ):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                validate_intervals([interval], 1.0)

    def test_splits_words_from_one_asr_segment_by_speaker(self):
        turns = assign_speakers(
            [{"start": 0, "end": 2, "text": " Привет мир", "words": [
                {"text": " Привет", "start": 0.1, "end": 0.6},
                {"text": " мир", "start": 1.1, "end": 1.7},
            ]}],
            [{"start": 0, "end": 0.8, "speaker": 8}, {"start": 1, "end": 2, "speaker": 3}],
        )
        self.assertEqual([turn["text"] for turn in turns], ["Привет", "мир"])
        self.assertEqual([turn["speakerIds"] for turn in turns], [[1], [2]])

    def test_overlap_and_silence_are_explicit(self):
        segments = [{"start": 0, "end": 3, "text": " один два три", "words": [
            {"text": " один", "start": 0.1, "end": 0.5},
            {"text": " два", "start": 1.1, "end": 1.5},
            {"text": " три", "start": 2.1, "end": 2.5},
        ]}]
        turns = assign_speakers(segments, [
            {"start": 0, "end": 0.8, "speaker": 1},
            {"start": 1, "end": 1.8, "speaker": 1},
            {"start": 1, "end": 1.8, "speaker": 2},
        ])
        self.assertEqual([turn["speakerIds"] for turn in turns], [[1], [1, 2], []])
        self.assertEqual(turns[2]["label"], "Говорящий не определён")

    def test_zero_or_invalid_word_time_is_unknown_without_losing_text(self):
        turns = assign_speakers(
            [{"start": 5, "end": 6, "text": " слово", "words": [
                {"text": " слово", "start": 0, "end": 0},
            ]}],
            [{"start": 0, "end": 8, "speaker": 1}],
        )
        self.assertEqual(turns[0]["text"], "слово")
        self.assertEqual(turns[0]["speakerIds"], [])

    def test_unicode_subtokens_reconstruct_exactly(self):
        turns = assign_speakers(
            [{"start": 0, "end": 1, "text": " привет", "words": [
                {"text": " при", "start": 0.1, "end": 0.3},
                {"text": "вет", "start": 0.3, "end": 0.7},
            ]}],
            [{"start": 0, "end": 1, "speaker": 1}],
        )
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["text"], "привет")

    def test_mismatched_word_reconstruction_falls_back_to_whole_asr_segment(self):
        turns = assign_speakers(
            [{"start": 1, "end": 2, "text": " весь текст", "words": [
                {"text": " потерянное", "start": 1.1, "end": 1.4},
            ]}],
            [{"start": 0, "end": 3, "speaker": 1}],
        )
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["text"], "весь текст")
        self.assertEqual(turns[0]["speakerIds"], [])

    def test_unknown_segments_keep_word_boundary_and_invalid_middle_word_order(self):
        turns = assign_speakers([
            {"start": 0, "end": 1, "text": "one", "words": []},
            {"start": 1, "end": 2, "text": "two", "words": []},
            {"start": 2, "end": 3, "text": "one bad three", "words": [
                {"text": "one", "start": 2.1, "end": 2.3},
                {"text": " bad", "start": 0, "end": 0},
                {"text": " three", "start": 2.6, "end": 2.9},
            ]},
        ], [])
        self.assertEqual([turn["text"] for turn in turns], ["one two one bad three"])
        self.assertEqual(turns[0]["speakerIds"], [])

    def test_render_uses_floored_second_and_speaker_label(self):
        rendered = render_turns([{"start": 3661.9, "end": 3662, "text": "да", "speakerIds": [1],
                                  "label": "Говорящий 1"}])
        self.assertEqual(rendered, "[01:01:01] Говорящий 1: да")


if __name__ == "__main__":
    unittest.main()
