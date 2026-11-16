from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import services.meeting_ocr_batches as meeting_ocr_batches
from services.meeting_ocr_batches import recognize_equal_width_batches


@dataclass
class Config:
    lang_type: str = "en"
    font_path: str | None = None


@dataclass
class Args:
    img: list[np.ndarray]
    return_word_box: bool = True


class FakeRecognizer:
    """Small RapidOCR 3.10 surface with the stock batch6 call contract."""

    def __init__(self, *, fail_session: bool = False):
        self.rec_batch_num = 2
        self.rec_image_shape = (3, 10, 20)
        self.cfg = Config()
        self.fail_session = fail_session
        self.session_inputs: list[np.ndarray] = []
        self.postprocess_calls: list[tuple[np.ndarray, bool, list[float], float]] = []
        self.stock_calls = 0

    def resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        channels, height, width = self.rec_image_shape
        padded_width = int(height * max_wh_ratio)
        resized_width = min(padded_width, int(np.ceil(height * img.shape[1] / img.shape[0])))
        resized = np.zeros((channels, height, padded_width), dtype=np.float32)
        # Distinct constant images make both source identity and padding visible.
        resized[:, :, :resized_width] = float(img[0, 0, 0])
        return resized

    def session(self, batch: np.ndarray) -> np.ndarray:
        if self.fail_session:
            raise RuntimeError("session failed")
        self.session_inputs.append(batch.copy())
        # The prediction is an identity token for this fake's postprocessor.
        return batch[:, 0, 0, 0].copy()

    def postprocess_op(self, preds, return_word_box, *, wh_ratio_list, max_wh_ratio):
        self.postprocess_calls.append(
            (np.asarray(preds).copy(), return_word_box, list(wh_ratio_list), max_wh_ratio)
        )
        lines = [(f"text-{int(value)}", float(value) / 100.0) for value in preds]
        words = [
            [(f"word-{int(value)}", float(value) / 100.0, [[int(value), 7]])]
            for value in preds
        ]
        return lines, words

    def __call__(self, args: Args):
        """Reference behavior copied from RapidOCR 3.10's __call__ grouping."""
        self.stock_calls += 1
        images = [args.img] if isinstance(args.img, np.ndarray) else args.img
        indices = np.argsort(np.array([im.shape[1] / float(im.shape[0]) for im in images]))
        records = [None] * len(images)
        for begin in range(0, len(images), self.rec_batch_num):
            end = min(len(images), begin + self.rec_batch_num)
            maximum = self.rec_image_shape[2] / self.rec_image_shape[1]
            ratios = []
            for position in range(begin, end):
                image = images[indices[position]]
                ratio = image.shape[1] * 1.0 / image.shape[0]
                maximum = max(maximum, ratio)
                ratios.append(ratio)
            batch = np.concatenate([
                self.resize_norm_img(images[indices[pos]], maximum)[np.newaxis, :]
                for pos in range(begin, end)
            ]).astype(np.float32)
            preds = self.session(batch)
            lines, words = self.postprocess_op(
                preds, args.return_word_box, wh_ratio_list=ratios, max_wh_ratio=maximum
            )
            for offset, line in enumerate(lines):
                records[int(indices[begin + offset])] = (line, words[offset] if args.return_word_box else None)
        return records


def _images() -> list[np.ndarray]:
    # Stock sorted groups have padded widths 20, 20, 30, and 40 pixels,
    # including a final one-image partial group. Input order is intentionally
    # different from ratio order so output restoration is exercised.
    dimensions = [(10, 30), (10, 10), (10, 40), (10, 20), (10, 30), (10, 20), (10, 20)]
    return [np.full((height, width, 3), index + 1, dtype=np.uint8)
            for index, (height, width) in enumerate(dimensions)]


def test_merges_only_adjacent_equal_width_groups_and_matches_stock_tensors():
    images = _images()
    args = Args(images)
    stock = FakeRecognizer()
    expected_records = stock(args)
    expected_inputs = [batch.copy() for batch in stock.session_inputs]
    expected_postprocess = stock.postprocess_calls.copy()

    merged = FakeRecognizer()
    result = recognize_equal_width_batches(merged, args, max_batch=4)

    # Four stock inferences become three; the first two intact groups merge.
    assert [len(batch) for batch in merged.session_inputs] == [4, 2, 1]
    assert len(merged.session_inputs) < len(expected_inputs)
    assert all(len(batch) <= 4 for batch in merged.session_inputs)
    np.testing.assert_array_equal(merged.session_inputs[0], np.concatenate(expected_inputs[:2]))
    np.testing.assert_array_equal(merged.session_inputs[1], expected_inputs[2])
    np.testing.assert_array_equal(merged.session_inputs[2], expected_inputs[3])

    # Decoder calls still receive each original stock group's predictions and
    # exact ratio metadata, preserving word-box coordinates.
    assert len(merged.postprocess_calls) == len(expected_postprocess)
    for actual, expected in zip(merged.postprocess_calls, expected_postprocess):
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
    assert result.txts == tuple(record[0][0] for record in expected_records)
    assert result.scores == tuple(record[0][1] for record in expected_records)
    assert result.word_results == tuple(record[1] for record in expected_records)
    assert result.imgs is images
    assert result.viser is not None


def test_max_batch_never_splits_a_source_group_or_exceeds_limit():
    recognizer = FakeRecognizer()
    recognize_equal_width_batches(recognizer, Args(_images()), max_batch=3)
    assert [len(batch) for batch in recognizer.session_inputs] == [2, 2, 2, 1]
    assert max(len(batch) for batch in recognizer.session_inputs) <= 3


def test_session_exception_escapes_without_stock_retry():
    recognizer = FakeRecognizer(fail_session=True)
    try:
        recognize_equal_width_batches(recognizer, Args(_images()), max_batch=4)
    except RuntimeError as error:
        assert str(error) == "session failed"
    else:
        raise AssertionError("session exception was swallowed")
    assert recognizer.stock_calls == 0


def test_one_stock_group_uses_the_recognizers_stock_call():
    recognizer = FakeRecognizer()
    result = recognize_equal_width_batches(recognizer, Args(_images()[:2]))
    assert recognizer.stock_calls == 1
    assert len(result) == 2


def test_elapse_includes_resize_preprocessing(monkeypatch):
    recognizer = FakeRecognizer()
    elapsed_clock = [0.0]
    real_resize = recognizer.resize_norm_img

    def timed_resize(image, max_wh_ratio):
        elapsed_clock[0] += 0.25
        return real_resize(image, max_wh_ratio)

    recognizer.resize_norm_img = timed_resize
    monkeypatch.setattr(meeting_ocr_batches.time, "perf_counter", lambda: elapsed_clock[0])

    result = recognize_equal_width_batches(recognizer, Args(_images()), max_batch=4)
    assert result.elapse == 7 * 0.25


def test_equal_padded_width_can_have_distinct_source_max_ratios():
    # The two original groups have max ratios 2.1 and 2.15. At imgH=10 both
    # produce int(10 * ratio) == 21, so they share one session batch while
    # retaining their distinct postprocessor metadata.
    dimensions = [(10, 20), (20, 42), (100, 211), (20, 43)]
    images = [np.full((h, w, 3), i + 1, dtype=np.uint8)
              for i, (h, w) in enumerate(dimensions)]
    recognizer = FakeRecognizer()
    recognize_equal_width_batches(recognizer, Args(images), max_batch=4)

    assert [len(batch) for batch in recognizer.session_inputs] == [4]
    assert [call[3] for call in recognizer.postprocess_calls] == [2.1, 2.15]
    assert [len(call[2]) for call in recognizer.postprocess_calls] == [2, 2]


def test_return_word_box_false_returns_none_for_each_word_result():
    recognizer = FakeRecognizer()
    args = Args(_images(), return_word_box=False)
    result = recognize_equal_width_batches(recognizer, args, max_batch=4)

    assert result.word_results == (None,) * len(args.img)
    assert all(call[1] is False for call in recognizer.postprocess_calls)
