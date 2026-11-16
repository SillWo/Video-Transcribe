"""Optional equal-width batching for RapidOCR 3.10 text recognition.

This module deliberately does not import RapidOCR. The adapter is only for a
ready-made TextRecognizer and leaves its engine, preprocessing, and decoder in
control. A source batch is always one of the recognizer's ordinary
``rec_batch_num`` groups; only adjacent source batches with the same padded
pixel width are joined for inference.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

DEFAULT_EQUAL_WIDTH_MAX_BATCH = 48


def recognize_equal_width_batches(
    recognizer: Any, args: Any, max_batch: int = DEFAULT_EQUAL_WIDTH_MAX_BATCH
) -> Any:
    """Recognize images while merging adjacent stock batches of equal width.

    The resized tensor for every image is produced with the same
    ``max_wh_ratio`` that RapidOCR's stock ``TextRecognizer.__call__`` would
    use for its original batch. Merging therefore changes only the number of
    session calls, not the input tensor or the decoder parameters. If the
    recognizer is not compatible with the RapidOCR 3.10 contract, or if the
    input fits in one stock batch, its own ``__call__`` is used unchanged.
    Session exceptions intentionally propagate; this adapter never retries on
    another engine or through the stock path.
    """
    start_time = time.perf_counter()
    if max_batch <= 0:
        raise ValueError("max_batch must be positive")

    if not _supports_contract(recognizer):
        return _stock_call(recognizer, args)

    img_list = [args.img] if isinstance(args.img, np.ndarray) else args.img
    if img_list is None:
        return _stock_call(recognizer, args)

    batch_num = recognizer.rec_batch_num
    if not isinstance(batch_num, int) or batch_num <= 0:
        return _stock_call(recognizer, args)

    img_num = len(img_list)
    if not img_num:
        return _stock_call(recognizer, args)
    if img_num <= batch_num:
        return _stock_call(recognizer, args)

    indices = np.argsort(
        np.array([img.shape[1] / float(img.shape[0]) for img in img_list])
    )
    channels, image_height, image_width = recognizer.rec_image_shape[:3]
    groups: list[dict[str, Any]] = []
    for begin in range(0, img_num, batch_num):
        end = min(img_num, begin + batch_num)
        max_wh_ratio = image_width / image_height
        wh_ratio_list = []
        for pos in range(begin, end):
            image = img_list[indices[pos]]
            ratio = image.shape[1] * 1.0 / image.shape[0]
            max_wh_ratio = max(max_wh_ratio, ratio)
            wh_ratio_list.append(ratio)

        normalized = [
            recognizer.resize_norm_img(img_list[indices[pos]], max_wh_ratio)[
                np.newaxis, :
            ]
            for pos in range(begin, end)
        ]
        normalized_batch = np.concatenate(normalized).astype(np.float32)
        padded_width = int(image_height * max_wh_ratio)
        # The stock call cannot exceed its own padded width. Treat malformed
        # custom resize contracts as incompatible and preserve stock behavior.
        if normalized_batch.shape[1:] != (channels, image_height, padded_width):
            return _stock_call(recognizer, args)
        groups.append(
            {
                "begin": begin,
                "end": end,
                "max_wh_ratio": max_wh_ratio,
                "wh_ratio_list": wh_ratio_list,
                "normalized": normalized_batch,
                "padded_width": padded_width,
            }
        )

    if len(groups) <= 1 or any(len(group["normalized"]) > max_batch for group in groups):
        return _stock_call(recognizer, args)

    rec_res: list[Any] = [("", 0.0)] * img_num
    return_word_box = args.return_word_box

    # Form maximal runs of consecutive original groups with identical padded
    # widths, respecting max_batch without splitting any source group.
    cursor = 0
    while cursor < len(groups):
        run = [groups[cursor]]
        merged_size = len(run[0]["normalized"])
        next_group = cursor + 1
        while next_group < len(groups):
            candidate = groups[next_group]
            if candidate["padded_width"] != run[0]["padded_width"]:
                break
            candidate_size = len(candidate["normalized"])
            if merged_size + candidate_size > max_batch:
                break
            run.append(candidate)
            merged_size += candidate_size
            next_group += 1

        merged_batch = np.concatenate([group["normalized"] for group in run])
        preds = recognizer.session(merged_batch)
        pred_offset = 0
        for group in run:
            group_size = len(group["normalized"])
            group_preds = preds[pred_offset : pred_offset + group_size]
            pred_offset += group_size
            line_results, word_results = recognizer.postprocess_op(
                group_preds,
                return_word_box,
                wh_ratio_list=group["wh_ratio_list"],
                max_wh_ratio=group["max_wh_ratio"],
            )
            for offset, line_result in enumerate(line_results):
                original_index = int(indices[group["begin"] + offset])
                if return_word_box:
                    rec_res[original_index] = (line_result, word_results[offset])
                else:
                    rec_res[original_index] = (line_result, None)
        cursor += len(run)

    all_line_results, all_word_results = list(zip(*rec_res))
    txts, scores = list(zip(*all_line_results))
    cfg = recognizer.cfg
    if _is_rtl(cfg.lang_type):
        from rapidocr.utils.utils import reorder_bidi_for_display

        txts = reorder_bidi_for_display(txts)

    from rapidocr.ch_ppocr_rec.typings import TextRecOutput
    from rapidocr.utils.vis_res import VisRes

    elapsed = time.perf_counter() - start_time
    return TextRecOutput(
        img_list,
        txts,
        scores,
        all_word_results,
        elapsed,
        viser=VisRes(lang_type=cfg.lang_type, font_path=cfg.font_path),
    )


def _supports_contract(recognizer: Any) -> bool:
    """Check only the ready recognizer surface used by RapidOCR 3.10."""
    try:
        shape = recognizer.rec_image_shape
        return (
            len(shape) >= 3
            and shape[0] > 0
            and shape[1] > 0
            and shape[2] > 0
            and callable(recognizer.session)
            and callable(recognizer.resize_norm_img)
            and callable(recognizer.postprocess_op)
            and hasattr(recognizer, "cfg")
            and hasattr(recognizer.cfg, "lang_type")
            and hasattr(recognizer.cfg, "font_path")
            and callable(recognizer)
        )
    except (AttributeError, TypeError):
        return False


def _stock_call(recognizer: Any, args: Any) -> Any:
    if not callable(recognizer):
        raise TypeError("recognizer does not satisfy the RapidOCR TextRecognizer contract")
    return recognizer(args)


def _is_rtl(lang_type: Any) -> bool:
    from rapidocr.utils.model_resolver import normalize_lang
    from rapidocr.ch_ppocr_rec.main import RTL_LANGS

    return normalize_lang(lang_type) in RTL_LANGS
