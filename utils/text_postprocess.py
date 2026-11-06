import re

from services.model_paths import HF_HOME  # configure before lazy Transformers/PyTorch imports
from services.task_errors import TaskFailure

try:
    import razdel
except Exception:
    razdel = None


_PUNCTUATION_MODEL = None


def load_punctuation_model() -> object:
    global _PUNCTUATION_MODEL

    if _PUNCTUATION_MODEL is not None:
        return _PUNCTUATION_MODEL

    try:
        from sbert_punc_case_ru import SbertPuncCase
    except ImportError as exc:
        raise TaskFailure(
            "Не найдена библиотека модели восстановления пунктуации.",
            "dependency_missing", stage="postprocess",
        ) from exc
    try:
        _PUNCTUATION_MODEL = SbertPuncCase()
        return _PUNCTUATION_MODEL
    except Exception as exc:
        raise TaskFailure(f"Unable to load kontur-ai/sbert_punc_case_ru model: {exc}",
                          "model_unavailable", stage="postprocess") from exc


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def restore_russian_segment_text(text: str) -> str:
    stripped_text = text.strip()
    if not stripped_text:
        return ""

    normalized_input = normalize_whitespace(re.sub(r"[^\w\s]", " ", stripped_text, flags=re.UNICODE)).lower()
    if not normalized_input:
        return ""
    model = load_punctuation_model()
    try:
        punctuated_text = model.punctuate(normalized_input)
    except Exception as exc:
        raise TaskFailure(f"Unable to run kontur-ai/sbert_punc_case_ru post-processing: {exc}",
                          "model_unavailable", stage="postprocess") from exc
    return normalize_whitespace(punctuated_text)


def polish_russian_full_text(text: str) -> str:
    stripped_text = text.strip()
    if not stripped_text:
        return ""

    if razdel is None:
        raise TaskFailure("Unable to import razdel for Russian text post-processing.",
                          "dependency_missing", stage="postprocess")

    try:
        sentences = [sentence.text.strip() for sentence in razdel.sentenize(stripped_text) if sentence.text.strip()]
    except Exception as exc:
        raise TaskFailure(f"Unable to run razdel sentence polishing: {exc}",
                          "transcription_failed", stage="postprocess") from exc
    return normalize_whitespace(" ".join(sentences))


def build_polished_russian_text_from_segments(segment_texts: list[str]) -> str:
    merged_text = " ".join(text.strip() for text in segment_texts if text and text.strip())
    return polish_russian_full_text(merged_text)
