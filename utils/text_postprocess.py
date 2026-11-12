import re

from services.model_paths import HF_HOME  # configure before lazy Transformers/PyTorch imports
from services.task_errors import TaskFailure

try:
    import razdel
except Exception:
    razdel = None


_PUNCTUATION_MODEL = None
_PUNCTUATION_DEVICE = "cpu"


def configure_punctuation_device(device: str, asr_device: str | None = None) -> None:
    """Select the supported postprocess device without loading PyTorch eagerly."""
    global _PUNCTUATION_DEVICE
    _PUNCTUATION_DEVICE = "cuda" if device == "cuda" else "cpu"
    if asr_device in {"directml", "vulkan"} and _PUNCTUATION_DEVICE == "cpu":
        print(f"Punctuation is using CPU with {asr_device} ASR: PyTorch Vulkan and a compatible DirectML backend are unavailable.",
              flush=True)
    if _PUNCTUATION_MODEL is not None:
        _move_punctuation_model(_PUNCTUATION_MODEL, _PUNCTUATION_DEVICE)


def _move_punctuation_model(model, device: str) -> bool:
    global _PUNCTUATION_DEVICE
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("Punctuation CUDA is unavailable; using CPU.", flush=True)
                model.to("cpu")
                _PUNCTUATION_DEVICE = "cpu"
                return False
            model.to("cuda")
            return True
        except Exception as exc:
            print(f"Punctuation CUDA failed ({exc}); using CPU.", flush=True)
            try:
                model.to("cpu")
            except Exception as cpu_exc:
                raise TaskFailure(f"Unable to move punctuation model to CPU: {cpu_exc}",
                                  "model_unavailable", stage="postprocess") from cpu_exc
            _PUNCTUATION_DEVICE = "cpu"
            return False
    try:
        model.to("cpu")
    except AttributeError:
        pass
    except Exception as exc:
        raise TaskFailure(f"Unable to move punctuation model to CPU: {exc}",
                          "model_unavailable", stage="postprocess") from exc
    return True


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
        _move_punctuation_model(_PUNCTUATION_MODEL, _PUNCTUATION_DEVICE)
        return _PUNCTUATION_MODEL
    except Exception as exc:
        raise TaskFailure(f"Unable to load kontur-ai/sbert_punc_case_ru model: {exc}",
                          "model_unavailable", stage="postprocess") from exc


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def restore_russian_segment_text(text: str) -> str:
    global _PUNCTUATION_DEVICE
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
        if _PUNCTUATION_DEVICE == "cuda":
            print(f"Punctuation CUDA inference failed ({exc}); retrying once on CPU.", flush=True)
            try:
                model.to("cpu")
            except Exception as cpu_move_exc:
                raise TaskFailure(f"Unable to move punctuation model to CPU after CUDA failure: {cpu_move_exc}",
                                  "model_unavailable", stage="postprocess") from cpu_move_exc
            _PUNCTUATION_DEVICE = "cpu"
            try:
                punctuated_text = model.punctuate(normalized_input)
            except Exception as cpu_exc:
                raise TaskFailure(f"Unable to run kontur-ai/sbert_punc_case_ru post-processing on CPU: {cpu_exc}",
                                  "model_unavailable", stage="postprocess") from cpu_exc
        else:
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
