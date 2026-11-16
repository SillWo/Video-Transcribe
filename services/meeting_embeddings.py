"""Lazy local ONNX embeddings for multilingual-e5-small.

This module never downloads models. The ONNX session and tokenizer are only
loaded when an explicit encode call is made.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


MODEL_NAME = "multilingual-e5-small"
MAX_TOKENS = 512
BATCH_SIZE = 8
VECTOR_SIZE = 384


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / ".runtime" / "models" / MODEL_NAME / "onnx"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MultilingualE5:
    """Encode query/passage text with a local ONNX E5 model on CPU."""

    def __init__(self, model_path: str | Path | None = None, batch_size: int = BATCH_SIZE):
        self.model_path = Path(model_path) if model_path is not None else default_model_path()
        self.batch_size = max(1, min(BATCH_SIZE, int(batch_size)))
        self._tokenizer = None
        self._session = None
        self._model_hash: str | None = None

    @property
    def available(self) -> bool:
        return (self.model_path / "model.onnx").is_file() and (self.model_path / "tokenizer.json").is_file()

    @property
    def model_hash(self) -> str:
        if not self.available:
            raise FileNotFoundError(f"Local {MODEL_NAME} ONNX model is unavailable at {self.model_path}")
        if self._model_hash is None:
            digest = hashlib.sha256()
            digest.update(_file_hash(self.model_path / "model.onnx").encode("ascii"))
            digest.update(_file_hash(self.model_path / "tokenizer.json").encode("ascii"))
            self._model_hash = digest.hexdigest()
        return self._model_hash

    def _load(self) -> None:
        if self._session is not None:
            return
        if not self.available:
            raise FileNotFoundError(f"Local {MODEL_NAME} ONNX model is unavailable at {self.model_path}")
        from tokenizers import Tokenizer
        import onnxruntime as ort

        self._tokenizer = Tokenizer.from_file(str(self.model_path / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(self.model_path / "model.onnx"),
                                             sess_options=options,
                                             providers=["CPUExecutionProvider"])

    def encode_queries(self, texts: Iterable[str]) -> np.ndarray:
        return self.encode(texts, prefix="query: ")

    def encode_passages(self, texts: Iterable[str]) -> np.ndarray:
        return self.encode(texts, prefix="passage: ")

    def encode(self, texts: Iterable[str], *, prefix: str) -> np.ndarray:
        values = [str(text or "") for text in texts]
        if not values:
            return np.empty((0, VECTOR_SIZE), dtype=np.float32)
        self._load()
        assert self._tokenizer is not None and self._session is not None
        input_names = {item.name for item in self._session.get_inputs()}
        vectors: list[np.ndarray] = []
        for offset in range(0, len(values), self.batch_size):
            batch = values[offset:offset + self.batch_size]
            encodings = self._tokenizer.encode_batch([prefix + text for text in batch])
            max_len = max(len(item.ids) for item in encodings)
            input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
            attention = np.zeros_like(input_ids)
            token_types = np.zeros_like(input_ids)
            for row, item in enumerate(encodings):
                length = len(item.ids)
                input_ids[row, :length] = item.ids
                attention[row, :length] = item.attention_mask
                if getattr(item, "type_ids", None) is not None:
                    token_types[row, :length] = item.type_ids
            feeds = {"input_ids": input_ids, "attention_mask": attention,
                     "token_type_ids": token_types}
            outputs = self._session.run(None, {key: value for key, value in feeds.items() if key in input_names})
            if not outputs:
                raise RuntimeError("E5 ONNX model returned no output")
            token_vectors = np.asarray(outputs[0], dtype=np.float32)
            if token_vectors.ndim != 3 or token_vectors.shape[0] != len(batch):
                raise RuntimeError(f"Unexpected E5 output shape: {token_vectors.shape}")
            mask = attention[:, :token_vectors.shape[1]].astype(np.float32)[..., None]
            pooled = (token_vectors * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalized = pooled / np.maximum(norms, 1e-12)
            vectors.extend(normalized.astype(np.float32))
        return np.asarray(vectors, dtype=np.float32)

