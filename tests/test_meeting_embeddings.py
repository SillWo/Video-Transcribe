import sys
from types import SimpleNamespace

import numpy as np
import pytest

from services.meeting_embeddings import MAX_TOKENS, MultilingualE5, content_hash


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1 if token != 0 else 0 for token in ids]
        self.type_ids = [0] * len(ids)


class FakeTokenizer:
    def __init__(self):
        self.inputs = []
        self.truncation = None
        self.padding = False

    @classmethod
    def from_file(cls, _path):
        return cls()

    def enable_truncation(self, *, max_length):
        self.truncation = max_length

    def enable_padding(self):
        self.padding = True

    def encode_batch(self, values):
        self.inputs.extend(values)
        return [FakeEncoding([1, 2, 0]) for _ in values]


class FakeSession:
    def __init__(self):
        self.batch_sizes = []

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask"),
                SimpleNamespace(name="token_type_ids")]

    def run(self, _outputs, feeds):
        ids = feeds["input_ids"]
        assert feeds["token_type_ids"].shape == ids.shape
        self.batch_sizes.append(ids.shape[0])
        output = np.zeros((ids.shape[0], ids.shape[1], 384), dtype=np.float32)
        output[:, 0, 0] = 1
        output[:, 1, 0] = 3
        output[:, 2, 0] = 1000  # masked padding must not affect the pooled vector
        return [output]


def test_constructor_is_lazy_and_missing_model_is_honest(tmp_path):
    model_dir = tmp_path / "missing"
    encoder = MultilingualE5(model_dir)
    assert encoder.available is False
    assert encoder._session is None
    with pytest.raises(FileNotFoundError):
        encoder.encode_queries(["a query"])


def test_onnx_encoding_uses_prefixes_masked_pooling_normalization_and_batches(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"onnx test bytes")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    tokenizer = FakeTokenizer()
    session = FakeSession()

    class TokenizerFactory:
        @staticmethod
        def from_file(path):
            assert path.endswith("tokenizer.json")
            return tokenizer

    class FakeOptions:
        intra_op_num_threads = None
        inter_op_num_threads = None

    def create_session(path, sess_options, providers):
        assert path.endswith("model.onnx")
        assert providers == ["CPUExecutionProvider"]
        assert sess_options.intra_op_num_threads == 4
        assert sess_options.inter_op_num_threads == 1
        return session

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=TokenizerFactory))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=create_session,
                                                                   SessionOptions=FakeOptions))
    encoder = MultilingualE5(model_dir, batch_size=8)
    values = [f"text {i}" for i in range(10)]
    vectors = encoder.encode_passages(values)

    assert tokenizer.truncation == MAX_TOKENS
    assert tokenizer.padding is True
    assert tokenizer.inputs == ["passage: " + value for value in values]
    assert session.batch_sizes == [8, 2]
    assert vectors.shape == (10, 384)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors[:, 0], 1.0)  # masked token 1000 did not influence mean pooling

    encoder.encode_queries(["Find this"])
    assert tokenizer.inputs[-1] == "query: Find this"


def test_empty_input_returns_an_empty_float32_matrix():
    vectors = MultilingualE5("missing").encode_passages([])
    assert vectors.shape == (0, 384)
    assert vectors.dtype == np.float32


def test_content_hash_is_stable_and_sensitive_to_literal_text():
    assert content_hash("ARCH-427") == content_hash("ARCH-427")
    assert content_hash("ARCH-427") != content_hash("ARCН-427")
