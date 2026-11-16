from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import pytest

from services import meeting_assets


class FakeResponse:
    def __init__(self, data: bytes, fail_after: int | None = None):
        self._stream = io.BytesIO(data)
        self._read = 0
        self._fail_after = fail_after

    def __enter__(self): return self
    def __exit__(self, *_): return False
    def read(self, size=-1):
        if self._fail_after is not None and self._read >= self._fail_after:
            raise OSError("interrupted download")
        chunk = self._stream.read(size)
        self._read += len(chunk)
        return chunk


def _configure_small_assets(monkeypatch, contents):
    rows = []
    for relative, data in contents.items():
        rows.append({"path": relative, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    monkeypatch.setattr(meeting_assets, "_FILES", tuple(rows))
    monkeypatch.setattr(meeting_assets, "DOWNLOAD_BYTES", sum(row["size"] for row in rows))
    meeting_assets._VERIFIED.clear()
    return rows


def test_status_is_explicit_and_reports_pinned_download_size(tmp_path, monkeypatch):
    monkeypatch.setattr(meeting_assets, "_FILES", ({"path": "onnx/model.onnx", "size": 4, "sha256": "0" * 64},))
    monkeypatch.setattr(meeting_assets, "DOWNLOAD_BYTES", 4)
    meeting_assets._VERIFIED.clear()
    status = meeting_assets.asset_status(tmp_path / "каталог с пробелом")
    assert status["available"] is False
    assert status["version"] == meeting_assets.REVISION
    assert status["download_bytes"] == 4


def test_prepare_downloads_each_pinned_file_atomically_with_unicode_path(tmp_path, monkeypatch):
    content = {"onnx/model.onnx": b"model-bytes", "onnx/tokenizer.json": "токенайзер".encode("utf-8"),
               "onnx/config.json": b"{}"}
    rows = _configure_small_assets(monkeypatch, content)
    urls = []
    def open_url(url, timeout):
        urls.append((url, timeout))
        relative = url.split(f"/{meeting_assets.REVISION}/", 1)[1].split("?", 1)[0]
        return FakeResponse(content[relative])
    monkeypatch.setattr(meeting_assets.urllib.request, "urlopen", open_url)
    progress = []
    project = tmp_path / "архив с пробелами"
    result = meeting_assets.prepare_semantic_search(project, progress=progress.append)
    assert result["available"] is True
    assert len(urls) == len(rows)
    assert all(meeting_assets.REVISION in url for url, _ in urls)
    assert all((meeting_assets.model_root(project) / row["path"]).read_bytes() == content[row["path"]] for row in rows)
    assert not list(meeting_assets.model_root(project).rglob("*.part"))
    assert progress[-1]["ready"] is True


def test_prepare_skips_existing_valid_files_without_network(tmp_path, monkeypatch):
    content = {"onnx/model.onnx": b"valid model", "onnx/tokenizer.json": b"valid tokenizer"}
    rows = _configure_small_assets(monkeypatch, content)
    root = meeting_assets.model_root(tmp_path)
    for row in rows:
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content[row["path"]])
    monkeypatch.setattr(meeting_assets.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("must not download"))
    result = meeting_assets.prepare_semantic_search(tmp_path)
    assert result["available"] is True


def test_bad_checksum_does_not_replace_an_existing_good_file(tmp_path, monkeypatch):
    content = {"onnx/model.onnx": b"good model", "onnx/tokenizer.json": b"good tokenizer"}
    rows = _configure_small_assets(monkeypatch, content)
    good_path = meeting_assets.model_root(tmp_path) / rows[0]["path"]
    good_path.parent.mkdir(parents=True, exist_ok=True)
    good_path.write_bytes(content[rows[0]["path"]])
    def open_url(url, timeout):
        relative = url.split(f"/{meeting_assets.REVISION}/", 1)[1].split("?", 1)[0]
        expected = content[relative]
        return FakeResponse(expected + b"corrupt")
    monkeypatch.setattr(meeting_assets.urllib.request, "urlopen", open_url)
    with pytest.raises(ValueError, match="larger than its pinned size"):
        meeting_assets.prepare_semantic_search(tmp_path)
    assert good_path.read_bytes() == content[rows[0]["path"]]
    assert not list(meeting_assets.model_root(tmp_path).rglob("*.part"))


def test_interrupted_download_keeps_existing_file_and_removes_part(tmp_path, monkeypatch):
    content = {"onnx/model.onnx": b"existing good", "onnx/tokenizer.json": b"tokenizer"}
    rows = _configure_small_assets(monkeypatch, content)
    root = meeting_assets.model_root(tmp_path)
    good_path = root / rows[0]["path"]
    good_path.parent.mkdir(parents=True, exist_ok=True)
    good_path.write_bytes(content[rows[0]["path"]])
    def open_url(url, timeout):
        relative = url.split(f"/{meeting_assets.REVISION}/", 1)[1].split("?", 1)[0]
        return FakeResponse(content[relative], fail_after=3)
    monkeypatch.setattr(meeting_assets.urllib.request, "urlopen", open_url)
    with pytest.raises(OSError, match="interrupted"):
        meeting_assets.prepare_semantic_search(tmp_path)
    assert good_path.read_bytes() == content[rows[0]["path"]]
    assert not list(root.rglob("*.part"))
