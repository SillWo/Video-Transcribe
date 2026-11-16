from __future__ import annotations

import threading
import time

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from services import meeting_assets_api
from services.meeting_assets import DOWNLOAD_BYTES


def test_assets_api_reports_download_size_then_polls_background_job(monkeypatch):
    monkeypatch.setattr(meeting_assets_api, "_status", lambda: {"available": False, "version": "pinned",
                                                                  "download_bytes": 10, "files": []})
    started, release = threading.Event(), threading.Event()
    def prepare(progress):
        progress({"file": "model.onnx", "completed_bytes": 5, "total_bytes": 10})
        started.set()
        release.wait(2)
        return {"available": True}
    monkeypatch.setattr(meeting_assets_api, "prepare_semantic_search", prepare)
    monkeypatch.setattr(meeting_assets_api, "_JOB", None)
    app = FastAPI()
    app.include_router(meeting_assets_api.router)
    client = TestClient(app)

    initial = client.get("/api/meetings/assets")
    assert initial.json()["semantic_search"]["download_bytes"] == 10
    assert client.get("/api/meetings/assets/prepare").json()["preparation"] is None
    response = client.post("/api/meetings/assets/prepare", json={"component": "semantic_search"})
    assert response.status_code == 202
    assert response.json()["estimated_download_bytes"] == DOWNLOAD_BYTES
    assert started.wait(1)
    polling = client.get("/api/meetings/assets/prepare").json()["preparation"]
    assert polling["status"] == "running"
    assert polling["progress"][-1]["completed_bytes"] == 5
    release.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        polling = client.get("/api/meetings/assets/prepare").json()["preparation"]
        if polling["status"] == "completed":
            break
        time.sleep(0.01)
    assert polling["status"] == "completed"
    assert polling["result"]["available"] is True


def test_assets_prepare_rejects_unknown_component_and_remote_peer(monkeypatch):
    monkeypatch.setattr(meeting_assets_api, "_status", lambda: {"available": False})
    app = FastAPI()
    app.include_router(meeting_assets_api.router)
    client = TestClient(app)
    bad_component = client.post("/api/meetings/assets/prepare", json={"component": "all_models"})
    assert bad_component.status_code == 400

    request = Request({"type": "http", "client": ("198.51.100.9", 1234), "headers": []})
    with pytest.raises(HTTPException) as error:
        meeting_assets_api._loopback(request)
    assert error.value.status_code == 403
