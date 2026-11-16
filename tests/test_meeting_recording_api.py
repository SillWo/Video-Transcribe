from __future__ import annotations

from fastapi import FastAPI
import pytest
from fastapi.testclient import TestClient

from services import meeting_recording_api


class FakeRecorder:
    def __init__(self):
        self.calls = []

    def status(self):
        self.calls.append(("status",))
        return {"available": False, "connected": False, "state": "idle", "duration_seconds": None,
                "monitors": [], "browsers": [], "microphones": [],
                "selection": {"monitor_id": None, "browser_id": None, "microphone_id": None},
                "preview_url": "/api/meetings/recorder/preview",
                "levels": {"browser": None, "microphone": None, "stale": True}}

    def prepare(self): self.calls.append(("prepare",)); return self.status()
    def configure(self, selection): self.calls.append(("configure", selection)); return self.status()
    def start(self, title): self.calls.append(("start", title)); return {"state": "recording"}
    def stop(self): self.calls.append(("stop",)); return {"meeting": {"meeting_id": "stable"}, "recorder": {"state": "idle"}}
    def preview(self): self.calls.append(("preview",)); return b"\xff\xd8jpeg\xff\xd9"


def test_recorder_http_contract_and_no_secret_leak(monkeypatch):
    service = FakeRecorder()
    monkeypatch.setattr(meeting_recording_api, "get_default_recorder", lambda: service)
    app = FastAPI()
    app.include_router(meeting_recording_api.router)
    client = TestClient(app)

    status = client.get("/api/meetings/recorder")
    assert status.status_code == 200
    assert status.json()["state"] == "idle"
    assert status.json()["download"]["size_bytes"] > 0
    assert "password" not in status.text.casefold()

    assert client.post("/api/meetings/recorder/prepare").json()["state"] == "idle"
    assert client.post("/api/meetings/recorder/configure", json={
        "monitor_id": "monitor", "browser_id": "browser#3A", "microphone_id": "default"}).status_code == 200
    assert client.post("/api/meetings/recorder/start", json={"title": "Встреча"}).json() == {"state": "recording"}
    stopped = client.post("/api/meetings/recorder/stop")
    assert stopped.status_code == 200
    assert stopped.json()["meeting"]["meeting_id"] == "stable"
    preview = client.get("/api/meetings/recorder/preview")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"
    assert preview.content.startswith(b"\xff\xd8")
    assert [call[0] for call in service.calls] == ["status", "prepare", "status", "configure", "status", "start", "stop", "preview"]


def test_recorder_routes_reject_remote_clients(monkeypatch):
    monkeypatch.setattr(meeting_recording_api, "get_default_recorder", lambda: pytest.fail("service must not run"))
    app = FastAPI()
    app.include_router(meeting_recording_api.router)
    client = TestClient(app)
    # Remote origins are blocked by the request peer address; the test client is local.
    # Exercise the explicit guard with a synthetic request.
    from starlette.requests import Request
    from fastapi import HTTPException
    request = Request({"type": "http", "client": ("198.51.100.12", 1234), "headers": []})
    with pytest.raises(HTTPException) as error:
        meeting_recording_api._local_only(request)
    assert error.value.status_code == 403
