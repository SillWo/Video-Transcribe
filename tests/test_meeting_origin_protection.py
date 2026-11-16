from fastapi.testclient import TestClient

from services import meeting_recording_api
from web_api import app


class RecorderStub:
    def __init__(self):
        self.calls = []

    def prepare(self):
        self.calls.append("prepare")
        return {"state": "idle"}

    def stop(self):
        self.calls.append("stop")
        return {"meeting": {"meeting_id": "x"}, "recorder": {"state": "idle"}}


def test_cross_site_empty_recorder_posts_block_before_service_call(monkeypatch):
    recorder = RecorderStub()
    monkeypatch.setattr(meeting_recording_api, "get_default_recorder", lambda: recorder)
    client = TestClient(app)

    prepare = client.post("/api/meetings/recorder/prepare", headers={"Origin": "https://attacker.example"})
    stop = client.post("/api/meetings/recorder/stop", headers={"Origin": "http://127.0.0.1.attacker.test"})
    assert prepare.status_code == stop.status_code == 403
    assert recorder.calls == []


def test_same_origin_and_dev_origin_mutations_are_allowed_and_no_origin_cli_remains_allowed(monkeypatch):
    recorder = RecorderStub()
    monkeypatch.setattr(meeting_recording_api, "get_default_recorder", lambda: recorder)
    client = TestClient(app)

    assert client.post("/api/meetings/recorder/prepare", headers={"Origin": "http://localhost:5173"}).status_code == 200
    assert client.post("/api/meetings/recorder/stop").status_code == 200
    assert recorder.calls == ["prepare", "stop"]
