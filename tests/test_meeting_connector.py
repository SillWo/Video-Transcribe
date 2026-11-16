import json
import subprocess

import pytest

from services import meeting_connector
from services import meeting_links


@pytest.fixture(autouse=True)
def avoid_host_protocol_registry_changes(monkeypatch):
    monkeypatch.setattr(meeting_connector, "register_evidence_protocol", lambda *_: {
        "supported": False, "registered": False, "owned": False, "conflict": False,
    })
    monkeypatch.setattr(meeting_connector, "evidence_protocol_status", lambda: {
        "supported": False, "registered": False, "owned": False, "conflict": False,
    })
    monkeypatch.setattr(meeting_connector, "unregister_evidence_protocol", lambda: {
        "supported": False, "registered": False, "owned": False, "conflict": False,
    })


def completed(code=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=stdout, stderr=stderr)


def test_registration_is_idempotent_and_does_not_claim_client_verified(tmp_path, monkeypatch):
    connector = meeting_connector.MeetingConnector(project_root=tmp_path, settings=tmp_path / "settings.json", python=tmp_path / "python.exe", codex="codex")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return completed(0, json.dumps({"name": meeting_connector.SERVER_NAME, "command": "python"}))

    monkeypatch.setattr(meeting_connector, "_run", run)
    result = connector.register()
    assert result["registered"] is True
    assert result["added"] is False
    assert result["client_verified"] is False
    assert not any("add" in call for call in calls)


def test_register_uses_cli_argument_list_and_stable_absolute_entrypoint(tmp_path, monkeypatch):
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "meeting_mcp.py").write_text("# stub", encoding="utf-8")
    connector = meeting_connector.MeetingConnector(project_root=tmp_path, settings=tmp_path / "settings.json", python=tmp_path / "python.exe", codex="codex")
    calls = []
    get_count = 0

    def run(args, **kwargs):
        nonlocal get_count
        calls.append(args)
        if args[1:3] == ["mcp", "get"]:
            get_count += 1
            if get_count == 1:
                return completed(1, stderr="not found")
            return completed(0, json.dumps({"name": meeting_connector.SERVER_NAME}))
        if args[1:3] == ["mcp", "add"]:
            return completed(0, "Added")
        return completed(1, stderr="unexpected")

    monkeypatch.setattr(meeting_connector, "_run", run)
    result = connector.register()
    assert result["added"] is True
    assert result["client_verified"] is False
    add_call = next(call for call in calls if call[1:3] == ["mcp", "add"])
    assert "--" in add_call
    assert str((tmp_path / "services" / "meeting_mcp.py").resolve()) in add_call
    assert "--settings" in add_call


def test_disconnect_removes_only_managed_server(tmp_path, monkeypatch):
    connector = meeting_connector.MeetingConnector(project_root=tmp_path, codex="codex")
    registered = True
    calls = []

    def run(args, **kwargs):
        nonlocal registered
        calls.append(args)
        if args[1:3] == ["mcp", "get"]:
            return completed(0, json.dumps({"name": meeting_connector.SERVER_NAME})) if registered else completed(1, stderr="not found")
        if args[1:3] == ["mcp", "remove"]:
            registered = False
            return completed(0, "Removed")
        return completed(1, stderr="unexpected")

    monkeypatch.setattr(meeting_connector, "_run", run)
    result = connector.disconnect()
    assert result["removed"] is True
    assert ["codex", "mcp", "remove", meeting_connector.SERVER_NAME] in calls


def test_status_distinguishes_registration_from_client_verification(tmp_path, monkeypatch):
    connector = meeting_connector.MeetingConnector(project_root=tmp_path, codex="codex")
    monkeypatch.setattr(meeting_connector, "_run", lambda args, **kwargs: completed(0, json.dumps({"name": meeting_connector.SERVER_NAME})))
    status = connector.status()
    assert status["registered"] is True
    assert status["protocol_checked"] is False
    assert status["client_verified"] is False


def test_evidence_protocol_registration_preserves_conflicts_and_removes_only_own_handler(tmp_path, monkeypatch):
    conflict_status = {"supported": True, "registered": True, "owned": False, "conflict": True}
    monkeypatch.setattr(meeting_links, "evidence_protocol_status", lambda: conflict_status)
    monkeypatch.setattr(meeting_links.os, "name", "nt")
    conflict = meeting_links.register_evidence_protocol(tmp_path / "python.exe", project_root=tmp_path)
    assert conflict["conflict"] is True
    assert conflict is conflict_status
    # A conflicting per-user protocol is left untouched; no registry API is called.


def test_evidence_protocol_registration_persists_its_archive_binding(tmp_path, monkeypatch):
    import sys
    import types

    values = []

    class Key:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake_registry = types.SimpleNamespace(
        HKEY_CURRENT_USER=object(), KEY_READ=1, KEY_WRITE=2, REG_SZ=3,
        CreateKeyEx=lambda _root, path, *_args: Key(path),
        SetValueEx=lambda key, name, _reserved, _kind, value: values.append((key.path, name, value)),
    )
    monkeypatch.setitem(sys.modules, "winreg", fake_registry)
    monkeypatch.setattr(meeting_links, "evidence_protocol_status", lambda: {
        "supported": True, "registered": True, "owned": True, "conflict": False,
    })

    settings = tmp_path / "meetings-settings.json"
    project = tmp_path / "project"
    meeting_links.register_evidence_protocol(tmp_path / "python.exe", settings, project)

    assert (meeting_links._PROTOCOL_REGISTRY_KEY, meeting_links._PROTOCOL_SETTINGS_VALUE,
            str(settings.resolve())) in values
    assert (meeting_links._PROTOCOL_REGISTRY_KEY, meeting_links._PROTOCOL_PROJECT_VALUE,
            str(project.resolve())) in values
