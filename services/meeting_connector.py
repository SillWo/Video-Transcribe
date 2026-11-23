"""Manage only this application's registration in the Codex MCP CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from services.meeting_mcp import SERVER_NAME, settings_path
from services.meeting_links import evidence_protocol_status, register_evidence_protocol, unregister_evidence_protocol


def _codex_command() -> str:
    configured = os.environ.get("CODEX_CLI")
    return configured or "codex"


def _run(args: Sequence[str], *, timeout: float = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), capture_output=True, text=True, timeout=timeout, check=False)


class MeetingConnector:
    """Idempotent registration and explicit protocol checks for Codex."""

    def __init__(self, project_root: str | Path | None = None, settings: str | Path | None = None,
                 python: str | Path | None = None, codex: str | None = None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.settings = Path(settings).resolve() if settings else settings_path()
        self.python = str(Path(python or sys.executable).resolve())
        self.codex = codex or _codex_command()

    def _get(self) -> tuple[bool, dict[str, Any] | None, str]:
        result = _run([self.codex, "mcp", "get", SERVER_NAME, "--json"])
        if result.returncode == 0:
            try:
                value = json.loads(result.stdout)
                return True, value if isinstance(value, dict) else {"value": value}, result.stdout
            except json.JSONDecodeError:
                return True, {"raw": result.stdout.strip()}, result.stdout
        # Older CLI versions may not implement `get --json`; use its read-only list.
        listed = _run([self.codex, "mcp", "list"])
        if listed.returncode == 0 and any(line.strip().split()[0:1] == [SERVER_NAME] for line in listed.stdout.splitlines() if line.strip()):
            return True, {"name": SERVER_NAME, "registration_details_available": False}, listed.stdout
        return False, None, (result.stderr or listed.stderr or result.stdout or listed.stdout).strip()

    def status(self) -> dict[str, Any]:
        registered, entry, detail = self._get()
        return {
            "server": SERVER_NAME,
            "registered": registered,
            "protocol_checked": False,
            "client_verified": False,
            "registration": entry,
            "evidence_protocol": evidence_protocol_status(),
            "detail": detail if not registered else None,
            "message": "Registration exists; the Codex client has not yet been verified." if registered else "The MCP server is not registered in Codex.",
        }

    def register(self) -> dict[str, Any]:
        exists, entry, detail = self._get()
        if exists:
            protocol = register_evidence_protocol(self.python, self.settings, self.project_root)
            return {"server": SERVER_NAME, "registered": True, "added": False, "protocol_checked": False,
                    "client_verified": False, "registration": entry,
                    "evidence_protocol": protocol,
                    "message": "Already registered. Restart or refresh Codex before treating it as connected."}
        script = self.project_root / "services" / "meeting_mcp.py"
        if not script.is_file():
            raise FileNotFoundError(f"MCP server entry point not found: {script}")
        command = [self.codex, "mcp", "add", SERVER_NAME, "--", self.python, str(script),
                   "--settings", str(self.settings), "--project-root", str(self.project_root)]
        added = _run(command)
        if added.returncode != 0:
            raise RuntimeError((added.stderr or added.stdout or "Codex MCP registration failed").strip())
        now_registered, now_entry, verify_detail = self._get()
        if not now_registered:
            raise RuntimeError("Codex CLI accepted the add command but the server registration could not be confirmed: " + (verify_detail or detail))
        protocol = register_evidence_protocol(self.python, self.settings, self.project_root)
        return {"server": SERVER_NAME, "registered": True, "added": True, "protocol_checked": False,
                "client_verified": False, "registration": now_entry, "evidence_protocol": protocol,
                "message": "Registered. Restart or refresh Codex, then run a real tool call to verify the client connection."}

    def disconnect(self) -> dict[str, Any]:
        exists, _, detail = self._get()
        if not exists:
            protocol = unregister_evidence_protocol()
            return {"server": SERVER_NAME, "registered": False, "removed": False, "detail": detail,
                    "evidence_protocol": protocol,
                    "message": "The meeting archive connector was already disconnected."}
        removed = _run([self.codex, "mcp", "remove", SERVER_NAME])
        if removed.returncode != 0:
            raise RuntimeError((removed.stderr or removed.stdout or "Codex MCP removal failed").strip())
        still_exists, _, failure = self._get()
        if still_exists:
            raise RuntimeError("Codex accepted removal but the meeting archive registration remains present.")
        protocol = unregister_evidence_protocol()
        return {"server": SERVER_NAME, "registered": False, "removed": True,
                "evidence_protocol": protocol,
                "message": "Removed only the meeting archive connector. The archive itself was not changed."}

    def verify_protocol(self, timeout: float = 40, meeting_id: str | None = None) -> dict[str, Any]:
        """Exercise this server via a real MCP SDK stdio client, including image payloads."""
        try:
            from anyio import fail_after
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.types import ImageContent
        except ImportError as exc:
            raise RuntimeError("Install the MCP Python SDK and anyio to run a protocol check.") from exc
        script = self.project_root / "services" / "meeting_mcp.py"
        params = StdioServerParameters(command=self.python, args=[str(script), "--settings", str(self.settings), "--project-root", str(self.project_root)])
        progress: dict[str, Any] = {"initialized": False, "tools_discovered": False,
                                    "get_frames_call": False, "actual_image_content_count": 0,
                                    "jpeg_payloads_valid": False,
                                    "conversion_tool_discovered": False}
        try:
            import anyio
            async def check() -> dict[str, Any]:
                with fail_after(timeout):
                    async with stdio_client(params) as (read_stream, write_stream):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            progress["initialized"] = True
                            tools = await session.list_tools()
                            names = {tool.name for tool in tools.tools}
                            progress["tools_discovered"] = True
                            progress["conversion_tool_discovered"] = "convert_media_file" in names and "get_media_conversion_status" in names
                            required = {"get_archive_status", "find_meetings", "find_archive_items", "read_archive_item",
                                        "search_archive", "search_evidence", "read_timeline", "get_frames",
                                        "get_glossary", "get_source_status", "open_evidence",
                                        "convert_media_file", "get_media_conversion_status"}
                            missing = sorted(required - names)
                            if missing:
                                raise RuntimeError("MCP server is missing tools: " + ", ".join(missing))
                            target_id = meeting_id
                            if target_id is None:
                                found = await session.call_tool("find_meetings", {"limit": 100, "offset": 0})
                                if not found.isError:
                                    for block in found.content:
                                        if getattr(block, "type", None) == "text":
                                            try:
                                                listing = json.loads(block.text)
                                                entries = listing.get("items", [])
                                                target_id = next((item.get("meeting_id") or item.get("id") for item in entries if item.get("meeting_id") or item.get("id")), None)
                                            except (json.JSONDecodeError, TypeError, AttributeError):
                                                continue
                                            if target_id:
                                                break
                            if target_id:
                                response = await session.call_tool("get_frames", {"meeting_id": target_id, "limit": 8})
                                progress["get_frames_call"] = True
                                if response.isError:
                                    raise RuntimeError("MCP get_frames tool returned an error")
                                images = [block for block in response.content if isinstance(block, ImageContent)]
                                progress["actual_image_content_count"] = len(images)
                                progress["jpeg_payloads_valid"] = bool(images) and all(
                                    _is_base64_jpeg(block.data) for block in images
                                )
                                if meeting_id and not progress["jpeg_payloads_valid"]:
                                    raise RuntimeError("The requested meeting did not return a valid base64 JPEG ImageContent payload")
                            return {"tools": sorted(names), **progress}
            checked = anyio.run(check)
        except Exception as exc:
            return {"server": SERVER_NAME, "registered": self._get()[0], "protocol_checked": bool(progress["initialized"] and progress["tools_discovered"]),
                    "client_verified": False, **progress, "error": str(exc)}
        return {"server": SERVER_NAME, "registered": self._get()[0], "protocol_checked": True,
                "client_verified": False, **checked,
                "message": "The MCP Python SDK STDIO handshake, tool discovery, and image call were checked. Conversion was discovered but not run. This is not a real Codex-client verification."}


def register(project_root: str | Path | None = None, settings: str | Path | None = None) -> dict[str, Any]:
    return MeetingConnector(project_root, settings).register()


def get_status(project_root: str | Path | None = None, settings: str | Path | None = None) -> dict[str, Any]:
    return MeetingConnector(project_root, settings).status()


def disconnect(project_root: str | Path | None = None, settings: str | Path | None = None) -> dict[str, Any]:
    return MeetingConnector(project_root, settings).disconnect()


def _is_base64_jpeg(data: str) -> bool:
    import base64
    try:
        decoded = base64.b64decode(data, validate=True)
        return decoded.startswith(b"\xff\xd8") and decoded.endswith(b"\xff\xd9")
    except (ValueError, TypeError):
        return False
