import json
import os
import signal
import subprocess
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from services import meeting_mcp


class FakeArchive:
    def __init__(self, root, read_only=False):
        assert read_only is True
        self.root = Path(root)

    def list_meetings(self, **kwargs):
        return [{"id": "one", "title": "Звонок"}]

    def get_meeting(self, meeting_id):
        return {"id": meeting_id, "title": "Звонок"}

    def meeting_dir(self, meeting_id):
        return self.root / meeting_id

    def original_path(self, meeting_id):
        return None

    def timeline(self, meeting_id, **kwargs):
        return {"items": [{"id": "e1", "kind": "screen", "start": 1.0, "end": 2.0, "frame": "frames/a.jpg", "text": "OCR", "regions": [], "tables": []}], "total": 1, "next_offset": None, "complete": True}

    def read_screen_events(self, meeting_id):
        return {"events": [{"id": "e1", "start": 1.0, "end": 2.0, "frame": "frames/a.jpg", "text": "OCR", "regions": [], "tables": []}], "status": "ready", "limitations": []}

    def glossary(self, meeting_id):
        return [{"id": "g1", "term": "API", "definition": "interface"}]


class FakeIndex:
    capabilities = {"search": ["hybrid"]}

    def __init__(self, root):
        self.root = root

    def search(self, query, **kwargs):
        return {"results": [{"query": query}], "total": 1, "next_offset": None}


def fake_core(monkeypatch):
    archive_module = types.ModuleType("services.meeting_archive")
    archive_module.MeetingArchive = FakeArchive
    monkeypatch.setitem(sys.modules, "services.meeting_archive", archive_module)
    import services.meeting_search as index_module
    monkeypatch.setattr(index_module, "ReadIndexProxy", FakeIndex)


def test_settings_resolution_uses_shared_contract(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"archive_root": str(tmp_path / "archive")}), encoding="utf-8")
    monkeypatch.delenv("VT_MEETING_ROOT", raising=False)
    assert meeting_mcp.resolve_archive_root(settings=settings) == (tmp_path / "archive").resolve()
    monkeypatch.setenv("VT_MEETING_ROOT", str(tmp_path / "override"))
    assert meeting_mcp.resolve_archive_root(settings=settings) == (tmp_path / "override").resolve()


def test_archive_tools_reload_settings_root_but_keep_explicit_root_fixed(tmp_path, monkeypatch):
    monkeypatch.delenv("VT_MEETING_ROOT", raising=False)
    settings = tmp_path / "meetings-settings.json"
    root_a, root_b = tmp_path / "archive-a", tmp_path / "archive-b"
    settings.write_text(json.dumps({"archive_root": str(root_a)}), encoding="utf-8")
    dynamic = meeting_mcp.ArchiveTools(None, settings=settings, project_root=tmp_path)
    explicit = meeting_mcp.ArchiveTools(root_a, settings=settings, project_root=tmp_path)
    assert dynamic.root == root_a.resolve()
    settings.write_text(json.dumps({"archive_root": str(root_b)}), encoding="utf-8")
    assert dynamic.root == root_b.resolve()
    assert explicit.root == root_a.resolve()


def test_read_tools_delegate_to_archive_and_index_read_only(tmp_path, monkeypatch):
    fake_core(monkeypatch)
    tools = meeting_mcp.ArchiveTools(tmp_path)
    assert json.loads(tools.find_meetings("call"))["items"][0]["id"] == "one"
    assert json.loads(tools.search_evidence("hello"))["results"][0]["query"] == "hello"
    assert json.loads(tools.read_timeline("one"))["total"] == 1
    assert json.loads(tools.glossary("one"))["items"][0]["term"] == "API"
    assert tools.archive_status()["read_only"] is True


def test_archive_status_detects_default_and_override_model_files_without_loading_runtime(tmp_path, monkeypatch):
    fake_core(monkeypatch)
    monkeypatch.delenv("MEETING_EMBEDDING_MODEL", raising=False)
    tools = meeting_mcp.ArchiveTools(tmp_path / "archive", project_root=tmp_path)
    default_model = tmp_path / ".runtime" / "models" / "multilingual-e5-small" / "onnx"
    default_model.mkdir(parents=True)
    (default_model / "model.onnx").write_bytes(b"onnx")
    (default_model / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert tools.archive_status()["capabilities"]["local_embeddings"] == "configured_lazy"

    override = tmp_path / "installed-model"
    override.mkdir()
    (override / "model.onnx").write_bytes(b"onnx")
    monkeypatch.setenv("MEETING_EMBEDDING_MODEL", str(override))
    assert tools.archive_status()["capabilities"]["local_embeddings"] == "not_configured"
    (override / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert tools.archive_status()["capabilities"]["local_embeddings"] == "configured_lazy"

    missing = tmp_path / "missing-model"
    monkeypatch.setenv("MEETING_EMBEDDING_MODEL", str(missing))
    assert tools.archive_status()["capabilities"]["local_embeddings"] == "not_configured"


def test_search_adapter_uses_readonly_proxy_without_loading_numeric_media_modules(tmp_path, monkeypatch):
    fake_core(monkeypatch)
    tools = meeting_mcp.ArchiveTools(tmp_path)
    index = tools.index()
    assert isinstance(index, FakeIndex)
    assert index.root == tmp_path.resolve()

    script = (
        "import sys; from pathlib import Path; from services.meeting_mcp import ArchiveTools; "
        "tools=ArchiveTools(Path.cwd() / '.runtime' / 'mcp-import-check'); tools.archive_status(); tools.index(); "
        "print(','.join(name for name in ('numpy','cv2','onnxruntime','services.meeting_embeddings') if name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=20, check=True,
    )
    assert result.stdout.strip() == ""


def test_get_frames_returns_bounded_image_bytes_and_continuation(tmp_path, monkeypatch):
    fake_core(monkeypatch)
    frame_module = types.ModuleType("services.meeting_images")
    frame_module.frame_image = lambda *args, **kwargs: b"jpeg-bytes"
    frame_module.extract_episode_frames = lambda *args, **kwargs: []
    frame_module.probe_media = lambda path: {"duration": 2}
    monkeypatch.setitem(sys.modules, "services.meeting_images", frame_module)
    tools = meeting_mcp.ArchiveTools(tmp_path)
    info, images = tools.get_frames("one", limit=99)
    decoded = json.loads(info)
    assert len(images) == 1
    assert images[0] == b"jpeg-bytes"
    assert decoded["items"][0]["source"] == "stored"
    assert decoded["complete"] is True


def test_stored_frames_use_half_open_ranges_before_pagination(tmp_path, monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace
    from PIL import Image

    meeting_dir = tmp_path / "one"
    frames_dir = meeting_dir / "frames"
    frames_dir.mkdir(parents=True)
    colors = {"a.jpg": "red", "b.jpg": "blue", "c.jpg": "green", "d.jpg": "yellow"}
    for filename, color in colors.items():
        Image.new("RGB", (32, 24), color).save(frames_dir / filename, "JPEG", quality=100)
    events = [
        {"id": "A", "kind": "screen", "start": 0, "end": 1, "frame": "frames/a.jpg", "text": "A"},
        {"id": "B", "kind": "screen", "start": 1, "end": 2, "frame": "frames/b.jpg", "text": "B"},
        {"id": "C", "kind": "screen", "start": 2, "end": 3, "frame": "frames/c.jpg", "text": "C"},
        {"id": "D", "kind": "screen", "start": 3, "frame": "frames/d.jpg", "text": "D"},
    ]

    archive = SimpleNamespace(meeting_dir=lambda _meeting_id: meeting_dir,
                              read_screen_events=lambda _meeting_id: {"events": events})
    tools = meeting_mcp.ArchiveTools(tmp_path)
    monkeypatch.setattr(tools, "archive", lambda: archive)

    def pixel(image_bytes):
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB").getpixel((16, 12))

    boundary_info, boundary_images = tools.get_frames("one", start=1, end=2, limit=1)
    boundary = json.loads(boundary_info)
    assert [item["event_id"] for item in boundary["items"]] == ["B"]
    assert boundary["total"] == 1
    assert boundary["next_offset"] is None
    assert pixel(boundary_images[0])[2] > 200  # The returned bytes are B's blue JPEG.

    first_info, first_images = tools.get_frames("one", start=1, end=None, limit=1)
    second_info, second_images = tools.get_frames("one", start=1, end=None, limit=1, offset=1)
    third_info, third_images = tools.get_frames("one", start=1, end=None, limit=1, offset=2)
    assert json.loads(first_info)["total"] == json.loads(second_info)["total"] == 3
    assert [json.loads(info)["items"][0]["event_id"] for info in
            (first_info, second_info, third_info)] == ["B", "C", "D"]
    assert json.loads(first_info)["next_offset"] == 1
    assert json.loads(second_info)["next_offset"] == 2
    assert json.loads(third_info)["next_offset"] is None
    assert pixel(first_images[0])[2] > 200
    assert pixel(second_images[0])[1] > 100  # The next actual JPEG is C's green frame.

    point_info, point_images = tools.get_frames("one", start=3, end=4, limit=1)
    assert [item["event_id"] for item in json.loads(point_info)["items"]] == ["D"]
    assert pixel(point_images[0])[0] > 200
    point_at_right_edge, point_at_right_edge_images = tools.get_frames("one", start=2, end=3, limit=2)
    assert [item["event_id"] for item in json.loads(point_at_right_edge)["items"]] == ["C"]
    assert len(point_at_right_edge_images) == 1

    empty_info, empty_images = tools.get_frames("one", start=2, end=2, limit=1)
    assert json.loads(empty_info)["items"] == []
    assert json.loads(empty_info)["total"] == 0
    assert empty_images == []


def test_stored_frames_read_large_screen_event_file_once(tmp_path, monkeypatch):
    from io import BytesIO
    from PIL import Image
    from services import meeting_archive as archive_module
    from services.meeting_archive import MeetingArchive

    source = tmp_path / "many-frames.wav"
    source.write_bytes(b"archive")
    archive = MeetingArchive(tmp_path / "archive")
    meeting_id = archive.import_file(source)["meeting_id"]
    meeting_dir = archive.meeting_dir(meeting_id)
    frames_dir = meeting_dir / "frames"
    frames_dir.mkdir()
    Image.new("RGB", (32, 24), "red").save(frames_dir / "shared.jpg", "JPEG", quality=100)
    Image.new("RGB", (32, 24), "blue").save(frames_dir / "selected.jpg", "JPEG", quality=100)
    events = [{"id": f"screen-{i:03}", "start": i, "end": i + 1,
               "frame": "frames/selected.jpg" if i == 150 else "frames/shared.jpg",
               "text": f"screen {i}"} for i in range(250)]
    archive.write_screen_events(meeting_id, events)

    read_names = []
    original_read = archive_module._json_read

    def spy(path, default=None):
        read_names.append(Path(path).name)
        return original_read(path, default)

    monkeypatch.setattr(archive_module, "_json_read", spy)
    metadata, images = meeting_mcp.ArchiveTools(archive.root).get_frames(
        meeting_id, start=0, end=None, limit=1, offset=150)

    decoded = json.loads(metadata)
    assert decoded["total"] == 250
    assert decoded["items"][0]["event_id"] == "screen-150"
    assert decoded["items"][0]["start"] == 150
    assert len(images) == 1
    with Image.open(BytesIO(images[0])) as frame:
        red, green, blue = frame.convert("RGB").getpixel((16, 12))
    assert blue > 200 and red < 80 and green < 80
    assert read_names.count("screen-events.json") == 1
    assert not ({"transcript.json", "speakers.json"} & set(read_names))


def test_get_frames_rejects_invalid_crop(tmp_path):
    tools = meeting_mcp.ArchiveTools(tmp_path)
    with pytest.raises(ValueError, match="crop"):
        tools.get_frames("one", crop=[1, -1, 3, 4])


def test_original_episode_crop_uses_native_pixels_and_overview_stays_bounded(tmp_path, monkeypatch):
    from io import BytesIO
    from PIL import Image, ImageDraw
    from services import meeting_images

    ffmpeg = meeting_images._executable("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg is not installed")

    source = tmp_path / "wide-source.mp4"
    still = Image.new("RGB", (2880, 128), "black")
    ImageDraw.Draw(still).rectangle((2630, 20, 2810, 108), fill=(255, 240, 0))
    encoded = BytesIO()
    still.save(encoded, format="JPEG", quality=96)
    encoded.seek(0)
    created = subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-loop", "1", "-f", "image2pipe", "-framerate", "1",
         "-i", "pipe:0", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "1", str(source)],
        input=encoded.read(), capture_output=True, timeout=25, check=False,
    )
    if created.returncode:
        pytest.skip("FFmpeg cannot encode the tiny synthetic fixture: " + created.stderr.decode("utf-8", "replace"))

    class OriginalArchive(FakeArchive):
        def original_path(self, meeting_id):
            assert meeting_id == "one"
            return source

    archive_module = types.ModuleType("services.meeting_archive")
    archive_module.MeetingArchive = OriginalArchive
    monkeypatch.setitem(sys.modules, "services.meeting_archive", archive_module)
    tools = meeting_mcp.ArchiveTools(tmp_path / "archive")

    overview_text, overview_images = tools.get_frames("one", start=0, end=1, source="original", limit=1)
    overview = json.loads(overview_text)["items"][0]
    assert overview["source_width"] == 2880
    assert overview["source_height"] == 128
    assert overview["image_width"] <= 1920
    assert overview["coordinate_space"] == "native_source_pixels"
    assert Image.open(BytesIO(overview_images[0])).width <= 1920

    crop = [2600, 10, 160, 100]
    crop_text, crop_images = tools.get_frames("one", start=0, end=1, source="original", crop=crop, limit=1)
    cropped = json.loads(crop_text)["items"][0]
    with Image.open(BytesIO(crop_images[0])) as image:
        assert image.size == (160, 100)
        red, green, blue = image.convert("RGB").getpixel((70, 40))
    assert red > 200 and green > 180 and blue < 100
    assert cropped["crop"] == crop
    assert cropped["coordinate_space"] == "native_source_pixels"
    assert cropped["source_width"] == 2880
    assert cropped["source_height"] == 128
    assert cropped["image_width"] == 160
    assert cropped["image_height"] == 100


def test_stored_frame_crop_keeps_native_coordinates_before_resize(tmp_path, monkeypatch):
    from io import BytesIO
    from PIL import Image, ImageDraw

    fake_core(monkeypatch)
    meeting_dir = tmp_path / "one"
    frames_dir = meeting_dir / "frames"
    frames_dir.mkdir(parents=True)
    frame = Image.new("RGB", (2880, 128), "black")
    ImageDraw.Draw(frame).rectangle((2630, 20, 2810, 108), fill=(255, 240, 0))
    frame.save(frames_dir / "wide.jpg", format="JPEG", quality=96)

    class StoredArchive(FakeArchive):
        def read_screen_events(self, meeting_id):
            return {"events": [{"id": "screen-wide", "start": 0, "end": 1,
                                 "frame": "frames/wide.jpg", "text": "right marker", "regions": [], "tables": []}],
                    "status": "ready", "limitations": []}

    archive_module = types.ModuleType("services.meeting_archive")
    archive_module.MeetingArchive = StoredArchive
    monkeypatch.setitem(sys.modules, "services.meeting_archive", archive_module)
    tools = meeting_mcp.ArchiveTools(tmp_path)
    metadata, images = tools.get_frames("one", start=0, end=1, crop=[2600, 10, 160, 100], limit=1)
    assert json.loads(metadata)["items"][0]["source"] == "stored"
    with Image.open(BytesIO(images[0])) as cropped:
        assert cropped.size == (160, 100)
        red, green, blue = cropped.convert("RGB").getpixel((70, 40))
    assert red > 200 and green > 180 and blue < 100


def test_server_import_is_lazy_and_stdio_has_no_startup_archive_access(tmp_path, monkeypatch):
    fake_core(monkeypatch)
    # Constructing server must not instantiate either archive or index.
    try:
        server = meeting_mcp.create_server(root=tmp_path)
    except ImportError:
        pytest.skip("MCP SDK is installed by the application dependency setup")
    assert server.name == meeting_mcp.SERVER_NAME


def test_real_mcp_sdk_stdio_call_delivers_imagecontent_and_readonly_annotations(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    pytest.importorskip("anyio")
    import anyio
    from base64 import b64decode
    from io import BytesIO
    from PIL import Image, ImageDraw
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import ImageContent
    from services.meeting_archive import MeetingArchive
    from services.meeting_connector import MeetingConnector

    monkeypatch.delenv("VT_MEETING_ROOT", raising=False)

    archive_root = tmp_path / "archive"
    archive = MeetingArchive(archive_root)
    original = tmp_path / "original.mp4"
    original.write_bytes(b"test media placeholder")
    meeting = archive.import_file(original, title="SDK image test")
    meeting_id = meeting["meeting_id"]
    meeting_dir = archive.meeting_dir(meeting_id)
    frames = meeting_dir / "frames"
    frames.mkdir()
    image = Image.new("RGB", (96, 64), "red")
    ImageDraw.Draw(image).rectangle((30, 20, 70, 50), fill="lime")
    image.save(frames / "screen.jpg", "JPEG", quality=100)
    archive.write_screen_events(meeting_id, [{"id": "screen-test", "start": 0, "end": 10,
                                             "frame": "frames/screen.jpg", "text": "Only OCR label"}])
    settings = tmp_path / "meetings-settings.json"
    settings.write_text(json.dumps({"archive_root": str(archive_root)}), encoding="utf-8")
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(meeting_mcp.__file__).resolve()), "--settings", str(settings),
              "--project-root", str(Path(meeting_mcp.__file__).resolve().parents[1])],
    )

    async def exercise():
        with anyio.fail_after(30):
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    expected_names = {"get_archive_status", "find_meetings", "search_evidence", "read_timeline",
                                      "get_frames", "get_glossary", "get_source_status", "open_evidence"}
                    by_name = {item.name: item for item in listing.tools}
                    assert expected_names <= set(by_name)
                    for name in expected_names:
                        ann = by_name[name].annotations
                        assert ann.readOnlyHint is True
                        assert ann.destructiveHint is False
                        assert ann.idempotentHint is True
                        assert ann.openWorldHint is False
                        assert by_name[name].outputSchema is None
                    status_result = await session.call_tool("get_archive_status", {})
                    assert status_result.isError is False
                    result = await session.call_tool("get_frames", {"meeting_id": meeting_id, "limit": 8})
                    assert result.isError is False
                    images = [item for item in result.content if isinstance(item, ImageContent)]
                    assert len(images) == 1
                    decoded = b64decode(images[0].data, validate=True)
                    assert decoded.startswith(b"\xff\xd8") and decoded.endswith(b"\xff\xd9")
                    delivered = Image.open(BytesIO(decoded)).convert("RGB")
                    red = delivered.getpixel((5, 5))
                    green = delivered.getpixel((45, 35))
                    assert red[0] > 180 and red[1] < 100
                    assert green[1] > 150 and green[0] < 100

    anyio.run(exercise)
    verifier = MeetingConnector(project_root=Path(meeting_mcp.__file__).resolve().parents[1],
                                settings=settings, python=sys.executable)
    monkeypatch.setattr(verifier, "_get", lambda: (False, None, "Codex registration is outside this SDK-only check"))
    checked = verifier.verify_protocol(timeout=15, meeting_id=meeting_id)
    assert checked["protocol_checked"] is True, repr(checked)
    assert checked["client_verified"] is False
    assert checked["actual_image_content_count"] == 1
    assert checked["jpeg_payloads_valid"] is True


def test_open_evidence_viewer_serves_escaped_html_media_ranges_and_missing_source(tmp_path, monkeypatch):
    from services import meeting_links

    source = tmp_path / "тест.mp4"
    source.write_bytes(b"0123456789")
    state = {"source": source}
    monkeypatch.setattr(meeting_links, "_get_evidence", lambda root, meeting_id, at: {
        "meeting": {"id": meeting_id, "title": '<Meeting "&">'},
        "source": state["source"], "frame": None, "at": at,
    })
    viewer = meeting_links.Viewer(tmp_path)
    try:
        url = viewer.evidence_url("meeting-ю", 14.5)
        with urllib.request.urlopen(url) as response:
            page = response.read().decode("utf-8")
        assert "&lt;Meeting &quot;&amp;&quot;&gt;" in page
        assert "currentTime=14.5" in page
        media_url = viewer.base_url + "source/meeting-%D1%8E"
        request = urllib.request.Request(media_url, headers={"Range": "bytes=2-5"})
        with urllib.request.urlopen(request) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 2-5/10"
            assert response.read() == b"2345"
        source.unlink()
        state["source"] = None
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(media_url)
        assert error.value.code == 404
    finally:
        viewer.close()


def test_durable_uri_requires_current_handler_archive_binding(tmp_path, monkeypatch):
    from services import meeting_links

    settings = tmp_path / "settings.json"
    archive_a = tmp_path / "archive-a"
    archive_b = tmp_path / "archive-b"
    settings.write_text(json.dumps({"archive_root": str(archive_a)}), encoding="utf-8")
    binding = {"settings_path": str(settings), "project_root": str(tmp_path)}
    monkeypatch.setattr(meeting_links, "evidence_protocol_binding", lambda: binding)

    monkeypatch.setenv("VT_MEETING_ROOT", str(archive_b))
    assert meeting_links.evidence_protocol_matches_archive(archive_a)
    assert not meeting_links.evidence_protocol_matches_archive(archive_b)
    settings.write_text(json.dumps({"archive_root": str(archive_b)}), encoding="utf-8")
    assert not meeting_links.evidence_protocol_matches_archive(archive_a)
    assert meeting_links.evidence_protocol_matches_archive(archive_b)

    monkeypatch.setattr(meeting_links, "evidence_protocol_binding", lambda: None)
    assert not meeting_links.evidence_protocol_matches_archive(archive_b)
    if os.name != "nt":
        assert not meeting_links.evidence_protocol_matches_archive(archive_b)


def test_open_evidence_only_promises_uri_for_matching_archive(tmp_path, monkeypatch):
    from services import meeting_links

    archive = tmp_path / "copy-of-archive"
    tools = meeting_mcp.ArchiveTools(archive, project_root=tmp_path)
    monkeypatch.setattr(meeting_links, "evidence_protocol_status", lambda: {
        "supported": True, "registered": True, "owned": True, "conflict": False,
    })
    monkeypatch.setattr(meeting_links, "open_evidence_url", lambda *args, **kwargs: "http://127.0.0.1/temp")
    monkeypatch.setattr(meeting_links, "viewer_launch_mode", lambda *args, **kwargs: "detached")
    monkeypatch.setattr(meeting_links, "evidence_protocol_matches_archive", lambda root: Path(root) == archive)

    same = json.loads(tools.open_evidence("meeting-1"))
    assert same["uri_handler_registered"] is True
    assert same["uri_handler_binding_matches_root"] is True

    monkeypatch.setattr(meeting_links, "evidence_protocol_matches_archive", lambda _root: False)
    different = json.loads(tools.open_evidence("meeting-1"))
    assert different["uri_handler_owned"] is True
    assert different["uri_handler_registered"] is False
    assert different["uri_handler_binding_matches_root"] is False
    assert "different archive" in different["http_url_lifetime"]


def test_viewer_rejects_non_loopback_host(tmp_path):
    from services.meeting_links import Viewer

    viewer = Viewer(tmp_path)
    try:
        request = urllib.request.Request(viewer.base_url, headers={"Host": "example.com"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
    finally:
        viewer.close()


def test_detached_viewer_persists_and_uses_retained_audio_frame(tmp_path, monkeypatch):
    from PIL import Image
    from services.meeting_archive import MeetingArchive
    from services import meeting_links

    settings = tmp_path / "settings" / "meetings-settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({"archive_root": str(tmp_path / "archive")}), encoding="utf-8")
    archive = MeetingArchive(tmp_path / "archive")
    source = tmp_path / "original.mp4"
    source.write_bytes(b"test-original")
    meeting = archive.import_file(source, title="Retained evidence")
    meeting_id = meeting["meeting_id"]
    meeting_dir = archive.meeting_dir(meeting_id)
    (meeting_dir / "frames").mkdir()
    frame_path = meeting_dir / "frames" / "one.jpg"
    next_frame_path = meeting_dir / "frames" / "two.jpg"
    Image.new("RGB", (64, 48), "red").save(frame_path, "JPEG")
    Image.new("RGB", (64, 48), "blue").save(next_frame_path, "JPEG")
    archive.write_screen_events(meeting_id, [
        {"id": "screen-1", "start": 0, "end": 13, "frame": "frames/one.jpg", "text": "earlier visible state"},
        {"id": "screen-2", "start": 13, "end": 18, "frame": "frames/two.jpg", "text": "later visible state"},
    ])
    audio = meeting_dir / "audio"
    audio.mkdir()
    (audio / "audio-01.opus").write_bytes(b"retained-audio")
    original = archive.original_path(meeting_id)
    original.unlink()

    monkeypatch.setattr(meeting_links.webbrowser, "open", lambda _url: True)
    uri = meeting_links.evidence_uri(meeting_id, 10.25)
    url = meeting_links.open_evidence_uri(uri, settings=settings,
                                          project_root=Path(meeting_mcp.__file__).resolve().parents[1])
    descriptor = settings.parent / "read-viewer.json"
    state = json.loads(descriptor.read_text(encoding="utf-8"))
    try:
        assert state["root"] == str(archive.root.resolve())
        with urllib.request.urlopen(url, timeout=3) as response:
            page = response.read().decode("utf-8")
        assert "<audio" in page
        assert "Saved screen evidence" in page
        audio_url = url.split("/evidence/", 1)[0] + f"/audio/{meeting_id}"
        with urllib.request.urlopen(audio_url, timeout=3) as response:
            assert response.read() == b"retained-audio"
        frame_url = url.split("/evidence/", 1)[0] + f"/frame/{meeting_id}?t=10.25"
        with urllib.request.urlopen(frame_url, timeout=3) as response:
            assert response.headers["Content-Type"] == "image/jpeg"
            from io import BytesIO
            opened = Image.open(BytesIO(response.read())).convert("RGB")
            assert opened.getpixel((3, 3))[0] > 200  # active [0, 13] interval, not future [13, 18]

        # A dead detached process restarts on the same URL so previously issued links survive.
        os.kill(int(state["pid"]), signal.SIGTERM)
        health_url = url.split("/evidence/", 1)[0] + "/health"
        deadline = __import__("time").monotonic() + 3
        while __import__("time").monotonic() < deadline:
            try:
                urllib.request.urlopen(health_url, timeout=.2)
                __import__("time").sleep(.05)
            except OSError:
                break
        restored_url = meeting_links.open_evidence_url(archive.root, meeting_id, 10.25, settings=settings)
        restored = json.loads(descriptor.read_text(encoding="utf-8"))
        assert restored_url.split("/evidence/", 1)[0] == url.split("/evidence/", 1)[0]
        with urllib.request.urlopen(url, timeout=3) as response:
            assert response.status == 200
        state = restored
    finally:
        try:
            os.kill(int(state["pid"]), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def test_viewer_survives_the_process_that_issued_its_url(tmp_path):
    script = (
        "from services.meeting_links import open_evidence_url; import sys; "
        "print(open_evidence_url(sys.argv[1], 'parent-exit-test', 0, settings=sys.argv[2]))"
    )
    root = tmp_path / "archive"
    root.mkdir()
    settings = tmp_path / "settings" / "meetings-settings.json"
    child = subprocess.run(
        [sys.executable, "-c", script, str(root), str(settings)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=15, check=False,
    )
    assert child.returncode == 0, (
        f"viewer issuer exited with {child.returncode}; "
        f"stdout={child.stdout!r}; stderr={child.stderr!r}"
    )
    url = child.stdout.strip()
    parsed = urllib.request.urlsplit(url)
    token = parsed.path.strip("/").split("/", 1)[0]
    health_url = f"{parsed.scheme}://{parsed.netloc}/{token}/health"
    descriptor = settings.parent / "read-viewer.json"
    state = json.loads(descriptor.read_text(encoding="utf-8"))
    try:
        with urllib.request.urlopen(health_url, timeout=4) as response:
            assert response.status == 200
            assert response.read() == b"ok"
    finally:
        try:
            os.kill(int(state["pid"]), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def test_viewer_startup_failure_has_private_diagnostic_without_token(tmp_path):
    from services.meeting_links import main

    diagnostic = tmp_path / "private" / "viewer-startup.log"
    token = "private-token-that-must-not-be-logged"
    result = main(["--serve", "--root", str(tmp_path), "--port", "70000", "--token", token,
                   "--diagnostic-log", str(diagnostic)])
    assert result == 1
    contents = diagnostic.read_text(encoding="utf-8")
    assert "OverflowError" in contents
    assert token not in contents


def test_registered_scheme_handler_subprocess_restarts_the_viewer(tmp_path):
    from services.meeting_links import evidence_uri

    root = tmp_path / "archive"
    root.mkdir()
    settings = tmp_path / "settings" / "meetings-settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"archive_root": str(root)}), encoding="utf-8")
    command = [sys.executable, str(Path(meeting_mcp.__file__).resolve().parent / "meeting_links.py"),
               "--open-uri", evidence_uri("handler-test", 5), "--settings", str(settings),
               "--project-root", str(Path(meeting_mcp.__file__).resolve().parents[1]), "--no-browser"]
    child = subprocess.run(command, cwd=Path(meeting_mcp.__file__).resolve().parents[1],
                           capture_output=True, text=True, timeout=15, check=False)
    assert child.returncode == 0, (
        f"URI handler exited with {child.returncode}; "
        f"stdout={child.stdout!r}; stderr={child.stderr!r}"
    )
    state = json.loads((settings.parent / "read-viewer.json").read_text(encoding="utf-8"))
    try:
        health = f"http://127.0.0.1:{state['port']}/{state['token']}/health"
        with urllib.request.urlopen(health, timeout=3) as response:
            assert response.status == 200
            assert response.read() == b"ok"
    finally:
        try:
            os.kill(int(state["pid"]), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def test_viewer_accepts_token_starting_with_dash_in_detached_subprocess(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    settings = tmp_path / "settings" / "meetings-settings.json"
    settings.parent.mkdir(parents=True)
    script = (
        "from services import meeting_links; import sys; "
        "meeting_links.secrets.token_urlsafe=lambda _size: '-test-token-leading-dash'; "
        "print(meeting_links.open_evidence_url(sys.argv[1], 'leading-dash-token', 0, settings=sys.argv[2]))"
    )
    child = subprocess.run(
        [sys.executable, "-c", script, str(root), str(settings)],
        cwd=Path(meeting_mcp.__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert child.returncode == 0, (
        f"viewer issuer exited with {child.returncode}; "
        f"stdout={child.stdout!r}; stderr={child.stderr!r}"
    )
    url = child.stdout.strip()
    parsed = urllib.request.urlsplit(url)
    token = parsed.path.strip("/").split("/", 1)[0]
    assert token == "-test-token-leading-dash"
    health_url = f"{parsed.scheme}://{parsed.netloc}/{token}/health"
    state = json.loads((settings.parent / "read-viewer.json").read_text(encoding="utf-8"))
    try:
        assert state["token"] == token
        with urllib.request.urlopen(health_url, timeout=3) as response:
            assert response.status == 200
            assert response.read() == b"ok"

        os.kill(int(state["pid"]), signal.SIGTERM)
        deadline = __import__("time").monotonic() + 3
        while __import__("time").monotonic() < deadline:
            try:
                urllib.request.urlopen(health_url, timeout=.2)
                __import__("time").sleep(.05)
            except OSError:
                break

        from services import meeting_links
        reopened = meeting_links._start_detached_viewer(root, settings)
        state = json.loads((settings.parent / "read-viewer.json").read_text(encoding="utf-8"))
        assert reopened.token == token
        with urllib.request.urlopen(reopened.base_url + "health", timeout=3) as response:
            assert response.status == 200
            assert response.read() == b"ok"
    finally:
        state = json.loads((settings.parent / "read-viewer.json").read_text(encoding="utf-8"))
        try:
            os.kill(int(state["pid"]), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def test_durable_evidence_uri_validates_and_reopens_only_the_viewer(tmp_path, monkeypatch):
    from services import meeting_links

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"archive_root": str(tmp_path / "archive")}), encoding="utf-8")
    opened = []
    monkeypatch.setattr(meeting_links, "open_evidence_url", lambda root, meeting_id, at, **kwargs: f"http://127.0.0.1/view/{meeting_id}?t={at}")
    monkeypatch.setattr(meeting_links.webbrowser, "open", lambda url: opened.append(url))

    uri = meeting_links.evidence_uri("meeting-1", 12.5)
    assert uri == "video-transcribe://evidence/meeting-1?t=12.5"
    url = meeting_links.open_evidence_uri(uri, settings=settings, project_root=tmp_path)
    assert url == "http://127.0.0.1/view/meeting-1?t=12.5"
    assert opened == [url]
    with pytest.raises(ValueError, match="Invalid meeting id"):
        meeting_links.open_evidence_uri("video-transcribe://evidence/%2e%2e%2fsecret?t=0", settings=settings)
    with pytest.raises(ValueError, match="Invalid evidence time"):
        meeting_links.open_evidence_uri("video-transcribe://evidence/meeting-1?t=NaN", settings=settings)
