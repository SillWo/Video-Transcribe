import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _synthetic_video(path: Path, seconds: int = 60) -> bool:
    ffmpeg = shutil_which("ffmpeg")
    if not ffmpeg:
        return False
    result = subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                             f"color=c=blue:s=160x120:r=2:d={seconds}", "-an",
                             "-c:v", "mpeg4", "-y", str(path)],
                            capture_output=True, timeout=60, check=False)
    return result.returncode == 0 and path.is_file()


def shutil_which(name: str) -> str | None:
    import shutil
    return shutil.which(name) or shutil.which(name + ".exe")


@pytest.mark.skipif(shutil_which("ffmpeg") is None or shutil_which("ffprobe") is None,
                    reason="FFmpeg is required for the isolated worker smoke")
def test_fresh_fastapi_host_runs_screen_worker_and_cancels_then_retries(tmp_path):
    video = tmp_path / "synthetic-screen.mp4"
    assert _synthetic_video(video)
    stubs = tmp_path / "worker-guards"
    stubs.mkdir()
    (stubs / "sitecustomize.py").write_text(textwrap.dedent("""
        import sys
        class BlockRapidOCR:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'rapidocr' or fullname.startswith('rapidocr.'):
                    raise RuntimeError('production frame extraction must not import RapidOCR')
        sys.meta_path.insert(0, BlockRapidOCR())
    """), encoding="utf-8")
    script = tmp_path / "fresh_host_smoke.py"
    script.write_text(textwrap.dedent("""
        import os, sys, time
        from pathlib import Path
        import web_api
        assert 'cv2' not in sys.modules, 'FastAPI host imported OpenCV during startup'
        assert 'numpy' not in sys.modules, 'FastAPI host imported NumPy during startup'
        from services.meeting_archive import MeetingArchive
        video = Path(sys.argv[1])
        archive = MeetingArchive(os.environ['VT_MEETING_ROOT'])
        mid = archive.import_file(video, title='Legacy screen-only rerun')['meeting_id']
        from fastapi.testclient import TestClient
        client = TestClient(web_api.app)
        first = client.post('/api/meetings/process', json={
            'meeting_ids':[mid], 'stages':['screen'], 'profile':'balanced',
            'device':'cpu', 'cpu_threads':2, 'diarization':False})
        assert first.status_code == 202, first.text
        first_id = first.json()['jobs'][0]['id']
        observed_screen_progress = False
        deadline = time.time() + 45
        first_status = None
        while time.time() < deadline:
            record = client.get(f'/api/simple/jobs/{first_id}').json()
            progress = record.get('progress') or {}
            if progress.get('stage') == 'screen':
                observed_screen_progress = True
                if record.get('status') in {'queued','running'}:
                    cancelled = client.post(f'/api/simple/jobs/{first_id}/cancel')
                    assert cancelled.status_code in {200,202}, cancelled.text
                    break
            if record.get('status') in {'completed','failed','cancelled','interrupted'}:
                first_status = record.get('status')
                break
            time.sleep(.05)
        else:
            raise AssertionError('screen progress did not arrive')
        deadline = time.time() + 20
        while time.time() < deadline:
            first_status = client.get(f'/api/simple/jobs/{first_id}').json().get('status')
            if first_status in {'completed','failed','cancelled','interrupted'}:
                break
            time.sleep(.05)
        assert observed_screen_progress
        assert first_status == 'cancelled', first_status
        deadline = time.time() + 5
        while first_id in web_api.job_process_controllers and time.time() < deadline:
            time.sleep(.05)
        assert first_id not in web_api.job_process_controllers, 'cancelled worker controller was not released'
        archive = MeetingArchive(os.environ['VT_MEETING_ROOT'], read_only=True)
        meta = archive.get_meeting(mid)
        assert meta['stage_state']['screen']['status'] == 'cancelled'

        retry = client.post('/api/meetings/process', json={
            'meeting_ids':[mid], 'stages':['screen'], 'profile':'balanced',
            'device':'cpu', 'cpu_threads':2, 'diarization':False})
        assert retry.status_code == 202, retry.text
        retry_id = retry.json()['jobs'][0]['id']
        deadline = time.time() + 90
        while time.time() < deadline:
            record = client.get(f'/api/simple/jobs/{retry_id}').json()
            if record.get('status') in {'completed','failed','cancelled','interrupted'}:
                break
            time.sleep(.1)
        assert record.get('status') == 'completed', record
        assert 'cv2' not in sys.modules, 'OpenCV leaked into the FastAPI host process'
        assert 'numpy' not in sys.modules, 'NumPy leaked into the FastAPI host process'
        timeline = archive.timeline(mid)
        assert any(item.get('kind') == 'screen' for item in timeline['items'])
        final = archive.get_meeting(mid)
        assert final['stage_state']['screen']['status'] in {'ready','partial'}
    """), encoding="utf-8")

    env = os.environ.copy()
    env["VT_SIMPLE_RUNTIME_DIR"] = str(tmp_path / "runtime")
    env["VT_MEETING_ROOT"] = str(tmp_path / "archive")
    env["VT_MEETING_SETTINGS"] = str(tmp_path / "settings.json")
    env["PYTHONPATH"] = os.pathsep.join([str(stubs), str(REPO), env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, str(script), str(video)], cwd=REPO,
                            env=env, capture_output=True, text=True, timeout=150, check=False)
    assert result.returncode == 0, f"fresh-host smoke failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
