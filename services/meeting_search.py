"""Lightweight stdlib-only proxy for isolated read-only meeting search."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class ReadIndexProxy:
    """Call the heavy search implementation in a bounded Python subprocess."""

    def __init__(self, root: str | Path, *, timeout: float = 120.0,
                 python_executable: str | None = None):
        self.root = Path(root).expanduser().resolve()
        self.timeout = max(1.0, float(timeout))
        self.python_executable = python_executable or sys.executable

    def search(self, query: str, limit: int = 20, offset: int = 0,
               meeting_id: str | None = None, mode: str = "hybrid") -> dict[str, Any]:
        request = {"root": str(self.root), "query": query, "limit": limit,
                   "offset": offset, "meeting_id": meeting_id, "mode": mode}
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        command = [self.python_executable, "-m", "services.meeting_index", "--search-stdio"]
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(request, ensure_ascii=False),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {"results": [], "total": 0, "next_offset": None, "status": "worker_timeout",
                    "error": {"type": "TimeoutExpired", "message": str(exc), "timeout_seconds": self.timeout}}
        except OSError as exc:
            return {"results": [], "total": 0, "next_offset": None, "status": "worker_error",
                    "error": {"type": type(exc).__name__, "message": str(exc)}}
        try:
            result = json.loads(completed.stdout)
        except (ValueError, json.JSONDecodeError):
            return {"results": [], "total": 0, "next_offset": None, "status": "worker_error",
                    "error": {"type": "InvalidWorkerResponse", "returncode": completed.returncode,
                              "stderr": completed.stderr[-4000:], "stdout": completed.stdout[-1000:]}}
        if not isinstance(result, dict):
            return {"results": [], "total": 0, "next_offset": None, "status": "worker_error",
                    "error": {"type": "InvalidWorkerResponse", "message": "worker response must be an object"}}
        if completed.returncode and "error" not in result:
            result["error"] = {"type": "WorkerExit", "returncode": completed.returncode,
                                "stderr": completed.stderr[-4000:]}
            result["status"] = "worker_error"
        return result

