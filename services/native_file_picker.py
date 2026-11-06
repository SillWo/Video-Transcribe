"""Native Windows file picker used by the simple UI."""

from __future__ import annotations

import os
import json
import subprocess
import sys
import threading
from pathlib import Path

_dialog_lock = threading.Lock()


def _run_tk_dialog(multiple: bool = False) -> str | list[str]:
    """Show an owned native file dialog and return its selected path or blank on cancel."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError(
            "В Python не установлен Tkinter. Установите Python для Windows с компонентом Tcl/Tk и повторите попытку."
        ) from exc

    owner = None
    selected = ""
    try:
        owner = tk.Tk()
        owner.title("Video Transcribe — выбор файла")
        owner.resizable(False, False)
        owner.attributes("-topmost", True)
        width, height = 420, 120
        screen_width, screen_height = owner.winfo_screenwidth(), owner.winfo_screenheight()
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        owner.geometry(f"{width}x{height}+{x}+{y}")
        tk.Label(
            owner,
            text="Выберите видео или аудиофайл в открывшемся окне.",
            wraplength=380,
            justify="center",
        ).pack(expand=True, fill="both", padx=16, pady=16)
        owner.update()
        selected = filedialog.askopenfilenames if multiple else filedialog.askopenfilename
        selected = selected(
            parent=owner,
            title="Выберите файл для расшифровки",
            filetypes=[
                ("Медиафайлы", "*.mp4 *.mkv *.mov *.avi *.webm *.mp3 *.wav *.m4a *.flac *.ogg *.wma *.aac"),
                ("Все файлы", "*.*"),
            ],
        )
    finally:
        if owner is not None:
            owner.destroy()
    return list(selected) if multiple else (selected or "")


def _show_windows_dialog(multiple: bool = False) -> str | list[str] | None:
    if os.name != "nt":
        raise RuntimeError("Выбор файла доступен только в Windows.")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "services.native_file_picker",
             "--dialog-multiple" if multiple else "--dialog"],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Время ожидания выбора файла истекло. Закройте окно выбора файла и попробуйте снова.") from exc
    if result.returncode:
        message = result.stderr.strip()
        raise RuntimeError(message or "Не удалось открыть системный выбор файла.")
    output = result.stdout.strip()
    if not output:
        return [] if multiple else None
    if not multiple:
        return output
    try:
        selected = json.loads(output)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("Не удалось получить список выбранных файлов.") from exc
    if not isinstance(selected, list) or any(not isinstance(path, str) for path in selected):
        raise RuntimeError("Не удалось получить список выбранных файлов.")
    return selected


def pick_file() -> str | None:
    if not _dialog_lock.acquire(blocking=False):
        raise RuntimeError("Окно выбора файла уже открыто.")
    try:
        return _show_windows_dialog()
    finally:
        _dialog_lock.release()


def pick_files() -> list[str]:
    if not _dialog_lock.acquire(blocking=False):
        raise RuntimeError("Окно выбора файла уже открыто.")
    try:
        return list(_show_windows_dialog(multiple=True) or [])
    finally:
        _dialog_lock.release()


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"--dialog", "--dialog-multiple"}:
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        multiple = sys.argv[1] == "--dialog-multiple"
        selected = _run_tk_dialog(multiple=multiple)
        sys.stdout.write(json.dumps(selected, ensure_ascii=False) if multiple else selected)
        sys.stdout.flush()
        return 0
    except Exception as exc:
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
        sys.stderr.write(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
