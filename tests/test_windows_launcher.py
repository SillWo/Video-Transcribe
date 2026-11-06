import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "start.ps1"
UTF8_BOM = b"\xef\xbb\xbf"


def _windows_powershell_51() -> str | None:
    configured = os.environ.get("VT_WINDOWS_POWERSHELL_51")
    if configured and Path(configured).is_file():
        return configured
    if os.name != "nt":
        return None
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    candidate = system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    return str(candidate) if candidate.is_file() else None


def _run_parser(executable: str, script: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VT_START_SCRIPT_PATH"] = str(script)
    command = (
        "$tokens=$null; $errors=$null; "
        "[Management.Automation.Language.Parser]::ParseFile("
        "$env:VT_START_SCRIPT_PATH,[ref]$tokens,[ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { "
        "$errors | ForEach-Object { [Console]::Error.WriteLine($_.Message) }; exit 1 }"
    )
    return subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )


class WindowsLauncherEncodingTests(unittest.TestCase):
    def test_start_ps1_is_utf8_with_bom_and_strictly_decodable(self):
        raw = SCRIPT.read_bytes()
        self.assertTrue(raw.startswith(UTF8_BOM), "start.ps1 must include a UTF-8 BOM for Windows PowerShell 5.1")
        text = raw.decode("utf-8-sig", errors="strict")
        self.assertTrue(text.startswith("[CmdletBinding()]"))

    def test_windows_powershell_51_parses_start_script(self):
        executable = _windows_powershell_51()
        if executable is None:
            self.skipTest("Windows PowerShell 5.1 is unavailable on this platform")
        result = _run_parser(executable, SCRIPT)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_pwsh_parser_accepts_start_script_when_available(self):
        executable = os.environ.get("VT_PWSH") or shutil.which("pwsh")
        if executable is None:
            self.skipTest("PowerShell 7 is unavailable")
        result = _run_parser(executable, SCRIPT)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
