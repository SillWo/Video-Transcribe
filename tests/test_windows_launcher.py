import os
import shutil
import subprocess
import tempfile
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


def _run_python_selection(executable: str, script: Path, root: Path, launcher: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VT_START_SCRIPT_PATH"] = str(script)
    env["VT_PYTHON_ROOT"] = str(root)
    env["VT_FAKE_PY_LAUNCHER"] = str(launcher)
    command = r'''
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($env:VT_START_SCRIPT_PATH, [ref]$tokens, [ref]$errors)
if ($errors.Count -gt 0) { throw 'start.ps1 did not parse' }
$testDefinition = $ast.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-PythonCommand' }, $true)
$definition = $ast.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Select-Python' }, $true)
if (-not $definition -or -not $testDefinition) { throw 'Python selection functions were not found' }
. ([ScriptBlock]::Create($testDefinition.Extent.Text))
. ([ScriptBlock]::Create($definition.Extent.Text))
$Root = $env:VT_PYTHON_ROOT
$CheckOnly = $true
function Get-PythonLauncher { return $env:VT_FAKE_PY_LAUNCHER }
function Find-CommandPath { return $null }
$selection = Select-Python
Write-Output (@($selection) -join '|')
'''
    return subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=root,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )


def _run_postprocess_check(executable: str, script: Path, runtime: Path, python_exe: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VT_START_SCRIPT_PATH"] = str(script)
    env["VT_TEST_RUNTIME"] = str(runtime)
    env["VT_TEST_PYTHON"] = str(python_exe)
    command = r'''
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($env:VT_START_SCRIPT_PATH, [ref]$tokens, [ref]$errors)
if ($errors.Count -gt 0) { throw 'start.ps1 did not parse' }
foreach ($name in @('Test-PythonCommand', 'Test-PostprocessPackages')) {
    $definition = $ast.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true)
    if (-not $definition) { throw "$name was not found" }
    . ([ScriptBlock]::Create($definition.Extent.Text))
}
$Runtime = $env:VT_TEST_RUNTIME
$result = Test-PostprocessPackages -PythonExe $env:VT_TEST_PYTHON
if ($result) { throw 'Missing postprocess packages were reported as installed' }
if (Test-Path (Join-Path $Runtime 'postprocess-check.py')) { throw 'Temporary check script was not removed' }
Write-Output 'False'
'''
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

    def test_missing_python_312_falls_back_to_py_three_with_stderr_under_powershell_51(self):
        executable = _windows_powershell_51()
        if executable is None:
            self.skipTest("Windows PowerShell 5.1 is unavailable")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "clean-project"
            root.mkdir()
            launcher = Path(temporary_directory) / "py.cmd"
            launcher.write_text(
                '@echo off\r\n'
                'if "%~1"=="-3.12" (echo No suitable Python runtime found 1>&2 & exit /b 1)\r\n'
                'if "%~1"=="-3" (echo Python 3.13.6 selected 1>&2 & exit /b 0)\r\n'
                'exit /b 9\r\n',
                encoding="utf-8",
            )
            result = _run_python_selection(executable, SCRIPT, root, launcher)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertEqual(result.stdout.strip(), f"{launcher}|-3")

    def test_missing_postprocess_packages_return_false_before_install_under_powershell_51(self):
        executable = _windows_powershell_51()
        if executable is None:
            self.skipTest("Windows PowerShell 5.1 is unavailable")
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory) / "runtime"
            runtime.mkdir()
            python_exe = Path(temporary_directory) / "missing-packages.cmd"
            python_exe.write_text(
                '@echo off\r\n'
                'echo Traceback (most recent call last): 1>&2\r\n'
                'echo ModuleNotFoundError: No module named razdel 1>&2\r\n'
                'exit /b 1\r\n',
                encoding="utf-8",
            )
            result = _run_postprocess_check(executable, SCRIPT, runtime, python_exe)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
