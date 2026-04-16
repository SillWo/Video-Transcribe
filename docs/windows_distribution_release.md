# Windows distribution release

## Branch model

- `develop` is the integration branch for product changes.
- `main` is the release-source branch.
- `windows_dist` stores the packaging layer only:
  - `packaging/windows/`
  - `scripts/windows/`
  - reusable workflow
  - release documentation

Tag-based automation becomes active only after the thin caller workflow file is delivered into `main`.

## Release guarantee

The Windows installer is built from two independent trees:

- `source/` is checked out from the exact tagged source in `main`.
- `packaging/` is checked out from `windows_dist`.

The packaging layer does not patch or rewrite files in `source/` during the build. Product code changes must reach `develop` and then `main` before they can be part of a tagged release.

## Activation requirements

The following must exist before tag-based release automation can work end to end:

- release tag on `main`
- packaging branch `windows_dist`
- reusable workflow in `windows_dist`
- thin caller workflow in `main`
- `contents: write` workflow permissions
- working `GITHUB_TOKEN` for release publishing

The file `.github/workflows/windows-release-main-caller.yml` is the minimal caller workflow that must be merged or cherry-picked into `main`.

## Runtime layout

Installer target:

- `C:\Program Files\Video Transcribe`

Per-user runtime data:

- `%LOCALAPPDATA%\VideoTranscribe\logs`
- `%LOCALAPPDATA%\VideoTranscribe\data`
- `%LOCALAPPDATA%\VideoTranscribe\huggingface`

Bundled binaries inside the install directory:

- `bin\yt-dlp.exe`
- `bin\ffmpeg.exe`
- `bin\ffprobe.exe`

The desktop launcher prepends the bundled `bin` directory to the process PATH only for the packaged process tree. It does not require global PATH changes.

## Local build

Pre-checks:

```powershell
git status --short --branch
git rev-parse --verify windows_dist
python --version
node --version
```

Build a release from the current local repository refs:

```powershell
$iscc = powershell -ExecutionPolicy Bypass -File .\scripts\windows\install-inno-setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\windows\build-release.ps1 `
  -ReleaseTag v1.1.0 `
  -SourceRef refs/heads/main `
  -PackagingRef windows_dist `
  -RepositoryRoot . `
  -IsccPath $iscc
```

The script creates two working trees under `build/windows/`:

- `build/windows/source`
- `build/windows/packaging`

Final local output:

- `build/windows/installer/VideoTranscribe-Setup-<version>.exe`
- `build/windows/installer/VideoTranscribe-Setup-<version>.exe.sha256`

## Runtime binaries

Pinned runtime binaries are declared in `packaging/windows/runtime-binaries.json`.

The fetch step:

1. downloads `yt-dlp.exe`
2. downloads the pinned ffmpeg archive
3. verifies SHA256
4. extracts `ffmpeg.exe` and `ffprobe.exe`
5. copies verified binaries into the staging payload

No runtime binaries are committed to git.

## Reusable workflow in `windows_dist`

Workflow file:

- `.github/workflows/windows-release-reusable.yml`

Inputs:

- `release_tag`
- `source_ref`
- `source_sha`
- `packaging_ref`

Main responsibilities:

1. checkout exact tagged source into `source/`
2. checkout packaging branch into `packaging/`
3. verify the source commit belongs to `main`
4. install Python and Node.js toolchains
5. install Inno Setup explicitly
6. build frontend
7. fetch and verify runtime binaries
8. build PyInstaller one-folder app
9. build Inno Setup installer
10. upload workflow artifact
11. publish installer as a GitHub Release asset

## Thin caller workflow in `main`

Caller file:

- `.github/workflows/windows-release-main-caller.yml`

Responsibilities:

- trigger on `push` for `v*` tags
- optionally allow `workflow_dispatch`
- forward `release_tag`, `source_ref`, and `source_sha`
- keep packaging logic out of `main`

This caller workflow must be merged into `main` separately. That is the activation step for tag-based release orchestration.

## Publishing flow

Automatic path:

1. product changes land in `develop`
2. product release source lands in `main`
3. packaging branch `windows_dist` is updated as needed
4. thin caller workflow exists in `main`
5. a release tag is pushed on `main`
6. caller workflow invokes the reusable workflow in `windows_dist`
7. installer is published as both artifact and release asset

Manual path:

```powershell
gh workflow run windows-release-main-caller.yml `
  -f release_tag=v1.1.0 `
  -f source_ref=refs/tags/v1.1.0
```

## Verification notes

Minimum checks to run after implementation:

- `npm run build` in `ui`
- backend unit tests in the project venv
- reusable workflow YAML validation
- local script dry-run review

Full verification still requires:

- a Windows environment with PyInstaller installed through the build script
- Inno Setup installation
- a real tagged release in `main`
- a real GitHub Actions run with release publishing permissions
