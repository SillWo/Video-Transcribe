# Video Transcribe

Локальный сервис для расшифровки аудио и видео с использованием `faster-whisper`, `FastAPI`, `yt-dlp`, `ffmpeg` и веб-интерфейса на React.

## Windows Packaging Pipeline

The Windows release flow is split into two branches:

- `main` contains the product source code.
- `windows_dist` contains packaging-only logic: launcher, PyInstaller spec, Inno Setup script, build scripts, workflows, and release docs.

This repository is a fork of [Video-Transcribe](https://github.com/a2nath/Video-Transcribe).

Development and release maintenance are done with Codex.

Release builds always use:

- `source/` from an exact tagged commit in `main`
- `packaging/` from `windows_dist`

Local installer build from the current checkout:

```powershell
$iscc = .\scripts\windows\install-inno-setup.ps1

powershell -ExecutionPolicy Bypass -File .\scripts\windows\build-release.ps1 `
  -ReleaseTag v0.0.0-local `
  -SourceRef refs/heads/develop `
  -PackagingDir . `
  -RepositoryRoot . `
  -IsccPath $iscc
```

Result:

- `build/windows/installer/VideoTranscribe-Setup-0.0.0-local.exe`
- `build/windows/installer/VideoTranscribe-Setup-0.0.0-local.exe.sha256`

User machines do not need manual installation of Python, Node.js, `ffmpeg`, or `yt-dlp`. The installer bundles the runtime and starts the local service automatically after installation.

### Codex refresh notes

If `main` changes product code, do not copy application source into `windows_dist`.

- Keep `windows_dist` packaging-only.
- Update `windows_dist` only when launcher, runtime config, PyInstaller, installer, or workflow logic must follow a product change.
- Rebuild release assets from exact tagged source in `main` plus packaging files from `windows_dist`.
- Never patch `source/` during the release build.

## Содержание

- [Описание проекта](#описание-проекта)
- [Описание функционала](#описание-функционала)
- [Пример пользовательского сценария при работе с удалёнными ссылками](#пример-пользовательского-сценария-при-работе-с-удалёнными-ссылками)
- [Пример пользовательского сценария при работе с локальным файлом](#пример-пользовательского-сценария-при-работе-с-локальным-файлом)
- [Гайд на установку под Windows](#гайд-на-установку-под-windows)
- [Гайд на установку под Linux](#гайд-на-установку-под-linux)

## Описание проекта

`Video Transcribe` — это локальный сервис для преобразования речи из аудио- и видеоисточников в текст без привязки к облачному SaaS-сценарию.

Сервис поддерживает два основных типа входных данных:
- локальные файлы, загружаемые с компьютера пользователя;
- удалённые ссылки на медиаресурсы, которые может обработать `yt-dlp`.

Проект включает:
- backend API на `FastAPI`;
- локальный web UI на `React + Vite`;
- пайплайн извлечения аудио через `ffmpeg`;
- распознавание речи через `faster-whisper`.

Для удалённых ссылок доступна отдельная предварительная проверка источника. Она позволяет до запуска полной расшифровки понять, поддерживается ли ссылка, доступны ли форматы и выглядит ли источник пригодным для извлечения аудио.

## Описание функционала

- Расшифровка локальных аудио- и видеофайлов.
- Расшифровка удалённых медиаисточников по URL через `yt-dlp`.
- Отдельная проверка ссылки перед запуском транскрибации.
- Поддержка форматов результата: `txt`, `srt`, `json`.
- Скачивание готового результата из интерфейса.
- Отображение этапов выполнения и live-лога обработки.
- Поддержка `cpu` и `cuda` режимов выполнения.
- Выбор языка распознавания или автоопределение.
- Опциональное сохранение промежуточного извлечённого аудио.

## Пример пользовательского сценария при работе с удалёнными ссылками

1. Пользователь запускает backend API и frontend UI.
2. Открывает интерфейс в браузере.
3. Выбирает режим `Ссылка на видео`.
4. Вставляет URL удалённого медиаисточника.
5. Нажимает `Проверить источник`.
6. Получает предварительный результат:
   - поддерживается ли ссылка;
   - есть ли форматы;
   - выглядит ли источник пригодным для извлечения аудио;
   - если нет, то по какой причине.
7. Если проверка успешна, настраивает параметры:
   - язык;
   - модель;
   - устройство выполнения;
   - формат результата;
   - необходимость сохранения промежуточного аудио.
8. Нажимает `Начать расшифровку`.
9. Сервис скачивает или извлекает источник, подготавливает аудио и запускает распознавание.
10. Пользователь просматривает результат в интерфейсе и скачивает готовый файл.

## Пример пользовательского сценария при работе с локальным файлом

1. Пользователь запускает backend API и frontend UI.
2. Открывает интерфейс в браузере.
3. Выбирает режим `Локальный файл`.
4. Загружает аудио- или видеофайл со своего компьютера.
5. Настраивает параметры расшифровки:
   - язык;
   - модель;
   - устройство выполнения;
   - формат результата;
   - необходимость сохранения промежуточного аудио.
6. Нажимает `Начать расшифровку`.
7. Backend сохраняет загруженный файл во временную рабочую директорию.
8. Если входной файл является видео, сервис извлекает из него аудио.
9. После завершения распознавания пользователь видит текст в интерфейсе.
10. Пользователь скачивает готовый результат в выбранном формате.

## Гайд на установку под Windows

### Системные требования

- Windows 10 или Windows 11
- Python 3.10 или новее
- Node.js 20 или новее
- `ffmpeg`, доступный из `PATH`
- Опционально: CUDA-окружение для запуска `faster-whisper` на GPU

### 1. Клонирование репозитория

```powershell
git clone https://github.com/SillWo/Video-Transcribe.git
cd Video-Transcribe
```

### 2. Создание и активация виртуального окружения Python

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Установка Python-зависимостей

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Установка frontend-зависимостей

```powershell
cd ui
npm install
cd ..
```

### 5. Запуск backend API

```powershell
.venv\Scripts\activate
uvicorn web_api:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Запуск frontend UI

Откройте второй терминал:

```powershell
cd ui
npm run dev
```

### 7. Открытие интерфейса

Откройте адрес, который покажет Vite. Обычно это:

```text
http://127.0.0.1:5173
```

### 8. Опциональная настройка адреса backend

Если frontend должен обращаться к backend по другому адресу или порту, создайте файл `ui/.env.local`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Гайд на установку под Linux

### Системные требования

- Linux-дистрибутив с `bash`
- Python 3.10 или новее
- Node.js 20 или новее
- установленный `ffmpeg`
- Опционально: CUDA и совместимые драйверы для GPU-режима

### 1. Клонирование репозитория

```bash
git clone https://github.com/SillWo/Video-Transcribe.git
cd Video-Transcribe
```

### 2. Создание и активация виртуального окружения Python

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установка Python-зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Установка frontend-зависимостей

```bash
cd ui
npm install
cd ..
```

### 5. Запуск backend API

```bash
source .venv/bin/activate
uvicorn web_api:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Запуск frontend UI

Откройте второй терминал:

```bash
cd ui
npm run dev
```

### 7. Открытие интерфейса

Откройте адрес, который покажет Vite. Обычно это:

```text
http://127.0.0.1:5173
```

### 8. Опциональная настройка адреса backend

Если frontend должен обращаться к backend по другому адресу или порту, создайте файл `ui/.env.local`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```
