import argparse
import json
import os
import shutil
import sys
import time
import timeit
from os.path import isfile, join
from pathlib import Path
from typing import TextIO

import psutil
import validators
from faster_whisper import WhisperModel as whisper
from faster_whisper.utils import available_models

from utils.download_best import Download
from utils.get_audio import AudioProcess

audio_file = "intermediate_audio_file.mp3"
JSON_EVENT_PREFIX = "__VT_JSON__ "


def sizes_supported() -> list[str]:
    return available_models()


def logical_cpu_count() -> int:
    return psutil.cpu_count(logical=True) or os.cpu_count() or 1


def srt_format_timestamp(seconds: float) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return (f"{hours}:") + f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(segments: list[dict], file: TextIO) -> None:
    print("\nBegin transcription and creating subtitle file:")
    print("-------------------------------------------------------")

    for count, segment in enumerate(segments, start=1):
        print(
            f"{count}\n"
            f"{srt_format_timestamp(segment['start'])} --> {srt_format_timestamp(segment['end'])}\n"
            f"{segment['text'].replace('-->', '->').strip()}\n",
            file=file,
            flush=True,
        )


def findarg(args, key: str) -> bool:
    return key in args and getattr(args, key)


def emit_event(args, event_type: str, **payload) -> None:
    if not getattr(args, "web_json", False):
        return

    event = {"type": event_type, **payload}
    print(f"{JSON_EVENT_PREFIX}{json.dumps(event, ensure_ascii=False)}", flush=True)


def configure_web_stdio(args) -> None:
    if not getattr(args, "web_json", False):
        return

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def get_fullpath(output_dir, output_file) -> tuple[Path, str]:
    output_filepath = Path(output_file).resolve()

    if output_filepath.parent != Path(output_dir).resolve():
        output_dir = output_filepath.parent
        output_file = Path(output_file).name

    return Path(output_dir, output_file), output_dir


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    suffix = path.suffix
    stem = path.stem
    parent = path.parent
    index = 2

    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def serialize_segments(segments) -> list[dict]:
    return [
        {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
        }
        for segment in segments
    ]


def build_plain_text(segments: list[dict], use_timestamps: bool) -> str:
    lines = []
    for segment in segments:
        if use_timestamps:
            lines.append(f"[{srt_format_timestamp(segment['start'])}] {segment['text']}")
        else:
            lines.append(segment["text"])
    return "\n".join(lines).strip() + "\n"


def build_output_payload(segments: list[dict], stats, use_timestamps: bool) -> dict:
    payload = {
        "language": stats.language,
        "language_probability": stats.language_probability,
        "text": "\n".join(segment["text"] for segment in segments).strip(),
    }

    if use_timestamps:
        payload["segments"] = segments
    else:
        payload["segments"] = [{"text": segment["text"]} for segment in segments]

    return payload


def _cuda_runtime_missing(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "cublas64_12.dll",
            "cudart64_12.dll",
            "cusparse64_12.dll",
            "cannot be loaded",
            "could not load",
            "not found",
        )
    )


def _create_model(args, device: str):
    if device != "cuda":
        threads = max(1, min(int(args.nproc or logical_cpu_count()), logical_cpu_count()))
        return whisper(
            args.model_size,
            cpu_threads=threads,
            num_workers=4,
            device=device,
            compute_type=args.precision,
        )

    return whisper(args.model_size, device=device, compute_type=args.precision)


def write_output(args, segments: list[dict], stats, filename: str) -> tuple[Path, str]:
    output_suffix = args.output_format
    output_language = stats.language if args.language == "auto" else args.language
    output_path = Path(
        args.output_dir,
        Path(filename).stem + "." + time.strftime("%Y%m%d-%H%M%S") + f".{output_language}.{output_suffix}",
    )

    if args.output_format == "srt":
        with open(output_path, "w", encoding="utf-8") as srt_file:
            write_srt(segments, file=srt_file)
        rendered_result = output_path.read_text(encoding="utf-8")

    elif args.output_format == "txt":
        rendered_result = build_plain_text(segments, use_timestamps=args.use_timestamps)
        output_path.write_text(rendered_result, encoding="utf-8")

    else:
        payload = build_output_payload(segments, stats, use_timestamps=args.use_timestamps)
        rendered_result = json.dumps(payload, ensure_ascii=False, indent=2)
        output_path.write_text(rendered_result, encoding="utf-8")

    return output_path, rendered_result


def transcribe(args, model, full_filepath: str, filename: str | None = None) -> dict:
    assert isinstance(full_filepath, str)

    if filename is None:
        filename = full_filepath

    emit_event(args, "stage", stage="recognition", status="in_progress", message=f"Transcribing {Path(filename).name}")

    start_time = timeit.default_timer()
    language = None if args.language == "auto" else args.language
    try:
        segments_iter, stats = model.transcribe(full_filepath, beam_size=args.beam_size, language=language)
    except RuntimeError as exc:
        if args.device == "cuda" and _cuda_runtime_missing(exc):
            print("[warning] CUDA runtime libraries are unavailable; retrying transcription on CPU.")
            args.device = "cpu"
            model = initialize(args)
            segments_iter, stats = model.transcribe(full_filepath, beam_size=args.beam_size, language=language)
        else:
            raise
    segments = serialize_segments(list(segments_iter))

    print("\nDetected language '%s' with probability %f" % (stats.language, stats.language_probability))

    output_path, rendered_result = write_output(args, segments, stats, filename)
    elapsed_seconds = timeit.default_timer() - start_time

    print(filename, " took ", "{:.1f}".format(elapsed_seconds), " seconds")
    print("Saved result to", output_path)
    print("-------------------------------------------------------")

    result = {
        "input": str(filename),
        "output_path": str(output_path),
        "output_format": args.output_format,
        "detected_language": stats.language,
        "detected_language_probability": stats.language_probability,
        "text": "\n".join(segment["text"] for segment in segments).strip(),
        "rendered_result": rendered_result,
        "segments": segments if args.use_timestamps else [{"text": segment["text"]} for segment in segments],
        "elapsed_seconds": round(elapsed_seconds, 2),
    }

    emit_event(
        args,
        "result",
        stage="result",
        status="completed",
        message=f"Completed {Path(filename).name}",
        result=result,
    )
    return result


def initialize(args):
    print("--------------------INITIALIZING-----------------------")

    if isfile(audio_file):
        os.remove(audio_file)

    try:
        return _create_model(args, args.device)
    except RuntimeError as exc:
        if args.device == "cuda" and _cuda_runtime_missing(exc):
            print("[warning] CUDA runtime libraries are unavailable; falling back to CPU.")
            args.device = "cpu"
            return _create_model(args, "cpu")
        raise


def close(args):
    if not findarg(args, "keep") and isfile(audio_file):
        print("Delete temp file")
        os.remove(audio_file)


def add_media_files(args, media_files, debug=False, verbose=False):
    if findarg(args, "filename"):
        if validators.url(args.filename):
            args.url = args.filename
            args.audio_only = True
            args.restrict_filenames = True
            args.overwrite = True
            args.verbose = False
            args.audio_format = "mp3"

            emit_event(args, "stage", stage="source", status="in_progress", message=f"Downloading source from {args.filename}")

            download = Download(args, debug)
            audio_file_list, retcode = download.run()

            if retcode == 0:
                media_files += audio_file_list
                emit_event(args, "stage", stage="source", status="completed", message=f"Downloaded {len(audio_file_list)} source file(s)")
            else:
                print(f"Could not process the URL, code {retcode} and audio file {audio_file_list} and args.output_name {args.output_name}")
                close(args)
                raise RuntimeError(f"URL download failed with code {retcode}")

        elif Path(args.filename).exists():
            args.filename = Path(args.filename).resolve()
            media_files.append(str(args.filename))
            emit_event(args, "stage", stage="source", status="completed", message=f"Loaded local source {Path(args.filename).name}")

    elif findarg(args, "input_dir"):
        if Path(args.input_dir).exists():
            for filename in os.listdir(args.input_dir):
                filename = Path(args.input_dir, filename)
                if filename.is_file():
                    media_files.append(str(filename))
        else:
            raise FileNotFoundError(f"--input_dir argument does not specify a valid dir: {args.input_dir}")

        emit_event(args, "stage", stage="source", status="completed", message=f"Discovered {len(media_files)} local file(s)")

    if len(media_files) == 0:
        if args.filename:
            print("URL is not valid")
            close(args)
            raise RuntimeError("URL is not valid")
        else:
            print("There were no media to process")
            close(args)
            raise RuntimeError("There were no media to process")

    print(f"Media files found {len(media_files)}")


def preserve_audio_copy(args, source_media_file: str) -> str | None:
    if not isfile(audio_file):
        return None

    target = Path(args.output_dir, Path(source_media_file).stem + Path(audio_file).suffix)
    target = next_available_path(target)
    shutil.move(audio_file, target)
    print(f"Saved intermediate audio to {target}")
    return str(target)


def main():
    global audio_file
    media_files = []

    parser = argparse.ArgumentParser("Generates subtitiles of the video file as an input")
    parser.add_argument("-f", "--filename", help="Name of the media file stored in the filesystem or URL of a video/audio file that needs to subtitles. URL can also be a list of media")
    parser.add_argument("-i", "--input_dir", help="Input directory where video files are. --filename overrides this")
    parser.add_argument("-af", "--audio_filter", help="Audio or video filters to use before transcription (for ffmpeg), no spaces, just comma-separated")
    parser.add_argument("-o", "--output_name", help="Output filename in case of issues with title")
    parser.add_argument("-od", "--output_dir", help="Ouput directory", default=os.getcwd())
    parser.add_argument("-l", "--language", help="Language to be translated from", default="ru", type=str)
    parser.add_argument("-b", "--beam_size", help="Beam size parameter or best_of equivalent from Open-AI whisper", type=int, default=5)
    parser.add_argument("-p", "--precision", help="Precision to use to create the model", type=str, default="auto")
    parser.add_argument("-d", "--device", help="Device to use such a CPU or GPU", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("-s", "--model_size", help="Size of the model, default is small.", choices=sizes_supported(), nargs="?", default="small")
    parser.add_argument("--start", help="Start time in 00:00:00 format", type=AudioProcess.valid_time)
    parser.add_argument("--end", help="End time in 00:00:00 format", type=AudioProcess.valid_time)
    parser.add_argument("-n", "--nproc", help="Number of CPUs to use", default=logical_cpu_count(), type=int)
    parser.add_argument("-k", "--keep", help="Keep intermediate files", action="store_true")
    parser.add_argument("--verbose", help="Verbose print from dependent processes", action="store_true")
    parser.add_argument("--quiet", help="Debug print off", action="store_true")
    parser.add_argument("--playlist_start", help="Starting position from a list of media, to start downloading from")
    parser.add_argument("--playlist_end", help="Ending position from a list of media, to stop downloading at")
    parser.add_argument("--codec", help="Audio codec to use")
    parser.add_argument("--bitrate", help="Audio bitrate to use")
    parser.add_argument("--output_format", help="Output format to create", choices=["txt", "srt", "json"], default="srt")
    parser.add_argument("--no_timestamps", help="Disable timestamps in txt/json output. Ignored for srt.", action="store_true")
    parser.add_argument("--web_json", help=argparse.SUPPRESS, action="store_true")

    args = parser.parse_args()
    configure_web_stdio(args)
    args.use_timestamps = False if args.no_timestamps else True

    if args.output_format == "srt":
        args.use_timestamps = True

    if "model_size" in args and getattr(args, "model_size") is None:
        supported = sizes_supported()
        print("\nSupported size in faster-whisper")
        print("-------------------------------------------------------")
        for size in supported:
            print(f"*  {size}")
        close(args)
        return 0

    if findarg(args, "output_name"):
        args.output_name, args.output_dir = get_fullpath(args.output_dir, args.output_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.quiet is False:
        print("-----------------------SETTINGS------------------------")
        arguments = vars(args)
        for arg in arguments:
            print(arg, "\t", getattr(args, arg))

    audio_file = str(Path(args.output_dir, audio_file))
    print(f"audio_file={audio_file}")

    try:
        add_media_files(args, media_files, debug=not args.quiet, verbose=args.verbose)
        model = initialize(args)
        results = []

        for media_file in media_files:
            if args.quiet is False:
                print(f"Processing file {media_file} and using audio filter")

            if args.quiet is False:
                if args.output_name:
                    print(f"Set file {args.output_name}")

                if args.audio_filter:
                    print(f"Set filter {args.audio_filter}")

                if args.codec:
                    print(f"Set codec {args.codec}")

                if args.bitrate:
                    print(f"Set bitrate {args.bitrate}")

            transcription_input = media_file
            extracted_audio = False
            audio_processor = AudioProcess(args)

            if (
                not audio_processor.audio_only(media_file)
                or findarg(args, "audio_filter")
                or findarg(args, "start")
                or findarg(args, "end")
                or findarg(args, "codec")
                or findarg(args, "bitrate")
            ):
                print("Extracting audio")
                emit_event(args, "stage", stage="audio", status="in_progress", message=f"Extracting audio from {Path(media_file).name}")
                audio_processor.extract_audio(input_filepath=media_file, output_filepath=audio_file, overwrite=True)
                transcription_input = audio_file
                extracted_audio = True
                emit_event(args, "stage", stage="audio", status="completed", message=f"Audio ready for {Path(media_file).name}")

            result = transcribe(args, model, transcription_input, media_file)

            if extracted_audio and isfile(audio_file):
                if findarg(args, "keep"):
                    result["saved_audio"] = preserve_audio_copy(args, media_file)
                else:
                    os.remove(audio_file)

            results.append(result)

        emit_event(args, "complete", stage="result", status="completed", message="All media processed", results=results)
        close(args)
        print("Done.")
        return 0

    except Exception as exc:
        emit_event(args, "error", stage="result", status="failed", message=str(exc))
        close(args)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
