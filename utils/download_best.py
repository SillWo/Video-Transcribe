import argparse
import shutil
import subprocess
import sys
import timeit
from pathlib import Path

default_model_bin = "yt-dlp.exe"
default_get_best_format = "bestvideo*+bestaudio/best"
default_get_video_format = "bestvideo*/best"
default_get_audio_format = "bestaudio/best"
default_get_merge_format = "mkv"


class Download:
    def findarg(self, args, key: str) -> bool:
        return key in args and getattr(args, key)

    def get_fullpath(self, output_dir, output_file) -> tuple[Path, str]:
        output_filepath = Path(output_file).resolve()

        if output_filepath.parent != Path(output_dir).resolve():
            output_dir = output_filepath.parent
            output_file = Path(output_file).name

        return Path(output_dir, output_file), output_dir

    def resolve_model_cmd(self, candidate: str | None) -> list[str]:
        if candidate:
            resolved = shutil.which(candidate)
            if resolved:
                return [resolved]

            explicit_path = Path(candidate)
            if explicit_path.exists():
                return [str(explicit_path.resolve())]

        for fallback in ("yt-dlp.exe", "yt-dlp"):
            resolved = shutil.which(fallback)
            if resolved:
                return [resolved]

        try:
            __import__("yt_dlp")
            return [sys.executable, "-m", "yt_dlp"]
        except ImportError as exc:
            raise FileNotFoundError("yt-dlp executable was not found and Python package yt_dlp is not installed") from exc

    def adjust_format(self, args):
        self.model_cmd = self.resolve_model_cmd(getattr(args, "bin", None))

        if self.findarg(args, "list"):
            self.opts = ["-F", args.url]
            return

        if self.findarg(args, "audio_only"):
            args.format = self.get_audio_format
            args.merge = False
        elif self.findarg(args, "video_only"):
            args.format = self.get_video_format
            args.merge = False

        self.opts = [args.url, "-f", args.format]

        if self.findarg(args, "keep"):
            self.opts += ["-k"]

        if self.findarg(args, "output_name"):
            self.filepath, self.output_dir = self.get_fullpath(self.output_dir, args.output_name)
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            self.opts += ["-o", str(self.filepath)]

        elif self.findarg(args, "output_dir"):
            self.output_dir = Path(args.output_dir).resolve()
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            self.opts += ["-P", str(self.output_dir)]

        if self.findarg(args, "verbose"):
            self.opts += ["--verbose"]

        if "merge" in args and args.merge:
            self.opts += ["--merge-output-format", args.merge[0]]
        elif "merge" in args:
            self.opts += ["--merge-output-format", self.get_merge_format]

        if self.findarg(args, "overwrite"):
            self.opts += ["--yes-overwrites"]

        if self.findarg(args, "username"):
            self.opts += ["-u", args.username]
            if self.findarg(args, "password"):
                self.opts += ["-p", args.password]

        if self.findarg(args, "playlist_start"):
            self.opts += ["--playlist-start", args.playlist_start]

        if self.findarg(args, "playlist_end"):
            self.opts += ["--playlist-end", args.playlist_end]

        if self.findarg(args, "audio_format"):
            self.opts += ["--extract-audio", "--audio-format", args.audio_format]

        if self.findarg(args, "restrict_filenames"):
            self.opts += ["--restrict-filenames"]

        if self.findarg(args, "timeout"):
            self.timeout = float(args.timeout)

    def normalize_output_path(self, output_file: str) -> str:
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = self.output_dir / output_path
        return str(output_path.resolve())

    def get_youtube_vid(self, filepath=None):
        if filepath:
            self.filepath, self.output_dir = self.get_fullpath(self.output_dir, filepath)
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            if "-o" not in self.opts:
                self.opts += ["-o", str(self.filepath)]

        process = subprocess.Popen(
            self.model_cmd + self.opts,
            encoding="utf8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )

        if self.debug_flag:
            print(f"Parameters: {self.model_cmd + self.opts}\n")

        media_list = []
        output_file = ""
        start_time = timeit.default_timer()

        merger_pattern = "[Merger] Merging formats into \""
        download_pattern = "[download] Destination: "
        post_audio_fix_pattern = "[ExtractAudio] Destination: "
        downloaded_pattern = " has already been downloaded"

        try:
            assert process.stdout is not None
            for stdout_line_str in process.stdout:
                stdout_line_str = stdout_line_str.rstrip()
                if not stdout_line_str:
                    continue

                if "-F" in self.opts:
                    print(stdout_line_str, flush=True)
                    continue

                if stdout_line_str.startswith(merger_pattern):
                    output_file = stdout_line_str[len(merger_pattern) : -1]
                elif stdout_line_str.startswith(post_audio_fix_pattern):
                    output_file = stdout_line_str[len(post_audio_fix_pattern) :]
                elif output_file == "" and stdout_line_str.startswith(download_pattern):
                    output_file = stdout_line_str[len(download_pattern) :]
                elif output_file == "" and downloaded_pattern in stdout_line_str:
                    output_file = stdout_line_str[len("[download] ") : stdout_line_str.index(downloaded_pattern)]

                if output_file and "Extracting URL:" in stdout_line_str:
                    media_list.append(self.normalize_output_path(output_file))
                    output_file = ""

                print(stdout_line_str, flush=True)

                if self.timeout is not None and timeit.default_timer() - start_time > self.timeout:
                    process.kill()
                    raise TimeoutError(f"Process timed out after {self.timeout} seconds")

        except KeyboardInterrupt:
            process.kill()
            raise

        return_code = process.wait(timeout=self.timeout)

        if output_file:
            normalized = self.normalize_output_path(output_file)
            if normalized not in media_list:
                media_list.append(normalized)

        if not media_list and return_code == 0 and "-F" not in self.opts:
            raise RuntimeError("Nothing was downloaded. Check the provided URL or format selection.")

        if self.debug_flag:
            for file in media_list:
                print(f"Downloaded file {file}")
            print(f"Returned code {return_code}")
            print("-----------------------FINISHED------------------------")

        return media_list, return_code

    def run(self, filepath=None):
        if self.debug_flag:
            print("---------------------DOWNLOADING-----------------------")

        video_names, retcode = self.get_youtube_vid(filepath)
        return video_names, retcode

    def __init__(self, args, debug=False):
        self.debug_flag = debug
        self.model_cmd = []
        self.get_best_format = default_get_best_format
        self.get_audio_format = default_get_audio_format
        self.get_video_format = default_get_video_format
        self.get_merge_format = default_get_merge_format
        self.opts = []
        self.output_dir = Path.cwd()
        self.filepath = None
        self.timeout = None
        self.adjust_format(args)


def main():
    parser = argparse.ArgumentParser("Downloads the best quality video from source", add_help=True)
    parser.add_argument("-l", "--url", help="URL or source of one or more videos", required=True)
    parser.add_argument("-F", "--list", help="List all formats that can be downloaded", action="store_true")
    parser.add_argument("--verbose", help="Verbose output", action="store_true")
    parser.add_argument("-k", "--keep", help="Keep intermediate files", action="store_true")
    parser.add_argument("-f", "--format", help="Format of the video to download", default=default_get_best_format)
    parser.add_argument("-u", "--username", help="Username to login with")
    parser.add_argument("-p", "--password", help="Password for the credentials. Used with username")
    parser.add_argument("-o", "--output_name", help="Ouput filename")
    parser.add_argument("-od", "--output_dir", help="Ouput directory", default=Path.cwd())
    parser.add_argument("-a", "--audio_only", help="Audio only download", action="store_true")
    parser.add_argument("-v", "--video_only", help="Video only download", action="store_true")
    parser.add_argument("-b", "--bin", help="Path to the yt-dlp binary to choose")
    parser.add_argument("-m", "--merge", help="Whether to merge the audio and video. Default format: mkv", nargs="*")
    parser.add_argument("--quiet", help="Debug print off", action="store_true")
    parser.add_argument("--overwrite", help="Overwrite an exising file", action="store_true")
    parser.add_argument("--timeout", help="Amount of time to wait for the download to finish (seconds)")
    parser.add_argument("--playlist_start", help="Starting position from a list of media, to start downloading from")
    parser.add_argument("--playlist_end", help="Ending position from a list of media, to stop downloading at")
    parser.add_argument("--audio_format", help="Specify the audio format to use in post-processing")
    parser.add_argument("--restrict_filenames", help="Restrict filenames to only ASCII characters, and avoid '&' and spaces in filenames", action="store_true")

    args = parser.parse_args()
    downloader = Download(args, debug=not args.quiet)

    if not args.quiet:
        print("-----------------------SETTINGS------------------------")
        arguments = vars(args)
        for key in arguments:
            value = getattr(args, key)
            if (key and type(value) is not bool and value is not None) or type(value) is bool:
                print(key, "\t", value)

        print("---------------------DOWNLOADING-----------------------")

    downloader.get_youtube_vid()


if __name__ == "__main__":
    main()
