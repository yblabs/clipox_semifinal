"""
FFmpeg / ffprobe / yt-dlp helpers shared by every feature.

Everything that shells out to a media tool lives here so the rest of the engine
can stay pure Python.
"""
from __future__ import annotations

import json
import logging
import re
import resource
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config import FFMPEG_BIN, FFPROBE_BIN, TARGET_H, TARGET_W, ffmpeg_thread_args


class MediaError(RuntimeError):
    """Raised when an ffmpeg/ffprobe/yt-dlp invocation fails."""


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------
def log_mem(stage: str) -> None:
    """Record live and peak RSS around an ffmpeg call.

    A host that kills the process for memory leaves no traceback, so this is
    what shows which call was running and how much headroom was left.
    """
    live = peak = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                live = int(line.split()[1]) / 1024
                break
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = peak / (1 << 20) if sys.platform == "darwin" else peak / 1024
    logging.getLogger(__name__).warning(
        "[mem] %-28s live %s MB, peak %.0f MB", stage,
        f"{live:.0f}" if live is not None else "?", peak)


def run(cmd: Sequence[str], *, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """Run a command, raising :class:`MediaError` with stderr on failure."""
    label = Path(cmd[0]).name
    log_mem(f"before {label}")
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    log_mem(f"after {label}")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        raise MediaError(f"{cmd[0]} exited with {proc.returncode}:\n{tail}")
    return proc


def have_binary(name: str) -> bool:
    return shutil.which(name) is not None


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def probe(path: Path | str) -> Dict[str, Any]:
    """Return ffprobe metadata (duration, streams, width/height, fps)."""
    proc = run(
        [FFPROBE_BIN, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    )
    return json.loads(proc.stdout)


def duration_of(path: Path | str) -> float:
    info = probe(path)
    if info.get("format", {}).get("duration"):
        return float(info["format"]["duration"])
    for stream in info.get("streams", []):
        if stream.get("duration"):
            return float(stream["duration"])
    raise MediaError(f"Could not determine duration of {path}")


def video_meta(path: Path | str) -> Dict[str, Any]:
    """Convenience: duration + first video stream geometry + title/fps."""
    info = probe(path)
    vstream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    fps = 30.0
    if vstream.get("avg_frame_rate") and "/" in vstream["avg_frame_rate"]:
        num, den = vstream["avg_frame_rate"].split("/")
        if float(den) > 0:
            fps = float(num) / float(den)
    return {
        "duration": float(info.get("format", {}).get("duration", 0) or 0),
        "width": int(vstream.get("width", 0) or 0),
        "height": int(vstream.get("height", 0) or 0),
        "fps": round(fps, 3),
        "title": info.get("format", {}).get("tags", {}).get("title", ""),
        "size": int(info.get("format", {}).get("size", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------
def extract_audio(src: Path | str, dst: Path | str, *, sample_rate: int = 16000) -> Path:
    """Pull a mono 16 kHz WAV track out of *src* (the input Whisper wants)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            FFMPEG_BIN, "-y", "-i", str(src),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-acodec", "pcm_s16le", str(dst),
        ]
    )
    return dst


def extract_loudness_audio(src: Path | str, dst: Path | str) -> Path:
    """Mono 48 kHz WAV used for silence detection (silencedetect)."""
    return extract_audio(src, dst, sample_rate=48000)


# ---------------------------------------------------------------------------
# Input acquisition
# ---------------------------------------------------------------------------
def save_upload(filename: str, chunk: Iterable[bytes], dest_dir: Path) -> Path:
    """Stream a multipart upload to disk (never load the whole file in RAM)."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name) or "input.mp4"
    path = dest_dir / safe
    with open(path, "wb") as fh:
        for part in chunk:
            fh.write(part)
    return path


# ---------------------------------------------------------------------------
# yt-dlp
# ---------------------------------------------------------------------------
# YouTube rotates which of its player clients actually serve video. The default
# ``web`` client regularly answers with:
#
#     ERROR: [youtube] <id>: The page needs to be reloaded.
#
# which is a client-side rejection, not a problem with the URL or the video.
# Rather than failing the job we walk a short list of alternative clients and
# looser format selectors until one combination works.
#
# Player clients to try, in order; ``None`` = yt-dlp's default selection.
_YTDLP_CLIENTS: List[Optional[str]] = [None, "android", "tv", "ios"]

# Format selectors from strictest (mp4 + m4a only) to loosest.
_YTDLP_FORMATS: List[str] = [
    "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
    "bv*[height<=1080]+ba/b[height<=1080]/b",
    "bv*+ba/b",
]

# Failures no client can fix — stop burning fallback attempts on them.
_YTDLP_FATAL = (
    "private video",
    "has been removed",
    "has been terminated",
    "copyright",
    "not made this video available in your country",
)


def _is_youtube(url: str) -> bool:
    low = (url or "").lower()
    return "youtube.com" in low or "youtu.be" in low


def _is_fatal(message: str) -> bool:
    low = (message or "").lower()
    return any(marker in low for marker in _YTDLP_FATAL)


def _ytdlp_options(fmt: str, client: Optional[str], outtmpl: Optional[str] = None) -> Dict[str, Any]:
    opts: Dict[str, Any] = {
        "format": fmt,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
    }
    if outtmpl:
        opts["outtmpl"] = outtmpl
        opts["merge_output_format"] = "mp4"
    if client:
        opts["extractor_args"] = {"youtube": {"player_client": [client]}}
    return opts


# Rejections that mean "this player client is broken", not "this format is
# wrong" — skip straight to the next client instead of re-trying formats.
_YTDLP_CLIENT_LEVEL = (
    "page needs to be reloaded",
    "missing a url",
    "sabr",
    "no longer supported",
)


def _clients_for(url: str) -> List[Optional[str]]:
    return list(_YTDLP_CLIENTS) if _is_youtube(url) else [None]


def _is_client_level(message: str) -> bool:
    """True when the rejection is about the player client, not the format."""
    low = (message or "").lower()
    return any(marker in low for marker in _YTDLP_CLIENT_LEVEL)


def _fail(errors: List[str], url: str):
    """Raise :class:`MediaError` with the first (most relevant) yt-dlp error."""
    seen: List[str] = []
    for err in errors:
        if err not in seen:
            seen.append(err)
    raise MediaError(seen[0] if seen else f"yt-dlp could not read {url}")


def download_url(url: str, dest_dir: Path) -> Path:
    """Download a remote video with yt-dlp and return the local file path.

    Walks the player-client fallback list when YouTube rejects the default
    client (``The page needs to be reloaded``) so a transient extractor break
    never fails the whole job.
    """
    import yt_dlp  # local import: keeps app boot fast if the dep is missing

    dest_dir.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []

    for client in _clients_for(url):
        for fmt in _YTDLP_FORMATS:
            opts = _ytdlp_options(fmt, client, outtmpl=str(dest_dir / "%(id)s.%(ext)s"))
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
            except Exception as exc:  # DownloadError and friends
                message = str(exc).splitlines()[0] if str(exc) else repr(exc)
                errors.append(message)
                if _is_fatal(message):
                    _fail(errors, url)
                if _is_client_level(message):
                    break  # this client is broken - move to the next one
                continue  # wrong format for this client - try a looser one

            path = Path(ydl.prepare_filename(info))
            if not path.exists():
                # merge_output_format can change the extension; ignore partials
                candidates = [
                    p for p in sorted(dest_dir.glob(f"{info.get('id', 'x')}.*"))
                    if not p.name.endswith((".part", ".ytdl"))
                ]
                if not candidates:
                    errors.append(f"yt-dlp produced no file for {url}")
                    continue
                path = candidates[-1]
            return path

    _fail(errors, url)


def fetch_metadata(url: str) -> Dict[str, Any]:
    """Title/thumbnail/duration without downloading the whole video."""
    import yt_dlp

    errors: List[str] = []
    info: Dict[str, Any] = {}
    for client in _clients_for(url):
        for fmt in _YTDLP_FORMATS:
            opts = _ytdlp_options(fmt, client)
            opts["skip_download"] = True
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                break
            except Exception as exc:
                message = str(exc).splitlines()[0] if str(exc) else repr(exc)
                errors.append(message)
                if _is_fatal(message):
                    _fail(errors, url)
                if _is_client_level(message):
                    break  # this client is broken - move to the next one
                continue  # wrong format for this client - try a looser one
        if info:
            break
    if not info:
        _fail(errors, url)
    return {
        "id": info.get("id"),
        "title": info.get("title", ""),
        "duration": info.get("duration", 0),
        "thumbnail": info.get("thumbnail", ""),
        "uploader": info.get("uploader", ""),
        "webpage_url": info.get("webpage_url", url),
    }


# ---------------------------------------------------------------------------
# Filters / rendering primitives
# ---------------------------------------------------------------------------
def ass_filter(ass_path: Path | str) -> str:
    """ffmpeg ``subtitles`` filter expression with proper path escaping."""
    p = str(Path(ass_path).resolve())
    p = p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return f"subtitles='{p}'"


def vertical_crop_filter(smart_offset: Optional[float] = None) -> str:
    """9:16 smart-crop chain.

    *smart_offset* is a normalised horizontal centre (0..1) computed by the
    face/subject tracker; ``None`` falls back to a centred crop.
    """
    cx = 0.5 if smart_offset is None else max(0.0, min(1.0, smart_offset))
    # crop a 9:16 window centred on cx, then scale to the target resolution
    return (
        f"crop='min(iw\\,ih*9/16)':'min(ih\\,iw*16/9)':"
        f"(iw-ow)*{cx:.4f}:'(ih-oh)/2',"
        f"scale={TARGET_W}:{TARGET_H}:flags=lanczos,setsar=1"
    )


def render_clip(
    src: Path | str,
    dst: Path | str,
    start: float,
    duration: float,
    *,
    vf: Optional[str] = None,
    ass: Optional[Path | str] = None,
    preset: str = "veryfast",
    crf: int = 20,
) -> Path:
    """Trim + re-encode a segment.

    ``-ss`` is placed *before* ``-i`` (input seeking) which is fast, and because
    we always re-encode, ffmpeg decodes from the preceding keyframe and drops it
    — so the cut stays frame-accurate with no A/V desync.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    filters: List[str] = []
    if vf:
        filters.append(vf)
    if ass:
        filters.append(f"ass='{str(Path(ass).resolve()).replace(':', chr(92) + ':')}'")

    cmd: List[str] = [
        FFMPEG_BIN, "-y",
        "-ss", f"{max(start, 0.0):.3f}",
        "-i", str(src),
        "-t", f"{max(duration, 0.1):.3f}",
        # Pinned decode/encoder/filter threads: the default one-thread-per-core
        # pool buffers full-size frames and does not fit a small container.
        *ffmpeg_thread_args(),
    ]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        "-shortest",
        str(dst),
    ]
    run(cmd, timeout=3600)
    return dst


def burn_captions(src: Path | str, dst: Path | str, ass: Path | str, *, preset: str = "veryfast", crf: int = 20) -> Path:
    """Second pass that only re-writes subtitles (used by the editor)."""
    return render_clip(src, dst, 0.0, duration_of(src), ass=ass, preset=preset, crf=crf)


def extract_frames(src: Path | str, out_dir: Path, *, every_sec: float = 2.0, limit: int = 24) -> List[Path]:
    """Grab one JPEG every *every_sec* seconds (capped at *limit* frames)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            FFMPEG_BIN, "-y", "-i", str(src),
            "-vf", f"fps=1/{every_sec}",
            "-frames:v", str(limit),
            "-q:v", "4",
            str(out_dir / "frame_%03d.jpg"),
        ]
    )
    return sorted(out_dir.glob("frame_*.jpg"))


def make_thumbnail(src: Path | str, dst: Path | str, at_sec: float = 1.0) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([FFMPEG_BIN, "-y", "-ss", f"{at_sec:.2f}", "-i", str(src), "-frames:v", "1", "-q:v", "3", str(dst)])
    return dst
