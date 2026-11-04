"""
Central configuration for the WayinVideo studio backend.

All values can be overridden with environment variables so the same code runs on
a laptop (Apple Silicon), a CUDA box or a CPU-only server.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # .../backend
WORK_DIR = Path(os.getenv("WV_WORK_DIR", str(BASE_DIR / "work")))
MEDIA_DIR = WORK_DIR / "media"                             # rendered outputs
UPLOAD_DIR = WORK_DIR / "uploads"                          # raw user files
DOWNLOAD_DIR = WORK_DIR / "downloads"                      # yt-dlp output

for _p in (WORK_DIR, MEDIA_DIR, UPLOAD_DIR, DOWNLOAD_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Directory that is exposed over HTTP so the browser can play rendered clips.
MEDIA_MOUNT = "media"

# ---------------------------------------------------------------------------
# Billing — the single central switch for the whole site
# ---------------------------------------------------------------------------
# UNLIMITED_CREDITS=true  (default) -> no action ever charges, no balance check
#                                   ever fails, every API answers "unlimited".
# UNLIMITED_CREDITS=false           -> the credit system behaves exactly as
#                                   before (charge, 402, upgrade prompt).
# Flip it from the environment or from this one constant; nothing else in the
# codebase needs to change.
UNLIMITED_CREDITS: bool = os.getenv(
    "UNLIMITED_CREDITS", "true"
).strip().lower() not in {"0", "false", "no", "off"}

# ---------------------------------------------------------------------------
# Media pipeline
# ---------------------------------------------------------------------------
FFPROBE_BIN = os.getenv("WV_FFPROBE", "ffprobe")


def _has_subtitle_filter(binary: str) -> bool:
    """True when *binary* was built with libass (the `subtitles` filter).

    Caption burning needs it, and some builds (notably a trimmed Homebrew
    ffmpeg 9) are compiled without libass. Feature-detecting beats assuming.
    """
    try:
        import subprocess

        out = subprocess.run(
            [binary, "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=25,
        ).stdout
        return " subtitles " in f" {out} "
    except Exception:
        return False


def discover_ffmpeg() -> str:
    """Pick an FFmpeg that can actually burn subtitles.

    Order: ``WV_FFMPEG`` override -> PATH ffmpeg (if libass-capable) -> the
    ffmpeg bundled with the ``imageio-ffmpeg`` wheel (static, includes libass
    and drawtext) -> PATH ffmpeg as a last resort.
    """
    forced = os.getenv("WV_FFMPEG", "").strip()
    if forced:
        return forced
    import shutil

    path_ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    if _has_subtitle_filter(path_ffmpeg):
        return path_ffmpeg
    try:
        import imageio_ffmpeg  # optional wheel with a full-featured static build

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and _has_subtitle_filter(bundled):
            return bundled
    except Exception:
        pass
    return path_ffmpeg


FFMPEG_BIN = discover_ffmpeg()

def _memory_limited_host() -> bool:
    """True when a cgroup caps this process below 1 GiB of RAM.

    Shared containers advertise many cores but little memory, and libx264
    sizes its thread pool (and frame buffers) from the core count. Knowing the
    real limit lets the encoder be told to stay small without penalising a
    laptop, which reports no cgroup limit.
    """
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if not raw.isdigit():
            continue                      # cgroup v2 writes "max" for unlimited
        limit = int(raw)
        if limit <= 0 or limit >= (1 << 50):
            continue                      # cgroup v1 sentinel for unlimited
        return limit <= (1 << 30)         # 1 GiB
    return False


# ffmpeg encoder threads. 0 lets ffmpeg decide (fastest on a laptop), 1 keeps
# the encoder inside a small memory budget. Constrained hosts get 1 by default;
# WV_FFMPEG_THREADS overrides either way.
_FFMPEG_THREADS_ENV = os.getenv("WV_FFMPEG_THREADS", "").strip()
FFMPEG_THREADS = int(_FFMPEG_THREADS_ENV) if _FFMPEG_THREADS_ENV else (
    1 if _memory_limited_host() else 0
)


def ffmpeg_thread_args() -> list[str]:
    """Encoder flags that pin ffmpeg to one thread pool (empty when unset)."""
    if FFMPEG_THREADS <= 0:
        return []
    n = str(FFMPEG_THREADS)
    # A short rate-control lookahead keeps x264 from buffering dozens of
    # frames (each several MB at 1080x1920) while it plans the bitrate.
    return ["-threads", n,
            "-x264-params", f"threads={n}:lookahead_threads=1:rc-lookahead=6"]


# Clip Maker
CLIP_COUNT = int(os.getenv("WV_CLIP_COUNT", "10"))         # clips to generate
CLIP_MIN_SEC = float(os.getenv("WV_CLIP_MIN_SEC", "18"))   # shortest clip
CLIP_MAX_SEC = float(os.getenv("WV_CLIP_MAX_SEC", "58"))   # longest clip
TARGET_W, TARGET_H = 1080, 1920                            # 9:16 output

# Silence detection (AI Video Editor)
SILENCE_DB = float(os.getenv("WV_SILENCE_DB", "-30"))      # noise floor in dBFS
SILENCE_MIN_SEC = float(os.getenv("WV_SILENCE_MIN_SEC", "1.0"))

# Transcription
WHISPER_MODEL = os.getenv("WV_WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WV_WHISPER_LANGUAGE", "")    # "" == auto detect
WHISPER_COMPUTE = os.getenv("WV_WHISPER_COMPUTE", "")      # "" == auto pick
# Greedy decoding (1) instead of beam search (5): word-level captions barely
# change, and on a CPU-only laptop it is the single biggest speed win.
WHISPER_BEAM = int(os.getenv("WV_WHISPER_BEAM", "1"))
WHISPER_THREADS = int(os.getenv("WV_WHISPER_THREADS", "0"))  # 0 == CTranslate2 default
# Windows decoded per batch: 8 is ~3.7x faster than one-at-a-time on a CPU and
# keeps every core busy. 0 disables batching.
WHISPER_BATCH = int(os.getenv("WV_WHISPER_BATCH", "8"))

# ---------------------------------------------------------------------------
# Hardware acceleration
# ---------------------------------------------------------------------------
def detect_device() -> str:
    """Return the best device faster-whisper/CTranslate2 can use here.

    CTranslate2 (faster-whisper) supports CUDA and CPU. Apple Silicon has no MPS
    build of CTranslate2, so on macOS we run the highly optimised CPU kernels
    (int8) which are fast on M-series chips. Set WV_DEVICE to override.
    """
    forced = os.getenv("WV_DEVICE", "").strip()
    if forced:
        return forced
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def detect_compute_type(device: str) -> str:
    """Pick a quantisation that the selected device actually supports."""
    forced = os.getenv("WV_WHISPER_COMPUTE", "").strip()
    if forced:
        return forced
    if device == "cuda":
        return "float16"
    return "int8"          # fastest CPU path, ~4x smaller than fp32


# ---------------------------------------------------------------------------
# LLM provider (metadata, summaries, notes, prompts)
# ---------------------------------------------------------------------------
# Works with any OpenAI-compatible endpoint (OpenAI, Z.ai, Ollama, vLLM...)
# plus the native Anthropic Messages API. When no key is configured the
# engine falls back to deterministic local generators so the app still works.
LLM_PROVIDER = os.getenv("WV_LLM_PROVIDER", "openai")      # openai | anthropic | none
LLM_API_KEY = os.getenv("WV_LLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("WV_LLM_BASE_URL", "https://api.z.ai/v1")
LLM_MODEL = os.getenv("WV_LLM_MODEL", "gpt-4o-mini")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY", "")
VISION_MODEL = os.getenv("WV_VISION_MODEL", "")

# ---------------------------------------------------------------------------
# CORS — the static site is served by another origin during development
# ---------------------------------------------------------------------------
CORS_ORIGINS = os.getenv(
    "WV_CORS_ORIGINS", "*"
).split(",")
