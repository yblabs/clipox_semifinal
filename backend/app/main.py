"""WayinVideo Studio — FastAPI application.

Layout::

    app/
      config.py        environment-driven settings (paths, models, LLM, CORS)
      tasks.py         in-memory background task store (poll /api/.../status)
      routes/          HTTP layer, one module per feature
      engine/          the media AI engine (ffmpeg, whisper, scenes, ass, llm)

Run::

    uvicorn app.main:app --host 127.0.0.1 --port 8000

The static frontend lives one level up (``wayin-site/``) and is mounted at ``/``
so a single origin serves both the UI and the API during development.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import BASE_DIR, CORS_ORIGINS, MEDIA_DIR, MEDIA_MOUNT
from .routes import api_router

SITE_DIR = BASE_DIR.parent

app = FastAPI(
    title="WayinVideo Studio API",
    description="Clip Maker, AI Video Editor, Summarizer, Video-to-Text and Note Taker.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()] or ["*"],
    allow_credentials=False,   # "*" origins cannot be combined with credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    """Liveness + capability report (what is installed, what is configured).

    Registered *before* the static mounts so the "/" mount cannot shadow it.
    """
    from .engine import transcribe
    from .engine.llm import configured as llm_configured, provider_label

    return {
        "status": "ok",
        "whisper": {
            "available": transcribe.available(),
            "model": os.getenv("WV_WHISPER_MODEL", "base"),
        },
        "llm": "on" if llm_configured() else "local-fallback",
        "llm_provider": provider_label(),
        "media_mount": f"/{MEDIA_MOUNT}",
        "fonts_mount": "/fonts",
        "host": host_report(),
        "ytdlp": ytdlp_report(),
    }


def ytdlp_report() -> dict:
    """Whether URL downloads have the credentials they need (values withheld)."""
    from .engine.media import auth_status

    return auth_status()


def host_report() -> dict:
    """What this machine looks like to the pipeline.

    Exposed so a host's real limits (container memory, encoder threads) can be
    confirmed from outside instead of inferred from the crash it causes.
    """
    from .config import FFMPEG_THREADS

    limit_mb = None
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and 0 < int(raw) < (1 << 50):
            limit_mb = int(raw) // (1 << 20)
            break
    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines()[:2]:
            key, value = line.split(":", 1)
            meminfo[key] = value.strip()
    except OSError:
        pass
    try:
        cpus = len(os.sched_getaffinity(0))       # type: ignore[attr-defined]
    except AttributeError:
        cpus = os.cpu_count()
    return {"cpu_count": cpus, "cgroup_mb": limit_mb,
            "meminfo": meminfo, "ffmpeg_threads": FFMPEG_THREADS}


# rendered clips, posters, exports (absolute paths under MEDIA_DIR)
app.mount(f"/{MEDIA_MOUNT}", StaticFiles(directory=str(MEDIA_DIR)), name="media")

# Vendored caption fonts. Serving the same files the renderer hands to libass
# keeps the burned-in export pixel-identical to the browser preview.
FONTS_DIR = BASE_DIR / "assets" / "fonts"
FONTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/fonts", StaticFiles(directory=str(FONTS_DIR)), name="fonts")

# the static frontend, if it sits next to the backend folder
if (SITE_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(SITE_DIR), html=True), name="site")
