"""
Scene detection.

Uses PySceneDetect when it is installed, otherwise falls back to FFmpeg's
built-in ``scene`` filter (which is dependency free and plenty good enough for
hook scoring — we only need a rough sense of how busy a passage is).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from ..config import FFMPEG_BIN, FFMPEG_THREADS

_SCENE_PTS = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")


def detect_scenes(src: Path | str, *, threshold: float = 0.32, limit: int = 400) -> Optional[List[float]]:
    """Return sorted scene-change timestamps (seconds). ``[]``/``None`` = unknown."""
    # PySceneDetect keeps a decoded frame buffer per core through OpenCV, which
    # is what pushed a 512 MB container over the edge. On a constrained host go
    # straight to the ffmpeg scan, whose threads we can pin.
    if FFMPEG_THREADS <= 0:
        try:
            return _pyscenedetect(src, threshold, limit)
        except Exception:
            pass
    try:
        return _ffmpeg_scenes(src, min(threshold, 0.4), limit)
    except Exception:
        return None


def _pyscenedetect(src: Path | str, threshold: float, limit: int) -> List[float]:
    from scenedetect import detect, ContentDetector  # type: ignore

    scenes = detect(str(src), ContentDetector(threshold=int(threshold * 100)))
    stamps = [float(s[0].get_seconds()) for s in scenes if s[0] is not None]
    return stamps[:limit]


def _ffmpeg_scenes(src: Path | str, threshold: float, limit: int) -> List[float]:
    if not shutil.which(FFMPEG_BIN):
        return []
    # A decoder thread pool sized from the core count is the biggest memory
    # spike in a scan of a long video, so cap it on constrained hosts.
    decode_threads = ["-threads", str(FFMPEG_THREADS)] if FFMPEG_THREADS > 0 else []
    proc = subprocess.run(
        [
            FFMPEG_BIN, "-hide_banner", *decode_threads, "-i", str(src),
            # commas inside the filter expression must be escaped for -vf
            "-vf", "select=gt(scene\\,{:.2f}),showinfo".format(threshold),
            "-an", "-f", "null", "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800,
    )
    found = [float(m) for m in _SCENE_PTS.findall(proc.stderr or "")]
    return sorted(set(found))[:limit]


def scene_density(scenes: Optional[List[float]], start: float, end: float) -> int:
    """How many cuts happen inside ``[start, end]`` — a decent energy proxy."""
    if not scenes:
        return 0
    return sum(1 for t in scenes if start <= t <= end)
