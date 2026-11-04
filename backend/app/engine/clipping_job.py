"""The AI clipping job runner.

Runs the seven-stage pipeline from Part 2 of the spec on a worker thread and
keeps the task's progress truthful at every stage::

    fetch/prepare (0-15) -> transcript (15-45) -> signals (45-60)
    -> AI selection (60-85) -> refinement (85-92) -> metadata (92-97)
    -> previews (97-100)

The finished document is persisted with :mod:`app.store` so the results page,
Library and export endpoints can reopen it by id.
"""
from __future__ import annotations

import logging
import resource
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .. import store
from ..config import DOWNLOAD_DIR, MEDIA_DIR, UPLOAD_DIR
from . import clipper, media, scenes
from .clip_maker import generate_hooks, generate_metadata as platform_metadata
from .transcribe import format_timestamp, release_model, transcribe_audio

STEPS = [
    "Fetch video",
    "Audio + transcript",
    "Signal analysis",
    "AI moment selection",
    "Refine boundaries",
    "Titles, hooks & metadata",
    "Render previews",
]

ProgressFn = Callable[[int, float, str], None]

log = logging.getLogger(__name__)

PREVIEW_HEIGHT = 854

DEFAULT_STYLE = {
    "motion": "moving", "palette": "brand", "font": "Montserrat",
    "size": "medium", "placement": "center", "preset": "classic",
}

MOTIONS = ("stationary", "moving")
PALETTES = ("classic", "white", "neon", "contrast", "brand", "ocean")
FONTS = ("Montserrat", "Impact", "Anton", "Poppins")
SIZES = ("small", "medium", "large")
PLACEMENTS = ("bottom", "center", "top")
PRESETS = ("classic", "hook", "karaoke")


def _pick(raw: Dict[str, Any], key: str, allowed, default: str) -> str:
    """Keep ``raw[key]`` when it is known; otherwise warn and use *default*.

    A silent fallback here is what made broken Colour / Placement selections
    look like a renderer bug: the UI showed one thing, the render used another
    and nothing was ever logged.
    """
    value = raw.get(key)
    if value in allowed:
        return value
    if value not in (None, ""):
        log.warning("caption style: unknown %s=%r — falling back to %r", key, value, default)
    return default


def clean_style(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = raw or {}
    return {
        "motion": _pick(raw, "motion", MOTIONS, DEFAULT_STYLE["motion"]),
        "palette": _pick(raw, "palette", PALETTES, DEFAULT_STYLE["palette"]),
        "font": _pick(raw, "font", FONTS, DEFAULT_STYLE["font"]),
        "size": _pick(raw, "size", SIZES, DEFAULT_STYLE["size"]),
        "placement": _pick(raw, "placement", PLACEMENTS, DEFAULT_STYLE["placement"]),
        "preset": _pick(raw, "preset", PRESETS, DEFAULT_STYLE["preset"]),
    }


def clean_settings(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = raw or {}
    language = str(raw.get("language") or "auto")
    if language != "auto" and language not in dict(clipper.LANGUAGES):
        language = "auto"
    clip_length = str(raw.get("clip_length") or "auto")
    if clip_length not in clipper.CLIP_LENGTHS:
        clip_length = "auto"
    platform = str(raw.get("platform") or "all")
    if platform not in clipper.PLATFORMS_UI:
        platform = "all"
    try:
        count = int(raw.get("count") or 6)
    except (TypeError, ValueError):
        count = 6
    kind = str(raw.get("clip_kind") or "normal")
    if kind not in ("normal", "hook"):
        kind = "normal"
    layout = str(raw.get("layout") or "Full")
    if layout not in clipper.LAYOUTS:
        layout = "Full"
    raw_style = dict(raw.get("style") or {})
    style = clean_style(raw_style)
    # Hook clips auto-select the Hook caption preset. An explicit preset from
    # the client always wins, so the user can override the auto-selection.
    if kind == "hook" and "preset" not in raw_style:
        style["preset"] = "hook"
    if style["preset"] in ("hook", "karaoke") and "placement" not in raw_style:
        # Hook and Karaoke captions default to lower-middle, above the platform
        # UI safe zone; an explicit placement from the client still wins.
        style["placement"] = "bottom"
    return {
        "language": language,
        "clip_length": clip_length,
        "count": max(1, min(count, 20)),
        "platform": platform,
        "instruction": str(raw.get("instruction") or "").strip()[:500],
        "clip_kind": kind,
        "layout": layout,
        "style": style,
        "translate": str(raw.get("translate") or ""),
    }


def estimate_credits(duration: float) -> int:
    """1 credit per started minute, with a 5 credit floor."""
    minutes = max(1.0, (float(duration or 0)) / 60.0)
    return max(5, int(round(minutes)))


def _fmt(sec: float) -> str:
    return format_timestamp(sec, ms=True)


def _pick_cold_open(moment: Dict[str, Any], words: List[Dict[str, Any]],
                    segments: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Teaser from the strongest line when the payoff lands late in the clip."""
    if moment.get("cold_open"):
        return moment["cold_open"]
    start, end = float(moment["start"]), float(moment["end"])
    span = max(1.0, end - start)
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for seg in segments:
        seg_start, seg_end = float(seg["start"]), float(seg["end"])
        if seg_start < start + span * 0.55 or seg_end > end - 1.0:
            continue
        text = (seg.get("text") or "").lower()
        score = sum(1.5 for t in clipper._tokens(text) if t in clipper.HOOK_WORDS)
        score += 2.0 if "?" in text else 0.0
        score += 1.0 if clipper.NUMBER_RE.search(text) else 0.0
        if score > best_score and seg_end - seg_start >= 1.5:
            best, best_score = seg, score
    if not best or best_score < 2.0:
        return None
    teaser_end = min(float(best["end"]), float(best["start"]) + 5.0)
    if teaser_end - float(best["start"]) < 1.5:
        return None
    return {"start": round(float(best["start"]), 3), "end": round(teaser_end, 3)}


def run(
    task_id: str,
    *,
    video_id: str,
    source: str,
    source_type: str,
    settings: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    on_progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Execute the pipeline and persist the finished document."""
    meta = meta or {}
    settings = clean_settings(settings)

    def report(step: int, pct: float, label: str) -> None:
        # Peak RSS per stage: a host that kills the process for memory leaves no
        # traceback, so the last stage's figure is the only evidence of where
        # the budget went.
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak_mb / (1 << 20) if sys.platform == "darwin" else peak_mb / 1024
        log.warning("[%s] %5.1f%% %s (peak rss %.0f MB)", task_id[:8], pct, label, peak_mb)
        if on_progress:
            on_progress(step, pct, label)

    out_dir = Path(MEDIA_DIR) / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. fetch -------------------------------------------------------------
    report(0, 2, "Fetching video")
    if source_type == "upload" or Path(source).is_file():
        src = Path(source)
        title = meta.get("title") or src.stem.replace("_", " ")
    else:
        report(0, 5, "Downloading video")
        src = media.download_url(source, DOWNLOAD_DIR)
        try:
            fetched = media.fetch_metadata(source)
            title = fetched.get("title") or src.stem
            meta.setdefault("thumbnail", fetched.get("thumbnail") or "")
            meta.setdefault("channel", fetched.get("uploader") or "")
        except Exception:
            title = src.stem
    try:
        probe = media.video_meta(src)
    except Exception:
        probe = {"duration": media.duration_of(src), "width": 0, "height": 0, "fps": 0}
    if not int(probe.get("width") or 0):
        # audio-only upload: build an audiogram so layouts/captions still render
        report(0, 10, "Building audiogram")
        src = clipper.ensure_video(src, out_dir)
        probe = media.video_meta(src)
    duration = float(probe.get("duration") or 0)
    if duration < 15:
        raise clipper.ClipperError("That video is shorter than 15 seconds — nothing to clip.")
    report(0, 15, "Video ready")

    # 2. audio + transcript -------------------------------------------------
    report(1, 17, "Extracting audio")
    audio = media.extract_audio(src, out_dir / "audio.wav")
    language = settings["language"] if settings["language"] != "auto" else None
    report(1, 22, "Transcribing with Whisper")
    transcript = transcribe_audio(
        audio,
        language=language,
        on_progress=lambda p: report(1, 22 + p * 0.18, "Transcribing with Whisper"),
    )
    transcript["duration"] = round(float(transcript.get("duration") or duration), 3)
    speakers = clipper.diarize(src, transcript["segments"])
    transcript["speakers"] = speakers["speakers"]
    transcript["speaker_count"] = speakers["count"]
    transcript["speaker_method"] = speakers["method"]
    # The transcript is all that is needed from here on; keeping the model
    # resident starves the scene detector and the encoder of RAM on small hosts.
    release_model()
    report(1, 45, "Transcript ready")

    # 3. signals ------------------------------------------------------------
    report(2, 47, "Detecting scenes")
    scene_list = scenes.detect_scenes(src) or []
    report(2, 52, "Measuring audio energy")
    signals = clipper.audio_signals(src)
    signals["scenes"] = scene_list
    signals["speaker_count"] = speakers["count"]
    report(2, 60, "Signals analysed")

    # 4. AI moment selection ------------------------------------------------
    report(3, 62, "Selecting the strongest moments")
    speaker_labels = [speakers["speakers"].get(i, "Speaker 1") for i in range(len(transcript["segments"]))]
    candidates, selector = clipper.all_candidates(
        transcript, signals, speaker_labels, settings,
        on_progress=lambda p: report(3, 62 + p * 22, "Selecting the strongest moments"),
    )
    if not candidates:
        raise clipper.ClipperError(
            "No strong standalone moments were found in this video. "
            "Try a longer clip length or a different instruction."
        )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    report(3, 85, f"{len(candidates)} candidate moments")

    # 5. refine boundaries ---------------------------------------------------
    report(4, 87, "Refining clip boundaries")
    lo, hi = clipper.length_bounds(settings["clip_length"])
    moments: List[Dict[str, Any]] = []
    for rank, cand in enumerate(candidates, start=1):
        refined = clipper.refine_bounds(
            cand["start"], cand["end"],
            words=transcript["words"], segments=transcript["segments"],
            silences=signals.get("silences") or [], duration=duration, lo=lo, hi=hi,
        )
        clip_text = clipper._clip_text_for(
            [{"start": s["start"], "end": s["end"], "text": s.get("text", "")}
             for s in transcript["segments"]],
            refined["start"], refined["end"],
        ) or cand.get("transcript", "")
        moment = {
            "id": f"m_{uuid.uuid4().hex[:10]}",
            "rank": rank,
            "title": cand.get("title") or "",
            "start": refined["start"],
            "end": refined["end"],
            "start_label": format_timestamp(refined["start"]),
            "end_label": format_timestamp(refined["end"]),
            "duration": round(refined["end"] - refined["start"], 2),
            "score": int(cand.get("score", 0)),
            "score_breakdown": cand.get("score_breakdown") or {},
            "hook_text": cand.get("hook_text") or "",
            "hook_type": cand.get("hook_type") or "",
            "why_it_works": cand.get("why_it_works") or "",
            "cold_open": cand.get("cold_open"),
            "suggested_caption": cand.get("suggested_caption") or "",
            "hashtags": cand.get("hashtags") or [],
            "best_platforms": cand.get("best_platforms") or ["shorts", "tiktok", "reels"],
            "risk_flags": cand.get("risk_flags") or [],
            "transcript": clip_text.strip(),
            "reasons": cand.get("reasons") or [],
            "speaker": cand.get("speaker") or "Speaker 1",
            "stats": cand.get("stats") or {},
            "layout": settings["layout"],
            "keep": refined["keep"],
        }
        moment["cold_open"] = _pick_cold_open(moment, transcript["words"], transcript["segments"])
        moments.append(moment)
    report(4, 92, "Boundaries snapped to words")

    # 6. metadata ------------------------------------------------------------
    report(5, 93, "Writing titles, hooks & metadata")
    for index, moment in enumerate(moments):
        info = clipper.generate_metadata(moment, title)
        moment.update({k: info.get(k, moment.get(k)) for k in
                       ("title", "hook_text", "hook_type", "why_it_works",
                        "suggested_caption", "hashtags")})
        if settings["clip_kind"] == "hook":
            try:
                moment["hooks"] = generate_hooks(
                    {"transcript": moment["transcript"], "duration": moment["duration"]},
                    title, index + 1,
                )
            except Exception:
                moment["hooks"] = None
        report(5, 93 + (index + 1) / max(1, len(moments)) * 4, "Writing titles, hooks & metadata")

    # 7. previews ------------------------------------------------------------
    report(6, 97, "Rendering previews")
    style = settings["style"]
    for index, moment in enumerate(moments):
        preview_name = f"preview_{index + 1:02d}"
        try:
            path = clipper.render_moment(
                src, moment, out_dir, name=preview_name, height=PREVIEW_HEIGHT,
                style=style, transcript=transcript, captions=True,
                preset="ultrafast", crf=30,
                on_progress=lambda p, i=index: report(
                    6, 97 + ((i + p) / max(1, len(moments))) * 3, "Rendering previews"),
            )
            moment["preview"] = path.name
            poster = media.make_thumbnail(path, out_dir / f"{preview_name}.jpg", at_sec=0.4)
            moment["poster"] = poster.name
        except Exception as exc:                     # a preview failure must not kill the job
            moment["preview"] = ""
            moment["preview_error"] = str(exc)[:300]

    thumbnail = meta.get("thumbnail") or ""
    if not thumbnail:
        try:
            poster = media.make_thumbnail(src, out_dir / "source.jpg", at_sec=min(3.0, duration / 3))
            thumbnail = store_url(poster)
        except Exception:
            thumbnail = ""

    doc = {
        "id": video_id,
        "task_id": task_id,
        "source": source,
        "source_type": source_type,
        "title": title,
        "channel": meta.get("channel", ""),
        "platform": meta.get("platform", clipper.detect_platform(source)["key"] if source_type == "url" else "upload"),
        "duration": round(duration, 2),
        "width": probe.get("width"),
        "height": probe.get("height"),
        "fps": probe.get("fps"),
        "thumbnail": thumbnail,
        "language": transcript.get("language", ""),
        "speaker_count": speakers["count"],
        "speaker_method": speakers["method"],
        "selector": selector,
        "settings": settings,
        "signals": {
            "scene_count": len(scene_list),
            "silence_count": len(signals.get("silences") or []),
            "peak_count": len(signals.get("peaks") or []),
            "has_music": bool(signals.get("has_music")),
        },
        "local_path": str(src),
        "created_at": time.time(),
        "moments": moments,
    }
    store.write_clip_transcript(video_id, transcript)
    store.write_clip(doc)
    report(6, 100, "Clips ready")
    return {"video_id": video_id, "title": title, "moments": len(moments),
            "selector": selector, "speaker_count": speakers["count"]}


def store_url(path: Path | str) -> str:
    from ..routes.common import media_url
    return media_url(path)


def rehydrate_moment_urls(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Add playable URLs to every moment (``/media/<task>/preview_01.mp4``)."""
    from ..routes.common import media_url
    task_id = doc.get("task_id")
    out: List[Dict[str, Any]] = []
    for moment in doc.get("moments") or []:
        item = dict(moment)
        if task_id and moment.get("preview"):
            base = Path(MEDIA_DIR) / task_id
            item["preview_url"] = media_url(base / moment["preview"])
            if moment.get("poster"):
                item["poster_url"] = media_url(base / moment["poster"])
        out.append(item)
    return out
