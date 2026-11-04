"""AI clipping engine.

Pipeline (Part 2 of the AI Clipping spec)::

    fetch/prepare -> audio + transcript -> signals -> AI moment selection
    -> boundary refinement -> metadata -> previews

Design notes
------------
* Everything works with **no API keys**. When no LLM is configured the moment
  selection, hooks, metadata and Find Moments all run on the deterministic
  writers in this module. With ``WV_LLM_API_KEY`` set, the two-pass LLM
  selection from Part 3 takes over and the local scorer becomes the fallback
  for anything the model gets wrong.
* Speaker labels come from a real (if simple) pitch-based diarisation: the
  median fundamental frequency of each transcript segment is clustered into
  two groups with k-means; when the groups are not separable everything is
  "Speaker 1" instead of inventing labels.
* Rendering uses FFmpeg only. Long internal silences (> 1.0 s) are removed by
  cutting the moment into keep-intervals and concatenating them, with the
  caption cues re-timed per interval, so audio and subtitles stay in sync.
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import FFMPEG_BIN, MEDIA_DIR, WORK_DIR
from . import media, scenes
from .ass import generate_ass, hex_to_ass, ass_time
from .llm import LLMUnavailable, complete_json, configured as llm_configured
from .transcribe import transcribe_audio

# ---------------------------------------------------------------------------
# Errors / constants
# ---------------------------------------------------------------------------
class ClipperError(RuntimeError):
    """User-facing failure (bad link, unsupported file, ...)."""


CLIP_LENGTHS: Dict[str, Dict[str, Any]] = {
    "auto":    {"label": "Auto (<90s)", "min": 15.0, "max": 90.0},
    "under30": {"label": "<30s",       "min": 10.0, "max": 30.0},
    "30to60":  {"label": "30-60s",     "min": 30.0, "max": 60.0},
    "60to90":  {"label": "60-90s",     "min": 60.0, "max": 90.0},
    "90to180": {"label": "90-180s",    "min": 90.0, "max": 180.0},
}

PLATFORMS_UI = ["tiktok", "shorts", "reels", "all"]
PLATFORM_LABEL = {
    "tiktok": "TikTok", "shorts": "YouTube Shorts", "reels": "Instagram Reels",
    "all": "All platforms",
}

# platform detection -------------------------------------------------------
PLATFORM_HOSTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("youtube",   "YouTube",        ("youtube.com", "youtu.be", "youtube-nocookie.com")),
    ("tiktok",    "TikTok",         ("tiktok.com",)),
    ("instagram", "Instagram",      ("instagram.com",)),
    ("facebook",  "Facebook",       ("facebook.com", "fb.watch")),
    ("x",         "X / Twitter",    ("x.com", "twitter.com")),
    ("twitch",    "Twitch",         ("twitch.tv",)),
    ("kick",      "Kick",           ("kick.com",)),
    ("rumble",    "Rumble",         ("rumble.com",)),
    ("vimeo",     "Vimeo",          ("vimeo.com",)),
    ("zoom",      "Zoom recording", ("zoom.us", "zoom.com")),
    ("podcast",   "Podcast",        ("podcasts.apple.com", "open.spotify.com", "anchor.fm",
                                     "buzzsprout.com", "podbean.com", "simplecast.com")),
]
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}

LANGUAGES = [
    ("auto", "Auto / No translation"), ("en", "English"), ("hi", "Hindi"),
    ("es", "Spanish"), ("fr", "French"), ("de", "German"), ("pt", "Portuguese"),
    ("it", "Italian"), ("nl", "Dutch"), ("ru", "Russian"), ("tr", "Turkish"),
    ("ar", "Arabic"), ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Chinese"),
    ("id", "Indonesian"), ("vi", "Vietnamese"), ("th", "Thai"), ("pl", "Polish"),
    ("uk", "Ukrainian"), ("bn", "Bengali"), ("ta", "Tamil"), ("te", "Telugu"),
    ("mr", "Marathi"), ("ur", "Urdu"),
]


def _host(url: str) -> str:
    m = re.match(r"^https?://([^/:?#]+)", url.strip(), re.I)
    return (m.group(1) if m else "").lower().replace("www.", "")


def detect_platform(url: str) -> Dict[str, str]:
    """Return ``{key, label}`` for a link; direct media files get ``direct``."""
    url = (url or "").strip()
    host = _host(url)
    for key, label, hosts in PLATFORM_HOSTS:
        if any(host == h or host.endswith("." + h) for h in hosts):
            return {"key": key, "label": label}
    path = re.sub(r"[?#].*$", "", url).lower()
    if any(path.endswith(ext) for ext in VIDEO_EXTS | AUDIO_EXTS):
        return {"key": "direct", "label": "Direct media link"}
    return {"key": "unknown", "label": "Web link"}


def validate_url(url: str) -> str:
    """Normalise and validate a pasted link, raising a helpful message."""
    url = (url or "").strip()
    if not url:
        raise ClipperError("Paste a video link first.")
    if not re.match(r"^https?://", url, re.I):
        raise ClipperError("That does not look like a link — it must start with http:// or https://")
    if len(url) > 2048:
        raise ClipperError("That link is too long.")
    if not _host(url):
        raise ClipperError("That link has no valid host.")
    return url


def inspect_url(url: str) -> Dict[str, Any]:
    """Metadata preview card for a pasted link (no download)."""
    url = validate_url(url)
    platform = detect_platform(url)
    if platform["key"] == "unknown":
        raise ClipperError(
            "This host is not in the supported list. Supported: YouTube, TikTok, "
            "Instagram, Facebook, X, Twitch, Kick, Rumble, Vimeo, Zoom recordings, "
            "podcast pages and direct video links."
        )
    try:
        meta = media.fetch_metadata(url)
    except Exception as exc:  # yt-dlp raises many types
        message = str(exc).splitlines()[0] if str(exc) else repr(exc)
        raise ClipperError(
            f"Could not read that video: {message}. Check the link is public and playable."
        ) from exc
    duration = float(meta.get("duration") or 0)
    if duration and duration < 15:
        raise ClipperError("That video is shorter than 15 seconds — nothing to clip.")
    return {
        "url": url,
        "source_type": "url",
        "platform": platform["key"],
        "platform_label": platform["label"],
        "title": meta.get("title") or "Untitled video",
        "duration": round(duration, 2),
        "duration_label": format_clock(duration),
        "thumbnail": meta.get("thumbnail") or "",
        "channel": meta.get("uploader") or "",
        "video_id": meta.get("id") or "",
    }


def format_clock(sec: float) -> str:
    sec = max(0, float(sec or 0))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return (f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}")


# ---------------------------------------------------------------------------
# Audio signals
# ---------------------------------------------------------------------------
def _pcm_mono(src: Path | str, sr: int = 8000) -> np.ndarray:
    """Decode the audio track to a mono float32 array at *sr*."""
    cmd = [FFMPEG_BIN, "-v", "error", "-i", str(src), "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=3600)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    raw = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    return raw / 32768.0


def audio_signals(src: Path | str, *, sr: int = 8000, win_sec: float = 0.25) -> Dict[str, Any]:
    """RMS envelope -> silence map, energy peaks and a crude music indicator."""
    pcm = _pcm_mono(src, sr)
    if pcm.size == 0:
        return {"silences": [], "peaks": [], "profile": [], "has_music": False}
    win = max(1, int(sr * win_sec))
    count = pcm.size // win
    if count == 0:
        return {"silences": [], "peaks": [], "profile": [], "has_music": False}
    frames = pcm[: count * win].reshape(count, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    times = np.arange(count) * win_sec

    floor = float(np.percentile(rms, 20))
    loud = float(np.percentile(rms, 95)) or 1e-6
    gate = max(floor * 2.2, loud * 0.08, 0.004)

    # silence runs >= 0.7s
    silences: List[Dict[str, float]] = []
    in_sil = False
    start = 0.0
    for t, value in zip(times, rms):
        quiet = value < gate
        if quiet and not in_sil:
            in_sil, start = True, float(t)
        elif not quiet and in_sil:
            in_sil = False
            if t - start >= 0.7:
                silences.append({"start": round(start, 2), "end": round(float(t), 2),
                                 "dur": round(float(t - start), 2)})
    if in_sil and times[-1] - start >= 0.7:
        silences.append({"start": round(start, 2), "end": round(float(times[-1] + win_sec), 2),
                         "dur": round(float(times[-1] + win_sec - start), 2)})

    # energy peaks: windows clearly above the running average
    mean = float(np.mean(rms)) or 1e-6
    peaks: List[float] = []
    for i in range(2, count - 2):
        if rms[i] > mean * 1.7 and rms[i] >= rms[i - 2:i + 3].max() - 1e-9:
            peaks.append(round(float(times[i]), 2))

    profile = [round(float(v), 4) for v in rms[:: max(1, count // 600)]]
    return {
        "silences": silences,
        "peaks": peaks[:400],
        "profile": profile,
        "has_music": bool(np.mean(rms > gate) > 0.82 and len(silences) < 3),
    }


# ---------------------------------------------------------------------------
# Speaker diarisation (pitch clustering)
# ---------------------------------------------------------------------------
def _segment_pitch(pcm: np.ndarray, sr: int, start: float, end: float) -> Optional[float]:
    a, b = int(start * sr), int(end * sr)
    if b - a < sr // 4:
        return None
    chunk = pcm[a:b]
    if chunk.size > sr * 2:                       # cap work per segment
        chunk = chunk[: sr * 2]
    window = int(sr * 0.04)
    hop = int(sr * 0.02)
    if window >= chunk.size:
        return None
    best: List[float] = []
    for i in range(0, chunk.size - window, hop * 6):
        frame = chunk[i:i + window]
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))
        if energy < 1e-5:
            continue
        corr = np.correlate(frame, frame, mode="full")[window - 1:]
        lo, hi = int(sr / 350), int(sr / 70)
        if hi >= corr.size:
            continue
        lag = int(np.argmax(corr[lo:hi])) + lo
        if lag:
            best.append(sr / lag)
    if len(best) < 3:
        return None
    return float(np.median(best))


def diarize(src: Path | str, segments: Sequence[Dict[str, Any]], *, sr: int = 16000,
            max_segments: int = 320) -> Dict[str, Any]:
    """Assign Speaker 1/2 per segment via 2-means on log pitch.

    Returns ``{"speakers": {seg_index: label}, "count": n, "method": ...}``.
    When pitch is not separable every segment gets "Speaker 1" — no invented
    labels.
    """
    if not segments:
        return {"speakers": {}, "count": 0, "method": "none"}
    pcm = _pcm_mono(src, sr)
    if pcm.size == 0:
        return {"speakers": {i: "Speaker 1" for i in range(len(segments))},
                "count": 1, "method": "single"}
    step = max(1, len(segments) // max_segments)
    sampled = list(range(0, len(segments), step))[:max_segments]
    pitches: Dict[int, float] = {}
    for idx in sampled:
        seg = segments[idx]
        p = _segment_pitch(pcm, sr, float(seg["start"]), float(seg["end"]))
        if p and 60 <= p <= 380:
            pitches[idx] = math.log(p)

    if len(pitches) >= 8:
        values = np.array(list(pitches.values()), dtype=np.float32)
        lo, hi = np.percentile(values, 10), np.percentile(values, 90)
        if hi - lo > 0.20:                        # ~22% pitch gap == two voices
            # 1-D 2-means, deterministic init at the percentiles
            c1, c2 = float(lo), float(hi)
            for _ in range(12):
                a = values[np.abs(values - c1) <= np.abs(values - c2)]
                b = values[np.abs(values - c1) > np.abs(values - c2)]
                if a.size == 0 or b.size == 0:
                    break
                c1, c2 = float(a.mean()), float(b.mean())
            # Require a real bimodal split, not just any pitch spread: the
            # density between the two centres must be a valley, and both
            # clusters must be substantial. One narrator with expressive
            # intonation then stays "Speaker 1" instead of being split.
            histogram, edges = np.histogram(values, bins=18)

            def _density(x: float) -> float:
                idx = int(np.clip(np.searchsorted(edges, x) - 1, 0, len(histogram) - 1))
                return float(histogram[idx])

            valley = _density((c1 + c2) / 2)
            peaks = min(_density(c1), _density(c2))
            is_bimodal = peaks > 0 and valley < peaks * 0.6
            if (abs(c2 - c1) > 0.16 and is_bimodal
                    and min(a.size, b.size) / values.size > 0.2):
                low_label = 1 if c1 <= c2 else 2   # deeper voice == Speaker 1
                speakers: Dict[int, str] = {}
                for i, seg in enumerate(segments):
                    if i in pitches:
                        speakers[i] = f"Speaker {low_label if pitches[i] <= (c1 + c2) / 2 else 3 - low_label}"
                    else:
                        speakers[i] = "Speaker 1"
                return {"speakers": speakers, "count": 2, "method": "pitch-cluster"}

    return {"speakers": {i: "Speaker 1" for i in range(len(segments))},
            "count": 1, "method": "single"}


# ---------------------------------------------------------------------------
# Vocabulary for the deterministic scorer
# ---------------------------------------------------------------------------
HOOK_WORDS = {
    "secret", "never", "nobody", "stop", "mistake", "mistakes", "insane", "crazy",
    "actually", "truth", "lie", "free", "money", "easy", "hack", "trick", "proof",
    "warning", "finally", "best", "worst", "biggest", "impossible", "shocking",
    "problem", "fix", "instantly", "guaranteed", "wrong", "why", "how", "what",
    "imagine", "listen", "before", "after", "million", "billion", "dollar",
}
EMOTION_WORDS = {
    "love", "hate", "laughed", "laugh", "funny", "hilarious", "crazy", "insane",
    "amazing", "awful", "terrible", "scary", "afraid", "angry", "furious", "cried",
    "crying", "shocked", "shock", "wow", "unbelievable", "ridiculous", "beautiful",
    "painful", "hurt", "proud", "embarrassing", "genius", "stupid", "honestly",
    "literally", "damn", "omg", "whoa", "hard", "broke", "failed", "won",
}
TREND_WORDS = {
    "ai", "money", "business", "startup", "productivity", "discipline", "gym",
    "dating", "relationship", "mental", "health", "sleep", "food", "travel",
    "crypto", "career", "salary", "interview", "coding", "python", "javascript",
    "iphone", "android", "gaming", "football", "cricket", "study", "exam",
    "weight", "diet", "skin", "fashion", "music", "podcast", "reaction",
}
SENSITIVE_WORDS = {
    "kill", "suicide", "rape", "nazi", "terrorist", "murder", "abuse", "overdose",
    "slur", "bomb", "shooting",
}
CONTEXT_STARTERS = {"it", "this", "that", "he", "she", "they", "them", "those",
                    "these", "there", "then", "so", "and", "but", "because"}
GREETINGS = {"hey", "hi", "hello", "welcome", "guys", "everyone", "thanks",
             "thank", "um", "uh", "okay", "ok", "so", "yeah", "today"}
INTRO_PHRASES = ("welcome back", "we are talking about", "we're talking about",
                 "in this video", "let me tell you", "today we", "so today")
QUESTIONISH = re.compile(r"\?|^(what|why|how|who|when|where|do|does|did|can|could|"
                         r"would|should|is|are|was|were|have|has)\b", re.I)
NUMBER_RE = re.compile(r"\b\d+([.,]\d+)?%?\b")
SENTENCE_END = re.compile(r"[.!?…]['\")\]]?$")

ProgressFn = Callable[[int, float, str], None]


def _words_in(words: Sequence[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    return [w for w in words if start <= float(w.get("s", 0)) < end]


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9']+", (text or "").lower()) if t]


# ---------------------------------------------------------------------------
# Candidate windows + rubric scoring
# ---------------------------------------------------------------------------
def _windows(segments: Sequence[Dict[str, Any]], lo: float, hi: float) -> List[Tuple[float, float]]:
    """Candidate windows aligned to transcript segment boundaries."""
    if not segments:
        return []
    out: List[Tuple[float, float]] = []
    starts = [float(s["start"]) for s in segments]
    for i, s0 in enumerate(starts):
        for target in (lo, (lo + hi) / 2, hi):
            end = None
            for j in range(i, len(segments)):
                span = float(segments[j]["end"]) - s0
                if span > hi:
                    break
                if span >= max(lo * 0.85, target * 0.75):
                    end = float(segments[j]["end"])
                if span >= target:
                    break
            if end and end - s0 >= max(6.0, lo * 0.8):
                out.append((round(s0, 3), round(end, 3)))
    return sorted(set(out))


def _silence_cost(silences: Sequence[Dict[str, float]], start: float, end: float) -> Tuple[int, float]:
    hits = [s for s in silences if s["start"] >= start and s["end"] <= end and s["dur"] > 1.0]
    return len(hits), round(sum(s["dur"] for s in hits), 2)


def score_candidate(
    window: Tuple[float, float],
    *,
    segments: Sequence[Dict[str, Any]],
    words: Sequence[Dict[str, Any]],
    speakers: Sequence[str],
    duration: float,
    signals: Dict[str, Any],
    platform: str,
    lo: float,
    hi: float,
) -> Optional[Dict[str, Any]]:
    """Score one window on the 0-100 rubric from Part 3."""
    start, end = window
    span = max(1.0, end - start)
    if start < 0 or end > duration + 0.5 or span < 5:
        return None
    clip_words = _words_in(words, start, end)
    if len(clip_words) < 8:
        return None
    text = " ".join(w["w"] for w in clip_words)
    tokens = _tokens(text)
    if not tokens:
        return None

    first_seg = next((s for s in segments if float(s["end"]) > start), None)
    opening = " ".join(w["w"] for w in clip_words[:14])
    opening_tokens = _tokens(opening)
    opening_lower = [t.strip(".,!?;:\"'") for t in opening.split()][:3]
    wps = len(tokens) / span
    scene_hits = scenes.scene_density(signals.get("scenes"), start, end)
    sil_count, sil_total = _silence_cost(signals.get("silences") or [], start, end)
    energy_inside = sum(1 for t in (signals.get("peaks") or []) if start <= t <= end)
    has_end_punct = bool(SENTENCE_END.search(text.strip()[-1] if text.strip() else ""))

    # 1. hook strength (25) -------------------------------------------------
    hook = 6.0
    if QUESTIONISH.search(opening):
        hook += 7
    if NUMBER_RE.search(opening):
        hook += 4
    hook_hits = [t for t in opening_tokens if t in HOOK_WORDS]
    hook += min(6.0, 2.0 * len(hook_hits))
    if any(t in {"you", "your"} for t in opening_tokens):
        hook += 3
    if opening_tokens and opening_tokens[0] in HOOK_WORDS:
        hook += 3
    if opening_lower and opening_lower[0].strip(".,!?") in GREETINGS:
        hook -= 6
    if any(phrase in opening.lower() for phrase in INTRO_PHRASES):
        hook -= 5
    hook = max(0.0, min(25.0, hook))

    # 2. pacing / retention (20) -------------------------------------------
    if 2.4 <= wps <= 4.8:
        pace = 18.0
    elif wps > 4.8:
        pace = 15.0
    elif wps >= 1.8:
        pace = 12.0
    else:
        pace = 6.0
    pace += min(4.0, scene_hits * 0.8)
    pace -= min(10.0, sil_count * 4.0)
    pace = max(0.0, min(20.0, pace))

    # 3. emotional peak / payoff (20) --------------------------------------
    emo = [t for t in tokens if t in EMOTION_WORDS]
    emotion = 5.0 + min(10.0, 2.5 * len(emo))
    if "!" in text:
        emotion += 3
    emotion += min(4.0, energy_inside * 1.2)
    if has_end_punct:
        emotion += 3
    emotion = max(0.0, min(20.0, emotion))

    # 4. standalone clarity (15) -------------------------------------------
    clarity = 8.0
    if has_end_punct:
        clarity += 4
    if first_seg and str(first_seg.get("text", "")).strip()[:1].isupper():
        clarity += 2
    if opening_tokens and opening_tokens[0] in CONTEXT_STARTERS:
        clarity -= 4
    if span < lo * 0.75 or span > hi * 1.05:
        clarity -= 3
    clarity = max(0.0, min(15.0, clarity))

    # 5. shareability / saveability (10) -----------------------------------
    share = 3.0
    if any(t in {"you", "your"} for t in tokens):
        share += 2.5
    if NUMBER_RE.search(text):
        share += 1.5
    if any(t in HOOK_WORDS for t in tokens):
        share += 1.5
    if re.search(r"\b(how to|the reason|because|steps?|tip|rule)\b", text, re.I):
        share += 1.5
    share = max(0.0, min(10.0, share))

    # 6. trend / topic fit (10) --------------------------------------------
    trend_hits = [t for t in tokens if t in TREND_WORDS]
    trend = 4.0 + min(6.0, 1.5 * len(set(trend_hits)))
    trend = max(0.0, min(10.0, trend))

    score = hook + pace + emotion + clarity + share + trend

    # penalties ------------------------------------------------------------
    reasons: List[str] = []
    risks: List[str] = []
    if hook < 12:
        score -= 10
        reasons.append("weak opening")
    if opening_tokens and opening_tokens[0] in CONTEXT_STARTERS and not any(
        t[0].isupper() for t in text.split()[:12] if t and t[0].isalpha()
    ):
        score -= 10
        reasons.append("needs prior context")
    if sil_count:
        reasons.append(f"{sil_count} long silence(s)")
    if not has_end_punct:
        score -= 15
        reasons.append("unfinished thought")
    sensitive = [t for t in tokens if t in SENSITIVE_WORDS]
    if sensitive:
        score -= 20
        risks.append("sensitive_topic")
    if signals.get("has_music"):
        risks.append("music_detected")

    score = int(max(0.0, min(100.0, round(score))))

    if hook >= 18:
        reasons.insert(0, "strong opening line")
    if emotion >= 14:
        reasons.insert(0, "emotional peak")
    if share >= 7:
        reasons.append("high share potential")
    if 2.4 <= wps <= 4.8:
        reasons.append("tight pacing")

    best_platforms = (["tiktok", "shorts", "reels"] if span <= 45
                      else ["shorts", "reels", "tiktok"] if span <= 90
                      else ["shorts", "reels"])
    if platform in ("tiktok", "shorts", "reels"):
        best_platforms = [platform] + [p for p in best_platforms if p != platform]

    clip_text = text.strip()
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "score": score,
        "score_breakdown": {
            "hook": round(hook, 1), "pacing": round(pace, 1), "emotion": round(emotion, 1),
            "clarity": round(clarity, 1), "shareability": round(share, 1), "trend": round(trend, 1),
        },
        "reasons": reasons[:4],
        "risk_flags": sorted(set(risks)),
        "best_platforms": best_platforms,
        "transcript": clip_text,
        "speaker": speakers[first_seg["id"]] if first_seg and first_seg.get("id") is not None
                   and first_seg["id"] < len(speakers) else "Speaker 1",
        "stats": {"wps": round(wps, 2), "scenes": scene_hits, "silences": sil_count,
                  "silence_sec": sil_total, "energy_peaks": energy_inside},
    }


def local_candidates(
    segments: Sequence[Dict[str, Any]],
    words: Sequence[Dict[str, Any]],
    speakers: Sequence[str],
    duration: float,
    signals: Dict[str, Any],
    *,
    lo: float,
    hi: float,
    platform: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for window in _windows(segments, lo, hi):
        scored = score_candidate(window, segments=segments, words=words, speakers=speakers,
                                 duration=duration, signals=signals, platform=platform,
                                 lo=lo, hi=hi)
        if scored:
            out.append(scored)
    return out


def pick_non_overlapping(candidates: Sequence[Dict[str, Any]], count: int,
                         *, max_overlap: float = 2.0) -> List[Dict[str, Any]]:
    """Greedy pick by score, dropping anything that shares > *max_overlap* seconds."""
    chosen: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: c["score"], reverse=True):
        clash = False
        for other in chosen:
            overlap = min(cand["end"], other["end"]) - max(cand["start"], other["start"])
            if overlap > max_overlap:
                clash = True
                break
        if not clash:
            chosen.append(cand)
        if len(chosen) >= count:
            break
    return chosen


# ---------------------------------------------------------------------------
# Prompt from Part 3
# ---------------------------------------------------------------------------
SELECTION_SYSTEM = """You are an expert short-form video editor and growth strategist for TikTok, YouTube Shorts and Instagram Reels. You receive a timestamped, speaker-labelled transcript plus signal data (audio energy peaks, laughter, silences, scene changes, speech pace). Find the moments most likely to hold attention and get shared, and return them as ranked clips.

WHAT THE PLATFORMS REWARD (use this to judge every moment)
- Watch time and completion rate are the heaviest signals, so favor moments that hold attention until the end and are short enough to finish.
- Rewatches are a very strong positive signal, so favor moments with a payoff or twist worth watching twice, and clips that loop naturally.
- Shares (especially sends to friends) and saves outweigh likes, so favor moments people want to send to someone ("this is so you", "you have to see this") or save (useful, surprising, quotable).
- Comments matter, so favor moments that provoke opinions, debate or reactions.
- Early drop-off and swipe-aways hurt reach, so the opening must create curiosity or tension immediately.

SCORING RUBRIC (0-100 total)
1. Hook strength, 25 points: does the first 1-3 seconds create a question, bold claim, contradiction, shock, or high stakes? Penalize slow intros, greetings, throat-clearing and context-dependent openings.
2. Retention and pacing, 20 points: constant new information or tension, fast pace, no dead air, no long tangents.
3. Emotional peak and payoff, 20 points: genuine laughter, surprise, anger, awe, vulnerability, or a clear insight with a satisfying payoff inside the clip.
4. Standalone clarity, 15 points: a viewer with zero context understands it; it has a clear beginning, middle and end; it does not depend on earlier or later parts.
5. Shareability and saveability, 10 points: relatable, quotable, controversial-but-safe, useful, or "send this to a friend" energy.
6. Trend and topic fit, 10 points: relevance to current popular themes, formats or audience interests for the target platform.
Penalties: -10 for a weak or slow opening, -10 for needing prior context, -5 for each silence longer than 1 second, -15 for an unfinished thought or a cut-off ending, -20 for sensitive content that could get the clip removed (hate, graphic harm, private personal data), and flag any music or copyrighted material heard in the audio.

SELECTION RULES
- Each clip must be self-contained, start at the strongest line (not at the beginning of the story if the best line comes later), and end right after the payoff or on a loopable beat.
- Respect the requested clip length: {clip_length}. Prefer the shortest version that keeps the full payoff (many top-performing clips finish under 30 seconds).
- Return about {n_clips} clips (about 3-5 strong moments per hour for general content, 8-12 for dense content such as podcasts, debates and reaction streams). If there are fewer genuinely strong moments, return fewer instead of padding with weak ones.
- No overlaps: two clips must not share more than 2 seconds.
- Follow the user's instruction if given: "{user_instruction}".
- Adapt to content type (podcast, gaming, comedy, education, interview, vlog, music) and to the target platform ({platform}).
- COLD OPEN OPTION: if a stronger line happens later in the same moment, give it as cold_open (2-5 seconds, not the full payoff) to be placed at the very start, followed by the normal beginning.
- Never invent words. Use only the transcript. Never change meaning.

OUTPUT: valid JSON only, an array of objects:
{{
 "rank": 1,
 "title": "max 70 chars, curiosity-driven, true to the clip",
 "start": "HH:MM:SS.mmm", "end": "HH:MM:SS.mmm",
 "score": 0-100,
 "score_breakdown": {{"hook":0,"pacing":0,"emotion":0,"clarity":0,"shareability":0,"trend":0}},
 "hook_text": "on-screen text, max 12 words",
 "hook_type": "curiosity gap | bold claim | pattern interrupt | contrarian | story cold-open | relatable pain | visual shock",
 "why_it_works": "one sentence, caption-ready, max 160 chars",
 "cold_open": {{"start":"...","end":"..."}} or null,
 "suggested_caption": "...",
 "hashtags": ["...", "..."],
 "best_platforms": ["tiktok","shorts","reels"],
 "risk_flags": ["music_detected","sensitive_topic"] or []
}}
Sort by score descending."""

RESCORE_SYSTEM = """You are the same short-form growth strategist doing a second pass over candidate clips.
You receive a JSON array of candidate moments with their transcripts, rubric scores and signal stats.
Do all of this and reply with JSON only:
1. Drop candidates that are weak, unfinished, context-dependent or duplicated.
2. Remove overlaps: two clips may not share more than 2 seconds - keep the stronger one and trim the other.
3. Re-score every survivor with the rubric (hook 25, pacing 20, emotion 20, clarity 15, shareability 10, trend 10) and the penalties.
4. Rank the survivors by score, best first, and renumber rank from 1.
5. Keep each clip inside the requested length, never cutting mid-word.

Reply with: {"clips": [{"title","start","end","score","score_breakdown","hook_text","hook_type","why_it_works","cold_open","suggested_caption","hashtags","best_platforms","risk_flags","rank"}]}
Times are absolute seconds from the start of the source video. Use the transcript words only."""

Hooks_SYSTEM = """You are a short-form hook specialist. Given one clip transcript, write 5 alternative on-screen hooks for A/B testing.
Each hook is under 12 words, true to the clip (no claim the clip cannot pay off), creates a question or tension the viewer can only resolve by watching, and lands in the first 1-2 seconds.
Use a different pattern for each: curiosity gap, bold claim, pattern interrupt, contrarian take, story cold-open.
Reply with JSON only: {"hooks":[{"text":"...","pattern":"...","why":"one line"}]}"""

FIND_SYSTEM = """You are a transcript search engine for a short-form clipping tool.
The user describes the moments they want in natural language. Search the timestamped transcript and return the best matching moments.
A moment must be self-contained, 10-90 seconds long, start on the matching line and end on a payoff.
Reply with JSON only: {"moments":[{"start":seconds,"end":seconds,"title":"max 70 chars","why":"why it matches, max 160 chars","score":0-100}]}
Never invent words or timestamps that are not in the transcript. Return at most 6 moments."""


def _hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _parse_hms(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] if parts else None


def _prompt_transcript(lines: Sequence[Dict[str, Any]], start: float, end: float,
                       *, max_chars: int = 14000) -> str:
    out: List[str] = []
    size = 0
    for line in lines:
        if float(line["end"]) < start or float(line["start"]) > end:
            continue
        text = f"[{_hms(line['start'])}] {line['speaker']}: {line['text']}"
        size += len(text)
        if size > max_chars:
            break
        out.append(text)
    return "\n".join(out)


def _signal_summary(signals: Dict[str, Any], start: float, end: float) -> str:
    sil = [s for s in (signals.get("silences") or []) if s["start"] >= start and s["end"] <= end]
    peaks = [t for t in (signals.get("peaks") or []) if start <= t <= end]
    cuts = scenes.scene_density(signals.get("scenes"), start, end)
    return (f"{len(sil)} silences >0.7s, {len(peaks)} energy peaks, {cuts} scene changes, "
            f"music-like audio: {bool(signals.get('has_music'))}")


def _chunks(lines: Sequence[Dict[str, Any]], span: float = 300.0, overlap: float = 45.0) -> List[Tuple[float, float]]:
    if not lines:
        return []
    end = float(lines[-1]["end"])
    out: List[Tuple[float, float]] = []
    start = 0.0
    while start < end:
        out.append((start, min(end, start + span)))
        if start + span >= end:
            break
        start += span - overlap
    return out


def _clip_text_for(lines: Sequence[Dict[str, Any]], start: float, end: float) -> str:
    parts = [line["text"] for line in lines if float(line["end"]) > start and float(line["start"]) < end]
    return " ".join(p.strip() for p in parts).strip()


def all_candidates(
    transcript: Dict[str, Any],
    signals: Dict[str, Any],
    speakers: Sequence[str],
    settings: Dict[str, Any],
    *,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Two-pass AI selection; returns ``(candidates, source)``."""
    segments = transcript["segments"]
    words = transcript["words"]
    duration = float(transcript.get("duration") or segments[-1]["end"])
    lo, hi = length_bounds(settings.get("clip_length", "auto"))
    platform = settings.get("platform", "all")
    count = int(settings.get("count") or 6)
    instruction = (settings.get("instruction") or "").strip()
    meta = {"source": "local"}
    lines = [
        {"start": float(s["start"]), "end": float(s["end"]),
         "text": (s.get("text") or "").strip(),
         "speaker": speakers[i] if i < len(speakers) else "Speaker 1"}
        for i, s in enumerate(segments)
    ]

    if not llm_configured():
        cands = local_candidates(segments, words, speakers, duration, signals,
                                 lo=lo, hi=hi, platform=platform)
        return pick_non_overlapping(cands, count), "local"

    system = SELECTION_SYSTEM.format(
        clip_length=f"{int(lo)}-{int(hi)} seconds",
        n_clips=count,
        user_instruction=instruction or "none",
        platform=PLATFORM_LABEL.get(platform, "all platforms"),
    )

    raw: List[Dict[str, Any]] = []
    chunks = _chunks(lines)
    for index, (c0, c1) in enumerate(chunks):
        chunk_text = _prompt_transcript(lines, c0, c1)
        if not chunk_text.strip():
            continue
        user = (
            f"Source duration: {_hms(duration)}\n"
            f"Transcript window: {_hms(c0)} to {_hms(c1)}\n"
            f"Signals in this window: {_signal_summary(signals, c0, c1)}\n"
            f"Speaker count: {len(set(speakers))}\n\n"
            f"Timestamped transcript (absolute times):\n{chunk_text}"
        )
        try:
            data = complete_json(system, user, temperature=0.4, max_tokens=3000)
        except LLMUnavailable:
            break
        except Exception:
            continue
        items = data.get("clips") if isinstance(data, dict) else data
        if isinstance(items, list):
            raw.extend(x for x in items if isinstance(x, dict))
        if on_progress:
            on_progress((index + 1) / max(1, len(chunks)))

    cleaned = [c for c in (_from_llm(x, lines, duration, lo, hi) for x in raw) if c]
    if not cleaned:
        cands = local_candidates(segments, words, speakers, duration, signals,
                                 lo=lo, hi=hi, platform=platform)
        return pick_non_overlapping(cands, count), "local"

    # pass 2 — re-score everything together
    payload = [
        {"rank": c["rank"], "title": c["title"], "start": round(c["start"], 2),
         "end": round(c["end"], 2), "score": c["score"],
         "breakdown": c["score_breakdown"], "transcript": c["transcript"][:1200]}
        for c in cleaned
    ]
    try:
        rescored = complete_json(
            RESCORE_SYSTEM,
            "Requested clip length: {}-{} seconds. Target platform: {}.\n"
            "User instruction: {}\nCandidates JSON:\n{}".format(
                int(lo), int(hi), PLATFORM_LABEL.get(platform, "all"),
                instruction or "none", _json(payload)),
            temperature=0.3, max_tokens=4000,
        )
        items = rescored.get("clips") if isinstance(rescored, dict) else rescored
        if isinstance(items, list) and items:
            refined = [c for c in (_merge_llm(old, new, lo, hi) for old, new in
                                   zip(sorted(cleaned, key=lambda c: c["score"], reverse=True), items)) if c]
            if refined:
                meta["source"] = "llm"
                return pick_non_overlapping(refined, count), "llm"
    except LLMUnavailable:
        pass
    except Exception:
        pass

    meta["source"] = "llm"
    cleaned.sort(key=lambda c: c["score"], reverse=True)
    return pick_non_overlapping(cleaned, count), "llm"


def _json(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False)


def length_bounds(key: str) -> Tuple[float, float]:
    item = CLIP_LENGTHS.get(key) or CLIP_LENGTHS["auto"]
    return float(item["min"]), float(item["max"])


def _from_llm(raw: Dict[str, Any], lines: Sequence[Dict[str, Any]], duration: float,
              lo: float, hi: float) -> Optional[Dict[str, Any]]:
    start = _parse_hms(raw.get("start"))
    end = _parse_hms(raw.get("end"))
    if start is None or end is None or end <= start:
        return None
    if start < 0 or end > duration + 1:
        return None
    if end - start < 5:
        return None
    # never let the model exceed the requested length
    if end - start > hi * 1.25:
        end = start + hi
    text = _clip_text_for(lines, start, end)
    if not text.strip():
        return None
    breakdown = raw.get("score_breakdown") or {}
    try:
        score = int(round(float(raw.get("score") or 0)))
    except (TypeError, ValueError):
        score = 60
    cold = raw.get("cold_open") or None
    cold_open = None
    if isinstance(cold, dict):
        cs, ce = _parse_hms(cold.get("start")), _parse_hms(cold.get("end"))
        if cs is not None and ce is not None and 1.0 <= ce - cs <= 6.0 and cs >= 0:
            cold_open = {"start": round(cs, 3), "end": round(ce, 3)}
    return {
        "start": round(start, 3), "end": round(end, 3),
        "score": max(0, min(100, score)),
        "score_breakdown": {
            "hook": float(breakdown.get("hook") or 0), "pacing": float(breakdown.get("pacing") or 0),
            "emotion": float(breakdown.get("emotion") or 0), "clarity": float(breakdown.get("clarity") or 0),
            "shareability": float(breakdown.get("shareability") or 0),
            "trend": float(breakdown.get("trend") or 0),
        },
        "title": str(raw.get("title") or "").strip()[:70],
        "hook_text": str(raw.get("hook_text") or "").strip()[:120],
        "hook_type": str(raw.get("hook_type") or "").strip()[:40],
        "why_it_works": str(raw.get("why_it_works") or "").strip()[:160],
        "cold_open": cold_open,
        "suggested_caption": str(raw.get("suggested_caption") or "").strip()[:300],
        "hashtags": [str(h).strip() for h in (raw.get("hashtags") or []) if str(h).strip()][:10],
        "best_platforms": [str(p).lower() for p in (raw.get("best_platforms") or [])][:4] or ["shorts", "tiktok", "reels"],
        "risk_flags": [str(r) for r in (raw.get("risk_flags") or [])][:4],
        "transcript": text.strip(),
        "reasons": [],
        "rank": 0,
        "speaker": "Speaker 1",
    }


def _merge_llm(old: Dict[str, Any], new: Dict[str, Any], lo: float, hi: float) -> Optional[Dict[str, Any]]:
    if not isinstance(new, dict):
        return old
    merged = dict(old)
    start = _parse_hms(new.get("start"))
    end = _parse_hms(new.get("end"))
    if start is not None and end is not None and end - start >= 5 and end - start <= hi * 1.25:
        merged["start"], merged["end"] = round(start, 3), round(end, 3)
    try:
        merged["score"] = max(0, min(100, int(round(float(new.get("score") or old["score"])))))
    except (TypeError, ValueError):
        pass
    if isinstance(new.get("score_breakdown"), dict):
        merged["score_breakdown"] = {k: float((new["score_breakdown"] or {}).get(k) or v)
                                     for k, v in old["score_breakdown"].items()}
    for key in ("title", "hook_text", "hook_type", "why_it_works", "suggested_caption"):
        if new.get(key):
            merged[key] = str(new[key]).strip()[:300 if key == "suggested_caption" else 160]
    if isinstance(new.get("hashtags"), list):
        merged["hashtags"] = [str(h) for h in new["hashtags"] if str(h).strip()][:10]
    if isinstance(new.get("best_platforms"), list) and new["best_platforms"]:
        merged["best_platforms"] = [str(p).lower() for p in new["best_platforms"]][:4]
    if isinstance(new.get("risk_flags"), list):
        merged["risk_flags"] = [str(r) for r in new["risk_flags"]][:4]
    return merged


# ---------------------------------------------------------------------------
# Boundary refinement
# ---------------------------------------------------------------------------
def refine_bounds(
    start: float,
    end: float,
    *,
    words: Sequence[Dict[str, Any]],
    segments: Sequence[Dict[str, Any]],
    silences: Sequence[Dict[str, float]],
    duration: float,
    lo: float,
    hi: float,
    padding: float = 0.3,
) -> Dict[str, Any]:
    """Snap to whole words/sentences, pad, trim edge silence, cap the length."""
    start = max(0.0, float(start))
    end = min(duration, float(end))

    # never cut mid-word: snap start back to the containing word's start and
    # end forward to the containing word's end
    for w in words:
        ws, we = float(w["s"]), float(w["e"])
        if ws < start < we:
            start = ws
        if ws < end < we:
            end = we

    # snap to a sentence boundary when one is within 1.2s
    for seg in segments:
        seg_start, seg_end = float(seg["start"]), float(seg["end"])
        if 0 < start - seg_start <= 1.2:
            start = seg_start
        if 0 < seg_end - end <= 1.2:
            end = seg_end

    # trim leading / trailing silence, keeping a small natural pad
    for sil in silences:
        if sil["start"] <= start < sil["end"]:
            start = min(sil["end"] + 0.05, end - 2)
        if sil["start"] < end <= sil["end"]:
            end = max(sil["start"] - 0.05, start + 2)

    start = max(0.0, start - padding)
    end = min(duration, end + padding)

    # respect the requested length without landing mid-word
    if end - start > hi:
        end = start + hi
        for w in words:
            if float(w["s"]) < end < float(w["e"]):
                end = float(w["s"])
                break
    if end - start < min(lo, 6.0):
        end = min(duration, start + min(lo, 6.0))

    start, end = round(start, 3), round(end, 3)
    return {"start": start, "end": end, "keep": keep_intervals(start, end, silences)}


def keep_intervals(start: float, end: float, silences: Sequence[Dict[str, float]],
                   *, max_silence: float = 1.0, keep: float = 0.35) -> List[Dict[str, float]]:
    """Cut long internal silences: returns ``[{start,end}]`` pieces to concatenate."""
    cuts = [s for s in silences if s["start"] > start + 0.6 and s["end"] < end - 0.6
            and s["dur"] > max_silence]
    if not cuts:
        return [{"start": round(start, 3), "end": round(end, 3)}]
    pieces: List[Dict[str, float]] = []
    cursor = start
    for sil in sorted(cuts, key=lambda s: s["start"]):
        cut_start = sil["start"] + keep / 2
        cut_end = sil["end"] - keep / 2
        if cut_end - cut_start < 0.2:
            continue
        if cut_start - cursor > 0.4:
            pieces.append({"start": round(cursor, 3), "end": round(cut_start, 3)})
        cursor = cut_end
    if end - cursor > 0.4:
        pieces.append({"start": round(cursor, 3), "end": round(end, 3)})
    return pieces or [{"start": round(start, 3), "end": round(end, 3)}]


# ---------------------------------------------------------------------------
# Metadata / hooks / find (local writers)
# ---------------------------------------------------------------------------
STOPWORDS = {
    "this", "that", "with", "have", "your", "youre", "they", "their", "them",
    "what", "when", "where", "which", "who", "why", "how", "from", "just",
    "like", "about", "there", "here", "would", "could", "should", "really",
    "need", "want", "know", "going", "thing", "things", "people", "because",
    "then", "also", "been", "were", "will", "does", "doing", "make", "made",
    "much", "many", "some", "very", "only", "even", "back", "over", "into",
    "each", "more", "than", "today", "week", "year", "next", "last", "said",
    "says", "tell", "told", "anyone", "something", "everything", "nothing",
    "because", "actually", "literally", "really", "gonna", "wanna",
}


def _top_keywords(text: str, n: int = 8) -> List[str]:
    freq: Dict[str, int] = {}
    for token in _tokens(text):
        if len(token) > 3 and token not in STOPWORDS:
            freq[token] = freq.get(token, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:n]


def _sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return (parts[0] if parts else text).strip()


_TRAILING_FILLER = {"is", "are", "was", "were", "the", "a", "an", "and", "of",
                    "to", "you", "that", "it", "in", "on", "for", "with", "but",
                    "so", "we", "i", "he", "she", "they", "this", "your"}


def _shorten(text: str, limit: int, *, ellipsis: bool = True) -> str:
    """Cut *text* at a word boundary without leaving a dangling filler word.

    Whisper often returns one long unpunctuated segment, so a naive slice ends
    mid-clause ("…the first thousand dollars is"). Dropping the trailing filler
    and marking the cut keeps titles readable.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    words = text[:limit].rsplit(" ", 1)[0].split()
    while words and words[-1].lower().strip(".,;:!?\"'") in _TRAILING_FILLER:
        words.pop()
    out = " ".join(words).rstrip(" ,.;:!?")
    return (out + "…") if (ellipsis and out) else out


def local_meta(candidate: Dict[str, Any], source_title: str) -> Dict[str, Any]:
    text = candidate.get("transcript", "")
    keywords = _top_keywords(text)
    topic = " ".join(keywords[:3]) or (source_title or "this clip")
    sentence = _sentence(text)
    title = _shorten(sentence, 70)
    if not title:
        title = f"The best part of {source_title or 'this video'}"
    title = title[0].upper() + title[1:] if title else title

    # The opening sentence *is* the hook when the clip already starts on the
    # strongest line, so use the clip's own words instead of a template.
    opening = _sentence(text).strip(" \n").rstrip(" ,;:")
    opening_words = opening.split()
    if len(opening_words) > 12:
        opening = " ".join(opening_words[:11]).rstrip(" ,;:")
    hook_type = "bold claim"
    if "?" in opening:
        hook_type = "curiosity gap"
    elif NUMBER_RE.search(opening):
        hook_type = "bold claim"
    elif re.search(r"\b(stop|wait|listen|watch|look)\b", opening, re.I):
        hook_type = "pattern interrupt"
    elif re.search(r"\b(never|nobody|wrong|mistake|lie|myth|but)\b", opening, re.I):
        hook_type = "contrarian take"
    hook_text = candidate.get("hook_text") or opening or f"Nobody talks about {topic} like this"
    if len(hook_text.split()) > 12:
        hook_text = " ".join(hook_text.split()[:12])
    hook_text = _shorten(hook_text, 62)
    hook_text = hook_text.strip().rstrip(".")
    if hook_text:
        hook_text = hook_text[0].upper() + hook_text[1:]

    # Pick the component that scored highest *relative to its own maximum* so
    # the reason varies with the clip instead of always naming pacing.
    maxima = {"hook": 25, "pacing": 20, "emotion": 20, "clarity": 15,
              "shareability": 10, "trend": 10}
    breakdown = candidate.get("score_breakdown") or {}
    dominant = max(breakdown, key=lambda k: breakdown[k] / maxima.get(k, 10)) if breakdown else "hook"
    why = {
        "hook": "Opens on a line that creates an immediate question, so the first second does the work.",
        "pacing": "Tight, information-dense delivery with no dead air, which protects completion rate.",
        "emotion": "Lands on a real emotional peak, the kind of moment people rewatch and send on.",
        "clarity": "Stands alone from a cold start — a viewer with zero context still follows it.",
        "shareability": "Quotable and relatable enough that viewers will send it to someone specific.",
        "trend": "Sits squarely on a topic the platform is currently rewarding.",
    }[dominant]

    tags = [f"#{re.sub(r'[^a-z0-9]', '', k)}" for k in keywords[:6]]
    platform_tags = {"tiktok": ["#fyp", "#foryou"], "reels": ["#reels", "#instareels"],
                     "shorts": ["#shorts", "#youtubeshorts"]}
    primary = (candidate.get("best_platforms") or ["shorts"])[0]
    hashtags = tags + platform_tags.get(primary, ["#shorts"])
    return {
        "title": title[:70],
        "hook_text": hook_text[:120],
        "hook_type": hook_type,
        "why_it_works": why[:160],
        "suggested_caption": (sentence[:200] + (" " + " ".join(hashtags[:4]) if hashtags else "")).strip(),
        "hashtags": hashtags[:8],
    }


def generate_metadata(candidate: Dict[str, Any], source_title: str) -> Dict[str, Any]:
    """Title/hook/why/caption per clip; the LLM path is optional."""
    base = local_meta(candidate, source_title)
    if not llm_configured():
        return base
    system = (
        "You write short-form clip metadata. Reply with JSON only: "
        '{"title":"max 70 chars, curiosity-driven, true to the clip",'
        '"hook_text":"on-screen hook, max 12 words",'
        '"hook_type":"curiosity gap | bold claim | pattern interrupt | contrarian | story cold-open | relatable pain | visual shock",'
        '"why_it_works":"one sentence, max 160 chars",'
        '"suggested_caption":"caption with 3-5 hashtags",'
        '"hashtags":["#..."]}. Never invent facts that are not in the transcript.'
    )
    user = (f"Source video: {source_title or 'untitled'}\n"
            f"Clip {candidate['start']:.1f}s-{candidate['end']:.1f}s\n"
            f"Transcript:\n\"\"\"\n{candidate.get('transcript', '')[:2200]}\n\"\"\"")
    try:
        data = complete_json(system, user, temperature=0.7, max_tokens=700)
    except Exception:
        return base
    if not isinstance(data, dict) or not data.get("title"):
        return base
    out = dict(base)
    for key in ("title", "hook_text", "hook_type", "why_it_works", "suggested_caption"):
        if data.get(key):
            out[key] = str(data[key]).strip()[:300 if key == "suggested_caption" else 160]
    if isinstance(data.get("hashtags"), list) and data["hashtags"]:
        out["hashtags"] = [str(h) for h in data["hashtags"]][:8]
    out["source"] = "llm"
    return out


def alt_hooks(transcript: str, *, title: str = "") -> Dict[str, Any]:
    """Five A/B hook alternatives for one clip."""
    lines = [l.strip() for l in re.split(r"(?<=[.!?])\s+", transcript.strip()) if l.strip()]
    first = lines[0] if lines else transcript[:80]
    words = " ".join(transcript.split()[:7]).strip(" ,.;:")
    if llm_configured():
        try:
            data = complete_json(Hooks_SYSTEM,
                                 f"Clip title: {title}\nClip transcript:\n\"\"\"\n{transcript[:2200]}\n\"\"\"",
                                 temperature=0.8, max_tokens=900)
            hooks = [h for h in (data.get("hooks") or []) if isinstance(h, dict) and h.get("text")]
            if hooks:
                return {"source": "llm", "hooks": [
                    {"text": str(h["text"]).strip()[:140],
                     "pattern": str(h.get("pattern") or "curiosity gap")[:40],
                     "why": str(h.get("why") or "")[:160]} for h in hooks[:5]]}
        except Exception:
            pass
    return {"source": "local", "hooks": [
        {"text": f"Wait until you hear this part", "pattern": "curiosity gap",
         "why": "Promises a withheld payoff and makes leaving feel costly."},
        {"text": f"{words} changed everything" if words else "This changed everything",
         "pattern": "bold claim", "why": "A confident claim the clip can pay off immediately."},
        {"text": "Stop scrolling — watch this", "pattern": "pattern interrupt",
         "why": "A direct command breaks the passive scroll."},
        {"text": "Everyone gets this wrong", "pattern": "contrarian take",
         "why": "Challenges an assumption and invites comments."},
        {"text": first[:90].rstrip(" ,.;:") or "It started with one line",
         "pattern": "story cold-open", "why": "Drops the viewer straight into the middle of the story."},
    ]}


def find_moments(transcript: Dict[str, Any], speakers: Sequence[str], query: str,
                 signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Semantic-ish search over the transcript for Find Moments."""
    query = (query or "").strip()
    if not query:
        raise ClipperError("Type what you are looking for first.")
    segments = transcript["segments"]
    lines = [{"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text", ""),
              "speaker": speakers[i] if i < len(speakers) else "Speaker 1"}
             for i, s in enumerate(segments)]
    duration = float(transcript.get("duration") or segments[-1]["end"])

    if llm_configured():
        try:
            data = complete_json(
                FIND_SYSTEM,
                f"User query: {query}\nSource duration: {_hms(duration)}\n\n"
                f"Transcript:\n{_prompt_transcript(lines, 0, duration, max_chars=18000)}",
                temperature=0.3, max_tokens=1800,
            )
            found = []
            for item in (data.get("moments") or []):
                start, end = _parse_hms(item.get("start")), _parse_hms(item.get("end"))
                if start is None or end is None or end - start < 4:
                    continue
                text = _clip_text_for(lines, start, end)
                if not text:
                    continue
                found.append({
                    "start": round(max(0.0, start), 3), "end": round(min(duration, end), 3),
                    "title": str(item.get("title") or text[:60])[:70],
                    "why_it_works": str(item.get("why") or "")[:160],
                    "score": int(max(0, min(100, float(item.get("score") or 70)))),
                    "transcript": text,
                })
            if found:
                return {"source": "llm", "moments": found[:6]}
        except Exception:
            pass

    # local: score every window against the query terms
    terms = [t for t in _tokens(query) if len(t) > 2]
    synonyms = {
        "funny": ["laugh", "funny", "hilarious", "joke", "comedy"],
        "price": ["price", "cost", "money", "dollar", "expensive", "cheap", "pay"],
        "money": ["money", "dollar", "revenue", "income", "salary", "cost"],
        "reaction": ["react", "reaction", "wow", "shock", "crazy"],
        "phone": ["phone", "iphone", "android", "mobile", "device"],
        "angry": ["angry", "mad", "furious", "hate"],
        "argument": ["argue", "argument", "disagree", "wrong"],
    }
    expanded = set(terms)
    for term in terms:
        expanded.update(synonyms.get(term, []))

    lo, hi = 10.0, 90.0
    results: List[Dict[str, Any]] = []
    for (start, end) in _windows(segments, lo, hi):
        text = _clip_text_for(lines, start, end)
        if not text:
            continue
        tokens = _tokens(text)
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in expanded)
        if hits == 0 and query.lower() not in text.lower():
            continue
        density = hits / max(1, len(tokens) / 40.0)
        base = score_candidate((start, end), segments=segments, words=transcript["words"],
                               speakers=speakers, duration=duration,
                               signals=signals or {}, platform="all", lo=lo, hi=hi)
        quality = base["score"] if base else 50
        score = int(max(0, min(100, round(quality * 0.55 + min(45.0, density * 6)))))
        if base:
            results.append({**base, "score": score, "title": _sentence(text)[:60],
                            "why_it_works": f"Matches '{query}' — {hits} relevant term(s) in the clip.",
                            "transcript": text})
    results.sort(key=lambda m: m["score"], reverse=True)
    picked: List[Dict[str, Any]] = []
    for cand in results:
        if all(min(cand["end"], p["end"]) - max(cand["start"], p["start"]) <= 2 for p in picked):
            picked.append(cand)
        if len(picked) >= 6:
            break
    return {"source": "local", "moments": picked}


# ---------------------------------------------------------------------------
# Layouts + rendering
# ---------------------------------------------------------------------------
LAYOUTS = ["Full", "Fit", "Split", "Trio", "Grid", "PiP"]


def layout_graph(layout: str, w: int, h: int) -> str:
    """Filter-complex producing a 9:16 ``[base]`` from ``[0:v]``."""
    layout = (layout or "Full").lower()
    fill = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    fit = f"scale={w}:{h}:force_original_aspect_ratio=decrease"
    if layout == "fit":
        return (
            f"[0:v]{fill},gblur=sigma=32,eq=brightness=-0.10[bg];"
            f"[0:v]{fit}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base]"
        )
    if layout == "split":
        return (
            f"[0:v]{fill},split=2[a][b];"
            f"[a]crop={w}:{h // 2}:0:0[top];"
            f"[b]crop={w}:{h // 2}:0:{h // 2}[bot];"
            f"[top][bot]vstack=inputs=2[base]"
        )
    if layout == "trio":
        main_h = int(h * 0.62)
        small_h = h - main_h
        return (
            f"[0:v]{fill},split=3[a][b][c];"
            f"[a]crop={w}:{main_h}:0:0[main];"
            f"[b]crop={w // 2}:{small_h}:0:{main_h}[s1];"
            f"[c]crop={w // 2}:{small_h}:{w // 2}:{main_h}[s2];"
            f"[s1][s2]hstack=inputs=2[bottom];"
            f"[main][bottom]vstack=inputs=2[base]"
        )
    if layout == "grid":
        return (
            f"[0:v]{fill},split=4[a][b][c][d];"
            f"[a]crop={w // 2}:{h // 2}:0:0[tl];"
            f"[b]crop={w // 2}:{h // 2}:{w // 2}:0[tr];"
            f"[c]crop={w // 2}:{h // 2}:0:{h // 2}[bl];"
            f"[d]crop={w // 2}:{h // 2}:{w // 2}:{h // 2}[br];"
            f"[tl][tr]hstack=inputs=2[top];"
            f"[bl][br]hstack=inputs=2[bot];"
            f"[top][bot]vstack=inputs=2[base]"
        )
    if layout == "pip":
        sw, sh = int(w * 0.78), int(h * 0.5)
        return (
            f"[0:v]{fill},gblur=sigma=26,eq=brightness=-0.06[bg];"
            f"[0:v]scale={sw}:{sh}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base]"
        )
    # full (default): crop-fill
    return f"[0:v]{fill},setsar=1[base]"


def _watermark_filter(text: str, w: int, h: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9 ._&-]", "", text or "WayinVideo")[:40] or "WayinVideo"
    size = max(12, int(h * 0.022))
    return (f"drawtext=text='{safe}':fontsize={size}:fontcolor=white@0.72:"
            f"x=w-tw-{int(w * 0.04)}:y=h-th-{int(h * 0.035)}:"
            f"box=1:boxcolor=black@0.25:boxborderw={max(4, size // 3)}")


def _retime_segments(segments: Sequence[Dict[str, Any]], piece_start: float,
                     piece_end: float) -> List[Dict[str, Any]]:
    out = []
    for seg in segments:
        s, e = float(seg["start"]), float(seg["end"])
        if e <= piece_start + 0.001 or s >= piece_end - 0.001:
            continue
        words = []
        for word in seg.get("words", []):
            ws, we = float(word["s"]), float(word["e"])
            if we <= piece_start or ws >= piece_end:
                continue
            words.append({**word, "s": round(max(ws, piece_start) - piece_start, 3),
                          "e": round(min(we, piece_end) - piece_start, 3)})
        out.append({
            **seg,
            "start": round(max(s, piece_start) - piece_start, 3),
            "end": round(min(e, piece_end) - piece_start, 3),
            "words": words,
        })
    return out


def _run(cmd: List[str], *, timeout: int = 3600) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-800:]
        raise ClipperError(f"FFmpeg failed: {tail}")


def render_piece(
    src: Path | str,
    dst: Path | str,
    *,
    start: float,
    duration: float,
    w: int,
    h: int,
    layout: str = "Full",
    captions: Optional[Path] = None,
    watermark: str = "",
    crf: int = 20,
    preset: str = "veryfast",
) -> Path:
    """Render one keep-interval: layout + burned captions + optional watermark."""
    graph = layout_graph(layout, w, h)
    chain = "[base]"
    extras: List[str] = []
    if captions:
        p = str(Path(captions).resolve()).replace(":", "\\:").replace("'", "\\'")
        extras.append(f"ass='{p}':fontsdir='{_fonts_dir().replace(chr(39), chr(92) + chr(39))}'")
    if watermark:
        extras.append(_watermark_filter(watermark, w, h))
    if extras:
        graph += ";" + chain + ",".join(extras) + "[vout]"
        vmap = "[vout]"
    else:
        graph += ";" + chain + "null[vout]"
        vmap = "[vout]"

    cmd = [
        FFMPEG_BIN, "-y", "-v", "error",
        "-ss", f"{max(0.0, start):.3f}", "-i", str(src),
        "-t", f"{max(0.2, duration):.3f}",
        "-filter_complex", graph,
        "-map", vmap, "-map", "0:a?",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", "-shortest", str(dst),
    ]
    _run(cmd)
    return Path(dst)


def _fonts_dir() -> str:
    from ..config import BASE_DIR
    return str(BASE_DIR / "assets" / "fonts")


def render_moment(
    src: Path | str,
    moment: Dict[str, Any],
    out_dir: Path,
    *,
    name: str,
    height: int = 1920,
    style: Optional[Dict[str, Any]] = None,
    transcript: Optional[Dict[str, Any]] = None,
    captions: bool = True,
    watermark: str = "",
    cold_open: bool = False,
    preset: str = "veryfast",
    crf: int = 20,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Path:
    """Render one moment (9:16) at *height*; returns the mp4 path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = int(round(height * 9 / 16 / 2) * 2), int(round(height / 2) * 2)
    layout = moment.get("layout") or "Full"
    style = style or {"motion": "moving", "palette": "brand", "font": "Montserrat",
                      "size": "medium", "placement": "center"}

    pieces = moment.get("keep") or [{"start": float(moment["start"]), "end": float(moment["end"])}]
    segments = (transcript or {}).get("segments") or []
    words = (transcript or {}).get("words") or []

    # cold open: prepend a 2-5s teaser, then the normal beginning
    cold = moment.get("cold_open") if cold_open else None
    if isinstance(cold, dict) and cold.get("end", 0) > cold.get("start", 0):
        first = pieces[0]
        pieces = [{"start": float(cold["start"]), "end": float(cold["end"])}, first]

    rendered: List[Path] = []
    total = len(pieces)
    for i, piece in enumerate(pieces):
        piece_start, piece_end = float(piece["start"]), float(piece["end"])
        span = max(0.3, piece_end - piece_start)
        cap_path = None
        if captions and segments:
            retimed = _retime_segments(segments, piece_start, piece_end)
            if retimed:
                cap_path = generate_ass(retimed, style, out_dir / f"{name}_cap_{i}.ass")
        piece_path = out_dir / f"{name}_p{i}.mp4"
        render_piece(src, piece_path, start=piece_start, duration=span, w=w, h=h,
                     layout=layout, captions=cap_path,
                     watermark=watermark, crf=crf, preset=preset)
        rendered.append(piece_path)
        if on_progress:
            on_progress((i + 1) / total)

    final = out_dir / f"{name}.mp4"
    if len(rendered) == 1:
        shutil.move(str(rendered[0]), final)
    else:
        list_file = out_dir / f"{name}_concat.txt"
        list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in rendered), "utf-8")
        _run([FFMPEG_BIN, "-y", "-v", "error", "-f", "concat", "-safe", "0",
              "-i", str(list_file), "-c", "copy", "-movflags", "+faststart", str(final)])
        for p in rendered:
            p.unlink(missing_ok=True)
        list_file.unlink(missing_ok=True)
    return final


def ensure_video(src: Path | str, out_dir: Path) -> Path:
    """Return a real video file for *src*.

    Audio-only uploads (MP3/WAV) get a generated audiogram: a dark canvas with
    a live waveform, so clipping, captions and layouts all still work.
    """
    src = Path(src)
    try:
        width = int(media.video_meta(src).get("width") or 0)
    except Exception:
        width = 0
    if width > 0:
        return src
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "audiogram.mp4"
    graph = (
        "[1:a]showwaves=s=1280x300:mode=cline:rate=25:colors=0x2BF5A0,"
        "format=yuva420p,colorchannelmixer=aa=0.85[w];"
        "[0:v][w]overlay=0:(H-h)/2:shortest=1[v]"
    )
    cmd = [FFMPEG_BIN, "-y", "-v", "error",
           "-f", "lavfi", "-i", "color=c=0x07120C:s=1280x720:r=25",
           "-i", str(src), "-filter_complex", graph,
           "-map", "[v]", "-map", "1:a", "-shortest",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
           "-movflags", "+faststart", str(dst)]
    _run(cmd)
    return dst


def make_poster(video: Path | str, dst: Path, at_sec: float = 0.5) -> Path:
    return media.make_thumbnail(video, dst, at_sec=at_sec)


def zip_files(files: Sequence[Path], dst: Path) -> Path:
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if Path(path).is_file():
                zf.write(path, arcname=Path(path).name)
    return dst
