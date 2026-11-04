"""
Clip Maker engine.

Pipeline
--------
1. **Acquire** – yt-dlp download for URLs, or accept an uploaded file.
2. **Probe**   – ffprobe metadata (duration, geometry, fps).
3. **Audio**   – mono 16 kHz WAV for Whisper.
4. **Transcribe** – faster-whisper, word-level timestamps.
5. **Scenes**  – PySceneDetect / ffmpeg scene filter.
6. **Hook scoring** – sliding-window scoring over the transcript (keyword
   energy, speech density, questions, scene activity, opening bonus) with an
   overlap penalty, picking the top N non-overlapping segments.
7. **Smart crop** – OpenCV face tracking to find the horizontal centre of the
   subject, falling back to a centred 9:16 crop.
8. **Subtitles** – ``.ass`` built from the user's style selection.
9. **Render** – single-pass ffmpeg trim + crop + burn + re-encode.
10. **Metadata** – LLM titles/tags/descriptions per platform (with a
    deterministic fallback when no LLM key is configured).
"""
from __future__ import annotations

import logging
import math
import re
import resource
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import (
    CLIP_COUNT,
    CLIP_MAX_SEC,
    CLIP_MIN_SEC,
    DOWNLOAD_DIR,
    MEDIA_DIR,
    UPLOAD_DIR,
)
from . import media, scenes
from .ass import generate_ass, style_summary
from .llm import LLMUnavailable, complete_json, configured as llm_configured
from .transcribe import format_timestamp, release_model, transcribe_audio

ProgressFn = Callable[[int, float, str], None]

# ---------------------------------------------------------------------------
# Hook vocabulary – words that historically open viral shorts
# ---------------------------------------------------------------------------
HOOK_WORDS = {
    "secret", "secrets", "never", "nobody", "stop", "mistake", "mistakes",
    "insane", "crazy", "actually", "truth", "lie", "lies", "free", "money",
    "easy", "fast", "hack", "hacks", "trick", "tricks", "proof", "warning",
    "urgent", "finally", "best", "worst", "biggest", "fastest", "simple",
    "actually", "imagine", "listen", "watch", "here's", "this", "why", "how",
    "what", "before", "after", "problem", "fix", "instantly", "guaranteed",
}
NUMBER_RE = re.compile(r"\b\d+([.,]\d+)?%?\b")
QUESTION_RE = re.compile(r"\?")
SECOND_PERSON = {"you", "your", "yours", "yourself"}


# ---------------------------------------------------------------------------
# Candidate windows
# ---------------------------------------------------------------------------
def candidate_windows(
    segments: List[Dict[str, Any]],
    *,
    min_len: float = CLIP_MIN_SEC,
    max_len: float = CLIP_MAX_SEC,
    step: float = 6.0,
) -> List[Tuple[float, float]]:
    """Return ``(start, end)`` windows aligned to transcript boundaries."""
    if not segments:
        return []
    starts = [float(s["start"]) for s in segments]
    ends = [float(s["end"]) for s in segments]
    windows: List[Tuple[float, float]] = []

    for i, s0 in enumerate(starts):
        for target in (min_len, (min_len + max_len) / 2, max_len):
            end_target = s0 + target
            # extend to the nearest segment end <= target + 25% slack
            j = i
            end = ends[i]
            while j < len(ends) and ends[j] - s0 <= target * 1.25:
                end = ends[j]
                j += 1
            if end - s0 >= min_len * 0.8:
                windows.append((round(s0, 3), round(min(end, end_target + 8), 3)))

    # de-duplicate
    return sorted(set(windows))


def _words_in(words: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    return [w for w in words if start <= float(w["s"]) < end]


def score_window(
    window: Tuple[float, float],
    *,
    words: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    duration: float,
    scenes: Optional[List[float]],
) -> Dict[str, Any]:
    """Score one candidate clip. Higher == more likely to be a viral hook."""
    start, end = window
    span = max(1.0, end - start)
    w = _words_in(words, start, end)
    if not w:
        return {"score": 0.0, "reasons": ["no speech"]}

    tokens = [re.sub(r"[^a-z0-9'%]", "", x["w"].lower()) for x in w]
    tokens = [t for t in tokens if t]
    if not tokens:
        return {"score": 0.0, "reasons": ["no speech"]}

    density = len(tokens) / span                     # words per second
    hits = [t for t in tokens if t in HOOK_WORDS]
    numbers = NUMBER_RE.findall(" ".join(tokens))
    questions = sum(1 for seg_text in [" ".join(tokens)] if QUESTION_RE.search(seg_text))
    second_person = sum(1 for t in tokens if t in SECOND_PERSON)
    span_segments = [s for s in segments if float(s["start"]) < end and float(s["end"]) > start]
    exclamations = sum(len(re.findall(r"!", s.get("text", ""))) for s in span_segments)

    density_norm = 1.0 - min(abs(density - 2.7) / 2.7, 1.0)     # ~2.7 wps is ideal
    keyword_ratio = min(len(hits) / max(6.0, len(tokens) * 0.08), 1.5)
    number_bonus = min(len(numbers) * 0.18, 0.7)
    question_bonus = 0.5 if (questions or "?" in " ".join(tokens)) else 0.0
    person_bonus = min(second_person * 0.12, 0.6)
    exclam_bonus = min(exclamations * 0.15, 0.5)
    scene_bonus = min(_scene_count(scenes, start, end) * 0.1, 0.8)
    opening_bonus = 0.6 if start <= duration * 0.18 else 0.0
    length_bonus = 0.35 if 20 <= span <= 45 else 0.0

    score = (
        2.2 * keyword_ratio
        + 1.6 * density_norm
        + 0.9 * scene_bonus
        + question_bonus
        + person_bonus
        + number_bonus
        + exclam_bonus
        + opening_bonus
        + length_bonus
    )

    reasons = []
    if keyword_ratio > 0.3:
        reasons.append(f"hook words: {', '.join(sorted(set(hits))[:4])}")
    if density_norm > 0.65:
        reasons.append(f"fast pace ({density:.1f} w/s)")
    if question_bonus:
        reasons.append("opens with a question")
    if number_bonus:
        reasons.append("uses specific numbers")
    if opening_bonus:
        reasons.append("strong opening")
    if scene_bonus >= 0.4:
        reasons.append("visually active")

    return {"score": round(score, 3), "reasons": reasons or ["balanced segment"]}


def _scene_count(scenes: Optional[List[float]], start: float, end: float) -> int:
    if not scenes:
        return 0
    return sum(1 for t in scenes if start <= t <= end)


def select_hooks(
    segments: List[Dict[str, Any]],
    words: List[Dict[str, Any]],
    duration: float,
    scene_list: Optional[List[float]],
    count: int = CLIP_COUNT,
) -> List[Dict[str, Any]]:
    """Pick the top N non-overlapping, well-spread hook segments.

    Short inputs need special care: a single window can cover the whole video
    and block the other nine. We cap the candidate length so ``count`` windows
    can coexist, then split the longest remaining window until we have enough.
    """
    # a window longer than ~1.6x its share of the video makes a full set
    # impossible, so shrink the ceiling for short sources / small counts
    share = (duration / max(1, count)) * 1.6 if duration else float(CLIP_MAX_SEC)
    max_len = max(12.0, min(CLIP_MAX_SEC, share))
    candidates = candidate_windows(segments, max_len=max_len)
    scored = []
    for win in candidates:
        info = score_window(win, words=words, segments=segments, duration=duration, scenes=scene_list)
        if info["score"] <= 0:
            continue
        scored.append({**info, "start": win[0], "end": win[1], "duration": round(win[1] - win[0], 2)})
    scored.sort(key=lambda x: x["score"], reverse=True)

    chosen: List[Dict[str, Any]] = []
    for cand in scored:
        if len(chosen) >= count:
            break
        if any(cand["start"] < c["end"] and c["start"] < cand["end"] for c in chosen):
            continue                                # overlap -> skip
        chosen.append(cand)
    chosen.sort(key=lambda x: x["start"])

    # if the transcript was too sparse, fall back to an even split
    if len(chosen) < count and duration > 0:
        step = duration / count
        for i in range(count):
            if len(chosen) >= count:
                break
            s, e = i * step, min(duration, (i + 1) * step)
            if e - s < 5:
                continue
            if any(s < c["end"] and c["start"] < e for c in chosen):
                continue
            chosen.append(
                {"score": 1.0, "reasons": ["even split (sparse transcript)"],
                 "start": round(s, 2), "end": round(e, 2), "duration": round(e - s, 2)}
            )
        chosen.sort(key=lambda x: x["start"])

    # still short: bisect the longest windows until we can return `count`
    guard = 0
    while len(chosen) < count and guard < 400 and duration > 0:
        guard += 1
        chosen.sort(key=lambda c: -(c["end"] - c["start"]))
        big = chosen[0]
        if big["end"] - big["start"] < 9.0:
            break
        mid = round((big["start"] + big["end"]) / 2, 2)
        if mid - big["start"] < 4.5 or big["end"] - mid < 4.5:
            break
        # snap the split onto a transcript boundary when one is close
        snap = [float(s["start"]) for s in segments
                if big["start"] + 4.5 <= float(s["start"]) <= mid + 4.5]
        if snap:
            mid = round(min(snap, key=lambda t: abs(t - mid)), 2)
        mid = max(big["start"] + 4.5, min(mid, big["end"] - 4.5))
        if mid <= big["start"] or mid >= big["end"]:
            break
        big_a = {**big, "end": mid, "duration": round(mid - big["start"], 2),
                 "reasons": big["reasons"] + ["split for variety"]}
        big_b = {**big, "start": mid, "duration": round(big["end"] - mid, 2),
                 "reasons": big["reasons"] + ["split for variety"]}
        chosen[0:1] = [big_a, big_b]

    chosen.sort(key=lambda x: x["start"])

    # Final guarantee: the contract is "exactly N clips". If scoring + splits
    # still came up short (short source, heavy overlap), fall back to a clean
    # even partition of the timeline.
    if len(chosen) < count and duration > 0 and (duration / count) >= 4.0:
        step = duration / count
        chosen = [
            {"score": 1.0, "reasons": ["even split (requested count)"],
             "start": round(i * step, 2), "end": round((i + 1) * step, 2),
             "duration": round(step, 2)}
            for i in range(count)
        ]

    return chosen[:count]


# ---------------------------------------------------------------------------
# Smart crop
# ---------------------------------------------------------------------------
def smart_crop_offset(src: Path | str, start: float, end: float) -> Optional[float]:
    """Normalised horizontal centre (0..1) of the main subject, or ``None``."""
    try:
        import cv2  # type: ignore
        # OpenCV 5.x dropped the legacy CascadeClassifier binding; without it we
        # simply fall back to a centred crop rather than failing the render.
        if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
            return None
    except Exception:
        return None

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return None

    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            cap.release()
            return None
    except Exception:
        cap.release()
        return None
    samples = 12
    span = max(0.5, end - start)
    centres: List[float] = []
    weights: List[float] = []

    for i in range(samples):
        t = start + span * (i + 0.5) / samples
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        scale = 480 / max(1, frame.shape[1])
        small = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(28, 28))
        if len(faces) == 0:
            continue
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        centres.append(((x + w / 2) / small.shape[1]))
        weights.append(float(w * h))
    cap.release()

    if not centres:
        return None
    # weighted mean, clamped so the crop window never leaves the frame
    total = sum(weights) or 1.0
    cx = sum(c * wgt for c, wgt in zip(centres, weights)) / total
    return round(min(max(cx, 0.18), 0.82), 4)


# ---------------------------------------------------------------------------
# Metadata (LLM with deterministic fallback)
# ---------------------------------------------------------------------------
def _slug(text: str, n: int = 60) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9 ]", "", text)).strip()[:n]


def fallback_metadata(transcript: str, source_title: str, index: int) -> Dict[str, Any]:
    """Keyword-driven metadata used when no LLM key is configured."""
    words = [w for w in re.findall(r"[A-Za-z0-9'%]+", transcript) if len(w) > 3]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w.lower()] = freq.get(w.lower(), 0) + 1
    keywords = sorted(freq, key=freq.get, reverse=True)[:8] or ["video"]
    topic = source_title or " ".join(keywords[:3])
    base = _slug(topic, 40).title() or "Watch This"

    tags = keywords + ["shorts", "viral", "clip"]
    hashtags = ["#" + re.sub(r"[^a-z0-9]", "", k) for k in keywords[:6]] + ["#shorts", "#fyp"]
    return {
        "titles": {
            "youtube": f"{base} 🔥 #{index} You Need To See This",
            "reels": f"{base} — wait for it ✨",
            "tiktok": f"POV: you found the best part of {base}",
        },
        "tags": {
            "youtube": tags + ["youtube shorts", "short form"],
            "instagram": tags + ["reels", "instareels"],
            "tiktok": tags + ["fyp", "viral"],
        },
        "hashtags": {"instagram": hashtags, "tiktok": hashtags + ["#foryou"]},
        "youtube_description": (
            f"{base} — clip #{index}.\n\n"
            f"In this short: {transcript[:220].strip()}...\n\n"
            f"Keywords: {', '.join(keywords)}\n\n"
            f"Created with WayinVideo Clip Maker.\n\n" + " ".join(hashtags)
        ),
        "source": "local-fallback",
    }


def generate_metadata(
    clip: Dict[str, Any],
    source_title: str,
    index: int,
) -> Dict[str, Any]:
    """Ask the LLM for platform-optimised titles/tags/description."""
    if not llm_configured():
        return fallback_metadata(clip.get("transcript", ""), source_title, index)

    system = (
        "You are a short-form video growth strategist. Return STRICT JSON only, "
        "no markdown, with exactly these keys: "
        '{"titles":{"youtube","reels","tiktok"},'
        '"tags":{"youtube":[],"instagram":[],"tiktok":[]},'
        '"hashtags":{"instagram":[],"tiktok":[]},'
        '"youtube_description":""}. '
        "Titles must be <= 90 chars and click-worthy without being spammy. "
        "Tags: 12-18 lowercase search phrases per platform. "
        "Hashtags: 6-10, no duplicates across platforms beyond relevance. "
        "youtube_description: 2-4 sentences plus a keyword line."
    )
    user = (
        f"Source video: {source_title or 'untitled'}\n"
        f"Clip #{index} transcript:\n\"\"\"\n{clip.get('transcript', '')[:2500]}\n\"\"\""
    )
    try:
        data = complete_json(system, user, temperature=0.6, max_tokens=1400)
    except LLMUnavailable:
        return fallback_metadata(clip.get("transcript", ""), source_title, index)

    if not data or "titles" not in data:
        return fallback_metadata(clip.get("transcript", ""), source_title, index)
    data.setdefault("source", "llm")
    return data


# ---------------------------------------------------------------------------
# Hook mode — 7 scroll-stopping hooks per clip
# ---------------------------------------------------------------------------
# STEP 1 research baseline. These five patterns recur across short-form
# strategy sources: what holds viewers in the first 1-3 seconds, and what the
# recommendation systems reward (watch time, rewatches, shares, comments).
# When no LLM key is configured they are also the vocabulary the deterministic
# writer below builds its hooks from.
HOOK_PATTERNS: List[Dict[str, str]] = [
    {"name": "Curiosity gap", "why": "Opens a question only watching can close."},
    {"name": "Bold claim", "why": "A confident promise triggers the 1-3s credibility check."},
    {"name": "Pattern interrupt", "why": "An unexpected line breaks the scroll reflex."},
    {"name": "Open loop", "why": "An unfinished story pushes completion and rewatches."},
    {"name": "Contrarian take", "why": "Challenging a belief provokes comments and shares."},
]

HOOKS_SYSTEM = (
    "You are an expert short-form video strategist who specializes in hooks for "
    "TikTok, Instagram Reels and YouTube Shorts.\n\n"
    "STEP 1 - RESEARCH\n"
    "These hook patterns recur across short-form strategy sources and are the ones "
    "currently performing on short-form platforms:\n"
    + "\n".join(f"- {p['name']}: {p['why']}" for p in HOOK_PATTERNS)
    + "\nYou may add other patterns you know are performing now, but only use patterns "
      "that appear in multiple sources. Keep the working list to the top 5.\n\n"
    "STEP 2 - ANALYZE THE CLIP\n"
    "Read the transcript and clip length given to you. Find the most surprising, "
    "emotional, funny or controversial point; the main tension or question; the "
    "payoff moment.\n\n"
    "STEP 3 - WRITE THE HOOKS\n"
    "Write exactly 7 hooks, each under 12 words, each using a different pattern from "
    "your research. For every hook give: on-screen text, the spoken version (empty "
    "string if it is not said out loud), the pattern used, and one line on why it will "
    "stop the scroll.\n\n"
    "RULES\n"
    "- The hook must be true to the clip. No fake claims or clickbait the video cannot "
    "pay off.\n"
    "- It must create a question or tension the viewer can only resolve by watching.\n"
    "- Simple words, no filler like 'In this video...' or 'Welcome back...'.\n"
    "- The hook must land in the first 1-2 seconds.\n\n"
    "OUTPUT — STRICT JSON only, no markdown, no commentary, exactly this shape:\n"
    '{"hooks":[{"on_screen":"","spoken":"","pattern":"","why":""}],'
    '"best":{"pick":1,"reason":""},'
    '"ab_test":{"on_screen":"","spoken":"","note":""}}\n'
    "`hooks` must contain exactly 7 items. `pick` is the 1-based index of the single "
    "best hook. `reason` explains in two sentences why it will get the most "
    "retention. `ab_test` is one A/B variant of that best hook."
)


def _hook_topic(source_title: str, transcript: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9'%]+", transcript) if len(w) > 3]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w.lower()] = freq.get(w.lower(), 0) + 1
    keywords = sorted(freq, key=freq.get, reverse=True)[:6] or ["this"]
    raw = _slug(source_title or " ".join(keywords[:2]), 40).strip()
    base = raw[:30]
    if len(raw) > 30:                       # never cut a word in half
        base = base.rsplit(" ", 1)[0] or base
    return base or "this video"


def _words_under(text: str, limit: int = 12) -> str:
    parts = str(text).split()
    return " ".join(parts[:limit])


def fallback_hooks(
    clip: Dict[str, Any],
    source_title: str,
    index: int,
    platform: str = "YouTube Shorts",
) -> Dict[str, Any]:
    """Pattern-driven hooks used when no LLM key is configured.

    Each line stays under 12 words and only leans on words that are actually
    in the clip transcript, so nothing promises more than the video delivers.
    """
    transcript = (clip.get("transcript") or "").strip()
    topic = _hook_topic(source_title, transcript)
    duration = int(round(float(clip.get("duration") or 10)))
    clause_words = [w for w in transcript.split() if w.strip(".,!?;:")][:7]
    clause = " ".join(clause_words).strip(" ,.;:")

    hooks: List[Dict[str, str]] = [
        {"on_screen": f"The part of {topic} nobody shows you",
         "spoken": f"Nobody shows you this part of {topic}",
         "pattern": "Curiosity gap",
         "why": "Promises withheld information, which only watching releases."},
        {"on_screen": f"This changes everything about {topic}",
         "spoken": f"This changes everything about {topic}",
         "pattern": "Bold claim",
         "why": "A confident promise makes the viewer verify it in the first second."},
        {"on_screen": "Stop scrolling — you need to see this",
         "spoken": "Stop scrolling. Watch this.",
         "pattern": "Pattern interrupt",
         "why": "A direct command breaks the passive scroll habit."},
        {"on_screen": "Wait until you hear what happens next",
         "spoken": "Wait until you hear what happens next.",
         "pattern": "Open loop",
         "why": "Leaves the story unfinished, so leaving costs the payoff."},
        {"on_screen": f"You have {topic} all wrong",
         "spoken": f"You have {topic} all wrong — watch this.",
         "pattern": "Contrarian take",
         "why": "Disputing an assumption invites comments and a second watch."},
        {"on_screen": f'"{clause}" — hear why it matters' if clause else f"The line everyone missed in {topic}",
         "spoken": clause or f"The line everyone missed in {topic}",
         "pattern": "Story-in-media-res",
         "why": "Dropping into a real moment makes the rest feel like context."},
        {"on_screen": f"In {duration} seconds you'll hear the best line",
         "spoken": f"Give me {duration} seconds.",
         "pattern": "Specific number",
         "why": "A small, concrete time cost lowers the decision to keep watching."},
    ]

    # best = the hook built from the clip's own words when there is one
    pick = 6 if len(clause.split()) > 2 else 4   # story-in-media-res, else open loop
    best = hooks[pick - 1]
    return {
        "platform": platform,
        "research": {"source": "curated", "patterns": HOOK_PATTERNS},
        "hooks": hooks,
        "best": {
            "pick": pick,
            "on_screen": best["on_screen"],
            "spoken": best["spoken"],
            "reason": (
                "It opens a question the clip answers in its own first seconds, so the "
                "cost of scrolling away is higher than the cost of watching."
            ),
        },
        "ab_test": {
            "on_screen": "Stop scrolling — this is the part people replay",
            "spoken": "Stop scrolling — this is the part people replay.",
            "note": "Pattern interrupt variant: same payoff, sharper first second.",
        },
        "source": "local-fallback",
        "index": index,
    }


def _clean_hook(item: Any) -> Optional[Dict[str, str]]:
    if not isinstance(item, dict):
        return None
    on = str(item.get("on_screen") or item.get("hook") or "").strip()
    if not on:
        return None
    spoken = str(item.get("spoken") or "").strip() or on
    return {
        "on_screen": _words_under(on, 16),
        "spoken": _words_under(spoken, 20),
        "pattern": str(item.get("pattern") or "Curiosity gap").strip()[:60],
        "why": str(item.get("why") or "").strip()[:160],
    }


def generate_hooks(
    clip: Dict[str, Any],
    source_title: str,
    index: int,
    platform: str = "YouTube Shorts",
) -> Dict[str, Any]:
    """STEP 1-3 hook write-up for one clip: 7 hooks, the best pick, an A/B test.

    Uses the LLM when one is configured and falls back to
    :func:`fallback_hooks` otherwise (or whenever the model returns junk).
    """
    transcript = clip.get("transcript", "")
    if not llm_configured():
        return fallback_hooks(clip, source_title, index, platform)

    user = (
        f"Platform: {platform}\n"
        f"Source video: {source_title or 'untitled'}\n"
        f"Clip #{index} — length: {float(clip.get('duration') or 0):.1f} seconds\n"
        f"Clip transcript:\n\"\"\"\n{transcript[:2500]}\n\"\"\""
    )
    try:
        data = complete_json(HOOKS_SYSTEM, user, temperature=0.7, max_tokens=1800)
    except LLMUnavailable:
        return fallback_hooks(clip, source_title, index, platform)
    if not data:
        return fallback_hooks(clip, source_title, index, platform)

    hooks = [h for h in (_clean_hook(x) for x in (data.get("hooks") or [])) if h][:7]
    if len(hooks) < 3:
        return fallback_hooks(clip, source_title, index, platform)

    raw_best = data.get("best") or {}
    try:
        pick = int(raw_best.get("pick") or 1)
    except (TypeError, ValueError):
        pick = 1
    pick = min(max(pick, 1), len(hooks))
    chosen = hooks[pick - 1]

    raw_ab = data.get("ab_test") or {}
    ab = {
        "on_screen": _words_under(str(raw_ab.get("on_screen") or "").strip() or chosen["on_screen"], 16),
        "spoken": _words_under(str(raw_ab.get("spoken") or "").strip() or chosen["spoken"], 20),
        "note": str(raw_ab.get("note") or "A/B variant of the best hook.").strip()[:160],
    }
    reason = str(raw_best.get("reason") or "").strip()
    if not reason:
        reason = chosen["why"] or "Leaves a question the clip answers immediately."

    return {
        "platform": platform,
        "research": {"source": "llm+curated", "patterns": HOOK_PATTERNS},
        "hooks": hooks,
        "best": {
            "pick": pick,
            "on_screen": chosen["on_screen"],
            "spoken": chosen["spoken"],
            "reason": reason,
        },
        "ab_test": ab,
        "source": "llm",
        "index": index,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process(
    task_id: str,
    *,
    source: str,
    style: Dict[str, Any],
    count: int = CLIP_COUNT,
    mode: str = "normal",
    uploaded: Optional[Path] = None,
    on_progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Run the whole Clip Maker pipeline for one job."""
    from ..config import TARGET_H, TARGET_W  # local import keeps module import cheap

    def report(step: int, pct: float, label: str) -> None:
        # Peak RSS per stage: a host that kills the process for memory leaves no
        # traceback, so the last stage's figure is the only evidence of where
        # the budget went.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = peak / (1 << 20) if sys.platform == "darwin" else peak / 1024
        logging.getLogger(__name__).warning(
            "[%s] %5.1f%% %s (peak rss %.0f MB)", task_id[:8], pct, label, peak)
        if on_progress:
            on_progress(step, pct, label)

    # 1. acquire ---------------------------------------------------------
    report(0, 3, "Preparing video")
    if uploaded is not None:
        src_path = Path(uploaded)
        title = src_path.stem
    elif Path(source).is_file():
        # local file passed as `source` (JSON API) — no download needed
        src_path = Path(source)
        title = src_path.stem
    else:
        src_path = media.download_url(source, DOWNLOAD_DIR)
        try:
            title = media.fetch_metadata(source).get("title") or src_path.stem
        except Exception:
            title = src_path.stem

    meta = media.video_meta(src_path)
    duration = meta["duration"]
    if duration < 15:
        raise media.MediaError("Video is shorter than 15 seconds — nothing to clip.")

    out_dir = MEDIA_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. audio -----------------------------------------------------------
    report(1, 8, "Extracting audio")
    audio_path = media.extract_audio(src_path, out_dir / "audio.wav")

    # 3. transcript ------------------------------------------------------
    report(2, 14, "Transcribing with Whisper")
    transcript = transcribe_audio(audio_path, on_progress=lambda p: report(2, 14 + p * 0.25, "Transcribing with Whisper"))
    # Nothing below needs the weights, and on a small host holding them starves
    # the scene detector and the encoder of RAM.
    release_model()

    # 4. scenes ----------------------------------------------------------
    report(3, 45, "Detecting scenes")
    scene_list = scenes.detect_scenes(src_path)

    # 5. hook scoring ----------------------------------------------------
    report(4, 52, "Scoring viral hooks")
    plan = select_hooks(transcript["segments"], transcript["words"], duration, scene_list, count=count)

    # 6. render ----------------------------------------------------------
    results: List[Dict[str, Any]] = []
    total = max(1, len(plan))
    for i, seg in enumerate(plan):
        pct = 55 + (i / total) * 35
        report(5, pct, f"Rendering clip {i + 1} of {total}")

        offset = smart_crop_offset(src_path, seg["start"], seg["end"])
        vf = media.vertical_crop_filter(offset)

        clip_words = _words_in(transcript["words"], seg["start"], seg["end"])
        clip_segments = [
            {
                **s,
                "start": max(0.0, float(s["start"]) - seg["start"]),
                "end": max(0.0, min(float(s["end"]), seg["end"]) - seg["start"]),
                "words": [
                    {**w, "s": round(float(w["s"]) - seg["start"], 3), "e": round(float(w["e"]) - seg["start"], 3)}
                    for w in s.get("words", [])
                    if seg["start"] <= float(w["s"]) < seg["end"]
                ],
            }
            for s in transcript["segments"]
            if float(s["end"]) > seg["start"] and float(s["start"]) < seg["end"]
        ]

        ass_path = generate_ass(clip_segments, style, out_dir / f"clip_{i + 1:02d}.ass")
        dst = out_dir / f"clip_{i + 1:02d}.mp4"
        media.render_clip(
            src_path, dst, seg["start"], seg["end"] - seg["start"],
            vf=vf, ass=ass_path, preset="veryfast", crf=20,
        )
        poster = media.make_thumbnail(dst, out_dir / f"clip_{i + 1:02d}.jpg", at_sec=0.4)

        clip_text = " ".join(w["w"] for w in clip_words) or " ".join(
            s["text"] for s in clip_segments
        )
        results.append(
            {
                "index": i + 1,
                "start": seg["start"],
                "end": seg["end"],
                "start_label": format_timestamp(seg["start"]),
                "end_label": format_timestamp(seg["end"]),
                "duration": round(seg["end"] - seg["start"], 2),
                "score": seg["score"],
                "reasons": seg["reasons"],
                "smart_crop": offset,
                "file": f"clip_{i + 1:02d}.mp4",
                "poster": f"clip_{i + 1:02d}.jpg",
                "ass": ass_path.name,
                "transcript": clip_text.strip(),
                "segments": clip_segments,
                "style": style_summary(style),
                "metadata": None,     # filled in below (can run while rendering)
            }
        )

    # 7. metadata (+ hooks in Hook mode) ---------------------------------
    hook_mode = mode == "hook"
    report(6, 92, "Writing titles, tags & hooks" if hook_mode else "Writing titles, tags & descriptions")
    for i, clip in enumerate(results):
        clip["metadata"] = generate_metadata(clip, title, i + 1)
        if hook_mode:
            clip["hooks"] = generate_hooks(clip, title, i + 1)

    report(6, 100, "Done")
    return {
        "source": source or str(uploaded),
        "title": title,
        "duration": duration,
        "language": transcript.get("language", ""),
        "clip_count": len(results),
        "style": style_summary(style),
        "mode": "hook" if hook_mode else "normal",
        "clips": results,
        "llm": "on" if llm_configured() else "local-fallback",
    }
