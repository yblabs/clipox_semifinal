"""
ASS (Advanced SubStation Alpha) subtitle generator.

Turns the frontend's style configuration into a ready-to-burn ``.ass`` file:

.. code-block:: json

    {"motion": "moving", "palette": "neon", "font": "Montserrat",
     "size": "medium", "placement": "bottom", "preset": "classic"}

* ``motion``  -> ``stationary`` (one line per segment) or ``moving``
                 (word highlight: white line, the spoken word pops out yellow).
* ``palette`` -> legacy colour schemes; word highlight ignores it and always
                 renders white text with a yellow active word.
* ``font``    -> any font installed on the render host.
* ``size``    -> small / medium / large (relative to a 1080x1920 canvas).
* ``placement`` -> bottom / center / top (``\\an2`` / ``\\an5`` / ``\\an8``).
* ``preset``  -> ``classic`` (plain captions), ``hook`` (viral 1-3 word
                 pop-in style used by Hook clips) or ``karaoke`` (2-4 words,
                 one line, one yellow word following the speech).

The file uses ``PlayResX: 1080 / PlayResY: 1920`` so libass scales it straight
onto vertical shorts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

PLAY_W, PLAY_H = 1080, 1920

# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------
# ``primary``   -> unspoken / stationary caption text
# ``highlight`` -> the active (spoken) word in ``moving`` motion
# ``outline``   -> border + soft shadow
# ``accent``    -> active-word colour of the Hook preset
PALETTES: Dict[str, Dict[str, str]] = {
    "classic":  {"primary": "#FFFFFF", "highlight": "#FFE600", "outline": "#000000", "accent": "#FFE600"},
    "white":    {"primary": "#FFFFFF", "highlight": "#FFFFFF", "outline": "#FFFFFF", "accent": "#FFE600"},
    "neon":     {"primary": "#39FF14", "highlight": "#FFFFFF", "outline": "#062033", "accent": "#39FF14"},
    "contrast": {"primary": "#FFE600", "highlight": "#FFFFFF", "outline": "#000000", "accent": "#FFE600"},
    "brand":    {"primary": "#00FF9C", "highlight": "#FFFFFF", "outline": "#04070A", "accent": "#00FF9C"},
    "ocean":    {"primary": "#00E5FF", "highlight": "#1E90FF", "outline": "#04263F", "accent": "#1E90FF"},
}

FONTS = ["Montserrat", "Impact", "Anton", "Poppins"]
SIZES = {"small": 56, "medium": 78, "large": 104}
# The Hook preset is deliberately much larger: 7-9% of the 1920px canvas.
HOOK_SIZES = {"small": 134, "medium": 154, "large": 173}

# ---------------------------------------------------------------------------
# Karaoke preset — one line of 2-4 words, exactly one word yellow at a time
# ---------------------------------------------------------------------------
# Spec size is 56px on a 720x1280 canvas: 7.8% of the width, so 84px here.
# Caps land at ~59px (Montserrat's cap height is ~0.70em) = the spec's 40px
# scaled by 1.5.
KARAOKE_SIZES = {"small": 76, "medium": 84, "large": 96}
KARAOKE_BASE = "#FFFFFF"         # unspoken words
KARAOKE_HIGHLIGHT = "#F8D000"    # the word being spoken (no other accent)
KARAOKE_OUTLINE = "#000000"
KARAOKE_DIM = "&H555555"         # grey the line fades to on its way out

# The spoken word pops: it lands ~14% larger in a brighter lemon with a yellow
# halo, then settles back to the accent yellow over ~120 ms. Negative letter
# spacing cancels the extra advance while it is enlarged so the rest of the
# line never shifts sideways.
KARAOKE_POP_SCALE = 112          # % size at the instant of the pop
KARAOKE_POP_MS = 120             # ms to settle back to 100%
KARAOKE_POP_OUTLINE = 7          # px outline at the instant of the pop
KARAOKE_FLASH = "#FFE94A"        # brighter yellow the pop starts from

KARAOKE_MIN_WORDS = 2
KARAOKE_MAX_WORDS = 4
KARAOKE_CHUNK_MAX = 1.45         # a line is never held longer than ~1.5s
KARAOKE_GAP_BREAK = 0.35         # a pause this long starts a new line
KARAOKE_HOLD = 0.12              # tail after the last word before a pause
KARAOKE_FADE_IN = 60             # ms, ~2 frames at 30fps
KARAOKE_FADE_OUT = 180           # ms of dim-and-vanish at the end of a line

# Advance widths of heavy geometric caps, as a fraction of the font size.
# Used to keep every line on ONE line (no wrapping) without pulling Pillow in.
_CAP_W = {
    "I": 0.32, "J": 0.50, "L": 0.53, "T": 0.61, "F": 0.57, "E": 0.57,
    "S": 0.62, "M": 0.93, "W": 0.95, "N": 0.78, "U": 0.75, "H": 0.77,
    "D": 0.77, "O": 0.81, "Q": 0.81, "G": 0.76, "C": 0.73, "B": 0.71,
    "P": 0.67, "R": 0.71, "A": 0.75, "V": 0.70, "X": 0.70, "Y": 0.69,
    "Z": 0.65, "K": 0.73, " ": 0.28, ".": 0.30, ",": 0.30, "?": 0.60,
    "!": 0.30, "'": 0.22, "-": 0.40, ":": 0.30, ";": 0.30, '"': 0.45,
}

PRESETS = ("classic", "hook", "karaoke")
MOTIONS = ("stationary", "moving")

# placement -> (alignment tag, MarginV as a fraction of the canvas height)
#
#   bottom: \an2, 18% above the bottom edge — clears the ~15-20% platform UI
#           safe zone (Shorts / Reels / TikTok captions, buttons and handles).
#   top:    \an8, 10% below the top edge — keeps clear of the status bar.
#   center: \an5, vertically centred (ASS ignores MarginV for middle alignment).
PLACEMENTS: Dict[str, tuple] = {
    "bottom": ("\\an2", 0.18),
    "center": ("\\an5", 0.00),
    "top": ("\\an8", 0.10),
}

# The Karaoke preset sits lower-middle: text baseline at 69% of the height
# (y ≈ 880 of 1280), i.e. 31% of MarginV for a bottom-anchored line.
KARAOKE_BOTTOM_MARGIN = 0.31
KARAOKE_MARGIN_LR = int(round(PLAY_W * 0.05))          # 5% safe side margins
KARAOKE_MARGIN_V = int(round(PLAY_H * KARAOKE_BOTTOM_MARGIN))   # 595px


# ---------------------------------------------------------------------------
# Colour helpers (hex -> ASS &HAABBGGRR)
# ---------------------------------------------------------------------------
def hex_to_ass(hex_color: str, alpha: int = 0) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b.upper()}{g.upper()}{r.upper()}"


def hex_to_bgr(hex_color: str) -> str:
    """``&HBBGGRR`` — the form an inline ``\\1c`` override tag wants."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return f"&H{h[4:6].upper()}{h[2:4].upper()}{h[0:2].upper()}"


def ass_time(sec: float) -> str:
    """``H:MM:SS.cc`` (ASS uses centiseconds)."""
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def palette_accent(palette: str) -> str:
    pal = PALETTES.get(palette, PALETTES["classic"])
    return pal.get("accent") or pal.get("highlight") or "#FFE600"


# ---------------------------------------------------------------------------
# Placement helpers
# ---------------------------------------------------------------------------
def placement_tag(placement: str) -> str:
    return PLACEMENTS.get(placement, PLACEMENTS["bottom"])[0]


def placement_margin(placement: str) -> int:
    return int(round(PLAY_H * PLACEMENTS.get(placement, PLACEMENTS["bottom"])[1]))


def karaoke_margin(placement: str) -> int:
    """Karaoke lines live higher up than plain captions (lower-middle)."""
    table = dict(PLACEMENTS)
    table["bottom"] = ("\\an2", KARAOKE_BOTTOM_MARGIN)
    return int(round(PLAY_H * table.get(placement, table["bottom"])[1]))


# ---------------------------------------------------------------------------
# Style block
# ---------------------------------------------------------------------------
def build_header(font: str, size_key: str, palette: str,
                 outline: Optional[int] = None, preset: str = "classic",
                 motion: str = "stationary") -> str:
    pal = PALETTES.get(palette, PALETTES["classic"])
    hook = preset == "hook"
    kara = preset == "karaoke"
    # Word highlight (Karaoke preset or "moving" motion) always renders white
    # body text with a single yellow word, whatever palette was stored.
    word_hi = kara or (motion == "moving" and not hook)
    table = KARAOKE_SIZES if kara else HOOK_SIZES if hook else SIZES
    px = table.get(size_key, table["medium"])
    if outline is None:
        outline = 5 if (hook or palette != "contrast") else 8
    if word_hi:
        # Word highlight: white body, one golden-yellow active word, chunky
        # black outline and a soft shadow. Nothing else is tinted.
        primary = KARAOKE_BASE
        highlight = KARAOKE_HIGHLIGHT
        outline_c = KARAOKE_OUTLINE
        outline = 6
    elif hook:
        # Hook style: white body text, accented active word and a black
        # 4-6px outline + soft shadow so it reads on any background.
        primary = "#FFFFFF"
        highlight = palette_accent(palette)
        outline_c = "#000000"
    else:
        primary = pal["primary"]
        highlight = pal["highlight"]
        outline_c = pal["outline"]
    font_name = font if font in FONTS else FONTS[0]
    p_c, h_c, o_c = hex_to_ass(primary), hex_to_ass(highlight), hex_to_ass(outline_c)
    back = hex_to_ass(outline_c, alpha=0x80)
    tail = f"-1,0,0,0,100,100,0,0,1,{outline},2,5,40,40,60,1"

    styles = [
        # dim context line / stationary captions
        f"Style: Base,{font_name},{px},{p_c},{h_c},{o_c},{back},{tail}",
        # SecondaryColour is the unspoken text, PrimaryColour the spoken fill,
        # so each word lights up exactly when it is said.
        f"Style: Karaoke,{font_name},{px},{h_c},{p_c},{o_c},{back},{tail}",
        f"Style: Active,{font_name},{px},{h_c},{h_c},{o_c},{back},{tail}",
    ]
    if hook:
        styles.append(f"Style: Hook,{font_name},{px},{p_c},{h_c},{o_c},{back},{tail}")
    if word_hi:
        # Spacing -1 (very slightly tight), BorderStyle 1 with a rounded 6px
        # outline, 4px soft shadow at ~65% opacity, Alignment 2 + 5% side
        # margins and a lower-middle MarginV so the line never jumps.
        kara_back = hex_to_ass(KARAOKE_OUTLINE, alpha=0x59)
        styles.append(
            f"Style: KaraokeLine,{font_name},{px},{p_c},{h_c},{o_c},{kara_back},"
            f"-1,0,0,0,100,100,-1,0,1,{outline},4,2,"
            f"{KARAOKE_MARGIN_LR},{KARAOKE_MARGIN_LR},{KARAOKE_MARGIN_V},1"
        )
    return (
        "[Script Info]\n"
        "; Generated by WayinVideo Clip Maker\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {PLAY_W}\n"
        f"PlayResY: {PLAY_H}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        + "\n".join(styles) + "\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def pos_tag(placement: str) -> str:
    """Backwards-compatible alias: the alignment tag only (no absolute \\pos)."""
    return placement_tag(placement)


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------
def _clean(word: str) -> str:
    return word.replace("{", "(").replace("}", ")").replace("\\", "")


def _dialogue(start: float, end: float, style: str, placement: str,
              margin_v: int, text: str, blur: int = 2) -> str:
    return (
        f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style},,0,0,{margin_v},,"
        f"{{{placement_tag(placement)}\\blur{blur}}}{text}"
    )


def events_stationary(segments: List[Dict[str, Any]], placement: str) -> List[str]:
    """One dialogue line per transcript segment, fixed position."""
    lines: List[str] = []
    margin = placement_margin(placement)
    for seg in segments:
        text = _clean(seg.get("text", "").strip())
        if not text:
            continue
        # keep each cue short enough to stay on one or two lines
        offset = 0.0
        seg_start, seg_end = float(seg["start"]), float(seg["end"])
        span_total = max(0.1, seg_end - seg_start)
        for i in range(0, len(text), 78):
            chunk = text[i : i + 78].strip()
            if not chunk:
                continue
            share = len(chunk) / max(1, len(text))
            start = seg_start + offset
            span = span_total * share
            offset += span
            lines.append(_dialogue(start, start + span, "Base", placement, margin, chunk))
    return lines


def events_moving(segments: List[Dict[str, Any]], placement: str,
                  size_key: str = "medium") -> List[str]:
    """Word highlight: a white caption line, the spoken word pops out yellow.

    This is the same renderer as the Karaoke preset (per-word events, one lit
    word at a time, pop + settle on the active word), so the "Word highlight"
    motion and the Karaoke preset always look alike. The only difference is
    that this path uses the normal size table.
    """
    return events_karaoke(segments, placement, size_key, sizes=SIZES)


# ---------------------------------------------------------------------------
# Hook preset (viral 1-3 word pop-in captions)
# ---------------------------------------------------------------------------
HOOK_MAX_WORDS = 3
HOOK_GAP_BREAK = 0.40        # start a new cue after a pause this long
HOOK_HOLD = 0.20             # keep the last word up for a beat
HOOK_POP = ("\\fscx80\\fscy80\\t(0,60,\\fscx110\\fscy110)"
            "\\t(60,120,\\fscx100\\fscy100)")
HOOK_POWER_WORDS = frozenset({
    "secret", "free", "never", "always", "stop", "why", "how", "best", "worst",
    "money", "hack", "truth", "lie", "nobody", "everyone", "instantly", "wrong",
    "mistake", "warning", "proof", "watch", "listen",
})


def _is_keyword(word: str) -> bool:
    w = word.strip(".,!?;:\"'()[]").lower()
    return bool(w) and (any(c.isdigit() for c in w) or w in HOOK_POWER_WORDS)


def _hook_chunks(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group word timestamps into cues of at most ``HOOK_MAX_WORDS`` words."""
    chunks: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for w in words:
        if cur and (len(cur) >= HOOK_MAX_WORDS
                    or float(w["s"]) - float(cur[-1]["e"]) > HOOK_GAP_BREAK):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def _hook_text(words: List[Dict[str, Any]], active: int, accent_bgr: str) -> str:
    """UPPERCASE cue text; the active word pops in and carries the accent."""
    parts: List[str] = []
    for i, w in enumerate(words):
        word = _clean(str(w.get("w", ""))).upper()
        if not word:
            continue
        if i == active:
            parts.append(f"{{\\1c{accent_bgr}&{HOOK_POP}}}{word}")
        elif _is_keyword(word):
            parts.append(f"{{\\1c{accent_bgr}&}}{word}")
        else:
            parts.append(word)
    return " ".join(parts)


def events_hook(segments: List[Dict[str, Any]], placement: str,
                palette: str = "classic") -> List[str]:
    """Viral hook captions.

    ALL CAPS, one to three words on screen at a time, the word being spoken in
    the accent colour while the rest stay white, every word popping in with a
    ``\\t`` 80% -> 110% -> 100% scale over ~120ms.
    """
    accent = hex_to_bgr(palette_accent(palette))
    margin = placement_margin(placement)
    lines: List[str] = []
    for seg in segments:
        seg_start, seg_end = float(seg["start"]), float(seg["end"])
        words = [w for w in seg.get("words", []) if str(w.get("w") or "").strip()]
        if not words:
            # no word timings — split the line into 3-word cues spread evenly
            text = _clean(seg.get("text", "").strip())
            if not text:
                continue
            raw = text.upper().split()
            groups = [raw[i : i + HOOK_MAX_WORDS] for i in range(0, len(raw), HOOK_MAX_WORDS)]
            span = max(0.1, seg_end - seg_start)
            for gi, group in enumerate(groups):
                a = seg_start + span * gi / max(1, len(groups))
                b = seg_start + span * (gi + 1) / max(1, len(groups))
                lines.append(_dialogue(a, max(a + 0.2, b), "Hook", placement, margin,
                                       "{" + HOOK_POP + "}" + " ".join(group)))
            continue

        chunks = _hook_chunks(words)
        for ci, chunk in enumerate(chunks):
            # a cue never runs past the first word of the next one
            limit = float(chunks[ci + 1][0]["s"]) if ci + 1 < len(chunks) else seg_end
            limit = max(seg_start, min(limit, seg_end))
            for i, w in enumerate(chunk):
                start = max(seg_start, min(float(w["s"]), seg_end))
                if i + 1 < len(chunk):
                    # keep the word up until the next one lands (no flicker)
                    end = max(start + 0.05, min(float(chunk[i + 1]["s"]), seg_end))
                else:
                    end = max(start + 0.05, min(float(w["e"]) + HOOK_HOLD, limit))
                if end <= start:
                    end = start + 0.08
                text = _hook_text(chunk[: i + 1], i, accent)
                lines.append(_dialogue(start, end, "Hook", placement, margin, text))
    return lines


# ---------------------------------------------------------------------------
# Karaoke preset (2-4 words, one line, one highlighted word at a time)
# ---------------------------------------------------------------------------
def _caps_width(text: str, px: int) -> float:
    """Approximate rendered width of an ALL-CAPS line in pixels."""
    return sum(_CAP_W.get(ch, 0.66) for ch in text) * px


def _karaoke_max_line() -> float:
    """Widest a line may get before it would have to wrap.

    An absolute pixel width: ``_caps_width`` already scales with the font size,
    so scaling the limit as well would shrink the safe width twice over.
    """
    return (PLAY_W - 2 * KARAOKE_MARGIN_LR) * 0.97


def _karaoke_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten the transcript into one time-ordered word list.

    Segments without word timings (older transcripts) are spread evenly over
    their own span so the preset still produces word-level cues.
    """
    out: List[Dict[str, Any]] = []
    for seg in segments:
        words = [w for w in (seg.get("words") or [])
                 if str(w.get("w") or "").strip()]
        if words:
            for w in words:
                out.append({"w": str(w["w"]).strip(),
                            "s": float(w["s"]), "e": float(w["e"])})
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start, end = float(seg["start"]), float(seg["end"])
        pieces = text.split()
        step = max(0.1, end - start) / max(1, len(pieces))
        for i, piece in enumerate(pieces):
            out.append({"w": piece, "s": start + i * step, "e": start + (i + 1) * step})
    out.sort(key=lambda w: w["s"])
    return out


def _karaoke_chunks(words: List[Dict[str, Any]], px: int) -> List[List[Dict[str, Any]]]:
    """Group words into speech-rhythm chunks of 2-4 words / ~1-1.5s."""
    limit = _karaoke_max_line()
    chunks: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for w in words:
        if cur:
            gap = w["s"] - cur[-1]["e"]
            span = w["e"] - cur[0]["s"]
            line = " ".join([c["w"] for c in cur] + [w["w"]]).upper()
            if (len(cur) >= KARAOKE_MAX_WORDS or gap > KARAOKE_GAP_BREAK
                    or span > KARAOKE_CHUNK_MAX or _caps_width(line, px) > limit):
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)

    # A lone trailing word reads badly: fold it into the previous line when
    # that line can still hold it (<=4 words, <=1.5s, still one line wide).
    merged: List[List[Dict[str, Any]]] = []
    for chunk in chunks:
        if (len(chunk) == 1 and merged and len(merged[-1]) < KARAOKE_MAX_WORDS
                and chunk[0]["s"] - merged[-1][-1]["e"] <= KARAOKE_GAP_BREAK):
            candidate = merged[-1] + chunk
            line = " ".join(c["w"] for c in candidate).upper()
            if (candidate[-1]["e"] - candidate[0]["s"] <= KARAOKE_CHUNK_MAX
                    and _caps_width(line, px) <= limit):
                merged[-1] = candidate
                continue
        merged.append(chunk)

    # Same idea forwards: a lone word joins the line that follows it instead
    # of sitting on screen by itself.
    out: List[List[Dict[str, Any]]] = []
    i = 0
    while i < len(merged):
        chunk = merged[i]
        if (len(chunk) == 1 and i + 1 < len(merged)
                and len(merged[i + 1]) < KARAOKE_MAX_WORDS
                and merged[i + 1][0]["s"] - chunk[0]["e"] <= KARAOKE_GAP_BREAK):
            candidate = chunk + merged[i + 1]
            line = " ".join(c["w"] for c in candidate).upper()
            if (candidate[-1]["e"] - candidate[0]["s"] <= KARAOKE_CHUNK_MAX + 0.15
                    and _caps_width(line, px) <= limit):
                out.append(candidate)
                i += 2
                continue
        out.append(chunk)
        i += 1
    return out


def _pop_tags(word: str, px: int, hi_bgr: str) -> str:
    """Inline override that makes the spoken word punch out in yellow.

    The line is centred, so growing the active word pushes its neighbours out
    symmetrically by a few pixels and pulls them back as it settles: a short
    breath on the line rather than a jump.
    """
    outline_bgr = hex_to_bgr(KARAOKE_OUTLINE)
    return (
        f"\\1c{hex_to_bgr(KARAOKE_FLASH)}&"
        f"\\3c{outline_bgr}\\bord{KARAOKE_POP_OUTLINE}"
        f"\\fscx{KARAOKE_POP_SCALE}\\fscy{KARAOKE_POP_SCALE}"
        f"\\t(0,{KARAOKE_POP_MS},"
        f"\\1c{hi_bgr}\\bord6\\fscx100\\fscy100)"
    )


# Every run restates the full set of tags the pop touches: an inline block
# applies to the rest of the line until it is overridden, so a plain colour tag
# after the popped word would leave that word scaled and haloed too.
def _plain_tags(px: int) -> str:
    return (f"\\1c{hex_to_bgr(KARAOKE_BASE)}&\\3c{hex_to_bgr(KARAOKE_OUTLINE)}"
            f"\\bord6\\fscx100\\fscy100")


def _karaoke_text(chunk: List[Dict[str, Any]], active: int, hi_bgr: str,
                  px: int) -> str:
    """Whole line in white, the spoken word popping out in yellow."""
    parts: List[str] = []
    plain = _plain_tags(px)
    for i, w in enumerate(chunk):
        word = _clean(str(w["w"])).upper()
        if not word:
            continue
        if i == active:
            parts.append(f"{{{_pop_tags(word, px, hi_bgr)}}}{word}")
        else:
            parts.append(f"{{{plain}}}{word}")
    return " ".join(parts)


def _karaoke_dim_text(chunk: List[Dict[str, Any]]) -> str:
    """The finished line in dim grey, for its fade-out."""
    words = [_clean(str(w["w"])).upper() for w in chunk]
    return " ".join(f"{{\\1c{KARAOKE_DIM}&}}{w}" for w in words if w)


def events_karaoke(segments: List[Dict[str, Any]], placement: str = "bottom",
                   size_key: str = "medium",
                   sizes: Optional[Dict[str, int]] = None) -> List[str]:
    """Word highlight: a white line where each spoken word pops out in yellow.

    Each word of a chunk gets its own event spanning that word's start to the
    next word's start, so the yellow word travels left to right exactly with
    the speech while every other word stays white. The active word punches in
    slightly larger with a yellow halo and settles over ~120 ms (the pop never
    moves the rest of the line). The first word of a chunk is already lit when
    the chunk appears, chunks butt up against each other with no gap, and a
    finished line dims to grey and fades before a pause.
    """
    table = sizes or KARAOKE_SIZES
    px = table.get(size_key, table["medium"])
    hi_bgr = hex_to_bgr(KARAOKE_HIGHLIGHT)
    tag = placement_tag(placement)
    margin = karaoke_margin(placement)
    lines: List[str] = []

    chunks = _karaoke_chunks(_karaoke_words(segments), px)
    for ci, chunk in enumerate(chunks):
        start = float(chunk[0]["s"])
        last_end = float(chunk[-1]["e"])
        nxt = float(chunks[ci + 1][0]["s"]) if ci + 1 < len(chunks) else None
        if nxt is not None and nxt - last_end <= KARAOKE_GAP_BREAK and nxt > start:
            end = nxt                       # continuous speech: next line starts here
        else:
            end = last_end + KARAOKE_HOLD    # pause: get the text off screen
        end = max(end, start + 0.12)

        # Speech stops here and the line starts to dim; the last word's event
        # must not run past it or the yellow would fight the grey.
        fade_from = max(last_end, end - KARAOKE_FADE_OUT / 1000.0)
        speech_end = min(end, fade_from) if fade_from > start else end

        for i, w in enumerate(chunk):
            w_start = max(start, float(w["s"]))
            if i + 1 < len(chunk):
                w_end = max(w_start + 0.05, float(chunk[i + 1]["s"]))
            else:
                w_end = max(w_start + 0.05, speech_end)

            tags = f"{tag}\\q2\\bord6\\shad4"
            if i == 0:
                # instant pop-in with at most a 2-frame fade
                tags += f"\\fad({KARAOKE_FADE_IN},0)"

            text = _karaoke_text(chunk, i, hi_bgr, px)
            lines.append(
                f"Dialogue: 0,{ass_time(w_start)},{ass_time(w_end)},KaraokeLine,,"
                f"{KARAOKE_MARGIN_LR},{KARAOKE_MARGIN_LR},{margin},,"
                f"{{{tags}}}{text}"
            )

        if end - speech_end >= 0.04:
            # Greyed to ~40% opacity, then gone before the pause ends.
            ms = int(round((end - speech_end) * 1000))
            tags = f"{tag}\\q2\\bord6\\shad4\\alpha&H99&"
            tags += f"\\t(0,{ms},\\alpha&HFF&)"
            lines.append(
                f"Dialogue: 0,{ass_time(speech_end)},{ass_time(end)},KaraokeLine,,"
                f"{KARAOKE_MARGIN_LR},{KARAOKE_MARGIN_LR},{margin},,"
                f"{{{tags}}}{_karaoke_dim_text(chunk)}"
            )
    return lines


def generate_ass(
    segments: List[Dict[str, Any]],
    style: Dict[str, Any],
    out_path: Path | str,
) -> Path:
    """Write a complete ``.ass`` file for *segments* using *style*."""
    style = dict(style or {})
    motion = _pick(style, "motion", MOTIONS, "stationary")
    palette = _pick(style, "palette", PALETTES, "classic")
    font = _pick(style, "font", FONTS, "Montserrat")
    size = _pick(style, "size", SIZES, "medium")
    placement = _pick(style, "placement", PLACEMENTS, "bottom")
    preset = _pick(style, "preset", PRESETS, "classic")

    default_outline = 5 if (preset in ("hook", "karaoke") or palette != "contrast") else 8
    try:
        outline = int(style.get("outline", default_outline))
    except (TypeError, ValueError):
        log.warning("ass: unusable outline=%r — using %s", style.get("outline"), default_outline)
        outline = default_outline

    if preset == "karaoke":
        events = events_karaoke(segments, placement, size)
    elif preset == "hook":
        events = events_hook(segments, placement, palette)
    elif motion == "moving":
        events = events_moving(segments, placement, size)
    else:
        events = events_stationary(segments, placement)

    content = (build_header(font, size, palette, outline, preset, motion)
               + "\n".join(events) + "\n")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def _pick(style: Dict[str, Any], key: str, allowed, default):
    """Return ``style[key]`` when it is known, else warn and use *default*."""
    value = style.get(key, default)
    if value in allowed:
        return value
    log.warning("ass: unknown caption %s=%r — falling back to %r", key, value, default)
    return default


def style_summary(style: Dict[str, Any]) -> str:
    """Human readable one-liner used in API responses / logs."""
    s = dict(style or {})
    parts = [
        s.get("motion", "stationary"),
        s.get("palette", "classic"),
        s.get("font", "Montserrat"),
        s.get("size", "medium"),
        s.get("placement", "bottom"),
    ]
    if s.get("preset") in PRESETS[1:]:
        parts.insert(0, str(s["preset"]))
    return " · ".join(str(p) for p in parts)
