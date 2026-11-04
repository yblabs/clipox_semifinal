"""Project timeline → FFmpeg filter graph → exported MP4.

One function does the whole thing: :func:`render_project`. Everything the editor
can author is compiled here:

* clips (trim, order, speed, volume, per-clip layout, transitions between clips)
* stackable effects, applied only inside their own time range
* B-roll overlays and the brand logo
* text items, the AI hook, subtitles and the watermark (one burned ASS track)
* music / sound effects with fades, looping and ducking under speech
* Enhance Speech on the base audio

The graph is built pairwise (concat/xfade accumulate one clip at a time) which
keeps it readable and well inside FFmpeg's limits even for 40+ clip timelines.

Preview parity: the frontend compositor reads the same project document and the
same registries in ``effects.py``/``captions.py``, so the layouts, captions,
effects and transitions you see are the ones exported here.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import FFMPEG_BIN, FFPROBE_BIN, ffmpeg_thread_args
from . import effects as fx
from .captions import build_ass, resolve_subtitle_style

ProgressFn = Optional[Callable[[float, str], None]]
CancelFn = Optional[Callable[[], bool]]


class RenderError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------
def probe(path: Path | str) -> Dict[str, Any]:
    """Duration / size / fps / stream kinds via ffprobe (never raises)."""
    info: Dict[str, Any] = {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0,
                            "has_video": False, "has_audio": False}
    try:
        proc = subprocess.run(
            [FFPROBE_BIN, "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=90,
        )
        data = json.loads(proc.stdout or "{}")
    except Exception:
        return info

    fmt = data.get("format") or {}
    try:
        info["duration"] = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        info["duration"] = 0.0
    for stream in data.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "video":
            info["has_video"] = True
            info["width"] = int(stream.get("width") or 0)
            info["height"] = int(stream.get("height") or 0)
            rate = str(stream.get("avg_frame_rate") or "0/1")
            try:
                num, den = rate.split("/")
                info["fps"] = float(num) / float(den or 1)
            except Exception:
                info["fps"] = 0.0
            if not info["duration"]:
                try:
                    info["duration"] = float(stream.get("duration") or 0.0)
                except (TypeError, ValueError):
                    pass
        elif kind == "audio":
            info["has_audio"] = True
    return info


def _esc_filter_path(path: str) -> str:
    """Escape a path for use inside an FFmpeg filter argument."""
    return (path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'"))


def atempo_chain(speed: float) -> str:
    """``atempo`` fragments for any factor in 0.25..4 (filter caps at 2 each)."""
    speed = max(0.25, min(4.0, float(speed or 1.0)))
    if abs(speed - 1.0) < 1e-3:
        return ""
    parts: List[str] = []
    remaining = speed
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.4f}")
    return ",".join(parts)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """A slice of the source, already placed on the output timeline."""

    clip_id: str
    src_start: float
    src_end: float
    out_start: float
    out_end: float
    clip: Dict[str, Any]
    effects: List[Dict[str, Any]] = field(default_factory=list)
    first_of_clip: bool = False
    last_of_clip: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.out_end - self.out_start)


class Timeline:
    """The ordered segments plus the source↔output time mapping."""

    def __init__(self, segments: List[Segment], clips: List[Dict[str, Any]], fps: int, w: int, h: int):
        self.segments = segments
        self.clips = clips
        self.fps = fps
        self.w = w
        self.h = h

    @property
    def duration(self) -> float:
        return max([s.out_end for s in self.segments] or [0.0])

    def clip_at(self, source_time: float) -> Optional[Dict[str, Any]]:
        for clip in self.clips:
            if float(clip.get("start", 0)) <= source_time <= float(clip.get("end", 0)):
                return clip
        return None

    def to_out(self, source_time: float) -> Optional[float]:
        """Source timestamp → output timestamp (``None`` when it was cut away)."""
        for seg in self.segments:
            if seg.src_start - 1e-6 <= source_time <= seg.src_end + 1e-6:
                speed = max(0.25, float(seg.clip.get("speed", 1.0) or 1.0))
                return seg.out_start + (source_time - seg.src_start) / speed
        return None

    def clip_durations(self) -> List[float]:
        """Output duration of each clip (summing its sub-segments)."""
        out: Dict[str, float] = {}
        for seg in self.segments:
            out[seg.clip_id] = out.get(seg.clip_id, 0.0) + seg.duration
        return [out.get(str(c.get("id")), 0.0) for c in self.clips]


def build_timeline(project: Dict[str, Any], *, fps: int, w: int, h: int) -> Timeline:
    """Expand the project's clips into output-timeline segments (effect-aware)."""
    segments: List[Segment] = []
    clips = list(project.get("clips") or [])
    cursor = 0.0
    for clip in clips:
        start = float(clip.get("start", 0.0))
        end = float(clip.get("end", 0.0))
        if end - start <= 0.05:
            continue
        speed = max(0.25, min(4.0, float(clip.get("speed", 1.0) or 1.0)))
        # split the clip where an effect starts/ends so looks are range-accurate
        bounds = {start, end}
        for effect in clip.get("effects") or []:
            for key in ("start", "end"):
                value = effect.get(key)
                if value is None:
                    continue
                value = max(start, min(end, float(value)))
                bounds.add(value)
        ordered = sorted(b for b in bounds if start - 1e-6 <= b <= end + 1e-6)
        made: List[Segment] = []
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            if b - a <= 0.02:
                continue
            mid = (a + b) / 2
            active = [
                e for e in (clip.get("effects") or [])
                if float(e.get("start", start)) - 1e-6 <= mid <= float(e.get("end", end)) + 1e-6
            ]
            made.append(Segment(
                clip_id=str(clip.get("id")),
                src_start=a, src_end=b,
                out_start=cursor, out_end=cursor + (b - a) / speed,
                clip=clip, effects=active,
            ))
            cursor = made[-1].out_end
        if made:
            made[0].first_of_clip = True
            made[-1].last_of_clip = True
            # a transition shortens the output: the next clip overlaps backwards
            transition = clip.get("transition") or {}
            if segments and fx.transition_key(transition.get("type", "none")):
                overlap = max(0.0, min(float(transition.get("duration", 0.4) or 0.4),
                                       made[0].duration / 2, 2.0))
                shift = -overlap
                for seg in made:
                    seg.out_start += shift
                    seg.out_end += shift
                cursor = made[-1].out_end
        segments.extend(made)
    return Timeline(segments, [c for c in clips], fps, w, h)


# ---------------------------------------------------------------------------
# Media helpers for overlays
# ---------------------------------------------------------------------------
def _media_path(project: Dict[str, Any], ref: str) -> Optional[str]:
    """Resolve an asset reference (``/media/...`` or a local path) to a file."""
    from ..config import MEDIA_DIR, MEDIA_MOUNT

    ref = str(ref or "").strip()
    if not ref:
        return None
    if ref.startswith(f"/{MEDIA_MOUNT}/"):
        candidate = Path(MEDIA_DIR) / ref[len(MEDIA_MOUNT) + 2:]
        return str(candidate) if candidate.is_file() else None
    if os.path.isfile(ref):
        return ref
    source = project.get("source") or {}
    if ref == "source" and source.get("path") and os.path.isfile(source["path"]):
        return source["path"]
    return None


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------
@dataclass
class RenderOptions:
    resolution: str = "1080p"
    fps: int = 30
    quality: str = "high"          # high | balanced | small
    burn_subtitles: bool = True
    watermark: bool = False
    filename: Optional[str] = None


def render_project(
    project: Dict[str, Any],
    out_path: Path | str,
    options: Optional[RenderOptions] = None,
    *,
    progress: ProgressFn = None,
    should_cancel: CancelFn = None,
) -> Dict[str, Any]:
    """Render *project* to *out_path* and return a small report dict."""
    opts = options or RenderOptions()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source = project.get("source") or {}
    src_path = source.get("path") or _media_path(project, source.get("media", ""))
    if not src_path or not os.path.isfile(str(src_path)):
        raise RenderError("project has no usable source video (re-upload or re-process it)")

    w, h = fx.canvas_size(project.get("aspect", "9:16"), opts.resolution)
    fps = max(12, min(60, int(opts.fps or 30)))
    timeline = build_timeline(project, fps=fps, w=w, h=h)
    if not timeline.segments:
        raise RenderError("the timeline is empty – add at least one clip")

    def report(pct: float, stage: str) -> None:
        if progress:
            progress(max(0.0, min(100.0, pct)), stage)

    def check() -> None:
        if should_cancel and should_cancel():
            raise Cancelled("render cancelled")

    report(2, "Preparing filter graph")
    src_info = probe(src_path)
    total_duration = timeline.duration
    clips = timeline.clips
    global_layout = project.get("layout", "fit")

    inputs: List[List[str]] = [["-i", str(src_path)]]
    graph: List[str] = []
    duration_map: Dict[str, float] = {}
    for seg in timeline.segments:
        duration_map.setdefault(seg.clip_id, 0.0)
        duration_map[seg.clip_id] += seg.duration

    def add_input(args: List[str]) -> int:
        inputs.append(args)
        return len(inputs) - 1

    def add_still(ref: str, kind: str = "") -> Optional[int]:
        path = _media_path(project, ref)
        if not path:
            return None
        if kind == "audio" or path.lower().endswith((".mp3", ".wav", ".m4a", ".aac", ".ogg")):
            return add_input(["-stream_loop", "-1", "-i", path])
        return add_input(["-i", path])

    # ---------------------------------------------------------------- video
    # One input feeds every trimmed slice, so it has to be split first.
    n_segments = len(timeline.segments)
    graph.append(f"[0:v]split={n_segments}" + "".join(f"[sv{i}]" for i in range(n_segments)))
    if src_info["has_audio"]:
        graph.append(f"[0:a]asplit={n_segments}" + "".join(f"[sa{i}]" for i in range(n_segments)))

    segment_labels: List[Tuple[str, str]] = []   # (video label, audio label)
    for i, seg in enumerate(timeline.segments):
        speed = max(0.25, min(4.0, float(seg.clip.get("speed", 1.0) or 1.0)))
        layout = seg.clip.get("layout") or global_layout
        chain = [
            f"[sv{i}]trim=start={seg.src_start:.3f}:end={seg.src_end:.3f},setpts=PTS-STARTPTS",
        ]
        if abs(speed - 1.0) > 1e-3:
            chain.append(f"setpts=PTS/{speed:.4f}")
        chain.append(f"fps={fps},format=yuv420p,setsar=1,settb=AVTB")

        # crop happens in SOURCE space so a reframe can follow the speaker,
        # then the layout fits the (cropped) frame onto the output canvas
        crop = seg.clip.get("crop") or {}
        transform = seg.clip.get("transform") or {}
        cw = float(crop.get("w", 1) or 1)
        ch = float(crop.get("h", 1) or 1)
        pre: List[str] = []
        if cw < 0.999 or ch < 0.999:
            cx = min(1.0, max(0.0, float(crop.get("x", 0.5) if crop.get("x") is not None else 0.5)))
            cy = min(1.0, max(0.0, float(crop.get("y", 0.5) if crop.get("y") is not None else 0.5)))
            cw = min(1.0, max(0.08, cw))
            ch = min(1.0, max(0.08, ch))
            pre.append(f"crop=w=iw*{cw:.4f}:h=ih*{ch:.4f}:x=(iw-ow)*{cx:.4f}:y=(ih-oh)*{cy:.4f}")
            pre.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
        # the Details panel nudges the framing by panning inside the slack
        pan_x = float(transform.get("x", 0) or 0)
        pan_y = float(transform.get("y", 0) or 0)
        if abs(pan_x) > 1e-3 or abs(pan_y) > 1e-3:
            pre.append(f"crop=w=iw:h=ih:x=(iw-ow)/2+iw*{pan_x:.4f}:y=(ih-oh)/2+ih*{pan_y:.4f}")
        if pre:
            graph.append(",".join(chain) + f"[raw{i}]")
            graph.append(f"[raw{i}]" + ",".join(pre) + f"[pre{i}]")
        else:
            graph.append(",".join(chain) + f"[pre{i}]")

        # layout fragments carry their own labels, so they join with ";"
        layout_chain = fx.video_layout(layout, w, h)
        graph.append(f"[pre{i}]" + ";".join(layout_chain) + f"[lay{i}]")
        current = f"lay{i}"

        intensity_default = 0.5
        for j, effect in enumerate(seg.effects):
            frags = fx.effect_chain(str(effect.get("type")), effect.get("intensity", intensity_default), w, h, fps)
            if not frags:
                continue
            graph.append(f"[{current}]" + ",".join(frags) + f"[eff{i}_{j}]")
            current = f"eff{i}_{j}"

        post: List[str] = [f"scale={w}:{h}"]
        scale_factor = float(transform.get("scale", 1) or 1)
        if abs(scale_factor - 1) > 1e-3:
            factor = max(0.1, min(3.0, scale_factor))
            post = [f"scale={w}:{h},scale=iw*{factor:.4f}:ih*{factor:.4f},"
                    f"crop={w}:{h}"]

        if transform.get("flip_h"):
            post.append("hflip")
        rotation = float(transform.get("rotation", 0) or 0)
        if abs(rotation) > 0.01:
            post.append(f"rotate={rotation}*PI/180:c=black@0:ow=iw:oh=ih")
        opacity = float(transform.get("opacity", 1) if transform.get("opacity") is not None else 1)
        if opacity < 0.999:
            post.append(f"format=rgba,colorchannelmixer=aa={max(0.05, min(1.0, opacity)):.3f}")
        post.append("format=yuv420p")
        # xfade needs identical timebases on both inputs
        post.append("settb=AVTB")
        graph.append(f"[{current}]" + ",".join(post) + f"[v{i}]")

        volume = max(0.0, float(seg.clip.get("volume", 1.0) if seg.clip.get("volume") is not None else 1.0))
        fades = ""
        clip_len = duration_map.get(seg.clip_id, seg.duration)
        if seg.first_of_clip and float(seg.clip.get("fade_in", 0) or 0) > 0.02:
            fades += f",afade=t=in:st=0:d={float(seg.clip['fade_in']):.2f}"
        if seg.last_of_clip and float(seg.clip.get("fade_out", 0) or 0) > 0.02:
            d = float(seg.clip["fade_out"])
            fades += f",afade=t=out:st={max(0.0, clip_len - d):.2f}:d={d:.2f}"
        if src_info["has_audio"]:
            audio_chain = [
                f"[sa{i}]atrim=start={seg.src_start:.3f}:end={seg.src_end:.3f},asetpts=PTS-STARTPTS",
            ]
            tempo = atempo_chain(speed)
            if tempo:
                audio_chain.append(tempo)
            audio_chain.append(f"volume={volume:.3f}{fades}")
            graph.append(f"{','.join(audio_chain)}[a{i}]")
            segment_labels.append((f"v{i}", f"a{i}"))
        else:
            graph.append(f"anullsrc=channel_layout=stereo:sample_rate=48000,atrim=0:{seg.duration:.3f}[a{i}]")
            segment_labels.append((f"v{i}", f"a{i}"))

    # ------------------------------------------------- clip level assembly
    groups: List[Tuple[List[str], List[str]]] = []
    for clip in clips:
        vids = [v for v, _ in segment_labels if timeline.segments[int(v[1:])].clip_id == str(clip.get("id"))]
        aids = [a for _, a in segment_labels if timeline.segments[int(a[1:])].clip_id == str(clip.get("id"))]
        if vids:
            groups.append((vids, aids))
    if not groups:
        raise RenderError("no clips to render")

    acc_v, acc_a = None, None
    acc_duration = 0.0
    for index, (vids, aids) in enumerate(groups):
        if len(vids) == 1:
            clip_v, clip_a = vids[0], aids[0]
        else:
            graph.append("".join(f"[{v}]" for v in vids) + f"concat=n={len(vids)}:v=1:a=0[cv{index}]")
            graph.append("".join(f"[{a}]" for a in aids) + f"concat=n={len(aids)}:v=0:a=1[ca{index}]")
            clip_v, clip_a = f"cv{index}", f"ca{index}"
            graph.append(f"[{clip_v}]format=yuv420p,setsar=1[cvf{index}]")
            clip_v = f"cvf{index}"

        clip_duration = duration_map.get(str(clips[index].get("id")), 0.0)
        if acc_v is None:
            acc_v, acc_a = clip_v, clip_a
            acc_duration = clip_duration
            continue

        transition = clips[index].get("transition") or {}
        key = fx.transition_key(transition.get("type", "none"))
        overlap = max(0.0, min(float(transition.get("duration", 0.4) or 0.4), clip_duration / 2, 2.0)) if key else 0.0
        if key and overlap > 0.05 and acc_duration > overlap:
            offset = max(0.0, acc_duration - overlap)
            graph.append(
                f"[{acc_v}][{clip_v}]xfade=transition={key}:duration={overlap:.3f}:offset={offset:.3f}[xv{index}]")
            graph.append(f"[{acc_a}][{clip_a}]acrossfade=d={overlap:.3f}:c1=tri:c2=tri[xa{index}]")
            acc_v, acc_a = f"xv{index}", f"xa{index}"
            acc_duration = offset + clip_duration
        else:
            graph.append(f"[{acc_v}][{clip_v}]concat=n=2:v=1:a=0[nv{index}]")
            graph.append(f"[{acc_a}][{clip_a}]concat=n=2:v=0:a=1[na{index}]")
            acc_v, acc_a = f"nv{index}", f"na{index}"
            acc_duration += clip_duration
        report(6 + (index / max(1, len(groups))) * 24, f"Assembling clip {index + 1}/{len(groups)}")

    total_duration = acc_duration

    # ------------------------------------------------------- B-roll overlay
    for k, item in enumerate(project.get("broll") or []):
        idx = add_still(str(item.get("media") or item.get("url") or ""))
        if idx is None:
            continue
        start = max(0.0, float(item.get("start", 0)))
        end = max(start + 0.2, float(item.get("end", start + 3)))
        from_time = max(0.0, float(item.get("from", 0) or 0))
        graph.append(
            f"[{idx}:v]trim=start={from_time:.3f}:duration={end - start:.3f},setpts=PTS-STARTPTS+{start:.3f}/TB,"
            f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps},"
            f"format=yuv420p,setsar=1[br{k}]")
        graph.append(f"[{acc_v}][br{k}]overlay=0:0:eof_action=pass:format=auto[ov{k}]")
        acc_v = f"ov{k}"

    # -------------------------------------------------------- logo watermark
    brand = project.get("brand") or {}
    logo = brand.get("logo") or {}
    if logo.get("enabled") and logo.get("asset"):
        idx = add_still(str(logo["asset"]))
        if idx is not None:
            scale = max(0.03, min(0.6, float(logo.get("scale", 0.16) or 0.16)))
            alpha = max(0.05, min(1.0, float(logo.get("opacity", 0.9) or 0.9)))
            lw = max(2, int(w * scale) // 2 * 2)
            lx = int(max(0.0, min(1.0, float(logo.get("x", 0.08)))) * w)
            ly = int(max(0.0, min(1.0, float(logo.get("y", 0.06)))) * h)
            graph.append(
                f"[{idx}:v]scale={lw}:-2,format=rgba,colorchannelmixer=aa={alpha:.2f},"
                f"loop=loop=-1:size=1:start=0,trim=duration={max(0.2, total_duration):.3f},"
                f"setpts=PTS-STARTPTS[lg]")
            graph.append(f"[{acc_v}][lg]overlay={lx}:{ly}:eof_action=repeat[ovl]")
            acc_v = "ovl"

    # ------------------------------------------------------- enhance speech
    audio_block = project.get("audio") if isinstance(project.get("audio"), dict) else {}
    baseline = bool(audio_block.get("enhance")) or bool((project.get("enhance") or {}).get("speech"))
    audio_filters: List[str] = []
    if baseline:
        audio_filters = ["afftdn=nr=12:nf=-25", "speechnorm=e=6.25:r=0.0001",
                         "loudnorm=I=-16:TP=-1.5:LRA=11"]
    if audio_filters:
        graph.append(f"[{acc_a}]" + ",".join(audio_filters) + "[aenh]")
        acc_a = "aenh"

    # ------------------------------------------------- music / sfx / ducking
    raw_audio = project.get("audio_items")
    if raw_audio is None:
        raw_audio = (project.get("audio") or {}).get("items") if isinstance(project.get("audio"), dict) else project.get("audio")
    music_labels: List[Tuple[str, bool]] = []
    for k, item in enumerate(raw_audio or []):
        if not isinstance(item, dict):
            continue
        idx = add_still(str(item.get("media") or item.get("url") or ""), "audio")
        if idx is None:
            continue
        start = max(0.0, float(item.get("start", 0)))
        end = max(start + 0.2, float(item.get("end", start + 10)))
        from_time = max(0.0, float(item.get("from", 0) or 0))
        volume = max(0.0, min(2.0, float(item.get("volume", 0.35) if item.get("volume") is not None else 0.35)))
        fade_in = max(0.0, float(item.get("fade_in", 0.5) or 0))
        fade_out = max(0.0, float(item.get("fade_out", 0.8) or 0))
        span = end - start
        chain = [
            f"atrim=start={from_time:.3f}:duration={span:.3f}",
            "asetpts=PTS-STARTPTS",
            f"volume={volume:.3f}",
        ]
        if fade_in > 0.02:
            chain.append(f"afade=t=in:st=0:d={min(fade_in, span / 2):.2f}")
        if fade_out > 0.02:
            chain.append(f"afade=t=out:st={max(0.0, span - min(fade_out, span / 2)):.2f}:"
                         f"d={min(fade_out, span / 2):.2f}")
        chain.append(f"adelay={int(start * 1000)}|{int(start * 1000)}")
        graph.append(f"[{idx}:a]" + ",".join(chain) + f"[mx{k}]")
        music_labels.append((f"mx{k}", bool(item.get("duck", True))))

    audio_out = acc_a
    if music_labels:
        bed = acc_a
        for k, (label, duck) in enumerate(music_labels):
            if duck:
                graph.append(f"[{bed}]asplit=2[sp{k}][sck{k}]")
                graph.append(f"[sck{k}]apad[scp{k}]")
                graph.append(f"[{label}][scp{k}]sidechaincompress="
                             f"threshold=0.035:ratio=8:attack=25:release=420:makeup=1[dk{k}]")
                label = f"dk{k}"
            else:
                graph.append(f"[{bed}]anull[sp{k}]")
            graph.append(f"[sp{k}][{label}]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[abase{k}]")
            bed = f"abase{k}"
        audio_out = bed

    # ------------------------------------------------------------- captions
    ass_path: Optional[Path] = None
    needs_text = bool(project.get("texts") or project.get("hooks"))
    subtitles_on = bool((project.get("subtitles") or {}).get("enabled", True))
    burn_subs = bool(opts.burn_subtitles and subtitles_on)
    if burn_subs or needs_text or opts.watermark:
        tmp_dir = Path(tempfile.mkdtemp(prefix="ass_"))
        ass_path = tmp_dir / "captions.ass"
        build_ass(
            project, ass_path, w=w, h=h, to_out=timeline.to_out,
            watermark=opts.watermark,
            subtitle_lines=None if burn_subs else [],
        )
        # no :shaping option – the bundled libass builds do not all expose it
        # fontsdir keeps libass on the vendored TTFs, the same files the browser
        # loads for the preview, so caption metrics (and therefore line wrapping)
        # match between preview and export.
        fontsdir = Path(__file__).resolve().parents[2] / "assets" / "fonts"
        font_arg = f":fontsdir='{_esc_filter_path(str(fontsdir))}'" if fontsdir.is_dir() else ""
        graph.append(f"[{acc_v}]subtitles=filename='{_esc_filter_path(str(ass_path))}'{font_arg}[vtxt]")
        acc_v = "vtxt"

    # -------------------------------------------------------------- encode
    crf = {"high": fx.RESOLUTIONS.get(opts.resolution, {"crf": 21})["crf"],
           "balanced": 24, "small": 28}.get(opts.quality, 21)
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error"]
    for args in inputs:
        cmd += args
    cmd += ["-filter_complex", ";".join(graph),
            "-map", f"[{acc_v}]", "-map", f"[{audio_out}]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            *ffmpeg_thread_args(),
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2",
            "-r", str(fps), "-g", str(fps * 2),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", "-progress", "pipe:1", "-nostats",
            str(out_path)]

    report(30, "Rendering video")
    started = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    tail: List[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if should_cancel and should_cancel():
                proc.kill()
                raise Cancelled("render cancelled")
            if line.startswith("out_time_ms="):
                try:
                    done = float(line.split("=", 1)[1]) / 1_000_000.0
                except ValueError:
                    continue
                pct = 30 + min(68.0, (done / max(0.1, total_duration)) * 68.0)
                report(pct, f"Rendering {int(min(100, done / max(0.1, total_duration) * 100))}%")
            elif line.startswith("progress=end"):
                report(99, "Finalising")
    finally:
        try:
            _, err = proc.communicate(timeout=30)
        except Exception:
            err = ""
        if err:
            tail = [ln for ln in err.strip().splitlines() if ln.strip()][-12:]
        if proc.poll() is None:
            proc.kill()
        if ass_path is not None:
            shutil.rmtree(ass_path.parent, ignore_errors=True)

    if proc.returncode != 0 or not out_path.is_file():
        raise RenderError("FFmpeg failed: " + (" | ".join(tail) or f"exit {proc.returncode}"))

    info = probe(out_path)
    report(100, "Done")
    return {
        "path": str(out_path),
        "duration": round(float(info.get("duration") or total_duration), 3),
        "width": w,
        "height": h,
        "fps": fps,
        "size_bytes": out_path.stat().st_size,
        "elapsed_sec": round(time.time() - started, 2),
        "clips": len(groups),
        "layout": global_layout,
    }


# ---------------------------------------------------------------------------
# Cheap helpers used by the API
# ---------------------------------------------------------------------------
def extract_frames(src: Path | str, out_dir: Path | str, *, count: int = 5,
                   timestamps: Optional[List[float]] = None, width: int = 480) -> List[str]:
    """Grab *count* evenly spaced frames (or explicit *timestamps*) as JPEGs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    duration = float(info.get("duration") or 0.0)
    if timestamps is None:
        if duration <= 0:
            timestamps = [0.0]
        else:
            count = max(1, count)
            timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]
    files: List[str] = []
    for i, ts in enumerate(timestamps):
        target = out_dir / f"frame_{i:02d}.jpg"
        proc = subprocess.run(
            [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, ts):.3f}",
             "-i", str(src), "-frames:v", "1", "-vf", f"scale={width}:-2", str(target)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0 and target.is_file():
            files.append(str(target))
    return files


def waveform(src: Path | str, *, buckets: int = 900) -> List[float]:
    """Peak envelope 0..1 for the timeline waveform (raw PCM, no numpy)."""
    from ..config import FFMPEG_BIN as FF

    try:
        proc = subprocess.run(
            [FF, "-hide_banner", "-loglevel", "error", "-i", str(src),
             "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
            capture_output=True, timeout=600,
        )
        raw = proc.stdout or b""
    except Exception:
        return []
    if len(raw) < 4:
        return []
    samples = len(raw) // 2
    step = max(1, samples // max(1, buckets))
    peaks: List[float] = []
    for start in range(0, samples, step):
        chunk = raw[start * 2:(start + step) * 2]
        peak = 0
        for i in range(0, len(chunk) - 1, 2):
            value = int.from_bytes(chunk[i:i + 2], "little", signed=True)
            peak = max(peak, abs(value))
        peaks.append(round(min(1.0, peak / 32767.0), 3))
    return peaks[:buckets]
