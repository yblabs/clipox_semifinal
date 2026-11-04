"""AI Clipping API — Part 1 to Part 7 of the clipping spec.

    POST /api/ai-clipping/inspect            link guard + preview card metadata
    POST /api/ai-clipping/upload/init        start a chunked, resumable upload
    PUT  /api/ai-clipping/upload/{id}/chunk  append one chunk (offset based)
    POST /api/ai-clipping/upload/{id}/complete  finalise + thumbnail
    GET  /api/ai-clipping/upload/{id}        resume state (bytes received)
    POST /api/ai-clipping/jobs               run the 7-stage pipeline
    GET  /api/ai-clipping/jobs/{task_id}     progress / result
    POST /api/ai-clipping/jobs/{task_id}/cancel
    GET  /api/ai-clipping/videos             Library of clipping projects
    GET  /api/ai-clipping/videos/{id}        one project with its moments
    ...  moments CRUD, hooks, platform copy, find moments ...
    POST /api/ai-clipping/export             render MP4s / ZIP (async job)
    POST /api/ai-clipping/montage            join clips with transitions
    POST /api/ai-clipping/share              shareable preview link
    ...  schedule + publish ...
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import credits, store
from ..config import FFMPEG_BIN, MEDIA_DIR, UPLOAD_DIR
from ..engine import clipper, clipping_job, media
from ..engine.clip_maker import generate_hooks, generate_metadata as platform_metadata
from ..engine.transcribe import format_timestamp
from ..tasks import STORE
from .common import media_url, start_task, status_of

def require_key(request: Request) -> None:
    """Optional API-key gate for the public REST surface.

    Local single-user mode stays open (no header, no check). Any request that
    *does* send ``X-API-Key`` must present a key created on the API page, and
    usage is counted so the usage table is real.
    """
    key = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if not key:
        return
    for item in store.apikeys.all():
        if item.get("key") == key:
            store.apikeys.patch(item["id"], {
                "last_used": store.now(),
                "calls": int(item.get("calls", 0)) + 1,
            })
            return
    raise HTTPException(status_code=401, detail="Invalid API key.")


router = APIRouter(prefix="/api/ai-clipping", tags=["ai-clipping"],
                   dependencies=[Depends(require_key)])

UPLOAD_MAX_MB = int(os.getenv("WV_UPLOAD_MAX_MB", "2048"))
CHUNK_SIZE = 4 * 1024 * 1024
CHUNK_DIR = Path(UPLOAD_DIR) / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
SHARE_DIR = Path(MEDIA_DIR) / "share"
SHARE_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTIONS = {"720p": 1280, "1080p": 1920, "4K": 3840}
ALLOWED_UPLOAD_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v",
                       ".mp3", ".wav", ".m4a", ".aac", ".ogg"}


def fire_webhooks(event: str, payload: Dict[str, Any]) -> None:
    """POST *event* to every registered webhook (fire and forget, one retry)."""
    hooks = store.Collection("webhooks").all()
    if not hooks:
        return
    body = json.dumps({"event": event, "sent_at": time.time(), "data": payload},
                      ensure_ascii=False).encode()

    def deliver(url: str) -> None:
        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    url, data=body, method="POST",
                    headers={"Content-Type": "application/json", "X-WV-Event": event},
                )
                urllib.request.urlopen(req, timeout=10).read()
                return
            except Exception:
                time.sleep(1.0 + attempt)

    for hook in hooks:
        if event == "test" or event in (hook.get("events") or [event]):
            threading.Thread(target=deliver, args=(hook.get("url", ""),), daemon=True).start()


# ---------------------------------------------------------------------------
# Meta / options
# ---------------------------------------------------------------------------
@router.get("/options")
def options() -> Dict[str, Any]:
    """Everything the input page needs to build its controls."""
    account = credits.account()
    return {
        "languages": [{"key": k, "label": v} for k, v in clipper.LANGUAGES],
        "clip_lengths": [{"key": k, **v} for k, v in clipper.CLIP_LENGTHS.items()],
        "platforms": [{"key": k, "label": clipper.PLATFORM_LABEL[k]} for k in clipper.PLATFORMS_UI],
        "layouts": clipper.LAYOUTS,
        "palettes": list(clipping_job.PALETTES),
        "fonts": list(clipping_job.FONTS),
        "sizes": list(clipping_job.SIZES),
        "placements": list(clipping_job.PLACEMENTS),
        "presets": list(clipping_job.PRESETS),
        "motions": list(clipping_job.MOTIONS),
        "resolutions": list(RESOLUTIONS),
        "upload": {"chunk_size": CHUNK_SIZE, "max_mb": UPLOAD_MAX_MB,
                   "accept": sorted(ALLOWED_UPLOAD_EXTS)},
        "credits": {"balance": account["balance"], "plan": account["plan"],
                    "unlimited": credits.UNLIMITED_CREDITS,
                    "price_per_minute": 1, "minimum": 5,
                    "upgrade_url": "/editor.html#pricing"},
    }


class InspectBody(BaseModel):
    url: str


@router.post("/inspect")
def inspect(body: InspectBody) -> Dict[str, Any]:
    """Validate a pasted link and return the preview card."""
    try:
        return clipper.inspect_url(body.url)
    except clipper.ClipperError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Chunked, resumable upload
# ---------------------------------------------------------------------------
def _chunk_meta_path(upload_id: str) -> Path:
    return CHUNK_DIR / f"{upload_id}.json"


def _chunk_file(upload_id: str) -> Path:
    return CHUNK_DIR / f"{upload_id}.part"


def _read_meta(upload_id: str) -> Dict[str, Any]:
    path = _chunk_meta_path(upload_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown upload — start it again.")
    return json.loads(path.read_text("utf-8"))


class UploadInit(BaseModel):
    filename: str
    size: int = 0


@router.post("/upload/init")
def upload_init(body: UploadInit) -> Dict[str, Any]:
    name = os.path.basename(body.filename or "video.mp4")
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(status_code=400, detail=(
            f"{ext or 'that file type'} is not supported. Use "
            + ", ".join(sorted(ALLOWED_UPLOAD_EXTS))))
    limit = UPLOAD_MAX_MB * 1024 * 1024
    if body.size and body.size > limit:
        raise HTTPException(status_code=413, detail=(
            f"File is {body.size / 1024 / 1024:.0f} MB — the limit is {UPLOAD_MAX_MB} MB."))
    upload_id = f"up_{uuid.uuid4().hex[:12]}"
    _chunk_file(upload_id).write_bytes(b"")
    _chunk_meta_path(upload_id).write_text(json.dumps({
        "id": upload_id, "filename": name, "size": body.size, "created_at": time.time(),
    }), "utf-8")
    return {"upload_id": upload_id, "chunk_size": CHUNK_SIZE,
            "received": 0, "size": body.size, "max_mb": UPLOAD_MAX_MB}


@router.get("/upload/{upload_id}")
def upload_state(upload_id: str) -> Dict[str, Any]:
    meta = _read_meta(upload_id)
    received = _chunk_file(upload_id).stat().st_size if _chunk_file(upload_id).is_file() else 0
    return {**meta, "received": received, "chunk_size": CHUNK_SIZE}


@router.put("/upload/{upload_id}/chunk")
async def upload_chunk(upload_id: str, request: Request, offset: int = 0) -> Dict[str, Any]:
    meta = _read_meta(upload_id)
    path = _chunk_file(upload_id)
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty chunk.")
    current = path.stat().st_size if path.is_file() else 0
    if offset > current:
        raise HTTPException(status_code=409, detail={
            "message": "Chunk arrived out of order.", "received": current})
    if offset < current:                     # retried chunk: rewind and rewrite
        with open(path, "r+b") as fh:
            fh.truncate(offset)
    with open(path, "ab") as fh:
        fh.write(payload)
    received = path.stat().st_size
    if meta.get("size") and received > int(meta["size"]):
        path.unlink(missing_ok=True)
        _chunk_meta_path(upload_id).unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="Upload is larger than declared size.")
    return {"upload_id": upload_id, "received": received, "size": meta.get("size", 0)}


@router.post("/upload/{upload_id}/complete")
def upload_complete(upload_id: str) -> Dict[str, Any]:
    meta = _read_meta(upload_id)
    part = _chunk_file(upload_id)
    if not part.is_file() or part.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="No bytes received for this upload.")
    if meta.get("size") and part.stat().st_size < int(meta["size"]):
        raise HTTPException(status_code=409, detail={
            "message": "Upload is incomplete — resume it.", "received": part.stat().st_size,
            "size": meta["size"]})

    ext = Path(meta["filename"]).suffix.lower()
    dest_dir = Path(UPLOAD_DIR) / "clips"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{int(time.time() * 1000)}_{os.path.basename(meta['filename']).replace(' ', '_')}"
    shutil.move(str(part), dest)
    _chunk_meta_path(upload_id).unlink(missing_ok=True)

    job_id = f"vid_{uuid.uuid4().hex[:12]}"
    out_dir = Path(MEDIA_DIR) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    duration, width, height, fps = 0.0, 0, 0, 0.0
    thumbnail = ""
    try:
        probe = media.video_meta(dest)
        duration = float(probe.get("duration") or 0)
        width, height, fps = int(probe.get("width") or 0), int(probe.get("height") or 0), float(probe.get("fps") or 0)
        if width:
            poster = media.make_thumbnail(dest, out_dir / "source.jpg", at_sec=min(2.0, max(0.2, duration / 4)))
            thumbnail = media_url(poster)
    except Exception:
        # audio-only files have no video stream but are still valid sources
        try:
            duration = media.duration_of(dest)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read that file: {exc}")
    if duration and duration < 15:
        raise HTTPException(status_code=400, detail="That file is shorter than 15 seconds — nothing to clip.")

    return {
        "source": str(dest),
        "source_type": "upload",
        "title": Path(meta["filename"]).stem.replace("_", " "),
        "duration": round(duration, 2),
        "duration_label": clipper.format_clock(duration),
        "width": width, "height": height, "fps": fps,
        "thumbnail": thumbnail,
        "platform": "upload",
        "platform_label": "Uploaded file",
        "size": dest.stat().st_size,
        "audio_only": width == 0,
    }


# ---------------------------------------------------------------------------
# Cost estimate + jobs
# ---------------------------------------------------------------------------
class EstimateBody(BaseModel):
    duration: float = 0.0
    url: Optional[str] = None


@router.post("/estimate")
def estimate(body: EstimateBody) -> Dict[str, Any]:
    duration = float(body.duration or 0)
    if not duration and body.url:
        try:
            duration = float(clipper.inspect_url(body.url).get("duration") or 0)
        except clipper.ClipperError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    cost = clipping_job.estimate_credits(duration)
    acct = credits.account()
    unlimited = credits.UNLIMITED_CREDITS
    return {"duration": round(duration, 2), "credits": cost,
            "balance": acct["balance"], "unlimited": unlimited,
            "affordable": unlimited or acct["balance"] >= cost,
            "minutes": round(duration / 60.0, 1)}


class JobBody(BaseModel):
    source: str
    source_type: str = "url"                    # url | upload
    meta: Dict[str, Any] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)


@router.post("/jobs")
def create_job(body: JobBody) -> Dict[str, Any]:
    source = (body.source or "").strip()
    if not source:
        raise HTTPException(status_code=400, detail="No video source given.")
    settings = clipping_job.clean_settings(body.settings)

    if body.source_type == "url":
        try:
            source = clipper.validate_url(source)
        except clipper.ClipperError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        meta = dict(body.meta or {})
    else:
        if not Path(source).is_file():
            raise HTTPException(status_code=404, detail="Uploaded file not found — upload it again.")
        meta = dict(body.meta or {})

    duration = float(meta.get("duration") or 0)
    if not duration:
        try:
            duration = float(media.video_meta(source).get("duration") or 0)
        except Exception:
            duration = 0.0
    cost = clipping_job.estimate_credits(duration)

    try:
        balance = credits.spend("ai_clipping", units=cost,
                                note=f"AI clipping · {clipper.format_clock(duration)}")
    except credits.InsufficientCredits as exc:
        raise HTTPException(status_code=402, detail={
            "message": str(exc), "required": exc.required, "available": exc.available,
            "upgrade": True, "upgrade_url": "/editor.html#pricing",
        })

    video_id = f"vid_{uuid.uuid4().hex[:12]}"

    def worker(task):
        def on_progress(step: int, pct: float, label: str) -> None:
            STORE.stage(task.id, min(step, len(clipping_job.STEPS) - 1), pct, label)

        try:
            result = clipping_job.run(
                task.id, video_id=video_id, source=source, source_type=body.source_type,
                settings=settings, meta=meta, on_progress=on_progress,
            )
        except Exception:
            credits.refund("ai_clipping", units=cost, note="AI clipping failed")
            store.notify("clipping", "Clipping failed", "Credits were refunded automatically.",
                         level="error", link="/clip-maker.html")
            raise
        store.notify("clipping", "Clips ready",
                     f"{result['moments']} moments from {result['title'][:60]}",
                     level="success", link=f"/clip-maker.html?v={video_id}")
        fire_webhooks("job.completed", {"video_id": video_id, **result, "credits": cost})
        return {**result, "credits_charged": cost}

    started = start_task("ai-clipping", clipping_job.STEPS, worker)
    return {**started, "video_id": video_id, "credits_charged": cost,
            "balance": balance, "steps": clipping_job.STEPS}


@router.get("/jobs/{task_id}")
def job_status(task_id: str) -> Dict[str, Any]:
    return status_of(task_id)


@router.post("/jobs/{task_id}/cancel")
def job_cancel(task_id: str) -> Dict[str, Any]:
    if not STORE.cancel(task_id):
        raise HTTPException(status_code=404, detail="Unknown job.")
    return {"ok": True, "status": "cancelling"}


@router.post("/jobs/{task_id}/retry")
def job_retry(task_id: str) -> Dict[str, Any]:
    """Re-run a finished job from its persisted document settings."""
    task = STORE.get(task_id)
    if task is None or not task.result:
        raise HTTPException(status_code=404, detail="Nothing to retry.")
    video_id = task.result.get("video_id")
    doc = store.read_clip(video_id) if video_id else None
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")
    return create_job(JobBody(source=doc["source"], source_type=doc["source_type"],
                              meta={"title": doc.get("title"), "duration": doc.get("duration"),
                                    "thumbnail": doc.get("thumbnail"), "channel": doc.get("channel"),
                                    "platform": doc.get("platform")},
                              settings=doc.get("settings") or {}))


# ---------------------------------------------------------------------------
# Videos (Library) + moments
# ---------------------------------------------------------------------------
def _load_video(video_id: str) -> Dict[str, Any]:
    doc = store.read_clip(video_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")
    return doc


def _save(doc: Dict[str, Any]) -> Dict[str, Any]:
    store.write_clip(doc)
    return doc


def _public(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc)
    out["moments"] = clipping_job.rehydrate_moment_urls(doc)
    out.pop("local_path", None)
    return out


@router.get("/videos")
def list_videos() -> Dict[str, Any]:
    items = store.clipvideos.all()
    items.sort(key=lambda v: v.get("updated_at") or v.get("created_at") or 0, reverse=True)
    return {"items": [
        {**item, "thumb_url": item.get("thumb") if str(item.get("thumb", "")).startswith("/media")
         else (item.get("thumb") or "")}
        for item in items], "count": len(items)}


@router.get("/videos/{video_id}")
def get_video(video_id: str) -> Dict[str, Any]:
    return _public(_load_video(video_id))


@router.delete("/videos/{video_id}")
def delete_video(video_id: str) -> Dict[str, Any]:
    doc = _load_video(video_id)
    if doc.get("task_id"):
        shutil.rmtree(Path(MEDIA_DIR) / doc["task_id"], ignore_errors=True)
    store.delete_clip(video_id)
    return {"ok": True}


def _find_moment(doc: Dict[str, Any], moment_id: str) -> Dict[str, Any]:
    for moment in doc.get("moments") or []:
        if moment.get("id") == moment_id:
            return moment
    raise HTTPException(status_code=404, detail="Moment not found.")


def _rerender_preview(doc: Dict[str, Any], moment: Dict[str, Any], index: int) -> None:
    transcript = store.read_clip_transcript(doc["id"]) or {}
    out_dir = Path(MEDIA_DIR) / doc["task_id"]
    name = f"preview_{index + 1:02d}"
    path = clipper.render_moment(
        doc["local_path"], moment, out_dir, name=name, height=clipping_job.PREVIEW_HEIGHT,
        style=(doc.get("settings") or {}).get("style"), transcript=transcript,
        captions=True, preset="ultrafast", crf=30,
    )
    moment["preview"] = path.name
    poster = media.make_thumbnail(path, out_dir / f"{name}.jpg", at_sec=0.4)
    moment["poster"] = poster.name


def _refine(doc: Dict[str, Any], moment: Dict[str, Any]) -> None:
    transcript = store.read_clip_transcript(doc["id"]) or {}
    settings = doc.get("settings") or {}
    lo, hi = clipper.length_bounds(settings.get("clip_length", "auto"))
    refined = clipper.refine_bounds(
        float(moment["start"]), float(moment["end"]),
        words=transcript.get("words") or [], segments=transcript.get("segments") or [],
        silences=[], duration=float(doc.get("duration") or 0), lo=min(lo, 6.0), hi=hi,
    )
    moment["start"], moment["end"] = refined["start"], refined["end"]
    moment["keep"] = refined["keep"]
    moment["duration"] = round(refined["end"] - refined["start"], 2)
    moment["start_label"] = format_timestamp(refined["start"])
    moment["end_label"] = format_timestamp(refined["end"])


class MomentPatch(BaseModel):
    title: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    layout: Optional[str] = None
    hook_text: Optional[str] = None
    hook_type: Optional[str] = None


@router.patch("/moments/{moment_id}")
def patch_moment(moment_id: str, body: MomentPatch, video_id: str) -> Dict[str, Any]:
    doc = _load_video(video_id)
    moment = _find_moment(doc, moment_id)
    changed = False
    if body.title is not None:
        moment["title"] = body.title.strip()[:70]
        changed = True
    if body.hook_text is not None:
        moment["hook_text"] = body.hook_text.strip()[:120]
        changed = True
    if body.hook_type is not None:
        moment["hook_type"] = body.hook_type.strip()[:40]
    if body.layout:
        if body.layout not in clipper.LAYOUTS:
            raise HTTPException(status_code=400, detail=f"Unknown layout {body.layout}")
        moment["layout"] = body.layout
        changed = True
    if body.start is not None or body.end is not None:
        start = float(body.start if body.start is not None else moment["start"])
        end = float(body.end if body.end is not None else moment["end"])
        if end - start < 4:
            raise HTTPException(status_code=400, detail="A clip must be at least 4 seconds long.")
        moment["start"], moment["end"] = start, end
        _refine(doc, moment)
        changed = True
    if changed:
        try:
            _rerender_preview(doc, moment, (doc.get("moments") or []).index(moment))
        except Exception as exc:
            moment["preview_error"] = str(exc)[:300]
    _save(doc)
    return {"moment": clipping_job.rehydrate_moment_urls(doc)[
        (doc.get("moments") or []).index(moment)], "video_id": doc["id"]}


@router.post("/moments/{moment_id}/duplicate")
def duplicate_moment(moment_id: str, video_id: str) -> Dict[str, Any]:
    doc = _load_video(video_id)
    moment = _find_moment(doc, moment_id)
    doc.setdefault("moments", [])
    copy = dict(moment)
    copy["id"] = f"m_{uuid.uuid4().hex[:10]}"
    copy["title"] = (moment.get("title") or "Untitled")[:60] + " (copy)"
    out_dir = Path(MEDIA_DIR) / doc["task_id"]
    index = len(doc["moments"])
    if moment.get("preview") and (out_dir / moment["preview"]).is_file():
        new_name = f"preview_{index + 1:02d}.mp4"
        shutil.copy2(out_dir / moment["preview"], out_dir / new_name)
        copy["preview"] = new_name
        if moment.get("poster"):
            new_poster = f"preview_{index + 1:02d}.jpg"
            shutil.copy2(out_dir / moment["poster"], out_dir / new_poster)
            copy["poster"] = new_poster
    doc["moments"].append(copy)
    for rank, item in enumerate(doc["moments"], start=1):
        item["rank"] = rank
    _save(doc)
    return {"moment": clipping_job.rehydrate_moment_urls(doc)[index], "video_id": doc["id"]}


@router.delete("/moments/{moment_id}")
def delete_moment(moment_id: str, video_id: str) -> Dict[str, Any]:
    doc = _load_video(video_id)
    moment = _find_moment(doc, moment_id)
    out_dir = Path(MEDIA_DIR) / doc["task_id"]
    for key in ("preview", "poster"):
        name = moment.get(key)
        if name:
            (out_dir / name).unlink(missing_ok=True)
    doc["moments"] = [m for m in doc["moments"] if m.get("id") != moment_id]
    for rank, item in enumerate(doc["moments"], start=1):
        item["rank"] = rank
    _save(doc)
    return {"ok": True, "moments": len(doc["moments"])}


@router.post("/moments/{moment_id}/regenerate")
def regenerate_moment(moment_id: str, video_id: str) -> Dict[str, Any]:
    """Re-pick this moment's boundaries/title for a small credit cost."""
    doc = _load_video(video_id)
    moment = _find_moment(doc, moment_id)
    transcript = store.read_clip_transcript(doc["id"])
    if not transcript:
        raise HTTPException(status_code=409, detail="Transcript is missing for this project.")
    try:
        balance = credits.spend("clip_regenerate", note="regenerate moment")
    except credits.InsufficientCredits as exc:
        raise HTTPException(status_code=402, detail={
            "message": str(exc), "required": exc.required, "available": exc.available,
            "upgrade": True, "upgrade_url": "/editor.html#pricing"})

    settings = doc.get("settings") or {}
    lo, hi = clipper.length_bounds(settings.get("clip_length", "auto"))
    centre = (float(moment["start"]) + float(moment["end"])) / 2
    window_start = max(0.0, centre - hi * 1.4)
    window_end = min(float(doc.get("duration") or 0), centre + hi * 1.4)
    speakers = [f"Speaker 1"] * len(transcript["segments"])
    signals = {}
    cands = clipper.local_candidates(
        [s for s in transcript["segments"] if s["end"] > window_start and s["start"] < window_end],
        [w for w in transcript["words"] if window_start <= w["s"] <= window_end],
        speakers, float(doc.get("duration") or 0), signals, lo=lo, hi=hi,
        platform=settings.get("platform", "all"),
    )
    others = [m for m in doc["moments"] if m["id"] != moment_id]
    fresh = [c for c in cands if all(
        min(c["end"], o["end"]) - max(c["start"], o["start"]) <= 1.0 for o in others)]
    if not fresh:
        credits.refund("clip_regenerate", note="nothing new found")
        raise HTTPException(status_code=409, detail="No alternative moment found nearby.")
    best = max(fresh, key=lambda c: c["score"])
    if best["score"] < max(25, int(moment.get("score", 60) * 0.6)):
        credits.refund("clip_regenerate", note="no strong alternative")
        raise HTTPException(status_code=409, detail=(
            "The alternatives nearby are weaker than this clip — keeping the current cut. "
            "Your credits were refunded."))
    meta = clipper.generate_metadata(best, doc.get("title", ""))
    moment.update({
        "start": best["start"], "end": best["end"],
        "start_label": format_timestamp(best["start"]),
        "end_label": format_timestamp(best["end"]),
        "duration": round(best["end"] - best["start"], 2),
        "score": best["score"], "score_breakdown": best["score_breakdown"],
        "transcript": best["transcript"], "reasons": best.get("reasons") or [],
        "why_it_works": meta["why_it_works"], "hook_text": meta["hook_text"],
        "hook_type": meta["hook_type"], "suggested_caption": meta["suggested_caption"],
        "hashtags": meta["hashtags"], "title": meta["title"],
    })
    _refine(doc, moment)
    try:
        _rerender_preview(doc, moment, (doc.get("moments") or []).index(moment))
    except Exception as exc:
        moment["preview_error"] = str(exc)[:300]
    _save(doc)
    return {"moment": clipping_job.rehydrate_moment_urls(doc)[
        (doc.get("moments") or []).index(moment)], "balance": balance}


@router.post("/moments/{moment_id}/hooks")
def moment_hooks(moment_id: str, video_id: str) -> Dict[str, Any]:
    doc = _load_video(video_id)
    moment = _find_moment(doc, moment_id)
    return clipper.alt_hooks(moment.get("transcript", ""), title=moment.get("title", ""))


@router.post("/moments/{moment_id}/platform")
def moment_platform(moment_id: str, video_id: str) -> Dict[str, Any]:
    """Platform copy (YouTube / Reels / TikTok titles, tags, description)."""
    doc = _load_video(video_id)
    moment = _find_moment(doc, moment_id)
    index = (doc.get("moments") or []).index(moment) + 1
    if not moment.get("platform_copy"):
        moment["platform_copy"] = platform_metadata(
            {"transcript": moment.get("transcript", ""), "duration": moment.get("duration", 0)},
            doc.get("title", ""), index,
        )
        _save(doc)
    return moment["platform_copy"]


class RestyleBody(BaseModel):
    palette: Optional[str] = None
    motion: Optional[str] = None
    font: Optional[str] = None
    size: Optional[str] = None
    placement: Optional[str] = None
    moment_ids: List[str] = Field(default_factory=list)


@router.post("/videos/{video_id}/restyle")
def restyle(video_id: str, body: RestyleBody) -> Dict[str, Any]:
    """Change the caption style for the project and re-render previews."""
    doc = _load_video(video_id)
    style = dict((doc.get("settings") or {}).get("style") or {})
    for key in ("palette", "motion", "font", "size", "placement"):
        value = getattr(body, key)
        if value:
            style[key] = value
    doc.setdefault("settings", {})["style"] = clipping_job.clean_style(style)
    _save(doc)

    moments = _selected_moments(doc, body.moment_ids) if body.moment_ids else (doc.get("moments") or [])

    def worker(task):
        for i, moment in enumerate(moments):
            STORE.stage(task.id, 0, (i / max(1, len(moments))) * 95,
                        f"Re-rendering preview {i + 1} of {len(moments)}")
            try:
                _rerender_preview(doc, moment, doc["moments"].index(moment))
            except Exception as exc:
                moment["preview_error"] = str(exc)[:300]
        _save(doc)
        store.notify("clipping", "Caption style updated", f"{len(moments)} preview(s) re-rendered",
                     level="success", link=f"/clip-maker.html?v={video_id}")
        return {"video_id": video_id, "restyled": len(moments)}

    return start_task("ai-clipping-restyle", ["Re-render previews"], worker)


# ---------------------------------------------------------------------------
# Find Moments
# ---------------------------------------------------------------------------
class FindBody(BaseModel):
    video_id: str
    query: str


@router.post("/find")
def find_moments(body: FindBody) -> Dict[str, Any]:
    doc = _load_video(body.video_id)
    transcript = store.read_clip_transcript(doc["id"])
    if not transcript:
        raise HTTPException(status_code=409, detail="Transcript is missing for this project.")
    speakers = [transcript.get("speakers", {}).get(str(i), "Speaker 1")
                for i in range(len(transcript["segments"]))]
    try:
        result = clipper.find_moments(transcript, speakers, body.query)
    except clipper.ClipperError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    created: List[Dict[str, Any]] = []
    out_dir = Path(MEDIA_DIR) / doc["task_id"]
    settings = doc.get("settings") or {}
    existing = doc.get("moments") or []
    for item in result["moments"]:
        # Skip only near-duplicates: a match that overlaps an existing clip by
        # most of its own length. Partial overlap is a different angle on the
        # same passage and is worth keeping.
        span = max(1.0, item["end"] - item["start"])
        duplicate = any(
            (min(item["end"], m["end"]) - max(item["start"], m["start"])) > span * 0.6
            for m in existing
        )
        if duplicate:
            continue
        moment = {
            "id": f"m_{uuid.uuid4().hex[:10]}",
            "rank": len(existing) + 1,
            "title": item.get("title") or "",
            "start": item["start"], "end": item["end"],
            "start_label": format_timestamp(item["start"]),
            "end_label": format_timestamp(item["end"]),
            "duration": round(item["end"] - item["start"], 2),
            "score": int(item.get("score") or 60),
            "score_breakdown": item.get("score_breakdown") or {},
            "hook_text": "", "hook_type": "",
            "why_it_works": item.get("why_it_works") or f"Matches “{body.query}”.",
            "suggested_caption": "", "hashtags": [], "best_platforms": ["shorts", "tiktok", "reels"],
            "risk_flags": [], "transcript": item.get("transcript", ""),
            "reasons": [f"matches “{body.query}”"], "speaker": "Speaker 1",
            "layout": settings.get("layout", "Full"), "source": "find",
            "keep": [{"start": item["start"], "end": item["end"]}],
        }
        meta = clipper.generate_metadata(moment, doc.get("title", ""))
        moment.update({k: meta[k] for k in ("hook_text", "hook_type", "suggested_caption", "hashtags")})
        _refine(doc, moment)
        try:
            _rerender_preview(doc, moment, len(existing))
        except Exception as exc:
            moment["preview_error"] = str(exc)[:300]
        existing.append(moment)
        created.append(moment)
        doc["moments"] = existing
    _save(doc)
    public = clipping_job.rehydrate_moment_urls(doc)
    ids = {m["id"] for m in created}
    return {"source": result["source"], "query": body.query,
            "moments": [m for m in public if m["id"] in ids],
            "added": len(created), "video_id": doc["id"]}


# ---------------------------------------------------------------------------
# Export / montage / share
# ---------------------------------------------------------------------------
class ExportBody(BaseModel):
    video_id: str
    moment_ids: List[str] = Field(default_factory=list)
    resolution: str = "1080p"
    options: Dict[str, Any] = Field(default_factory=dict)
    zip: bool = True


def _selected_moments(doc: Dict[str, Any], ids: List[str]) -> List[Dict[str, Any]]:
    moments = doc.get("moments") or []
    if not ids:
        return moments
    wanted = [m for m in moments if m.get("id") in set(ids)]
    if not wanted:
        raise HTTPException(status_code=400, detail="None of the selected clips exist.")
    return wanted


def _clean_export_options(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "captions": bool(raw.get("captions", True)),
        "hook": bool(raw.get("hook", True)),
        "watermark": bool(raw.get("watermark", False)),
        "cold_open": bool(raw.get("cold_open", False)),
        "layout": raw.get("layout") if raw.get("layout") in clipper.LAYOUTS else None,
    }


@router.post("/export")
def export_clips(body: ExportBody) -> Dict[str, Any]:
    doc = _load_video(body.video_id)
    if body.resolution not in RESOLUTIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported resolution {body.resolution}")
    moments = _selected_moments(doc, body.moment_ids)
    height = RESOLUTIONS[body.resolution]
    options = _clean_export_options(body.options)
    transcript = store.read_clip_transcript(doc["id"]) or {}
    cost = max(1, len(moments))
    try:
        credits.spend("export", units=cost, note=f"export {len(moments)} clip(s)")
    except credits.InsufficientCredits as exc:
        raise HTTPException(status_code=402, detail={
            "message": str(exc), "required": exc.required, "available": exc.available,
            "upgrade": True, "upgrade_url": "/editor.html#pricing"})

    steps = ["Render clips", "Package download"]

    def worker(task):
        out_dir = Path(MEDIA_DIR) / doc["task_id"] / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        files: List[Dict[str, Any]] = []
        for i, moment in enumerate(moments):
            STORE.stage(task.id, 0, 3 + (i / len(moments)) * 85, f"Rendering clip {i + 1} of {len(moments)}")
            name = f"clip_{moment['rank']:02d}_{moment['id']}"
            path = clipper.render_moment(
                doc["local_path"], moment, out_dir, name=name, height=height,
                style=(doc.get("settings") or {}).get("style"), transcript=transcript,
                captions=options["captions"],
                watermark="ClipOX" if options["watermark"] else "",   # brand matches the site
                cold_open=options["cold_open"], preset="veryfast", crf=20,
                on_progress=lambda p: STORE.update(task.id, progress=3 + ((i + p) / len(moments)) * 85),
            )
            files.append({"moment_id": moment["id"], "title": moment.get("title", ""),
                          "file": path.name, "url": media_url(path),
                          "size": path.stat().st_size, "resolution": body.resolution})
        STORE.stage(task.id, 1, 92, "Packaging download")
        zip_url = ""
        if body.zip and len(files) > 1:
            zip_path = out_dir / f"clips_{int(time.time())}.zip"
            clipper.zip_files([out_dir / f["file"] for f in files], zip_path)
            zip_url = media_url(zip_path)
        record = {"id": f"exp_{uuid.uuid4().hex[:10]}", "video_id": doc["id"],
                  "video_title": doc.get("title", ""), "resolution": body.resolution,
                  "files": files, "zip_url": zip_url, "created_at": time.time(),
                  "moment_ids": [f["moment_id"] for f in files]}
        store.Collection("clipexports").upsert(record)
        doc.setdefault("exports", []).insert(0, record)
        _save(doc)
        store.notify("export", "Export ready",
                     f"{len(files)} clip(s) at {body.resolution}",
                     level="success", link=f"/clip-maker.html?v={doc['id']}")
        fire_webhooks("export.ready", {"video_id": doc["id"], **record})
        return record

    return start_task("ai-clipping-export", steps, worker)


class MontageBody(BaseModel):
    video_id: str
    moment_ids: List[str] = Field(default_factory=list)
    resolution: str = "1080p"
    transition: str = "fade"
    music_url: str = ""
    captions: bool = True


def _xffade_concat(files: List[Path], dst: Path, *, transition: str = "fade",
                   duration: float = 0.35, music: str = "") -> Path:
    """Join rendered clips with FFmpeg xfade (video) + acrossfade (audio)."""
    if len(files) == 1:
        shutil.copy2(files[0], dst)
        return dst
    probes = [media.duration_of(f) for f in files]
    inputs: List[str] = []
    for f in files:
        inputs += ["-i", str(f)]
    graph: List[str] = []
    # normalise every clip to the first one's geometry / fps / pixel format
    first = media.video_meta(files[0])
    w = int(first.get("width") or 1080)
    h = int(first.get("height") or 1920)
    for i in range(len(files)):
        graph.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                     f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v{i}]")
        graph.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]")
    # chain xfade: each step shortens the total by the transition duration
    last_v, last_a = "v0", "a0"
    offset = max(0.0, probes[0] - duration)
    for i in range(1, len(files)):
        out_v, out_a = f"vx{i}", f"ax{i}"
        graph.append(f"[{last_v}][v{i}]xfade=transition={transition}:duration={duration}:"
                     f"offset={offset:.3f}[{out_v}]")
        graph.append(f"[{last_a}][a{i}]acrossfade=d={duration}[{out_a}]")
        last_v, last_a = out_v, out_a
        offset += max(0.1, probes[i] - duration)
    audio_map = f"[{last_a}]"
    cmd = [FFMPEG_BIN, "-y", "-v", "error",
           *inputs, "-filter_complex", ";".join(graph),
           "-map", f"[{last_v}]", "-map", audio_map,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)]
    if music and Path(music).is_file():
        cmd = [FFMPEG_BIN, "-y", "-v", "error",
               *inputs, "-i", str(music), "-filter_complex",
               ";".join(graph) + f";[{last_a}][{len(files)}:a]amix=inputs=2:duration=first:"
                                  f"dropout_transition=2,volume=1.4[amix]",
               "-map", f"[{last_v}]", "-map", "[amix]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(dst)]
    clipper._run(cmd)
    return dst


@router.post("/montage")
def montage(body: MontageBody) -> Dict[str, Any]:
    doc = _load_video(body.video_id)
    moments = _selected_moments(doc, body.moment_ids)
    if len(moments) < 2:
        raise HTTPException(status_code=400, detail="Select at least two clips for a montage.")
    height = RESOLUTIONS.get(body.resolution, 1920)
    transcript = store.read_clip_transcript(doc["id"]) or {}
    try:
        credits.spend("montage", note=f"montage {len(moments)} clips")
    except credits.InsufficientCredits as exc:
        raise HTTPException(status_code=402, detail={
            "message": str(exc), "required": exc.required, "available": exc.available,
            "upgrade": True, "upgrade_url": "/editor.html#pricing"})

    def worker(task):
        out_dir = Path(MEDIA_DIR) / doc["task_id"] / "montage"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        pieces: List[Path] = []
        for i, moment in enumerate(moments):
            STORE.stage(task.id, 0, 4 + (i / len(moments)) * 80, f"Rendering clip {i + 1} of {len(moments)}")
            pieces.append(clipper.render_moment(
                doc["local_path"], moment, out_dir, name=f"part_{i + 1:02d}", height=height,
                style=(doc.get("settings") or {}).get("style"), transcript=transcript,
                captions=body.captions, preset="veryfast", crf=20,
            ))
        STORE.stage(task.id, 1, 88, "Joining clips")
        music = ""
        if body.music_url:
            try:
                from .common import resolve_media
                music = resolve_media(body.music_url)
            except Exception:
                music = ""
        dst = out_dir / f"montage_{stamp}.mp4"
        _xffade_concat(pieces, dst, transition=body.transition, music=music)
        for piece in pieces:
            piece.unlink(missing_ok=True)
        record = {"id": f"mon_{uuid.uuid4().hex[:10]}", "video_id": doc["id"],
                  "url": media_url(dst), "file": dst.name, "clips": len(moments),
                  "resolution": body.resolution, "created_at": time.time(),
                  "title": f"{doc.get('title', 'Montage')} — montage"}
        store.Collection("clipexports").upsert(record)
        store.notify("export", "Montage ready", f"{len(moments)} clips joined",
                     level="success", link=f"/clip-maker.html?v={doc['id']}")
        return record

    return start_task("ai-clipping-montage", ["Render clips", "Join clips"], worker)


@router.get("/exports")
def list_exports() -> Dict[str, Any]:
    items = store.Collection("clipexports").all()
    items.sort(key=lambda e: e.get("created_at") or 0, reverse=True)
    return {"items": items[:60]}


class ShareBody(BaseModel):
    video_id: str
    moment_id: str


@router.post("/share")
def share(body: ShareBody) -> Dict[str, Any]:
    doc = _load_video(body.video_id)
    moment = _find_moment(doc, body.moment_id)
    out_dir = Path(MEDIA_DIR) / doc["task_id"]
    source = out_dir / (moment.get("preview") or "")
    if not source.is_file():
        raise HTTPException(status_code=409, detail="Render this clip first — no preview exists.")
    token = uuid.uuid4().hex[:16]
    dest = SHARE_DIR / f"{token}.mp4"
    shutil.copy2(source, dest)
    return {"url": media_url(dest), "token": token,
            "title": moment.get("title", ""), "expires": "never"}


# ---------------------------------------------------------------------------
# Schedule / publish
# ---------------------------------------------------------------------------
class ScheduleBody(BaseModel):
    video_id: str
    moment_ids: List[str] = Field(default_factory=list)
    platforms: List[str] = Field(default_factory=list)
    start_at: float = 0.0          # unix seconds; 0 = now
    spacing_minutes: int = 60
    title: str = ""
    description: str = ""
    hashtags: List[str] = Field(default_factory=list)


@router.post("/schedule")
def schedule_clips(body: ScheduleBody) -> Dict[str, Any]:
    doc = _load_video(body.video_id)
    moments = _selected_moments(doc, body.moment_ids)
    if not body.platforms:
        raise HTTPException(status_code=400, detail="Choose at least one platform.")
    out_dir = Path(MEDIA_DIR) / doc["task_id"]
    start = body.start_at or time.time()
    created: List[Dict[str, Any]] = []
    for index, moment in enumerate(moments):
        when = start + index * max(0, body.spacing_minutes) * 60
        video = ""
        if moment.get("preview"):
            video = media_url(out_dir / moment["preview"])
        post = store.posts.upsert({
            "id": store.new_id("post"),
            "kind": "clip",
            "project_id": doc["id"],
            "video_id": doc["id"],
            "moment_id": moment["id"],
            "project_name": doc.get("title", ""),
            "platforms": body.platforms,
            "title": body.title or moment.get("title", ""),
            "description": body.description or moment.get("suggested_caption", ""),
            "hashtags": body.hashtags or moment.get("hashtags", []),
            "thumbnail": moment.get("poster_url") or moment.get("poster") or "",
            "video": video,
            "when": when,
            "status": "scheduled" if when > time.time() else "ready",
            "history": [{"at": time.time(), "event": "created", "detail": "bulk schedule"}],
        })
        created.append(post)
    store.notify("publish", f"{len(created)} clip(s) scheduled",
                 ", ".join(body.platforms), level="success", link="/clip-maker.html")
    return {"items": created, "count": len(created)}


@router.get("/schedule")
def list_clipping_schedule() -> Dict[str, Any]:
    items = [p for p in store.posts.all() if p.get("kind") == "clip"]
    items.sort(key=lambda p: p.get("when", 0))
    counts: Dict[str, int] = {}
    for item in items:
        counts[item.get("status", "draft")] = counts.get(item.get("status", "draft"), 0) + 1
    return {"items": items, "counts": counts, "count": len(items)}


class SchedulePatch(BaseModel):
    when: Optional[float] = None
    title: Optional[str] = None
    description: Optional[str] = None
    platforms: Optional[List[str]] = None
    status: Optional[str] = None


@router.patch("/schedule/{post_id}")
def patch_clipping_schedule(post_id: str, body: SchedulePatch) -> Dict[str, Any]:
    changes = {k: v for k, v in body.dict().items() if v is not None}
    if "when" in changes:
        changes["status"] = "scheduled" if float(changes["when"]) > time.time() else "ready"
    post = store.posts.patch(post_id, changes)
    if not post:
        raise HTTPException(status_code=404, detail="Unknown scheduled post.")
    return post


@router.delete("/schedule/{post_id}")
def delete_clipping_schedule(post_id: str) -> Dict[str, Any]:
    if not store.posts.delete(post_id):
        raise HTTPException(status_code=404, detail="Unknown scheduled post.")
    return {"ok": True}


class PublishBody(BaseModel):
    post_id: str


@router.post("/publish")
def publish_clip(body: PublishBody) -> Dict[str, Any]:
    post = store.posts.get(body.post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Unknown post.")
    connected = {a.get("platform") for a in store.accounts.all()}
    missing = [p for p in post.get("platforms", []) if p not in connected]
    if missing:
        store.posts.patch(post["id"], {
            "status": "failed",
            "error": f"no connected account for {', '.join(missing)}",
            "history": (post.get("history") or []) + [
                {"at": time.time(), "event": "publish_failed", "detail": str(missing)}],
        })
        raise HTTPException(status_code=503, detail={
            "message": f"Connect {', '.join(missing)} first — this server holds no OAuth "
                       f"credentials for them. Set WV_{missing[0].upper()}_CLIENT_ID and "
                       f"WV_{missing[0].upper()}_CLIENT_SECRET to enable it.",
            "platforms": missing, "configure": "WV_<PLATFORM>_CLIENT_ID / _SECRET",
        })
    credits.spend("publish", note="clip publish")
    post = store.posts.patch(post["id"], {
        "status": "published", "published_at": time.time(),
        "history": (post.get("history") or []) + [
            {"at": time.time(), "event": "published", "detail": ",".join(post.get("platforms", []))}],
    })
    store.notify("publish", "Clip published", post.get("title", "")[:60], level="success")
    fire_webhooks("post.published", post)
    return post


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------
class WebhookBody(BaseModel):
    url: str
    events: List[str] = Field(default_factory=lambda: ["job.completed", "export.ready",
                                                      "post.published"])


@router.get("/webhooks")
def list_webhooks() -> Dict[str, Any]:
    return {"items": store.Collection("webhooks").all()}


@router.post("/webhooks")
def add_webhook(body: WebhookBody) -> Dict[str, Any]:
    url = (body.url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Webhook URL must start with http:// or https://")
    hook = store.Collection("webhooks").upsert({
        "id": store.new_id("hook"), "url": url,
        "events": [e for e in body.events if e] or ["job.completed"],
    })
    return hook


@router.delete("/webhooks/{hook_id}")
def delete_webhook(hook_id: str) -> Dict[str, Any]:
    if not store.Collection("webhooks").delete(hook_id):
        raise HTTPException(status_code=404, detail="Unknown webhook.")
    return {"ok": True}


@router.post("/webhooks/{hook_id}/test")
def test_webhook(hook_id: str) -> Dict[str, Any]:
    hook = store.Collection("webhooks").get(hook_id)
    if not hook:
        raise HTTPException(status_code=404, detail="Unknown webhook.")
    fire_webhooks("test", {"hook_id": hook_id, "message": "WayinVideo webhook test"})
    return {"ok": True, "url": hook["url"]}


@router.get("/accounts")
def clipping_accounts() -> Dict[str, Any]:
    connected = {a.get("platform") for a in store.accounts.all()}
    out = []
    for key, label, env in [
        ("youtube", "YouTube", ("WV_YOUTUBE_CLIENT_ID", "WV_YOUTUBE_CLIENT_SECRET")),
        ("tiktok", "TikTok", ("WV_TIKTOK_CLIENT_ID", "WV_TIKTOK_CLIENT_SECRET")),
        ("instagram", "Instagram", ("WV_INSTAGRAM_CLIENT_ID", "WV_INSTAGRAM_CLIENT_SECRET")),
        ("facebook", "Facebook", ("WV_FACEBOOK_CLIENT_ID", "WV_FACEBOOK_CLIENT_SECRET")),
    ]:
        out.append({"platform": key, "label": label, "connected": key in connected,
                    "configured": all(os.getenv(v) for v in env), "env": list(env)})
    return {"platforms": out}
