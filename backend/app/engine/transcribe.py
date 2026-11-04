"""
Whisper transcription layer.

Wraps ``faster-whisper`` (CTranslate2) with:
  * lazy model loading (the model is only downloaded once),
  * automatic device / compute-type selection,
  * word-level timestamps (required for karaoke subtitles and text-based edit),
  * a stable JSON schema consumed by every other feature.

Schema
------
.. code-block:: json

    {
      "language": "en",
      "duration": 612.4,
      "text": "...",
      "segments": [
        {"id": 0, "start": 0.0, "end": 4.2, "text": "...",
         "words": [{"w": "Hello", "s": 0.0, "e": 0.32}]}
      ]
    }
"""
from __future__ import annotations

import gc
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import (
    WHISPER_BATCH,
    WHISPER_BEAM,
    WHISPER_COMPUTE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
    WHISPER_THREADS,
    detect_compute_type,
    detect_device,
)


def _ensure_pyav_compat() -> None:
    """Keep faster-whisper working against newer PyAV releases.

    faster-whisper 1.2.1 decodes audio with
    ``av.open(path, mode="r", metadata_errors="ignore")``, but PyAV 14+
    removed the ``metadata_errors`` argument, so every transcription died
    with ``TypeError: open() got an unexpected keyword argument
    'metadata_errors'``.  No upstream release fixes it yet, so strip the
    argument -- but only when the installed PyAV actually rejects it.
    """
    import io

    try:
        import av
    except ImportError:  # pragma: no cover - faster-whisper reports this itself
        return

    try:
        av.open(io.BytesIO(b""), format="wav", metadata_errors="ignore")
    except TypeError:
        pass  # unsupported -> patch below
    except Exception:
        return  # argument accepted, failure was just the empty stream
    else:
        return

    _open = av.open

    def _open_compat(file, *args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _open(file, *args, **kwargs)

    av.open = _open_compat


_ensure_pyav_compat()

_model = None
_model_key: Optional[str] = None
_model_lock = threading.Lock()
_batched = None
_batched_key: Optional[str] = None


def release_model() -> None:
    """Drop the cached weights so their RAM is free for ffmpeg.

    On a 512 MB host the model and an encoder cannot coexist; reloading costs a
    couple of seconds, an OOM kill costs the whole job.
    """
    global _model, _model_key, _batched, _batched_key
    with _model_lock:
        _model = None
        _model_key = None
        _batched = None
        _batched_key = None
    gc.collect()


class TranscriptionError(RuntimeError):
    pass


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:
        return False


def load_model(model_size: Optional[str] = None):
    """Load (and cache) a ``WhisperModel`` instance."""
    global _model, _model_key

    model_size = model_size or WHISPER_MODEL
    device = detect_device()
    compute = WHISPER_COMPUTE or detect_compute_type(device)
    key = f"{model_size}:{device}:{compute}"

    with _model_lock:
        if _model is not None and _model_key == key:
            return _model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - env dependent
            raise TranscriptionError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from exc

        threads = {"cpu_threads": WHISPER_THREADS} if WHISPER_THREADS > 0 else {}
        _model = WhisperModel(model_size, device=device, compute_type=compute,
                              download_root=None, **threads)
        _model_key = key
        _batched = None
        return _model


def load_batched(model_size: Optional[str] = None):
    """Batched pipeline over the cached model (same weights, parallel windows).

    On a CPU the sequential decoder leaves most cores idle, which is why a
    46-minute lecture used to spend ~14 minutes in Whisper. Batching 8 windows
    together measures ~3.7x faster on this machine with identical output.
    """
    global _batched, _batched_key

    model = load_model(model_size)
    key = f"{_model_key}:{WHISPER_BATCH}"
    if _batched is not None and _batched_key == key:
        return _batched
    from faster_whisper import BatchedInferencePipeline

    _batched = BatchedInferencePipeline(model=model)
    _batched_key = key
    return _batched


def _run(audio_path: Path | str, *, model_size: Optional[str], lang: Optional[str],
         word_timestamps: bool, vad_filter: bool):
    """Start a transcription, preferring the batched pipeline."""
    common = {
        "language": lang,
        "word_timestamps": word_timestamps,
        "vad_filter": vad_filter,
    }
    if WHISPER_BATCH > 0:
        try:
            pipe = load_batched(model_size)
            return pipe.transcribe(str(audio_path), batch_size=WHISPER_BATCH, **common)
        except Exception:  # pragma: no cover - fall back to the plain decoder
            pass
    model = load_model(model_size)
    try:
        return model.transcribe(
            str(audio_path),
            beam_size=WHISPER_BEAM,
            condition_on_previous_text=False,
            **common,
        )
    except Exception as exc:  # pragma: no cover - model runtime errors
        raise TranscriptionError(str(exc)) from exc


def transcribe_audio(
    audio_path: Path | str,
    *,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    word_timestamps: bool = True,
    vad_filter: bool = True,
    on_progress=None,
) -> Dict[str, Any]:
    """Transcribe *audio_path* and return word/segment level timestamps."""
    lang = language or (WHISPER_LANGUAGE or None)

    t0 = time.time()
    segments_iter, info = _run(
        audio_path, model_size=model_size, lang=lang,
        word_timestamps=word_timestamps, vad_filter=vad_filter,
    )

    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    total = float(getattr(info, "duration", 0) or 0)

    for idx, seg in enumerate(segments_iter):
        entry: Dict[str, Any] = {
            "id": idx,
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": seg.text.strip(),
            "words": [],
        }
        for w in seg.words or []:
            item = {
                "w": w.word.strip(),
                "s": round(float(w.start), 3),
                "e": round(float(w.end), 3),
            }
            entry["words"].append(item)
            words.append({**item, "seg": idx})
        segments.append(entry)

        if on_progress and total > 0:
            on_progress(min(99.0, (seg.end / total) * 100.0))

    if not segments:
        raise TranscriptionError(
            "No speech detected in the audio track (empty transcript)."
        )

    return {
        "language": getattr(info, "language", "") or "",
        "language_probability": round(float(getattr(info, "language_probability", 0) or 0), 3),
        "duration": round(total or segments[-1]["end"], 3),
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
        "words": words,
        "elapsed_sec": round(time.time() - t0, 2),
        "device": detect_device(),
    }


def format_timestamp(sec: float, *, ms: bool = False) -> str:
    """``HH:MM:SS.mmm`` (or ``MM:SS`` when *ms* is False)."""
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if ms:
        return f"{h:02d}:{m:02d}:{s:02d}.{int((sec - int(sec)) * 1000):03d}"
    return (f"{h:02d}:" if h else "") + f"{m:02d}:{s:02d}"
