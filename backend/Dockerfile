FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.hf

# ffmpeg from Debian ships libass (burned-in captions) and libx264; libgomp1 is
# needed by ctranslate2, libgl1/libglib by opencv, fonts-dejavu is a fallback face.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg libgomp1 libgl1 libglib2.0-0 ca-certificates fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY assets ./assets

# glibc keeps freed model memory in per-thread arenas and does not hand it
# back by default; on a 512 MB host that is the difference between a job
# finishing and being killed mid-render.
ENV MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=131072 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    MALLOC_TOP_PAD_=0 \
    WV_WORK_DIR=/data \
    WV_FFMPEG=/usr/bin/ffmpeg \
    WV_FFPROBE=/usr/bin/ffprobe \
    WV_WHISPER_MODEL=tiny \
    WV_WHISPER_COMPUTE=int8 \
    WV_WHISPER_BATCH=8

RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# Render (and most PaaS) inject PORT; fall back to 8000 for local docker runs.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
