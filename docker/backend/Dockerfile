# ================================================================
# Backend Dockerfile - 360° Construction Intelligence Engine
# Multi-stage: builder → runtime
# ================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    build-essential git curl libgl1-mesa-glx libglib2.0-0 \
    libgomp1 ffmpeg libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY backend/requirements.txt .

RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --prefix=/install -r requirements.txt

# ── Runtime stage ────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/install/bin:$PATH" \
    LD_LIBRARY_PATH="/install/lib:$LD_LIBRARY_PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 libgl1-mesa-glx libglib2.0-0 libgomp1 \
    ffmpeg libsm6 libxext6 curl wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /install

WORKDIR /app

# Copy application code
COPY backend/app ./app
COPY ml ./ml
COPY configs ./configs

# Create non-root user
RUN useradd -m -u 1000 panoramic && \
    mkdir -p /tmp/panoramic_uploads /app/ml/models && \
    chown -R panoramic:panoramic /app /tmp/panoramic_uploads

USER panoramic

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--loop", "uvloop", "--http", "httptools"]
