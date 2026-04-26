# syntax=docker/dockerfile:1
# scene-analysis-service
#
# Build stages
# ─────────────
# uv     - UV binary (from official Astral image)
# base   - Python 3.13-slim + system deps + UV
# deps   - uv pip install (split for layer caching)
# final  - non-root user, copy app, set entrypoint
#
# GPU note: swap the base image for nvidia/cuda:12.x-cudnn-runtime-ubuntu22.04
# and install the matching torch+torchvision wheels for CUDA inference.
#
# Build:
#   docker build -t scene-analysis-service .
#
# Build (with inference deps):
#   docker build --build-arg EXTRAS=inference -t scene-analysis-service .
#
# Run (CPU):
#   docker run -p 8300:8300 scene-analysis-service
#
# Run (GPU):
#   docker run --gpus all -p 8300:8300 -e SAS_DEVICE=cuda scene-analysis-service

# ── UV binary ──────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:latest AS uv

# ── base stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS base

# Copy UV from the official image so all stages share the same binary.
COPY --from=uv /uv /uvx /usr/local/bin/

# System-level dependencies for Pillow, OpenCV (if added), etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── deps stage ─────────────────────────────────────────────────────────────
FROM base AS deps

COPY pyproject.toml .
# Install base (non-inference) deps only for the default Docker image.
# Override by passing --build-arg EXTRAS=inference to also pull torch/ultralytics.
ARG EXTRAS=""
RUN if [ -n "${EXTRAS}" ]; then \
        uv pip install --system --no-cache ".[${EXTRAS}]"; \
    else \
        uv pip install --system --no-cache .; \
    fi

# ── final stage ────────────────────────────────────────────────────────────
FROM base AS final

# Copy installed packages from deps stage.
COPY --from=deps /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=deps /usr/local/bin /usr/local/bin

# Non-root user for security.
RUN groupadd -r sas && useradd -r -g sas sas

WORKDIR /app
COPY --chown=sas:sas . .

# Writable cache directory owned by sas (HuggingFace, YOLO, torch).
RUN mkdir -p /app/.cache && chown sas:sas /app/.cache

# Default config directory is expected at /app/config/
# Mount an alternative with: -v /host/config:/app/config

USER sas

EXPOSE 8300

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8300/health')" \
    || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8300"]
