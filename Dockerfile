# syntax=docker/dockerfile:1
# scene-analysis-service — Triton Inference Server client
#
# Build:
#   docker build -t scene-analysis-service .
#
# Run:
#   docker run -p 8300:8300 -e SAS_TRITON_URL=host.docker.internal:8701 scene-analysis-service

# ── UV binary ──────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:latest AS uv

# ── base stage ─────────────────────────────────────────────────────────────
FROM python:3.14-slim AS base

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── deps stage ─────────────────────────────────────────────────────────────
FROM base AS deps

COPY pyproject.toml .
RUN uv pip install --system --no-cache .

# ── final stage ────────────────────────────────────────────────────────────
FROM base AS final

COPY --from=deps /usr/local/lib/python3.14 /usr/local/lib/python3.14
COPY --from=deps /usr/local/bin /usr/local/bin

RUN groupadd -r sas && useradd -r -g sas sas

WORKDIR /app
COPY --chown=sas:sas . .

USER sas

EXPOSE 8300

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8300/health')" \
    || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8300"]
