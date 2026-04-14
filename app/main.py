"""scene-analysis-service FastAPI application.

Startup sequence
----------------
1. Load :class:`~app.config.Settings` from ``config/config.yaml`` (+ env vars).
2. Build a :class:`~app.services.analyzer.SceneAnalyzer` via
   :func:`~app.services.analyzer.create_from_settings`.  Model loading happens
   here — the service is not healthy until this completes.
3. Store the analyzer on ``app.state`` so every route can access it through
   ``request.app.state.analyzer``.

Graceful degradation
--------------------
If inference dependencies (``ultralytics``, ``transformers``, ``open_clip_torch``)
are not installed, the corresponding ``Null*`` implementations are used and
the service starts successfully.  The ``/health`` endpoint reflects which
components are available.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.config import Settings
from app.routers import analyze, describe, detect, health
from app.services.analyzer import SceneAnalyzer, create_from_settings

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models on startup; clean up on shutdown."""
    cfg = Settings()
    _configure_logging(cfg.get("log_level", "info"))

    logger.info("scene_analysis_service_starting")
    analyzer: SceneAnalyzer = create_from_settings(cfg)
    app.state.analyzer = analyzer
    logger.info(
        "scene_analysis_service_ready "
        "detector=%s describer=%s embedder=%s",
        analyzer.detector_available,
        analyzer.describer_available,
        analyzer.embedder_available,
    )

    yield

    logger.info("scene_analysis_service_stopping")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Scene Analysis Service",
        description=(
            "Fast multi-modal scene analysis: YOLO object detection, "
            "Florence-2 structured descriptions, and CLIP image embeddings."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(detect.router)
    app.include_router(describe.router)
    app.include_router(analyze.router)
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    cfg = Settings()
    uvicorn.run(
        "app.main:app",
        host=cfg.get("host", "0.0.0.0"),
        port=cfg.get("port", 8100),
        reload=False,
    )
