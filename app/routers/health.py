"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    """Return service health and per-model availability."""
    analyzer = request.app.state.analyzer
    status = (
        "ok"
        if analyzer.detector_available
        or analyzer.describer_available
        or analyzer.embedder_available
        else "degraded"
    )
    return HealthResponse(
        status=status,
        detector_available=analyzer.detector_available,
        describer_available=analyzer.describer_available,
        embedder_available=analyzer.embedder_available,
    )
