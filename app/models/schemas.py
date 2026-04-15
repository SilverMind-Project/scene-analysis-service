"""Pydantic request/response models for the scene-analysis-service API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class DetectionOut(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4)
    class_id: int


# ---------------------------------------------------------------------------
# Hazard
# ---------------------------------------------------------------------------


class HazardAlertOut(BaseModel):
    name: str
    severity: str
    description: str
    detection: DetectionOut


# ---------------------------------------------------------------------------
# /detect endpoint
# ---------------------------------------------------------------------------


class DetectResponse(BaseModel):
    detections: list[DetectionOut]
    detector_available: bool


# ---------------------------------------------------------------------------
# /describe endpoint
# ---------------------------------------------------------------------------


class DescribeResponse(BaseModel):
    description: str
    describer_available: bool


# ---------------------------------------------------------------------------
# /embed endpoint
# ---------------------------------------------------------------------------


class EmbedResponse(BaseModel):
    embedding: list[float]
    embedding_dim: int
    embedder_available: bool


# ---------------------------------------------------------------------------
# /analyze endpoint  (full pipeline)
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Optional flags to enable/disable individual pipeline stages."""

    run_detect: bool = True
    run_describe: bool = True
    run_embed: bool = True
    run_hazards: bool = True


class AnalyzeResponse(BaseModel):
    detections: list[DetectionOut]
    description: str
    embedding: list[float]
    hazards: list[HazardAlertOut]
    detector_available: bool
    describer_available: bool
    embedder_available: bool


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    detector_available: bool
    describer_available: bool
    embedder_available: bool
