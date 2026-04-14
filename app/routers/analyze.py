"""Full-pipeline analysis endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.params import Depends

from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DetectionOut,
    HazardAlertOut,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _parse_analyze_flags(
    run_detect: bool = True,
    run_describe: bool = True,
    run_embed: bool = True,
    run_hazards: bool = True,
) -> AnalyzeRequest:
    """Dependency that maps query-string flags to :class:`AnalyzeRequest`."""
    return AnalyzeRequest(
        run_detect=run_detect,
        run_describe=run_describe,
        run_embed=run_embed,
        run_hazards=run_hazards,
    )


@router.post("/analyze", response_model=AnalyzeResponse, tags=["inference"])
async def analyze(
    request: Request,
    image: UploadFile = File(..., description="Image file (JPEG/PNG)"),
    flags: AnalyzeRequest = Depends(_parse_analyze_flags),
) -> AnalyzeResponse:
    """Run the full analysis pipeline: detect, describe, embed, and check hazards.

    Individual stages can be disabled via query-string flags:
    ``run_detect``, ``run_describe``, ``run_embed``, ``run_hazards``.
    """
    analyzer = request.app.state.analyzer
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Empty image upload")

    try:
        result = analyzer.analyze(
            image_bytes,
            run_detect=flags.run_detect,
            run_describe=flags.run_describe,
            run_embed=flags.run_embed,
            run_hazards=flags.run_hazards,
        )
    except Exception as exc:
        logger.exception("analyze_failed error=%s", exc)
        raise HTTPException(status_code=500, detail="Analysis failed") from exc

    return AnalyzeResponse(
        detections=[DetectionOut(**d.to_dict()) for d in result.detections],
        description=result.description,
        embedding=result.embedding,
        hazards=[
            HazardAlertOut(
                name=h.name,
                severity=h.severity,
                description=h.description,
                detection=DetectionOut(**h.detection.to_dict()),
            )
            for h in result.hazards
        ],
        detector_available=result.detector_available,
        describer_available=result.describer_available,
        embedder_available=result.embedder_available,
    )
