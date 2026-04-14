"""Object detection endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models.schemas import DetectResponse, DetectionOut

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect", response_model=DetectResponse, tags=["inference"])
async def detect(
    request: Request,
    image: UploadFile = File(..., description="Image file (JPEG/PNG)"),
) -> DetectResponse:
    """Run object detection on the uploaded image.

    Returns bounding boxes, class labels, and confidence scores for all
    detected objects above the configured confidence threshold.
    """
    analyzer = request.app.state.analyzer
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Empty image upload")

    try:
        result = analyzer.analyze(
            image_bytes,
            run_detect=True,
            run_describe=False,
            run_embed=False,
            run_hazards=False,
        )
    except Exception as exc:
        logger.exception("detect_failed error=%s", exc)
        raise HTTPException(status_code=500, detail="Detection failed") from exc

    return DetectResponse(
        detections=[DetectionOut(**d.to_dict()) for d in result.detections],
        detector_available=result.detector_available,
    )
