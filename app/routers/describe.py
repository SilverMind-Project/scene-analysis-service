"""Scene description endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models.schemas import DescribeResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/describe", response_model=DescribeResponse, tags=["inference"])
async def describe(
    request: Request,
    image: UploadFile = File(..., description="Image file (JPEG/PNG)"),
) -> DescribeResponse:
    """Generate a structured scene description for the uploaded image.

    Uses Florence-2-large to produce a detailed, structured caption.
    Returns an empty string when the Florence model is not loaded.
    """
    analyzer = request.app.state.analyzer
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Empty image upload")

    try:
        result = await analyzer.analyze(
            image_bytes,
            run_detect=False,
            run_describe=True,
            run_embed=False,
            run_hazards=False,
        )
    except Exception as exc:
        logger.exception("describe_failed error=%s", exc)
        raise HTTPException(status_code=500, detail="Description failed") from exc

    return DescribeResponse(
        description=result.description,
        describer_available=result.describer_available,
    )
