"""Object detection via YOLO26L on Triton Inference Server.

Design
------
``Detector`` is an abstract base class with two implementations:

- ``TritonDetector`` -- calls YOLO26L on Triton via gRPC (default backend).
- ``NullDetector`` -- zero-dep stub; always returns an empty list.  Used when
  ``yolo_enabled`` is false or Triton is unavailable.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data transfer object
# ---------------------------------------------------------------------------


class Detection:
    """A single object-detection result.

    Attributes:
        label: YOLO class name (e.g. ``"person"``).
        confidence: Score in ``[0, 1]``.
        bbox: ``[x1, y1, x2, y2]`` in *pixel* coordinates relative to the
            input image size.
        class_id: Integer YOLO class index.
    """

    __slots__ = ("label", "confidence", "bbox", "class_id")

    def __init__(
        self,
        label: str,
        confidence: float,
        bbox: list[float],
        class_id: int,
    ) -> None:
        self.label = label
        self.confidence = confidence
        self.bbox = bbox
        self.class_id = class_id

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox],
            "class_id": self.class_id,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Detector(ABC):
    """Abstract object detector."""

    @abstractmethod
    async def detect(self, image: Image.Image) -> list[Detection]:
        """Run detection on a PIL image and return a list of detections."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True when the underlying model is loaded and ready."""
        ...


# ---------------------------------------------------------------------------
# Null implementation (graceful degradation)
# ---------------------------------------------------------------------------


class NullDetector(Detector):
    """No-op detector used when YOLO is disabled or unavailable."""

    async def detect(self, image: Image.Image) -> list[Detection]:
        return []

    @property
    def is_available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_detector(
    *,
    enabled: bool,
    model_name: str,
    confidence_threshold: float,
    triton_client: Any | None = None,
) -> Detector:
    """Construct a :class:`Detector` from config values.

    Args:
        enabled: When ``False`` a :class:`NullDetector` is returned immediately.
        model_name: Triton model name (default ``"person-detector"``).
        confidence_threshold: Minimum detection confidence.
        triton_client: Pre-connected Triton client.

    Returns:
        A :class:`TritonDetector`, or :class:`NullDetector` on failure.
    """
    if not enabled:
        logger.info("yolo_disabled returning_null_detector")
        return NullDetector()

    if triton_client is None:
        logger.warning("triton_backend_requires_client returning_null_detector")
        return NullDetector()

    try:
        from app.services.triton_detector import TritonDetector

        return TritonDetector(
            client=triton_client,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
        )
    except ImportError as exc:
        logger.warning(
            "triton_detector_import_failed error=%s returning_null_detector", exc
        )
        return NullDetector()
