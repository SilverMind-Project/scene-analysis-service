"""Unit tests for Pydantic schemas.

Validates field constraints without any inference deps.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    AnalyzeResponse,
    DetectionOut,
    HazardAlertOut,
    HealthResponse,
)


class TestDetectionOut:
    def test_valid(self):
        d = DetectionOut(label="person", confidence=0.9, bbox=[0, 0, 100, 100], class_id=0)
        assert d.label == "person"

    def test_confidence_ge_zero(self):
        with pytest.raises(ValidationError):
            DetectionOut(label="x", confidence=-0.1, bbox=[0, 0, 1, 1], class_id=0)

    def test_confidence_le_one(self):
        with pytest.raises(ValidationError):
            DetectionOut(label="x", confidence=1.1, bbox=[0, 0, 1, 1], class_id=0)

    def test_bbox_must_have_four_elements(self):
        with pytest.raises(ValidationError):
            DetectionOut(label="x", confidence=0.5, bbox=[0, 0, 1], class_id=0)


class TestHealthResponse:
    def test_ok_status(self):
        r = HealthResponse(
            status="ok",
            detector_available=True,
            describer_available=True,
            embedder_available=True,
        )
        assert r.status == "ok"

    def test_degraded_status(self):
        r = HealthResponse(
            status="degraded",
            detector_available=False,
            describer_available=False,
            embedder_available=False,
        )
        assert r.status == "degraded"


class TestAnalyzeResponse:
    def test_empty_lists_valid(self):
        r = AnalyzeResponse(
            detections=[],
            description="",
            embedding=[],
            hazards=[],
            detector_available=False,
            describer_available=False,
            embedder_available=False,
        )
        assert r.detections == []
        assert r.embedding == []

    def test_with_detections_and_hazard(self):
        det = DetectionOut(label="fire", confidence=0.85, bbox=[0, 0, 50, 50], class_id=99)
        hazard = HazardAlertOut(
            name="fire",
            severity="critical",
            description="Fire!",
            detection=det,
        )
        r = AnalyzeResponse(
            detections=[det],
            description="Flames visible",
            embedding=[0.1, 0.2],
            hazards=[hazard],
            detector_available=True,
            describer_available=True,
            embedder_available=False,
        )
        assert len(r.hazards) == 1
        assert r.hazards[0].severity == "critical"
