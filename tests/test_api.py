"""Integration tests for the scene-analysis-service FastAPI endpoints.

Uses the ``test_client`` fixture (defined in conftest.py) which pre-wires
a :class:`~app.services.analyzer.SceneAnalyzer` with all Null components
onto ``app.state`` — no real model inference required.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_jpeg(width: int = 32, height: int = 32) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _upload(client, endpoint: str, image_bytes: bytes):
    return client.post(
        endpoint,
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_status_degraded_with_null_components(self, test_client):
        data = resp = test_client.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["detector_available"] is False
        assert data["describer_available"] is False
        assert data["embedder_available"] is False


# ---------------------------------------------------------------------------
# /detect
# ---------------------------------------------------------------------------


class TestDetect:
    def test_returns_200(self, test_client):
        resp = _upload(test_client, "/detect", _tiny_jpeg())
        assert resp.status_code == 200

    def test_empty_detections_with_null_detector(self, test_client):
        data = _upload(test_client, "/detect", _tiny_jpeg()).json()
        assert data["detections"] == []
        assert data["detector_available"] is False

    def test_rejects_empty_upload(self, test_client):
        resp = test_client.post(
            "/detect",
            files={"image": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /describe
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_returns_200(self, test_client):
        resp = _upload(test_client, "/describe", _tiny_jpeg())
        assert resp.status_code == 200

    def test_empty_description_with_null_describer(self, test_client):
        data = _upload(test_client, "/describe", _tiny_jpeg()).json()
        assert data["description"] == ""
        assert data["describer_available"] is False

    def test_rejects_empty_upload(self, test_client):
        resp = test_client.post(
            "/describe",
            files={"image": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_returns_200(self, test_client):
        resp = _upload(test_client, "/analyze", _tiny_jpeg())
        assert resp.status_code == 200

    def test_full_result_structure(self, test_client):
        data = _upload(test_client, "/analyze", _tiny_jpeg()).json()
        assert "detections" in data
        assert "description" in data
        assert "embedding" in data
        assert "hazards" in data
        assert "detector_available" in data
        assert "describer_available" in data
        assert "embedder_available" in data

    def test_all_empty_with_null_components(self, test_client):
        data = _upload(test_client, "/analyze", _tiny_jpeg()).json()
        assert data["detections"] == []
        assert data["description"] == ""
        assert data["embedding"] == []
        assert data["hazards"] == []

    def test_run_detect_false_flag(self, test_client):
        resp = test_client.post(
            "/analyze?run_detect=false",
            files={"image": ("test.jpg", _tiny_jpeg(), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["detections"] == []

    def test_rejects_empty_upload(self, test_client):
        resp = test_client.post(
            "/analyze",
            files={"image": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 422
