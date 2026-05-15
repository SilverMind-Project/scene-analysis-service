"""Integration tests for the scene-analysis-service FastAPI endpoints.

Uses the ``test_client`` fixture (defined in conftest.py) which replaces the
real lifespan with a null one that wires a Null-component
:class:`~app.services.analyzer.SceneAnalyzer` — no real model inference or
Triton connection required.
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from app.services.analyzer import SceneAnalyzer
from app.services.describer import NullDescriber
from app.services.detector import NullDetector
from app.services.embedder import NullEmbedder
from app.services.hazards import HazardRuleEngine

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
        data = test_client.get("/health").json()
        assert data["status"] == "degraded"
        assert data["detector_available"] is False
        assert data["describer_available"] is False
        assert data["embedder_available"] is False

    def test_status_ok_when_any_component_available(self, tmp_path):
        """Health reports 'ok' when at least one component is available."""
        from contextlib import asynccontextmanager

        class _AvailableDetector(NullDetector):
            @property
            def is_available(self) -> bool:
                return True

        analyzer = SceneAnalyzer(
            detector=_AvailableDetector(),
            describer=NullDescriber(),
            embedder=NullEmbedder(),
            hazard_engine=HazardRuleEngine(config_path=tmp_path / "none.yaml"),
        )
        from app.main import create_app

        application = create_app()

        @asynccontextmanager
        async def _lifespan(app):
            app.state.analyzer = analyzer
            yield

        application.router.lifespan_context = _lifespan

        with TestClient(application, raise_server_exceptions=True) as client:
            data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["detector_available"] is True


# ---------------------------------------------------------------------------
# /detect
# ---------------------------------------------------------------------------


class TestDetect:
    def test_returns_200(self, test_client):
        assert _upload(test_client, "/detect", _tiny_jpeg()).status_code == 200

    def test_empty_detections_with_null_detector(self, test_client):
        data = _upload(test_client, "/detect", _tiny_jpeg()).json()
        assert data["detections"] == []
        assert data["detector_available"] is False

    def test_rejects_empty_upload(self, test_client):
        resp = test_client.post("/detect", files={"image": ("e.jpg", b"", "image/jpeg")})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /describe
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_returns_200(self, test_client):
        assert _upload(test_client, "/describe", _tiny_jpeg()).status_code == 200

    def test_empty_description_with_null_describer(self, test_client):
        data = _upload(test_client, "/describe", _tiny_jpeg()).json()
        assert data["description"] == ""
        assert data["describer_available"] is False

    def test_rejects_empty_upload(self, test_client):
        resp = test_client.post("/describe", files={"image": ("e.jpg", b"", "image/jpeg")})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_returns_200(self, test_client):
        assert _upload(test_client, "/analyze", _tiny_jpeg()).status_code == 200

    def test_full_result_structure(self, test_client):
        data = _upload(test_client, "/analyze", _tiny_jpeg()).json()
        for key in ("detections", "description", "embedding", "hazards",
                    "detector_available", "describer_available", "embedder_available"):
            assert key in data, f"missing key: {key}"

    def test_all_empty_with_null_components(self, test_client):
        data = _upload(test_client, "/analyze", _tiny_jpeg()).json()
        assert data["detections"] == []
        assert data["description"] == ""
        assert data["embedding"] == []
        assert data["hazards"] == []

    def test_run_detect_false_flag_via_query_param(self, test_client):
        resp = test_client.post(
            "/analyze?run_detect=false",
            files={"image": ("t.jpg", _tiny_jpeg(), "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["detections"] == []

    def test_rejects_empty_upload(self, test_client):
        resp = test_client.post("/analyze", files={"image": ("e.jpg", b"", "image/jpeg")})
        assert resp.status_code == 422
