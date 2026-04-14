"""Shared pytest fixtures for scene-analysis-service tests.

Tests run without inference dependencies (torch, ultralytics, transformers,
open_clip_torch) — all model components are replaced with Null* stubs so
tests focus on orchestration logic, hazard rules, schemas, and the API layer.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.services.analyzer import SceneAnalyzer
from app.services.describer import NullDescriber
from app.services.detector import NullDetector
from app.services.embedder import NullEmbedder
from app.services.hazards import HazardRuleEngine


@pytest.fixture()
def null_analyzer(tmp_path) -> SceneAnalyzer:
    """A SceneAnalyzer with all Null components (no inference deps required)."""
    return SceneAnalyzer(
        detector=NullDetector(),
        describer=NullDescriber(),
        embedder=NullEmbedder(),
        hazard_engine=HazardRuleEngine(config_path=tmp_path / "nonexistent.yaml"),
    )


@pytest.fixture()
def test_client(null_analyzer) -> TestClient:
    """FastAPI TestClient with the null_analyzer pre-wired on app.state."""
    from app.main import create_app

    application = create_app()
    application.state.analyzer = null_analyzer

    # Bypass the lifespan (which would load real models) by using the
    # TestClient without entering startup events.
    with TestClient(application, raise_server_exceptions=True) as client:
        yield client
