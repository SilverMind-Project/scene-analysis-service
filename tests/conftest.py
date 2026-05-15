"""Shared pytest fixtures for scene-analysis-service tests.

Tests run without Triton dependencies — all model components are replaced
with Null* stubs so tests focus on orchestration logic, hazard rules,
schemas, and the API layer.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

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
def test_client(null_analyzer: SceneAnalyzer):
    """FastAPI TestClient with a null lifespan that pre-wires the null_analyzer.

    The real lifespan is replaced so no config files or Triton connections are
    needed — component availability is determined entirely by ``null_analyzer``.
    """
    from app.main import create_app

    application = create_app()

    @asynccontextmanager
    async def _null_lifespan(app):
        app.state.analyzer = null_analyzer
        yield

    application.router.lifespan_context = _null_lifespan

    with TestClient(application, raise_server_exceptions=True) as client:
        yield client
