"""Tests for :class:`~app.services.analyzer.SceneAnalyzer` and its helpers.

All inference components are replaced with Null stubs or minimal subclasses
so no Triton connection is required.  Tests cover orchestration logic, the
run_* flags, image downscaling, and the factory functions added for the
TritonGrpcClient lifecycle fix.
"""

from __future__ import annotations

import contextlib
import io

import pytest
from PIL import Image

from app.services.analyzer import (
    AnalysisResult,
    SceneAnalyzer,
    create_from_settings,
    create_triton_client,
)
from app.services.describer import NullDescriber
from app.services.detector import NullDetector
from app.services.embedder import NullEmbedder
from app.services.hazards import HazardRuleEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_image_bytes(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_analyzer(
    detector=None,
    describer=None,
    embedder=None,
    hazard_engine=None,
    max_px: int = 1920,
) -> SceneAnalyzer:
    return SceneAnalyzer(
        detector=detector or NullDetector(),
        describer=describer or NullDescriber(),
        embedder=embedder or NullEmbedder(),
        hazard_engine=hazard_engine or HazardRuleEngine(config_path="nonexistent.yaml"),
        max_image_px=max_px,
    )


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------


class TestAnalysisResult:
    def test_to_dict_includes_all_fields(self):
        result = AnalysisResult(
            detections=[],
            description="empty room",
            embedding=[0.1, 0.2],
            hazards=[],
            detector_available=False,
            describer_available=True,
            embedder_available=True,
        )
        d = result.to_dict()
        assert d["description"] == "empty room"
        assert d["embedding"] == [0.1, 0.2]
        assert d["describer_available"] is True
        assert d["detector_available"] is False
        assert d["embedder_available"] is True
        assert d["detections"] == []
        assert d["hazards"] == []


# ---------------------------------------------------------------------------
# Null-component behaviour
# ---------------------------------------------------------------------------


class TestNullComponents:
    async def test_analyze_with_all_nulls_returns_empty_result(self):
        analyzer = _make_analyzer()
        result = await analyzer.analyze(_tiny_image_bytes())
        assert result.detections == []
        assert result.description == ""
        assert result.embedding == []
        assert result.hazards == []
        assert result.detector_available is False
        assert result.describer_available is False
        assert result.embedder_available is False


# ---------------------------------------------------------------------------
# run_* flags
# ---------------------------------------------------------------------------


class TestRunFlags:
    """Spy subclasses verify that components are called/skipped per flag."""

    def _make_spy_detector(self):
        class _SpyDetector(NullDetector):
            def __init__(self):
                self.calls = 0

            @property
            def is_available(self) -> bool:
                return True

            async def detect(self, image):
                self.calls += 1
                return []

        return _SpyDetector()

    def _make_spy_describer(self):
        class _SpyDescriber(NullDescriber):
            def __init__(self):
                self.calls = 0

            @property
            def is_available(self) -> bool:
                return True

            async def describe(self, image):
                self.calls += 1
                return "a kitchen"

        return _SpyDescriber()

    def _make_spy_embedder(self):
        class _SpyEmbedder(NullEmbedder):
            def __init__(self):
                self.calls = 0

            @property
            def is_available(self) -> bool:
                return True

            async def embed(self, image):
                self.calls += 1
                return [0.1] * 768

        return _SpyEmbedder()

    async def test_run_detect_false_skips_detector(self):
        spy = self._make_spy_detector()
        await _make_analyzer(detector=spy).analyze(_tiny_image_bytes(), run_detect=False)
        assert spy.calls == 0

    async def test_run_detect_true_calls_detector(self):
        spy = self._make_spy_detector()
        await _make_analyzer(detector=spy).analyze(_tiny_image_bytes(), run_detect=True)
        assert spy.calls == 1

    async def test_run_describe_false_skips_describer(self):
        spy = self._make_spy_describer()
        await _make_analyzer(describer=spy).analyze(_tiny_image_bytes(), run_describe=False)
        assert spy.calls == 0

    async def test_run_embed_false_skips_embedder(self):
        spy = self._make_spy_embedder()
        await _make_analyzer(embedder=spy).analyze(_tiny_image_bytes(), run_embed=False)
        assert spy.calls == 0

    async def test_run_all_false_returns_empty_result(self):
        spy_det = self._make_spy_detector()
        spy_desc = self._make_spy_describer()
        spy_emb = self._make_spy_embedder()
        result = await _make_analyzer(
            detector=spy_det, describer=spy_desc, embedder=spy_emb
        ).analyze(
            _tiny_image_bytes(),
            run_detect=False,
            run_describe=False,
            run_embed=False,
            run_hazards=False,
        )
        assert spy_det.calls == 0
        assert spy_desc.calls == 0
        assert spy_emb.calls == 0
        assert result.detections == []
        assert result.description == ""
        assert result.embedding == []

    async def test_hazards_not_evaluated_when_no_detections(self):
        """run_hazards=True with NullDetector → hazard engine never fires."""
        result = await _make_analyzer().analyze(_tiny_image_bytes(), run_hazards=True)
        assert result.hazards == []


# ---------------------------------------------------------------------------
# Image downscaling
# ---------------------------------------------------------------------------


class TestImageDownscaling:
    async def test_image_larger_than_max_is_resized(self):
        received: list[tuple[int, int]] = []

        class _SpyDetector(NullDetector):
            @property
            def is_available(self) -> bool:
                return True

            async def detect(self, image: Image.Image):
                received.append((image.width, image.height))
                return []

        await _make_analyzer(detector=_SpyDetector(), max_px=32).analyze(
            _tiny_image_bytes(width=200, height=100)
        )
        w, h = received[0]
        assert max(w, h) == 32

    async def test_image_within_max_is_not_resized(self):
        received: list[tuple[int, int]] = []

        class _SpyDetector(NullDetector):
            @property
            def is_available(self) -> bool:
                return True

            async def detect(self, image: Image.Image):
                received.append((image.width, image.height))
                return []

        await _make_analyzer(detector=_SpyDetector(), max_px=1920).analyze(
            _tiny_image_bytes(width=64, height=64)
        )
        assert received[0] == (64, 64)


# ---------------------------------------------------------------------------
# create_triton_client — lifecycle helpers introduced by the gRPC fix
# ---------------------------------------------------------------------------


class TestCreateTritonClient:
    def test_returns_none_and_async_exit_stack_when_url_empty(self, tmp_path):
        from app.config import Settings

        cfg = Settings(yaml_path=tmp_path / "nonexistent.yaml")
        client, ctx = create_triton_client(cfg)
        assert client is None
        assert isinstance(ctx, contextlib.AsyncExitStack)

    @pytest.mark.asyncio
    async def test_empty_url_context_enters_and_exits_cleanly(self, tmp_path):
        from app.config import Settings

        cfg = Settings(yaml_path=tmp_path / "nonexistent.yaml")
        _, ctx = create_triton_client(cfg)
        async with ctx:
            pass  # must not raise


# ---------------------------------------------------------------------------
# create_from_settings — factory
# ---------------------------------------------------------------------------


class TestCreateFromSettings:
    def test_null_client_produces_null_analyzer(self, tmp_path):
        from app.config import Settings

        cfg = Settings(yaml_path=tmp_path / "nonexistent.yaml")
        analyzer = create_from_settings(cfg, triton_client=None)
        assert analyzer.detector_available is False
        assert analyzer.describer_available is False
        assert analyzer.embedder_available is False

    async def test_analyze_runs_without_error_with_null_client(self, tmp_path):
        from app.config import Settings

        cfg = Settings(yaml_path=tmp_path / "nonexistent.yaml")
        analyzer = create_from_settings(cfg, triton_client=None)
        result = await analyzer.analyze(_tiny_image_bytes())
        assert isinstance(result, AnalysisResult)
        assert result.detections == []
