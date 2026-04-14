"""Tests for :class:`~app.services.analyzer.SceneAnalyzer` and its helpers.

All inference components are replaced with Null stubs so no GPU or model
files are required.  Tests focus on orchestration logic and the
``analyze()`` method flags.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.services.analyzer import AnalysisResult, SceneAnalyzer
from app.services.describer import NullDescriber
from app.services.detector import Detection, NullDetector
from app.services.embedder import NullEmbedder
from app.services.hazards import HazardRuleEngine


# ---------------------------------------------------------------------------
# Fixtures
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
    max_px=1920,
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
    def test_to_dict_structure(self):
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


# ---------------------------------------------------------------------------
# Null-component behaviour
# ---------------------------------------------------------------------------


class TestNullComponents:
    def test_analyze_with_all_nulls_returns_empty_result(self):
        analyzer = _make_analyzer()
        result = analyzer.analyze(_tiny_image_bytes())
        assert result.detections == []
        assert result.description == ""
        assert result.embedding == []
        assert result.hazards == []
        assert result.detector_available is False
        assert result.describer_available is False
        assert result.embedder_available is False

    def test_analyze_still_succeeds_on_large_image(self):
        """Images larger than max_px should be downscaled without error."""
        analyzer = _make_analyzer(max_px=32)
        result = analyzer.analyze(_tiny_image_bytes(width=200, height=200))
        assert isinstance(result, AnalysisResult)


# ---------------------------------------------------------------------------
# run_* flags
# ---------------------------------------------------------------------------


class TestRunFlags:
    """Use subclasses (not class-level property mutation) so tests don't leak state."""

    def _spy_detector(self):
        class _SpyDetector(NullDetector):
            """Detector that reports as available and records calls."""
            @property
            def is_available(self) -> bool:
                return True

            def detect(self, image):
                self.__class__._called += 1
                return []

        _SpyDetector._called = 0
        d = _SpyDetector()
        d.detect = MagicMock(wraps=d.detect)
        return d

    def _spy_describer(self):
        class _SpyDescriber(NullDescriber):
            @property
            def is_available(self) -> bool:
                return True

            def describe(self, image):
                return "a kitchen"

        d = _SpyDescriber()
        d.describe = MagicMock(wraps=d.describe)
        return d

    def _spy_embedder(self):
        class _SpyEmbedder(NullEmbedder):
            @property
            def is_available(self) -> bool:
                return True

            def embed(self, image):
                return [0.1] * 768

        e = _SpyEmbedder()
        e.embed = MagicMock(wraps=e.embed)
        return e

    def test_run_detect_false_skips_detector(self):
        detector = self._spy_detector()
        analyzer = _make_analyzer(detector=detector)
        analyzer.analyze(_tiny_image_bytes(), run_detect=False)
        detector.detect.assert_not_called()

    def test_run_describe_false_skips_describer(self):
        describer = self._spy_describer()
        analyzer = _make_analyzer(describer=describer)
        analyzer.analyze(_tiny_image_bytes(), run_describe=False)
        describer.describe.assert_not_called()

    def test_run_embed_false_skips_embedder(self):
        embedder = self._spy_embedder()
        analyzer = _make_analyzer(embedder=embedder)
        analyzer.analyze(_tiny_image_bytes(), run_embed=False)
        embedder.embed.assert_not_called()

    def test_run_detect_true_calls_detector(self):
        detector = self._spy_detector()
        analyzer = _make_analyzer(detector=detector)
        analyzer.analyze(_tiny_image_bytes(), run_detect=True)
        detector.detect.assert_called_once()


# ---------------------------------------------------------------------------
# Image downscaling
# ---------------------------------------------------------------------------


class TestImageDownscaling:
    def test_image_larger_than_max_is_resized(self):
        captured_sizes: list[tuple[int, int]] = []

        class _SpyDetector(NullDetector):
            @property
            def is_available(self) -> bool:
                return True

            def detect(self, image: Image.Image):
                captured_sizes.append((image.width, image.height))
                return []

        analyzer = _make_analyzer(detector=_SpyDetector(), max_px=32)
        analyzer.analyze(_tiny_image_bytes(width=200, height=100))
        # longest edge was 200; max_px=32 → should be scaled to 32×16
        w, h = captured_sizes[0]
        assert max(w, h) == 32

    def test_image_within_max_is_not_resized(self):
        captured_sizes: list[tuple[int, int]] = []

        class _SpyDetector(NullDetector):
            @property
            def is_available(self) -> bool:
                return True

            def detect(self, image: Image.Image):
                captured_sizes.append((image.width, image.height))
                return []

        analyzer = _make_analyzer(detector=_SpyDetector(), max_px=1920)
        analyzer.analyze(_tiny_image_bytes(width=64, height=64))
        assert captured_sizes[0] == (64, 64)
