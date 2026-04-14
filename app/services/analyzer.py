"""Scene analyzer orchestrator.

:class:`SceneAnalyzer` wires together the detector, describer, embedder, and
hazard engine into a single ``analyze`` call.  Each component is optional: a
``NullDetector`` / ``NullDescriber`` / ``NullEmbedder`` is returned when the
corresponding config flag is false, so the caller always gets a valid
:class:`AnalysisResult` regardless of which models are loaded.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

from PIL import Image

from app.config import Settings
from app.services.describer import SceneDescriber, build_describer
from app.services.detector import Detection, Detector, build_detector
from app.services.device import onnxruntime_providers, resolve_device
from app.services.embedder import ImageEmbedder, build_embedder
from app.services.hazards import HazardAlert, HazardRuleEngine

logger = logging.getLogger(__name__)

_MAX_IMAGE_PX = 1920


@dataclass
class AnalysisResult:
    """Bundled output of :meth:`SceneAnalyzer.analyze`.

    All lists default to empty when the corresponding component is disabled.
    """

    detections: list[Detection] = field(default_factory=list)
    description: str = ""
    embedding: list[float] = field(default_factory=list)
    hazards: list[HazardAlert] = field(default_factory=list)
    detector_available: bool = False
    describer_available: bool = False
    embedder_available: bool = False

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "description": self.description,
            "embedding": self.embedding,
            "hazards": [h.to_dict() for h in self.hazards],
            "detector_available": self.detector_available,
            "describer_available": self.describer_available,
            "embedder_available": self.embedder_available,
        }


class SceneAnalyzer:
    """Orchestrates all inference components for a single image.

    Instantiated once at application startup via :func:`create_from_settings`
    and stored on the FastAPI ``app.state``.
    """

    def __init__(
        self,
        detector: Detector,
        describer: SceneDescriber,
        embedder: ImageEmbedder,
        hazard_engine: HazardRuleEngine,
        max_image_px: int = _MAX_IMAGE_PX,
    ) -> None:
        self._detector = detector
        self._describer = describer
        self._embedder = embedder
        self._hazards = hazard_engine
        self._max_px = max_image_px

    # ------------------------------------------------------------------
    # Availability properties (avoid reaching into private component attrs)
    # ------------------------------------------------------------------

    @property
    def detector_available(self) -> bool:
        return self._detector.is_available

    @property
    def describer_available(self) -> bool:
        return self._describer.is_available

    @property
    def embedder_available(self) -> bool:
        return self._embedder.is_available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        image_bytes: bytes,
        *,
        run_detect: bool = True,
        run_describe: bool = True,
        run_embed: bool = True,
        run_hazards: bool = True,
    ) -> AnalysisResult:
        """Analyse raw image bytes and return an :class:`AnalysisResult`.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.).
            run_detect: Whether to run object detection.
            run_describe: Whether to run scene description.
            run_embed: Whether to run CLIP embedding.
            run_hazards: Whether to evaluate hazard rules.

        Returns:
            :class:`AnalysisResult` with available fields populated.
        """
        image = self._load_image(image_bytes)

        detections: list[Detection] = []
        if run_detect and self._detector.is_available:
            detections = self._detector.detect(image)
            logger.debug("detect_done count=%d", len(detections))

        description = ""
        if run_describe and self._describer.is_available:
            description = self._describer.describe(image)
            logger.debug("describe_done length=%d", len(description))

        embedding: list[float] = []
        if run_embed and self._embedder.is_available:
            embedding = self._embedder.embed(image)
            logger.debug("embed_done dim=%d", len(embedding))

        hazards: list[HazardAlert] = []
        if run_hazards and detections:
            hazards = self._hazards.evaluate(detections)
            logger.debug("hazards_done count=%d", len(hazards))

        return AnalysisResult(
            detections=detections,
            description=description,
            embedding=embedding,
            hazards=hazards,
            detector_available=self._detector.is_available,
            describer_available=self._describer.is_available,
            embedder_available=self._embedder.is_available,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Downscale very large images to keep inference latency predictable.
        longest = max(image.width, image.height)
        if longest > self._max_px:
            scale = self._max_px / longest
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image


def create_from_settings(cfg: Settings) -> SceneAnalyzer:
    """Build a :class:`SceneAnalyzer` from a :class:`~app.config.Settings` instance.

    This is the canonical factory used by ``app.main`` during application startup.
    """
    device_setting = cfg.get("device", "auto")
    device = resolve_device(device_setting)

    inference_backend: str = cfg.get("inference_backend", "ultralytics")
    ort_providers_cfg: list[str] = cfg.get("ort_providers", []) or []
    ort_providers = (
        ort_providers_cfg
        if ort_providers_cfg
        else onnxruntime_providers(device_setting)
    )

    detector = build_detector(
        enabled=cfg.get("yolo_enabled", True),
        model_name=cfg.get("yolo_model_name", "yolo26x.pt"),
        device=device,
        confidence_threshold=cfg.get("yolo_confidence_threshold", 0.25),
        iou_threshold=cfg.get("yolo_iou_threshold", 0.45),
        max_detections=cfg.get("yolo_max_detections", 100),
        backend=inference_backend,
        ort_providers=ort_providers if inference_backend == "onnxruntime" else None,
        ort_input_size=cfg.get("ort_input_size", 640),
    )
    describer = build_describer(
        enabled=cfg.get("florence_enabled", True),
        model_name=cfg.get("florence_model_name", "microsoft/Florence-2-large"),
        device=device,
        task=cfg.get("florence_task", "<DETAILED_CAPTION>"),
    )
    embedder = build_embedder(
        enabled=cfg.get("clip_enabled", True),
        model_name=cfg.get("clip_model_name", "ViT-L-14"),
        pretrained=cfg.get("clip_pretrained", "openai"),
        device=device,
    )
    hazard_engine = HazardRuleEngine(
        config_path=cfg.get("hazards_config_path", "config/hazards.yaml")
    )

    return SceneAnalyzer(
        detector=detector,
        describer=describer,
        embedder=embedder,
        hazard_engine=hazard_engine,
        max_image_px=cfg.get("max_image_size_px", _MAX_IMAGE_PX),
    )
