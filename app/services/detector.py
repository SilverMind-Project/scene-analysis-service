"""Object detection via YOLO26x (Ultralytics) or ONNX Runtime.

Design
------
``Detector`` is an abstract base class with three concrete implementations:

- ``UltralyticsDetector`` -- wraps ``ultralytics.YOLO`` for PyTorch-based
  inference.  Supports ``cuda``, ``xpu`` (Intel Arc via IPEX), and ``cpu``
  device strings, plus any model format that Ultralytics handles natively
  (``.pt``, ``.onnx``, OpenVINO model directory).

- ``OnnxRuntimeDetector`` -- uses ``onnxruntime.InferenceSession`` directly,
  giving explicit control over the execution provider (EP).  Use this when you
  need deterministic EP selection, e.g.::

      providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"]   # Intel Arc
      providers=["VulkanExecutionProvider", "CPUExecutionProvider"]     # Vulkan (experimental)
      providers=["CUDAExecutionProvider", "CPUExecutionProvider"]       # NVIDIA explicit

  The ``onnxruntime`` inference_backend requires a pre-exported ONNX model::

      from ultralytics import YOLO
      YOLO("yolo26x.pt").export(format="onnx")   # produces yolo26x.onnx

- ``NullDetector`` -- zero-dep stub; always returns an empty list.  Used when
  ``yolo_enabled`` is false or when the chosen backend is unavailable.

Both real implementations are safe to call from async context: inference is
synchronous but short enough (tens of ms on GPU) not to block the event loop.
A future improvement could use ``asyncio.to_thread`` for long CPU-bound batches.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
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
    def detect(self, image: Image.Image) -> list[Detection]:
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

    def detect(self, image: Image.Image) -> list[Detection]:
        return []

    @property
    def is_available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Ultralytics implementation
# ---------------------------------------------------------------------------


class UltralyticsDetector(Detector):
    """YOLO26x object detector via the ``ultralytics`` library.

    Supports any model format that Ultralytics handles natively:

    - ``yolo26x.pt`` -- PyTorch weights, runs on ``cuda`` / ``xpu`` / ``cpu``
    - ``yolo26x.onnx`` -- ONNX, Ultralytics uses onnxruntime automatically
    - ``yolo26x_openvino_model/`` -- OpenVINO IR, Ultralytics uses OpenVINO

    To export a model for alternative formats::

        from ultralytics import YOLO
        YOLO("yolo26x.pt").export(format="onnx")
        YOLO("yolo26x.pt").export(format="openvino")

    Args:
        model_name: Model path or name (e.g. ``"yolo26x.pt"``).
        device: PyTorch device string (``"cuda"``, ``"xpu"``, or ``"cpu"``).
        confidence_threshold: Minimum detection confidence to include in results.
        iou_threshold: IoU threshold for NMS.
        max_detections: Maximum number of detections per image.
    """

    def __init__(
        self,
        model_name: str = "yolo26x.pt",
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for UltralyticsDetector. "
                "Install it with: pip install ultralytics"
            ) from exc

        logger.info(
            "loading_yolo_model model=%s device=%s",
            model_name,
            device,
        )
        self._model = YOLO(model_name)
        self._model.to(device)
        self._device = device
        self._conf = confidence_threshold
        self._iou = iou_threshold
        self._max_det = max_detections
        logger.info("yolo_model_loaded model=%s", model_name)

    def detect(self, image: Image.Image) -> list[Detection]:
        """Run inference on *image* and return filtered detections."""
        results = self._model.predict(
            source=image,
            conf=self._conf,
            iou=self._iou,
            max_det=self._max_det,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                label = result.names.get(cls, str(cls))
                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        bbox=xyxy,
                        class_id=cls,
                    )
                )
        return detections

    @property
    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# ONNX Runtime implementation
# ---------------------------------------------------------------------------


class OnnxRuntimeDetector(Detector):
    """YOLO detector backed by ``onnxruntime.InferenceSession``.

    Unlike :class:`UltralyticsDetector`, this implementation accepts an
    explicit execution-provider list so callers have full control over which
    hardware backend is used:

    - Intel Arc via OpenVINO EP:
      ``providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"]``
    - Experimental Vulkan:
      ``providers=["VulkanExecutionProvider", "CPUExecutionProvider"]``
    - NVIDIA explicit CUDA:
      ``providers=["CUDAExecutionProvider", "CPUExecutionProvider"]``

    The ONNX model must be exported from an Ultralytics YOLO model first::

        from ultralytics import YOLO
        YOLO("yolo26x.pt").export(format="onnx")   # produces yolo26x.onnx

    Post-processing (coordinate decoding and NMS) is implemented in pure
    NumPy so no PyTorch dependency is needed at runtime.

    Args:
        model_path: Path to the ``.onnx`` model file.
        providers: ORT execution provider list.  ``None`` uses
            ``ort.get_available_providers()`` (auto-detect).
        confidence_threshold: Minimum class score to keep a detection.
        iou_threshold: IoU threshold for class-agnostic NMS.
        input_size: Square input dimension expected by the model (default 640).
        max_detections: Maximum detections to return per image.
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        max_detections: int = 100,
    ) -> None:
        try:
            import numpy as np  # noqa: F401 — verify available at init time
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "numpy and onnxruntime are required for OnnxRuntimeDetector. "
                "Install them with: uv pip install onnxruntime openvino"
            ) from exc

        resolved_providers = providers or ort.get_available_providers()
        logger.info(
            "loading_onnx_model path=%s providers=%s",
            model_path,
            resolved_providers,
        )
        self._session = ort.InferenceSession(
            str(model_path), providers=resolved_providers
        )
        self._conf = confidence_threshold
        self._iou = iou_threshold
        self._input_size = input_size
        self._max_det = max_detections
        self._input_name: str = self._session.get_inputs()[0].name

        # Class names are embedded in the ONNX model metadata by Ultralytics.
        meta = self._session.get_modelmeta().custom_metadata_map
        raw_names = meta.get("names", "{}")
        try:
            parsed = json.loads(raw_names)
            # Ultralytics stores names as {int: str} but JSON keys are strings.
            self._names: dict[int, str] = {int(k): v for k, v in parsed.items()}
        except (ValueError, TypeError):
            self._names = {}

        logger.info(
            "onnx_model_loaded path=%s classes=%d providers=%s",
            model_path,
            len(self._names),
            resolved_providers,
        )

    # ------------------------------------------------------------------
    # Detector ABC
    # ------------------------------------------------------------------

    def detect(self, image: Image.Image) -> list[Detection]:
        """Run ONNX inference and return filtered detections."""
        import numpy as np

        orig_w, orig_h = image.size
        inp, scale, pad_x, pad_y = self._preprocess(image)
        outputs = self._session.run(None, {self._input_name: inp})
        pred = outputs[0]  # [1, num_attrs, num_boxes] from Ultralytics export
        if pred.ndim == 3:
            pred = pred[0].T  # [num_boxes, num_attrs]

        if pred.shape[0] == 0:
            return []

        boxes_xywh = pred[:, :4]  # cx, cy, w, h in model-input coords
        class_scores = pred[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores.max(axis=1)

        mask = confidences >= self._conf
        if not mask.any():
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert cx,cy,w,h -> x1,y1,x2,y2
        half_w = boxes_xywh[:, 2] / 2
        half_h = boxes_xywh[:, 3] / 2
        boxes_xyxy = np.stack(
            [
                boxes_xywh[:, 0] - half_w,
                boxes_xywh[:, 1] - half_h,
                boxes_xywh[:, 0] + half_w,
                boxes_xywh[:, 1] + half_h,
            ],
            axis=1,
        )

        keep = self._nms(boxes_xyxy, confidences, self._iou)[: self._max_det]

        detections: list[Detection] = []
        for i in keep:
            bx1 = max(0.0, min(float((boxes_xyxy[i, 0] - pad_x) / scale), orig_w))
            by1 = max(0.0, min(float((boxes_xyxy[i, 1] - pad_y) / scale), orig_h))
            bx2 = max(0.0, min(float((boxes_xyxy[i, 2] - pad_x) / scale), orig_w))
            by2 = max(0.0, min(float((boxes_xyxy[i, 3] - pad_y) / scale), orig_h))
            cls_id = int(class_ids[i])
            label = self._names.get(cls_id, str(cls_id))
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidences[i]),
                    bbox=[bx1, by1, bx2, by2],
                    class_id=cls_id,
                )
            )
        return detections

    @property
    def is_available(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(
        self, image: Image.Image
    ) -> tuple[Any, float, int, int]:
        """Letterbox-resize and normalise to a float32 NCHW tensor.

        Returns:
            Tuple of ``(tensor, scale, pad_x, pad_y)`` where ``scale`` is
            the ratio applied to both axes and ``pad_x/pad_y`` are the
            pixel offsets of the image within the padded canvas.
        """
        w, h = image.size
        scale = self._input_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        import numpy as np

        canvas = Image.new(
            "RGB", (self._input_size, self._input_size), (114, 114, 114)
        )
        pad_x = (self._input_size - new_w) // 2
        pad_y = (self._input_size - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))

        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(arr, 0), scale, pad_x, pad_y

    @staticmethod
    def _nms(
        boxes: Any, scores: Any, iou_threshold: float
    ) -> list[int]:
        """Class-agnostic greedy NMS in pure NumPy.

        Args:
            boxes: ``[N, 4]`` array of ``x1,y1,x2,y2`` coordinates.
            scores: ``[N]`` confidence array.
            iou_threshold: Suppress boxes with IoU above this threshold.

        Returns:
            Indices of kept boxes in descending score order.
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[rest] - inter)
            order = order[1:][iou <= iou_threshold]

        return keep


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_detector(
    *,
    enabled: bool,
    model_name: str,
    device: str,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
    backend: str = "ultralytics",
    ort_providers: list[str] | None = None,
    ort_input_size: int = 640,
) -> Detector:
    """Construct a :class:`Detector` from config values.

    Args:
        enabled: When ``False`` a :class:`NullDetector` is returned immediately.
        model_name: Ultralytics model name / path, or path to ``.onnx`` file
            when ``backend="onnxruntime"``.
        device: PyTorch device for the ``ultralytics`` backend.
        confidence_threshold: Minimum detection confidence.
        iou_threshold: IoU threshold for NMS.
        max_detections: Cap on returned detections per image.
        backend: ``"ultralytics"`` (default) or ``"onnxruntime"``.
        ort_providers: Explicit ORT execution provider list for the
            ``onnxruntime`` backend.  ``None`` defers to
            :func:`~app.services.device.onnxruntime_providers`.
        ort_input_size: Square input dimension for the ONNX model (default 640).

    Returns:
        A ready :class:`Detector`, or :class:`NullDetector` on failure.
    """
    if not enabled:
        logger.info("yolo_disabled returning_null_detector")
        return NullDetector()

    if backend == "onnxruntime":
        onnx_path = Path(model_name)
        if not onnx_path.suffix == ".onnx":
            logger.warning(
                "onnxruntime_backend_requires_onnx_file model_name=%s "
                "hint='export with: YOLO(\"%s\").export(format=\"onnx\")'",
                model_name,
                model_name,
            )
            return NullDetector()
        try:
            return OnnxRuntimeDetector(
                model_path=onnx_path,
                providers=ort_providers,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                input_size=ort_input_size,
                max_detections=max_detections,
            )
        except RuntimeError as exc:
            logger.warning(
                "onnxruntime_detector_init_failed error=%s returning_null_detector", exc
            )
            return NullDetector()

    # Default: ultralytics backend
    try:
        return UltralyticsDetector(
            model_name=model_name,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
    except RuntimeError:
        logger.warning("ultralytics_not_installed returning_null_detector")
        return NullDetector()
