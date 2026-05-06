"""YOLO26L object detection via Triton Inference Server.

Implements the :class:`Detector` ABC using the shared Triton client and
YOLO26L decode logic from ``triton_shared``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any

import numpy as np
from PIL import Image

from app.services.detector import Detection, Detector

logger = logging.getLogger(__name__)


def _run_in_thread(coro: Any) -> Any:
    """Run an async coroutine from sync code by spinning a fresh event loop in a worker thread.

    This avoids ``asyncio.run()`` clashing with an already-running event loop
    (e.g. inside a FastAPI async route handler that calls synchronous
    ``SceneAnalyzer.analyze()`` — see tech-debt #1 in CLAUDE.md).
    """

    def _target() -> Any:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_target).result()


class TritonDetector(Detector):
    """YOLO26L person detector via Triton Inference Server.

    Args:
        client: A :class:`~triton_shared.client.TritonClientProtocol` instance
            (typically :class:`~triton_shared.client.TritonGrpcClient`).
        model_name: Triton model name (default ``"person-detector"``).
        confidence_threshold: Minimum detection confidence.
    """

    def __init__(
        self,
        client: Any,  # TritonClientProtocol (avoid import at module level)
        model_name: str = "person-detector",
        confidence_threshold: float = 0.25,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._conf = confidence_threshold

    def detect(self, image: Image.Image) -> list[Detection]:
        """Run YOLO26L detection on a PIL image via Triton.

        The Triton gRPC call is dispatched from a worker thread so this
        method remains synchronous (required by the ``Detector`` ABC).
        """
        from triton_shared.inference.detection import (
            DETECTOR_INPUT_SIZE,
            DETECTOR_PERSON_CLASS,
            decode_output,
            letterbox_preprocess,
        )

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        tensor, pad_x, pad_y, scale = letterbox_preprocess(rgb, DETECTOR_INPUT_SIZE)
        batch = np.expand_dims(tensor, axis=0)  # (1, 3, 640, 640)

        outputs = _run_in_thread(
            self._client.infer(
                model_name=self._model_name,
                inputs=[("images", batch)],
                output_names=["output0"],
            )
        )
        raw = outputs["output0"][0]  # (300, 6)

        boxes = decode_output(
            raw,
            orig_h=image.height,
            orig_w=image.width,
            pad_x=pad_x,
            pad_y=pad_y,
            scale=scale,
            conf_threshold=self._conf,
            person_class=DETECTOR_PERSON_CLASS,
        )

        # Convert normalised DetectionBox → SAS Detection (pixel coordinates).
        w, h = float(image.width), float(image.height)
        return [
            Detection(
                label="person",
                confidence=box.confidence,
                bbox=[
                    round(box.x1 * w, 1),
                    round(box.y1 * h, 1),
                    round(box.x2 * w, 1),
                    round(box.y2 * h, 1),
                ],
                class_id=DETECTOR_PERSON_CLASS,
            )
            for box in boxes
        ]

    @property
    def is_available(self) -> bool:
        return True
