"""CLIP ViT-L/14 image embedding via Triton Inference Server.

Implements the :class:`ImageEmbedder` ABC using the shared Triton client and
CLIP preprocessing from ``triton_shared``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from app.services.embedder import ImageEmbedder

logger = logging.getLogger(__name__)


class TritonClipEmbedder(ImageEmbedder):
    """CLIP ViT-L/14 image embedder via Triton Inference Server.

    Args:
        client: A :class:`~triton_shared.client.TritonClientProtocol` instance.
        model_name: Triton model name (default ``"clip-vision"``).
    """

    def __init__(
        self,
        client: Any,  # TritonClientProtocol
        model_name: str = "clip-vision",
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._dim: int = 768

    async def embed(self, image: Image.Image) -> list[float]:
        """Return an L2-normalised CLIP embedding for *image*."""
        from triton_shared.inference.embedding import (
            CLIP_INPUT_SIZE,
            clip_postprocess,
            clip_preprocess,
        )

        tensor = clip_preprocess(image, CLIP_INPUT_SIZE)
        batch = np.expand_dims(tensor, axis=0)

        outputs = await self._client.infer(
            model_name=self._model_name,
            inputs=[("input", batch)],
            output_names=["output"],
        )
        raw = outputs["output"]  # (1, 768)
        return clip_postprocess(raw[0])

    @property
    def is_available(self) -> bool:
        return True

    @property
    def embedding_dim(self) -> int:
        return self._dim
