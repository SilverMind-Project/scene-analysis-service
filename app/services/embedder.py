"""Dense image embeddings via CLIP ViT-L/14 on Triton Inference Server.

Embeddings are 768-dimensional float32 vectors suitable for storing in a
pgvector column and performing cosine-similarity searches.

Design
------
``ImageEmbedder`` is the ABC.  ``TritonClipEmbedder`` calls CLIP on Triton
via gRPC.  ``NullEmbedder`` returns an empty list and is used when CLIP is
disabled or Triton is unavailable.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


class ImageEmbedder(ABC):
    """Abstract image embedder."""

    @abstractmethod
    async def embed(self, image: Image.Image) -> list[float]:
        """Return a normalised embedding vector for *image*.

        Returns an empty list when the model is unavailable.
        """
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True when the model is loaded and ready."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of the embedding vectors produced by this model."""
        ...


class NullEmbedder(ImageEmbedder):
    """No-op embedder for graceful degradation."""

    async def embed(self, image: Image.Image) -> list[float]:
        return []

    @property
    def is_available(self) -> bool:
        return False

    @property
    def embedding_dim(self) -> int:
        return 0


def build_embedder(
    *,
    enabled: bool,
    model_name: str,
    triton_client: Any | None = None,
) -> ImageEmbedder:
    """Construct an :class:`ImageEmbedder` from config values.

    Args:
        enabled: When ``False`` a :class:`NullEmbedder` is returned.
        model_name: Triton model name.
        triton_client: Pre-connected Triton client.
    """
    if not enabled:
        logger.info("clip_disabled returning_null_embedder")
        return NullEmbedder()

    if triton_client is None:
        logger.warning("triton_backend_requires_client returning_null_embedder")
        return NullEmbedder()

    try:
        from app.services.triton_embedder import TritonClipEmbedder

        return TritonClipEmbedder(client=triton_client, model_name=model_name)
    except ImportError as exc:
        logger.warning(
            "triton_embedder_import_failed error=%s returning_null_embedder", exc
        )
        return NullEmbedder()
