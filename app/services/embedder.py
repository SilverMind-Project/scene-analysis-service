"""Dense image embeddings via CLIP ViT-L/14 (OpenCLIP).

Embeddings are 768-dimensional float32 vectors suitable for storing in a
pgvector column and performing cosine-similarity searches.

Design
------
``ImageEmbedder`` is the ABC.  ``CLIPEmbedder`` wraps ``open_clip_torch``.
``NullEmbedder`` returns an empty list and is used when CLIP is disabled or
``open_clip_torch`` is not installed.
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
    def embed(self, image: Image.Image) -> list[float]:
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

    def embed(self, image: Image.Image) -> list[float]:
        return []

    @property
    def is_available(self) -> bool:
        return False

    @property
    def embedding_dim(self) -> int:
        return 0


class CLIPEmbedder(ImageEmbedder):
    """CLIP ViT-L/14 image embedder via ``open_clip_torch``.

    Args:
        model_name: OpenCLIP architecture name (default ``"ViT-L-14"``).
        pretrained: Pretrained weights tag (default ``"openai"``).
        device: PyTorch device string.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        try:
            import open_clip  # type: ignore[import-untyped]
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "open_clip_torch and torch are required for CLIPEmbedder. "
                "Install them with: pip install open_clip_torch torch"
            ) from exc

        logger.info(
            "loading_clip model=%s pretrained=%s device=%s",
            model_name,
            pretrained,
            device,
        )
        self._device = device
        self._torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        # Probe embedding dimension from the model.
        self._dim: int = model.visual.output_dim
        logger.info("clip_loaded model=%s dim=%d", model_name, self._dim)

    def embed(self, image: Image.Image) -> list[float]:
        """Return an L2-normalised CLIP embedding for *image*."""
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        # device_type must be the base device name ("cuda", "cpu", "xpu") without
        # an ordinal suffix — torch.amp.autocast replaced the old cuda-only API.
        device_type = self._device.split(":")[0]
        with self._torch.no_grad(), self._torch.amp.autocast(device_type=device_type):
            features = self._model.encode_image(tensor)
            features = self._torch.nn.functional.normalize(features, dim=-1)
        return features[0].cpu().float().tolist()

    @property
    def is_available(self) -> bool:
        return True

    @property
    def embedding_dim(self) -> int:
        return self._dim


def build_embedder(
    *,
    enabled: bool,
    model_name: str,
    pretrained: str,
    device: str,
    backend: str = "openclip",
    triton_client: Any | None = None,
) -> ImageEmbedder:
    """Construct an :class:`ImageEmbedder` from config values.

    Args:
        enabled: When ``False`` a :class:`NullEmbedder` is returned.
        model_name: OpenCLIP architecture name or Triton model name.
        pretrained: Pretrained weights tag (OpenCLIP backend only).
        device: PyTorch device string (OpenCLIP backend only).
        backend: ``"openclip"`` (default) or ``"triton"``.
        triton_client: Pre-connected Triton client for the ``triton`` backend.
    """
    if not enabled:
        logger.info("clip_disabled returning_null_embedder")
        return NullEmbedder()

    if backend == "triton":
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

    # Default: openclip backend
    try:
        return CLIPEmbedder(model_name=model_name, pretrained=pretrained, device=device)
    except RuntimeError:
        logger.warning("open_clip_not_installed returning_null_embedder")
        return NullEmbedder()
