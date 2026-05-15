"""Structured scene description via Florence-2-large on Triton Inference Server.

Florence-2-large (232M parameters, Microsoft) produces dense, structured
captions given an image and a task prompt.  It is not a conversational VLM
and runs in ~1-2 s per image on GPU — well within the target latency budget.

Design
------
``SceneDescriber`` is the ABC.  ``TritonFlorenceDescriber`` calls Florence-2
on Triton via gRPC.  ``NullDescriber`` returns an empty string and is used
when Florence is disabled or Triton is unavailable.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# Florence task prompts supported by this service.
TASK_DETAILED_CAPTION = "<DETAILED_CAPTION>"
TASK_CAPTION = "<CAPTION>"
TASK_OBJECT_DETECTION = "<OD>"


class SceneDescriber(ABC):
    """Abstract structured scene describer."""

    @abstractmethod
    async def describe(self, image: Image.Image) -> str:
        """Return a structured text description of *image*."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True when the model is loaded and ready."""
        ...


class NullDescriber(SceneDescriber):
    """No-op describer for graceful degradation."""

    async def describe(self, image: Image.Image) -> str:
        return ""

    @property
    def is_available(self) -> bool:
        return False


def build_describer(
    *,
    enabled: bool,
    model_name: str,
    task: str,
    triton_client: Any | None = None,
    tokenizer_dir: Any | None = None,
) -> SceneDescriber:
    """Construct a :class:`SceneDescriber` from config values.

    Args:
        enabled: When ``False`` a :class:`NullDescriber` is returned.
        model_name: Triton model name.
        task: Florence task prompt.
        triton_client: Pre-connected Triton client.
        tokenizer_dir: Path to directory containing ``tokenizer.json``.
    """
    if not enabled:
        logger.info("florence_disabled returning_null_describer")
        return NullDescriber()

    if triton_client is None:
        logger.warning("triton_backend_requires_client returning_null_describer")
        return NullDescriber()

    try:
        from app.services.triton_describer import TritonFlorenceDescriber

        return TritonFlorenceDescriber(
            client=triton_client,
            model_name=model_name,
            task=task,
            tokenizer_dir=tokenizer_dir,
        )
    except (ImportError, RuntimeError) as exc:
        logger.warning(
            "triton_describer_init_failed error=%s returning_null_describer", exc
        )
        return NullDescriber()
