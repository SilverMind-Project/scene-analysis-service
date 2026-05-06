"""Structured scene description via Florence-2-large.

Florence-2-large (232M parameters, Microsoft) produces dense, structured
captions given an image and a task prompt.  It is not a conversational VLM
and runs in ~1-2 s per image on GPU — well within the target latency budget.

Design
------
``SceneDescriber`` is the ABC.  ``FlorenceDescriber`` loads Florence-2-large
via ``transformers`` and runs ``AutoProcessor`` + ``AutoModelForCausalLM``.
``NullDescriber`` returns an empty string and is used when Florence is
disabled or ``transformers`` is not installed.
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
    def describe(self, image: Image.Image) -> str:
        """Return a structured text description of *image*."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True when the model is loaded and ready."""
        ...


class NullDescriber(SceneDescriber):
    """No-op describer for graceful degradation."""

    def describe(self, image: Image.Image) -> str:
        return ""

    @property
    def is_available(self) -> bool:
        return False


class FlorenceDescriber(SceneDescriber):
    """Florence-2-large scene describer.

    Args:
        model_name: HuggingFace model ID (default ``"microsoft/Florence-2-large"``).
        device: PyTorch device string.
        task: Florence task prompt (default ``"<DETAILED_CAPTION>"``).
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-large",
        device: str = "cpu",
        task: str = TASK_DETAILED_CAPTION,
    ) -> None:
        try:
            import torch
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForCausalLM,
                AutoProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for FlorenceDescriber. "
                "Install them with: pip install transformers torch"
            ) from exc

        logger.info("loading_florence model=%s device=%s task=%s", model_name, device, task)
        self._device = device
        self._task = task

        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self._model.eval()
        self._torch = torch
        logger.info("florence_loaded model=%s", model_name)

    def describe(self, image: Image.Image) -> str:
        """Return a structured caption for *image*."""
        inputs = self._processor(text=self._task, images=image, return_tensors="pt").to(
            self._device
        )
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)

        with self._torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = self._processor.post_process_generation(
            generated_text,
            task=self._task,
            image_size=(image.width, image.height),
        )
        # parsed is a dict like {"<DETAILED_CAPTION>": "text..."}
        return parsed.get(self._task, generated_text)

    @property
    def is_available(self) -> bool:
        return True


def build_describer(
    *,
    enabled: bool,
    model_name: str,
    device: str,
    task: str,
    backend: str = "transformers",
    triton_client: Any | None = None,
    tokenizer_dir: Any | None = None,
) -> SceneDescriber:
    """Construct a :class:`SceneDescriber` from config values.

    Args:
        enabled: When ``False`` a :class:`NullDescriber` is returned.
        model_name: HF model ID or Triton model name.
        device: PyTorch device string (``transformers`` backend only).
        task: Florence task prompt.
        backend: ``"transformers"`` (default) or ``"triton"``.
        triton_client: Pre-connected Triton client for the ``triton`` backend.
        tokenizer_dir: Path to directory containing ``tokenizer.json``
            (``triton`` backend only).
    """
    if not enabled:
        logger.info("florence_disabled returning_null_describer")
        return NullDescriber()

    if backend == "triton":
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

    # Default: transformers backend
    try:
        return FlorenceDescriber(model_name=model_name, device=device, task=task)
    except RuntimeError:
        logger.warning("transformers_not_installed returning_null_describer")
        return NullDescriber()
