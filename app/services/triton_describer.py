"""Florence-2-large scene description via Triton Inference Server.

Implements the :class:`SceneDescriber` ABC using the shared Triton client and
Florence-2 ONNX models served by Triton's Python backend.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.services.describer import SceneDescriber

logger = logging.getLogger(__name__)


def _run_in_thread(coro: Any) -> Any:
    """Run an async coroutine from sync code by spinning a fresh event loop in a worker thread."""

    def _target() -> Any:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_target).result()


class TritonFlorenceDescriber(SceneDescriber):
    """Florence-2-large scene describer via Triton Inference Server.

    Args:
        client: A :class:`~triton_shared.client.TritonClientProtocol` instance.
        model_name: Triton model name (default ``"florence-2"``).
        task: Florence task prompt (default ``"<DETAILED_CAPTION>"``).
        tokenizer_dir: Directory containing ``tokenizer.json``.
    """

    def __init__(
        self,
        client: Any,  # TritonClientProtocol
        model_name: str = "florence-2",
        task: str = "<DETAILED_CAPTION>",
        tokenizer_dir: Path | None = None,
    ) -> None:
        from triton_shared.inference.description import TASK_DETAILED_CAPTION

        self._client = client
        self._model_name = model_name
        self._task = task

        # Load tokenizer (Rust-based, no torch dependency).
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise RuntimeError(
                "tokenizers is required for TritonFlorenceDescriber. "
                "Install with: pip install tokenizers"
            )

        tokenizer_path = tokenizer_dir / "tokenizer.json" if tokenizer_dir else None
        if tokenizer_path is None or not tokenizer_path.exists():
            raise RuntimeError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Download from onnx-community/Florence-2-large or set florence_tokenizer_dir."
            )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Load special tokens for decoding.
        config_path = tokenizer_dir / "tokenizer_config.json"
        self._skip_special_tokens = False
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            self._skip_special_tokens = cfg.get("skip_special_tokens", False)

        logger.info(
            "florence_triton_loaded model=%s task=%s tokenizer=%s",
            model_name,
            task,
            tokenizer_path,
        )

    def describe(self, image: Image.Image) -> str:
        """Return a structured caption for *image*."""
        from triton_shared.inference.description import (
            florence_preprocess,
            tokenize_task_prompt,
        )

        # Preprocess image.
        pixel_values = florence_preprocess(image)

        # Tokenize task prompt.
        input_ids_list = tokenize_task_prompt(self._tokenizer, self._task)
        input_ids = np.array([input_ids_list], dtype=np.int64)

        # Run inference via Triton.
        outputs = _run_in_thread(
            self._client.infer(
                model_name=self._model_name,
                inputs=[
                    ("pixel_values", pixel_values.astype(np.float32)),
                    ("input_ids", input_ids),
                ],
                output_names=["output_ids"],
            )
        )
        raw_ids = outputs["output_ids"][0]  # (max_len,)

        # Decode token IDs to text.
        # Strip EOS and any trailing tokens.
        eos_id = 2  # Florence-2 EOS token
        ids_list = raw_ids.tolist()
        if eos_id in ids_list:
            ids_list = ids_list[: ids_list.index(eos_id)]
        # Skip the input prompt tokens — return only generated text.
        prompt_len = len(input_ids_list)
        generated = ids_list[prompt_len:]

        return self._tokenizer.decode(generated, skip_special_tokens=False)

    @property
    def is_available(self) -> bool:
        return True
