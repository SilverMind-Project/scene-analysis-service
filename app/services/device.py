"""Device and inference-backend selection.

Two concepts are managed here:

1. **PyTorch device** (``resolve_device``): the string passed to ``.to(device)``
   on PyTorch models (Florence-2, CLIP, and Ultralytics YOLO with a ``.pt``
   model). Supported values:

   - ``auto``    -- CUDA → XPU → CPU (highest-capability available)
   - ``cuda``    -- NVIDIA GPU via PyTorch CUDA
   - ``xpu``     -- Intel Arc GPU via Intel Extension for PyTorch (IPEX)
   - ``openvino``-- mapped to ``cpu`` for PyTorch components; YOLO can use
                    OpenVINO natively by loading a ``_openvino_model/`` export
   - ``vulkan``  -- mapped to ``cpu`` for PyTorch components; the ONNX Runtime
                    backend uses the Vulkan EP directly
   - ``cpu``     -- CPU fallback

2. **ONNX Runtime execution providers** (``onnxruntime_providers``): the
   providers list passed to ``onnxruntime.InferenceSession`` when
   ``inference_backend=onnxruntime`` is configured.  Derived from the device
   setting unless overridden by ``ort_providers`` in config.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Devices that are routed to CPU for PyTorch components (Florence-2, CLIP).
# These devices are only meaningful for the ONNX Runtime detector path.
_ORT_ONLY_DEVICES = frozenset({"openvino", "vulkan"})


def resolve_device(device_setting: str = "auto") -> str:
    """Return a concrete PyTorch device string.

    For ``openvino`` and ``vulkan`` the function returns ``"cpu"`` for use
    with PyTorch-based components (Florence-2, CLIP).  The ONNX Runtime
    detector uses :func:`onnxruntime_providers` instead of a PyTorch device.

    Args:
        device_setting: Value of the ``device`` config key.

    Returns:
        One of ``"cuda"``, ``"xpu"``, or ``"cpu"``.
    """
    setting = (device_setting or "auto").lower().strip()

    if setting == "cpu":
        logger.info("device_selected device=cpu")
        return "cpu"

    if setting in _ORT_ONLY_DEVICES:
        logger.info(
            "device_ort_only device=%s pytorch_fallback=cpu", setting
        )
        return "cpu"

    if setting == "xpu":
        return _resolve_xpu()

    if setting in ("cuda", "auto"):
        result = _resolve_cuda_or_auto(setting)
        if result == "cuda":
            return result
        if setting == "auto":
            # Try XPU before falling back to CPU.
            xpu = _resolve_xpu(silent=True)
            if xpu == "xpu":
                return "xpu"
        return result

    logger.warning("unknown_device_setting value=%s falling_back=cpu", setting)
    return "cpu"


def onnxruntime_providers(
    device_setting: str,
    override: list[str] | None = None,
) -> list[str]:
    """Return an ONNX Runtime execution-provider list for the given device.

    Args:
        device_setting: Value of the ``device`` config key.
        override: Explicit provider list from config (``ort_providers``).
            When non-empty this is returned as-is.

    Returns:
        Ordered list of ORT provider strings.  ``CPUExecutionProvider`` is
        always appended as the final fallback.
    """
    if override:
        return list(override)

    setting = (device_setting or "auto").lower().strip()

    mapping: dict[str, list[str]] = {
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "xpu": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        "vulkan": ["VulkanExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }

    if setting == "auto":
        try:
            import torch  # noqa: F401

            if torch.cuda.is_available():
                return mapping["cuda"]
        except ImportError:
            pass
        return mapping["cpu"]

    return mapping.get(setting, ["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_xpu(*, silent: bool = False) -> str:
    """Return ``"xpu"`` when Intel Extension for PyTorch (IPEX) is available
    and an XPU device is present, otherwise ``"cpu"``."""
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore[import-untyped]  # noqa: F401
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            if not silent:
                logger.info("device_selected device=xpu")
            return "xpu"
        if not silent:
            logger.warning(
                "xpu_requested_but_no_device_found falling_back=cpu"
            )
    except ImportError:
        if not silent:
            logger.warning(
                "ipex_not_installed device=xpu falling_back=cpu "
                "hint='pip install intel-extension-for-pytorch'"
            )
    return "cpu"


def _resolve_cuda_or_auto(setting: str) -> str:
    """Return ``"cuda"`` when a CUDA device is present, otherwise ``"cpu"``."""
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("device_selected device=cuda")
            return "cuda"
        if setting == "cuda":
            logger.warning(
                "cuda_requested_but_not_available falling_back=cpu"
            )
    except ImportError:
        logger.warning("torch_not_installed falling_back=cpu")

    logger.info("device_selected device=cpu")
    return "cpu"
