# scene-analysis-service - Claude guidance

Standalone FastAPI microservice for multi-modal scene analysis. Part of the
cognitive-companion monorepo (`/home/sriram/code/nanai/`).

---

## Commands

```bash
# Install base deps (API only, no inference)
uv sync

# Install full inference stack (CPU PyTorch)
uv sync --extra inference

# Install with NVIDIA CUDA
uv sync --extra inference
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Check https://pytorch.org/get-started/locally/ for the current CUDA wheel URL.

# Install with Intel Arc (IPEX XPU)
uv sync --extra intel

# Install with ONNX Runtime + OpenVINO EP (Intel Arc, YOLO only)
uv sync --extra onnxruntime

# Install dev deps
uv sync --group dev

# Run (development)
uv run uvicorn app.main:app --reload --port 8300

# Lint
uv run ruff check app/ tests/
uv run ruff format app/ tests/

# Tests (no inference deps required)
uv run pytest

# Tests with coverage
uv run pytest --cov=app --cov-report=term-missing

# Export YOLO model to ONNX (required for onnxruntime backend)
uv run python -c "from ultralytics import YOLO; YOLO('yolo26x.pt').export(format='onnx')"

# Export YOLO model to OpenVINO IR (for native Ultralytics OpenVINO path)
uv run python -c "from ultralytics import YOLO; YOLO('yolo26x.pt').export(format='openvino')"

# Build Docker image (CPU) - uses UV internally
docker build -t scene-analysis-service .

# Build Docker image (with inference)
docker build --build-arg EXTRAS=inference -t scene-analysis-service .
```

---

## Architecture

### Service layout

```text
app/
├── config.py       Settings - YAML + SAS_ env overrides
├── main.py         FastAPI factory + lifespan (model loading)
├── models/
│   └── schemas.py  Pydantic I/O models
├── routers/        One file per endpoint group
└── services/
    ├── analyzer.py     SceneAnalyzer orchestrator (wired at startup)
    ├── detector.py     Detector ABC + UltralyticsDetector + OnnxRuntimeDetector + NullDetector
    ├── describer.py    SceneDescriber ABC + FlorenceDescriber + NullDescriber
    ├── embedder.py     ImageEmbedder ABC + CLIPEmbedder + NullEmbedder
    ├── device.py       resolve_device() + onnxruntime_providers()
    └── hazards.py      HazardRuleEngine - pure YAML-driven rule evaluator
```

### Core pattern: ABC + Null implementation

Every inference component follows this structure:

```python
class Detector(ABC):          # Abstract base
    @abstractmethod
    def detect(self, image): ...
    @property
    @abstractmethod
    def is_available(self) -> bool: ...

class UltralyticsDetector(Detector):   # PyTorch / Ultralytics backend
    ...

class OnnxRuntimeDetector(Detector):   # ONNX Runtime backend (explicit EP)
    ...

class NullDetector(Detector):          # Graceful fallback
    def detect(self, image): return []
    @property
    def is_available(self): return False
```

`build_detector()` (and equivalents) catches `RuntimeError` from import
failures and returns `NullDetector`. The service always starts successfully.

### SceneAnalyzer

`SceneAnalyzer` is instantiated once at startup via `create_from_settings(cfg)`
and stored on `app.state.analyzer`. Routes access it via
`request.app.state.analyzer`. It accepts `run_*` flags to skip components.

Public availability properties (`analyzer.detector_available`, etc.) should be
used instead of reaching into the private `_detector`, `_describer`, or
`_embedder` attributes.

### HazardRuleEngine

Pure function: `evaluate(detections) -> list[HazardAlert]`. No I/O. Loads
rules from `config/hazards.yaml` at startup. Missing file = zero rules (no
crash). Supports label matching, aspect-ratio heuristic, and L-infinity
proximity checks.

---

## Device and inference backend

### Device resolution

`resolve_device(device_setting)` in `device.py` returns a concrete PyTorch
device string (`"cuda"`, `"xpu"`, or `"cpu"`):

| Config `device` | PyTorch device | Notes |
| --------------- | -------------- | ----- |
| `auto` | CUDA → XPU → CPU | Picks highest-capability available |
| `cuda` | `cuda` | Falls back to `cpu` if CUDA unavailable |
| `xpu` | `xpu` | Intel Arc via IPEX; falls back to `cpu` |
| `openvino` | `cpu` | ORT-only device; PyTorch components use CPU |
| `vulkan` | `cpu` | ORT-only device; PyTorch components use CPU |
| `cpu` | `cpu` | Always CPU |

### ONNX Runtime providers

`onnxruntime_providers(device_setting, override)` returns the ORT execution
provider list for `OnnxRuntimeDetector`. Set `ort_providers` in config to
override the default derivation.

| `device` | Default providers |
| -------- | ----------------- |
| `cuda` | `[CUDAExecutionProvider, CPUExecutionProvider]` |
| `xpu` / `openvino` | `[OpenVINOExecutionProvider, CPUExecutionProvider]` |
| `vulkan` | `[VulkanExecutionProvider, CPUExecutionProvider]` |
| `cpu` | `[CPUExecutionProvider]` |

### Inference backends

`inference_backend` in config selects the YOLO detector implementation:

- `ultralytics` (default): `UltralyticsDetector` wraps `ultralytics.YOLO`.
  Accepts `.pt`, `.onnx`, or `_openvino_model/` model names - Ultralytics
  handles the format automatically.

- `onnxruntime`: `OnnxRuntimeDetector` uses `onnxruntime.InferenceSession`
  with an explicit provider list. Requires a `.onnx` model file. Use when you
  need deterministic EP selection (e.g., force OpenVINO EP on Intel Arc, or
  use the experimental Vulkan EP).

When `inference_backend=onnxruntime` and `yolo_model_name` does not end in
`.onnx`, `build_detector()` logs a warning and returns `NullDetector`.

---

## Configuration

`config/config.yaml` is the source of truth. Every key maps to a `SAS_`
prefixed environment variable:

```bash
SAS_DEVICE=xpu
SAS_INFERENCE_BACKEND=onnxruntime
SAS_YOLO_MODEL_NAME=yolo26x.onnx
```

`Settings.__getattr__` raises `AttributeError` for missing keys - do not add
fallback logic, raise early.

---

## Tests

- **No inference deps required** - all tests use `NullDetector`, `NullDescriber`,
  `NullEmbedder` via fixtures in `tests/conftest.py`.
- `asyncio_mode = "auto"` is set in `pyproject.toml` - all test methods can be
  `async def` without decoration.
- `null_analyzer` fixture creates a `SceneAnalyzer` with all `Null*` components
  and a non-existent hazards path.
- `test_client` fixture creates the app, pre-assigns `null_analyzer` to
  `app.state`, then uses `with TestClient(app, raise_server_exceptions=True) as client:`. The lifespan is bypassed to avoid loading real models.

### Class property override rule

**Never mutate a class-level property in a test** (`type(obj).prop = ...`).
This poisons all subsequent instantiations of that class within the test
session. Use local subclasses instead:

```python
# BAD - mutates NullDetector for all later tests
type(detector).is_available = property(lambda self: True)

# GOOD - isolated subclass
class _SpyDetector(NullDetector):
    @property
    def is_available(self) -> bool:
        return True
```

---

## Coding conventions

- All public service classes have an ABC, a real implementation, and a `Null*`
  stub. Follow this pattern for any new inference component.
- `build_*()` factory functions are the only place where `ImportError` /
  `RuntimeError` from missing deps should be caught.
- Logging uses the stdlib `logging` module (not structlog). Format:
  `logger.info("event_name key=%s", value)`.
- `to_dict()` methods on dataclasses are the serialisation boundary - keep
  Pydantic schemas in `models/schemas.py` and service dataclasses in
  `services/*.py` separately.
- Image downscaling happens in `SceneAnalyzer._load_image()` before any
  component sees the image. Do not resize inside individual components.
- `OnnxRuntimeDetector._preprocess()` does its own letterbox resize because it
  operates on raw PIL images before the analyzer downscale. Always letterbox to
  `ort_input_size` (default 640).
- `HazardRuleEngine` must remain pure (no I/O in `evaluate()`). Add new
  constraint types as `_check_*` static/instance methods.
- Use `Image.Resampling.LANCZOS` / `Image.Resampling.BILINEAR` - the bare
  `Image.LANCZOS` / `Image.BILINEAR` constants are deprecated since Pillow 10.
- Use `torch.amp.autocast(device_type=...)` - `torch.cuda.amp.autocast()` is
  deprecated since PyTorch 2.4.

---

## Common tasks

### Add a new inference component (e.g. pose estimator)

1. Create `app/services/pose.py` with `PoseEstimator(ABC)`, `MediaPipePose`,
   and `NullPose`.
2. Add `build_pose_estimator()` factory with `RuntimeError` catch.
3. Add fields to `AnalysisResult` and wire in `SceneAnalyzer.__init__` and
   `analyze()`.
4. Expose via new router or extend `/analyze` response schema.

### Add a new hazard rule type

1. Add a `_check_<rule_type>()` method to `HazardRuleEngine`.
2. Call it from `_check_rule()` using a new YAML key.
3. Add a test case to `tests/test_hazard_engine.py`.

### Change the default YOLO model

Edit `config/config.yaml` → `yolo_model_name`. The `[inference]` extras pull
Ultralytics which auto-downloads `.pt` models on first use. Override the cache
directory with the `YOLO_CONFIG_DIR` env var.

YOLO26x is the current default. For faster inference at lower accuracy use
`yolo26l.pt` or `yolo26m.pt`. Refer to the [Ultralytics docs](https://docs.ultralytics.com/) for
current benchmark numbers.

### Use Intel Arc with OpenVINO EP (onnxruntime backend)

```bash
# 1. Export the model
uv run python -c "from ultralytics import YOLO; YOLO('yolo26x.pt').export(format='onnx')"

# 2. Install ONNX Runtime + OpenVINO
uv sync --extra onnxruntime
```

`config/config.yaml`:

```yaml
device: openvino
inference_backend: onnxruntime
yolo_model_name: "yolo26x.onnx"
ort_providers: [OpenVINOExecutionProvider, CPUExecutionProvider]
```

Florence-2 and CLIP will run on CPU in this configuration. To run all models
on Intel Arc, use `device: xpu` with the `intel` extra (IPEX) instead.

---

## Known tech debt

Items intentionally deferred - fix before shipping anything performance-critical.

| # | File | Issue | Effort |
| - | ---- | ----- | ------ |
| 1 | `services/analyzer.py` | `analyze()` is synchronous; heavy CPU inference (Florence-2 on CPU) blocks the uvicorn event loop. Wrap each component call in `asyncio.to_thread` and make `analyze` async. | Medium |
| 2 | `services/device.py` | `onnxruntime_providers()` returns CPU fallback for `auto` when torch isn't installed, even if an ORT CUDA build is present. Should probe ORT providers directly. | Small |
| 3 | `app/config.py` | `_coerce()` has no list handler - `SAS_ORT_PROVIDERS` cannot be set via env var (the YAML default is a list, so the env key is silently skipped). | Small |
| 4 | `pyproject.toml` | The `cuda` extra is a no-op alias for `inference` - no CUDA-specific wheels are added. The real CUDA install still requires a manual `uv pip install torch ...` step, making the extra misleading. | Small |
| 5 | `tests/test_analyzer.py` | `_SpyDetector._called` counter in `TestRunFlags` is dead code - the MagicMock wrapper is used for all assertions. Remove the counter. | Trivial |
| 6 | `services/detector.py` | Post-processing NMS is reimplemented in pure NumPy for `OnnxRuntimeDetector`. When Ultralytics ships YOLO26 with end-to-end NMS baked into the ONNX export, this can be removed. | Deferred |
| 7 | `app/models/schemas.py` | `EmbedResponse` schema exists but there is no `/embed` standalone endpoint - only `/analyze` returns embeddings. Add `app/routers/embed.py`. | Small |

---

## Relationship to cognitive-companion

- `cognitive-companion/backend/integrations/scene_analysis_client.py` -
  HTTP client that calls this service
- `cognitive-companion/backend/steps/builtin/scene_analysis.py` -
  Pipeline step that invokes the client
- `cognitive-companion/backend/steps/base.py` `ServiceContainer` -
  holds `scene_analysis_client`
- `cognitive-companion/docker-compose.yml` - commented-out service block
