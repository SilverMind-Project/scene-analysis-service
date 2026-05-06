# scene-analysis-service - Claude guidance

Standalone FastAPI microservice for multi-modal scene analysis. Part of the
cognitive-companion monorepo (`/home/sriram/code/nanai/`).

All inference runs via **Triton Inference Server** (shared with
`continuous-tracking`). SAS itself has no GPU runtime or PyTorch dependency.

---

## Commands

```bash
# Install base deps (API + Triton client + tokenizers)
uv sync

# Install Triton gRPC client (required for production inference)
uv sync --extra triton

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

# Build Docker image (lightweight — no PyTorch)
docker build -t scene-analysis-service .

# Legacy: Install full in-process inference stack (PyTorch-based, non-Triton)
uv sync --extra inference
```

---

## Architecture

### Service layout

```text
app/
├── config.py           Settings - YAML + SAS_ env overrides
├── main.py             FastAPI factory + lifespan (model loading)
├── models/
│   └── schemas.py      Pydantic I/O models
├── routers/            One file per endpoint group
└── services/
    ├── analyzer.py         SceneAnalyzer orchestrator (wired at startup)
    ├── detector.py         Detector ABC + TritonDetector + UltralyticsDetector + NullDetector
    ├── triton_detector.py  YOLO26L via Triton gRPC (default backend)
    ├── describer.py        SceneDescriber ABC + TritonFlorenceDescriber + FlorenceDescriber + NullDescriber
    ├── triton_describer.py Florence-2 via Triton Python backend (default)
    ├── embedder.py         ImageEmbedder ABC + TritonClipEmbedder + CLIPEmbedder + NullEmbedder
    ├── triton_embedder.py  CLIP ViT-L/14 via Triton gRPC (default backend)
    ├── device.py           resolve_device() — PyTorch device for legacy in-process backends only
    └── hazards.py          HazardRuleEngine - pure YAML-driven rule evaluator
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

class TritonDetector(Detector):        # Triton gRPC backend (default)
    ...

class UltralyticsDetector(Detector):   # Legacy PyTorch backend
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

## Triton Inference Server

SAS defaults to Triton for all three models. GPU vendor portability is
handled by Triton's ONNX Runtime backend — no client-side GPU detection.

| Model | Triton model name | Backend | Input shape | Output shape |
|-------|-------------------|---------|-------------|-------------|
| YOLO26L | `person-detector` | ONNX Runtime | [N,3,640,640] | [N,300,6] NMS-free |
| CLIP ViT-L/14 | `clip-vision` | ONNX Runtime | [N,3,224,224] | [N,768] |
| Florence-2 | `florence-2` | Python (ORT) | [1,3,H,W] + [1,seq] | [1,max_len] |

Models are INT8-quantized for performance. See
`../continuous-tracking/triton-models/` for configs and export/download scripts.

The shared Triton client library lives in `../triton-shared/` and is imported
by both SAS and CTS. It provides:
- `TritonClientProtocol` — structural interface for test mocking
- `TritonGrpcClient` — async gRPC client
- Pre/post processing functions for each model

### Triton client lifecycle

The `TritonGrpcClient` is created in `create_from_settings()` at startup.
It is **not** opened as an async context manager — instead, each sync
`detect()` / `embed()` / `describe()` call dispatches the async gRPC call
through a worker thread (see tech-debt #1).

---

## Inference Backends

Each component supports multiple backends via its `build_*()` factory.

### Detector backends

| Backend | Config key | Implementation | Notes |
|---------|-----------|----------------|-------|
| `triton` *(default)* | `inference_backend: triton` | `TritonDetector` | Calls YOLO26L on Triton via gRPC |
| `ultralytics` | `inference_backend: ultralytics` | `UltralyticsDetector` | Legacy in-process PyTorch |
| `onnxruntime` | `inference_backend: onnxruntime` | `OnnxRuntimeDetector` | Legacy in-process ONNX Runtime |

### Embedder backends

| Backend | Config key | Implementation | Notes |
|---------|-----------|----------------|-------|
| `triton` *(default)* | `clip_backend: triton` | `TritonClipEmbedder` | Calls CLIP on Triton via gRPC |
| `openclip` | `clip_backend: openclip` | `CLIPEmbedder` | Legacy in-process OpenCLIP + PyTorch |

### Describer backends

| Backend | Config key | Implementation | Notes |
|---------|-----------|----------------|-------|
| `triton` *(default)* | `florence_backend: triton` | `TritonFlorenceDescriber` | Calls Florence-2 on Triton via gRPC |
| `transformers` | `florence_backend: transformers` | `FlorenceDescriber` | Legacy in-process HF Transformers + PyTorch |

---

## Configuration

`config/config.yaml` is the source of truth. Every key maps to a `SAS_`
prefixed environment variable:

```bash
SAS_TRITON_URL=localhost:8001
SAS_INFERENCE_BACKEND=triton
SAS_CLIP_BACKEND=triton
SAS_FLORENCE_BACKEND=triton
```

`Settings.__getattr__` raises `AttributeError` for missing keys - do not add
fallback logic, raise early.

### Key settings

| Key | Default | Description |
| --- | ------- | ----------- |
| `triton_url` | `""` | Triton gRPC endpoint (empty = Triton backends fall back to Null) |
| `inference_backend` | `triton` | `triton` / `ultralytics` / `onnxruntime` |
| `yolo_model_name` | `person-detector` | Triton model name or .pt/.onnx path |
| `clip_backend` | `triton` | `triton` / `openclip` |
| `clip_model_name` | `clip-vision` | Triton model name or OpenCLIP architecture |
| `florence_backend` | `triton` | `triton` / `transformers` |
| `florence_model_name` | `florence-2` | Triton model name or HF model ID |
| `florence_tokenizer_dir` | `../continuous-tracking/triton-models/florence-2/1` | Path to tokenizer.json |
| `device` | `auto` | PyTorch device for legacy backends only |

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
- TritonDetector, TritonClipEmbedder, and TritonFlorenceDescriber each
  preprocess their own input (letterbox, CLIP resize, Florence resize).
  Preprocessing functions live in `triton_shared/inference/`.
- `HazardRuleEngine` must remain pure (no I/O in `evaluate()`). Add new
  constraint types as `_check_*` static/instance methods.
- Use `Image.Resampling.LANCZOS` / `Image.Resampling.BILINEAR` - the bare
  `Image.LANCZOS` / `Image.BILINEAR` constants are deprecated since Pillow 10.
- Triton gRPC calls are dispatched from worker threads via `_run_in_thread()`
  to bridge the async Triton client with the sync `Detector`/`Embedder`/`Describer` ABCs.

---

## Common tasks

### Add a new inference component (e.g. pose estimator)

1. Create `app/services/pose.py` with `PoseEstimator(ABC)`, `TritonPoseEstimator`,
   and `NullPose`.
2. Add `build_pose_estimator()` factory with `RuntimeError` catch.
3. Add fields to `AnalysisResult` and wire in `SceneAnalyzer.__init__` and
   `analyze()`.
4. Expose via new router or extend `/analyze` response schema.

### Add a new hazard rule type

1. Add a `_check_<rule_type>()` method to `HazardRuleEngine`.
2. Call it from `_check_rule()` using a new YAML key.
3. Add a test case to `tests/test_hazard_engine.py`.

### Switch a component to legacy in-process backend

Set the relevant backend config key and ensure the inference extras are
installed:

```bash
uv sync --extra inference
```

```yaml
inference_backend: ultralytics
clip_backend: openclip
florence_backend: transformers
```

---

## Known tech debt

Items intentionally deferred - fix before shipping anything performance-critical.

| # | File | Issue | Effort |
| - | ---- | ----- | ------ |
| 1 | `services/analyzer.py` | `analyze()` is synchronous; blocks the uvicorn event loop. Triton backends dispatch gRPC calls from worker threads as a workaround. Make `analyze` async and `await` Triton calls directly. | Medium |
| 2 | `services/device.py` | `resolve_device()` and `onnxruntime_providers()` only needed for legacy in-process backends. Can be removed once those backends are retired. | Small |
| 3 | `app/config.py` | `_coerce()` has no list handler - `SAS_ORT_PROVIDERS` cannot be set via env var (the YAML default is a list, so the env key is silently skipped). | Small |
| 4 | `tests/test_analyzer.py` | `_SpyDetector._called` counter in `TestRunFlags` is dead code - the MagicMock wrapper is used for all assertions. Remove the counter. | Trivial |
| 5 | `app/models/schemas.py` | `EmbedResponse` schema exists but there is no `/embed` standalone endpoint - only `/analyze` returns embeddings. Add `app/routers/embed.py`. | Small |
| 6 | `services/triton_*.py` | `_run_in_thread()` pattern is duplicated across triton_detector, triton_embedder, triton_describer. Extract to a shared utility. | Small |

---

## Relationship to cognitive-companion

- `cognitive-companion/backend/integrations/scene_analysis_client.py` -
  HTTP client that calls this service
- `cognitive-companion/backend/steps/builtin/scene_analysis.py` -
  Pipeline step that invokes the client
- `cognitive-companion/backend/steps/base.py` `ServiceContainer` -
  holds `scene_analysis_client`
- `cognitive-companion/docker-compose.yml` - commented-out service block
