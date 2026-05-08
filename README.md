# scene-analysis-service

A lightweight FastAPI microservice for real-time multi-modal scene analysis.
All inference runs via **Triton Inference Server** (ONNX, INT8-quantized)
shared with the `continuous-tracking` system. No GPU runtime or PyTorch
required in this container.

Provides object detection (YOLO26L), structured scene description
(Florence-2-large), dense image embeddings (CLIP ViT-L/14), and
YAML-configured hazard alerting via a single HTTP API.

---

## Features

| Component | Model | Backend | Notes |
| --------- | ----- | ------- | ----- |
| Object detection | YOLO26L | Triton (ONNX Runtime, INT8) | Up to 100 detections per frame, NMS-free |
| Scene description | Florence-2-large | Triton (Python backend, INT8) | Structured natural-language caption |
| Image embeddings | CLIP ViT-L/14 | Triton (ONNX Runtime, INT8) | 768-dim L2-normalised vector |
| Hazard alerting | Rule engine (YAML) | In-process | Label match + aspect-ratio + proximity checks |

All four components are optional and independently togglable via config or
per-request flags. The service starts and remains healthy even when Triton is
unavailable (graceful degradation via `Null*` stubs).

---

## Quick start

### Prerequisites

- Triton Inference Server running with the model repository from
  `../continuous-tracking/triton-models/`. See that project's README for
  model export/download and Triton setup.
- Python 3.13+

### With Triton (production)

```bash
uv sync --extra triton
SAS_TRITON_URL=localhost:8701 uv run uvicorn app.main:app --host 0.0.0.0 --port 8300
```

### Without Triton (API only, all Null stubs)

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8300
```

The service starts and all endpoints return empty results with
`*_available: false`.

### Docker

```bash
docker build -t scene-analysis-service .
docker run -p 8300:8300 -e SAS_TRITON_URL=triton:8701 scene-analysis-service
```

The Docker image is ~200 MB — no PyTorch, no GPU drivers.

---

## API endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/health` | Service health + component availability |
| `POST` | `/detect` | YOLO object detection only |
| `POST` | `/describe` | Florence-2 scene description only |
| `POST` | `/analyze` | Full pipeline (detect + describe + embed + hazards) |

All `POST` endpoints accept `multipart/form-data` with an `image` field
(JPEG, PNG, or any Pillow-supported format).

### `/analyze` query parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `run_detect` | bool | `true` | Run YOLO detection |
| `run_describe` | bool | `true` | Run Florence-2 description |
| `run_embed` | bool | `true` | Run CLIP embedding |
| `run_hazards` | bool | `true` | Evaluate hazard rules |

### Example

```bash
curl -s -X POST http://localhost:8300/analyze \
  -F "image=@photo.jpg" \
  "?run_embed=false" | jq .
```

```json
{
  "detections": [
    {"label": "person", "confidence": 0.92, "bbox": [10, 20, 200, 400], "class_id": 0}
  ],
  "description": "A person standing in a kitchen near a stove.",
  "embedding": [],
  "hazards": [],
  "detector_available": true,
  "describer_available": true,
  "embedder_available": true
}
```

---

## Configuration

Configuration is read from `config/config.yaml` at startup. Every key can be
overridden with an environment variable prefixed `SAS_` (uppercased), e.g.:

```bash
SAS_TRITON_URL=triton:8701
SAS_YOLO_ENABLED=false
SAS_PORT=8200
```

### Key settings

| Key | Default | Description |
| --- | ------- | ----------- |
| `triton_url` | `""` | Triton gRPC endpoint (required for Triton backends) |
| `inference_backend` | `triton` | Detector: `triton` / `ultralytics` / `onnxruntime` |
| `yolo_model_name` | `person-detector` | Triton model name |
| `clip_backend` | `triton` | Embedder: `triton` / `openclip` |
| `clip_model_name` | `clip-vision` | Triton model name |
| `florence_backend` | `triton` | Describer: `triton` / `transformers` |
| `florence_model_name` | `florence-2` | Triton model name |
| `florence_tokenizer_dir` | `/models/florence-2/1` | Tokenizer path |
| `device` | `auto` | PyTorch device (legacy backends only) |
| `yolo_confidence_threshold` | `0.25` | Detection confidence floor |
| `max_image_size_px` | `1920` | Longest edge limit; larger images downscaled |
| `port` | `8300` | Listening port |
| `log_level` | `info` | `debug` / `info` / `warning` / `error` |

---

## GPU vendor support

All GPU-specific logic lives in Triton configs, not in SAS. Both NVIDIA and
Intel Arc GPUs are supported with identical SAS client code:

| GPU | Triton setup |
| --- | ------------ |
| NVIDIA | Default `config.pbtxt` (TensorRT EP / CUDA EP) |
| Intel Arc | `python triton-models/scripts/configure_gpu.py --vendor intel` |

SAS itself needs no GPU drivers, PyTorch, or vendor-specific libraries.

---

## Legacy: in-process inference (without Triton)

SAS retains legacy in-process backends for development and fallback. Install
the full PyTorch stack:

```bash
uv sync --extra inference
```

Then set backends in `config/config.yaml`:

```yaml
triton_url: ""                     # disable Triton
inference_backend: ultralytics     # YOLO via PyTorch
clip_backend: openclip             # CLIP via OpenCLIP
florence_backend: transformers     # Florence-2 via HF Transformers
```

GPU acceleration requires additional setup (CUDA wheels or Intel IPEX).
Prefer Triton backends for production.

---

## Project layout

```text
scene-analysis-service/
├── app/
│   ├── config.py              # Settings (YAML + SAS_ env overrides)
│   ├── main.py                # FastAPI app factory + lifespan
│   ├── models/
│   │   └── schemas.py         # Pydantic request/response models
│   ├── routers/
│   │   ├── analyze.py         # POST /analyze  (full pipeline)
│   │   ├── detect.py          # POST /detect
│   │   ├── describe.py        # POST /describe
│   │   └── health.py          # GET  /health
│   └── services/
│       ├── analyzer.py        # SceneAnalyzer orchestrator
│       ├── detector.py        # Detector ABC + build_detector()
│       ├── triton_detector.py # TritonDetector (YOLO via gRPC)
│       ├── describer.py       # SceneDescriber ABC + build_describer()
│       ├── triton_describer.py# TritonFlorenceDescriber (Florence-2 via gRPC)
│       ├── embedder.py        # ImageEmbedder ABC + build_embedder()
│       ├── triton_embedder.py # TritonClipEmbedder (CLIP via gRPC)
│       ├── device.py          # PyTorch device resolution (legacy backends)
│       └── hazards.py         # HazardRuleEngine (pure, YAML-driven)
├── config/
│   ├── config.yaml            # Default service config
│   └── hazards.yaml           # Default hazard rules
├── tests/
│   ├── conftest.py            # null_analyzer + TestClient fixtures
│   ├── test_analyzer.py       # SceneAnalyzer unit tests
│   ├── test_api.py            # HTTP endpoint tests
│   └── test_hazard_engine.py  # HazardRuleEngine unit tests
├── Dockerfile
└── pyproject.toml
```

---

## Development

### Install dev dependencies

```bash
uv sync --group dev
```

### Run tests

```bash
uv run pytest
# with coverage
uv run pytest --cov=app --cov-report=term-missing
```

Tests do not require Triton or inference dependencies. All model components
are replaced by `Null*` stubs.

### Lint and format

```bash
uv run ruff check app/ tests/
uv run ruff format app/ tests/
```

---

## Integration with cognitive-companion

The cognitive-companion backend communicates with this service via
`SceneAnalysisClient` (`backend/integrations/scene_analysis_client.py`).
Enable it by setting `scene_analysis.enabled: true` and
`scene_analysis.base_url: http://scene-analysis:8300` in the backend's
`settings.yaml`.

---

## Dependencies

| Dependency | Purpose | Required? |
|-----------|---------|-----------|
| `triton-shared` | Shared Triton client + pre/post processing | Yes (base) |
| `tritonclient[all]` | Triton gRPC client | Yes (production) |
| `tokenizers` | Florence-2 task prompt tokenization | Yes (base) |
| `fastapi` + `uvicorn` | HTTP API server | Yes (base) |
| `pillow` + `numpy` | Image handling | Yes (base) |
| `torch` + `transformers` + `open_clip_torch` | Legacy in-process backends | Optional (`inference` extra) |
