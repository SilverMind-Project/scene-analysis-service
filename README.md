# scene-analysis-service

A standalone FastAPI microservice for real-time multi-modal scene analysis.
Provides fast object detection (YOLO26x), structured scene description
(Florence-2-large), dense image embeddings (CLIP ViT-L/14), and
YAML-configured hazard alerting via a single HTTP API.

Designed to run alongside the cognitive-companion backend and be called from
the `scene_analysis` pipeline step.

---

## Features

| Component | Model | Notes |
| --------- | ----- | ----- |
| Object detection | YOLO26x (Ultralytics) | Up to 100 detections per frame |
| Scene description | Florence-2-large (Microsoft) | Structured natural-language caption |
| Image embeddings | CLIP ViT-L/14 (OpenCLIP) | 768-dim L2-normalised vector |
| Hazard alerting | Rule engine (YAML) | Label match + aspect-ratio + proximity checks |

All four components are optional and independently togglable via config or
per-request flags. The service starts and remains healthy even when inference
dependencies are not installed (graceful degradation via `Null*` stubs).

---

## Quick start

### Without inference dependencies (API only)

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8100
```

### With inference dependencies (CPU)

```bash
uv sync --extra inference
uv run uvicorn app.main:app --host 0.0.0.0 --port 8100
```

### NVIDIA GPU (CUDA)

```bash
uv sync --extra inference
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Check https://pytorch.org/get-started/locally/ for the current CUDA wheel URL.
SAS_DEVICE=cuda uv run uvicorn app.main:app --host 0.0.0.0 --port 8100
```

### Intel Arc GPU

Two paths are available. Choose based on whether you want PyTorch-level
integration (XPU) or explicit execution-provider control (ONNX Runtime).

#### Path 1: IPEX XPU (all models on Intel Arc)

```bash
uv sync --extra intel
SAS_DEVICE=xpu uv run uvicorn app.main:app --host 0.0.0.0 --port 8100
```

Requires Intel Extension for PyTorch (IPEX) and an Arc-compatible driver.
All three inference components (YOLO, Florence-2, CLIP) run on the XPU device.

#### Path 2: ONNX Runtime with OpenVINO EP (YOLO on Intel Arc)

```bash
# Export the YOLO model to ONNX format first
uv run python -c "from ultralytics import YOLO; YOLO('yolo26x.pt').export(format='onnx')"

uv sync --extra onnxruntime
```

Then set in `config/config.yaml`:

```yaml
device: openvino
inference_backend: onnxruntime
yolo_model_name: "yolo26x.onnx"
ort_providers: [OpenVINOExecutionProvider, CPUExecutionProvider]
```

Florence-2 and CLIP fall back to CPU on the `openvino` device. Use Path 1
(IPEX) to run all models on Intel Arc.

### Docker (CPU)

The Docker image uses [UV](https://github.com/astral-sh/uv) for fast,
reproducible package installation.

```bash
docker build -t scene-analysis-service .
docker run -p 8100:8100 scene-analysis-service
```

### Docker (GPU)

```bash
docker build --build-arg EXTRAS=inference -t scene-analysis-service .
docker run --gpus all -p 8100:8100 -e SAS_DEVICE=cuda scene-analysis-service
```

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
curl -s -X POST http://localhost:8100/analyze \
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
SAS_DEVICE=cuda
SAS_YOLO_ENABLED=false
SAS_PORT=8200
```

### Key settings

| Key | Default | Description |
| --- | ------- | ----------- |
| `device` | `auto` | `auto \| cuda \| xpu \| openvino \| vulkan \| cpu` |
| `inference_backend` | `ultralytics` | `ultralytics \| onnxruntime` |
| `ort_providers` | `[]` | ONNX Runtime EP list (empty = derived from device) |
| `ort_input_size` | `640` | Square input size for ONNX Runtime detector |
| `yolo_enabled` | `true` | Enable YOLO detection |
| `yolo_model_name` | `yolo26x.pt` | Model file: `.pt`, `.onnx`, or `_openvino_model/` |
| `yolo_confidence_threshold` | `0.25` | Detection confidence floor |
| `florence_enabled` | `true` | Enable Florence-2 description |
| `florence_model_name` | `microsoft/Florence-2-large` | HuggingFace model ID |
| `florence_task` | `<DETAILED_CAPTION>` | Florence prompt task token |
| `clip_enabled` | `true` | Enable CLIP embedding |
| `clip_model_name` | `ViT-L-14` | OpenCLIP model name |
| `clip_pretrained` | `openai` | OpenCLIP pretrained weights tag |
| `hazards_config_path` | `config/hazards.yaml` | Hazard rules file |
| `max_image_size_px` | `1920` | Longest edge limit; larger images are downscaled |
| `port` | `8100` | Listening port |
| `log_level` | `info` | `debug \| info \| warning \| error` |

### Device and backend matrix

| Target hardware | `device` | `inference_backend` | Additional setup |
| --------------- | -------- | ------------------- | ---------------- |
| NVIDIA GPU | `cuda` | `ultralytics` | Install torch+cuda wheels |
| Intel Arc (all models) | `xpu` | `ultralytics` | Install `intel` extra (IPEX) |
| Intel Arc (YOLO only, explicit EP) | `openvino` | `onnxruntime` | Export ONNX model; install `onnxruntime` extra |
| Vulkan (experimental) | `vulkan` | `onnxruntime` | ORT build with Vulkan EP |
| CPU | `cpu` | `ultralytics` | No extras needed |

### Hazard rules (`config/hazards.yaml`)

Each rule matches one or more YOLO class labels and fires a `HazardAlert`
when a detection satisfies all constraints:

```yaml
hazards:
  - name: fire
    labels: [fire, flame]
    severity: critical
    description: "Active fire or open flame detected."

  - name: person_on_floor
    labels: [person]
    severity: high
    description: "Person in fallen posture."
    aspect_ratio_min: 1.5   # width/height >= 1.5 -> likely prone

  - name: medication_near_stove
    labels: [bottle]
    severity: medium
    near_labels: [oven]
    proximity_px: 200        # L-infinity distance between bbox centres
    description: "Medication-like container near cooking appliance."
```

Rule fields:

| Field | Required | Description |
| ----- | -------- | ----------- |
| `name` | yes | Hazard identifier surfaced in the API |
| `labels` | yes | YOLO class names that trigger this rule |
| `severity` | yes | `low \| medium \| high \| critical` |
| `description` | yes | Human-readable alert text |
| `near_labels` | no | Only fire when near one of these labels |
| `proximity_px` | no | L-infinity pixel distance threshold (default 200) |
| `aspect_ratio_min` | no | `bbox_width / bbox_height` minimum (fallen-person heuristic) |

---

## YOLO model selection

YOLO26 is the current generation. Ultralytics model files are downloaded
automatically on first use. Refer to the
[Ultralytics documentation](https://docs.ultralytics.com/) for current
benchmark numbers.

| Model | Use case |
| ----- | -------- |
| `yolo26x.pt` | Highest accuracy (default) |
| `yolo26l.pt` | Good accuracy/speed balance |
| `yolo26m.pt` | Lower-power deployments |

Set `SAS_YOLO_MODEL_NAME=yolo26l.pt` to trade accuracy for speed.

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
│       ├── detector.py        # UltralyticsDetector + OnnxRuntimeDetector + NullDetector
│       ├── describer.py       # Florence-2 wrapper + NullDescriber
│       ├── embedder.py        # CLIP wrapper + NullEmbedder
│       ├── device.py          # Device resolution (auto/cuda/xpu/openvino/vulkan/cpu)
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

Tests do not require inference dependencies. All model components are replaced
by `Null*` stubs in the test fixtures.

### Lint and format

```bash
uv run ruff check app/ tests/
uv run ruff format app/ tests/
```

### Interactive API docs

With the service running, visit:

- Swagger UI: `http://localhost:8100/docs`
- ReDoc: `http://localhost:8100/redoc`

---

## Integration with cognitive-companion

The cognitive-companion backend communicates with this service via
`SceneAnalysisClient` (`backend/integrations/scene_analysis_client.py`).
Enable it by setting `scene_analysis.enabled: true` and
`scene_analysis.base_url: http://scene-analysis:8100` in the backend's
`settings.yaml`.

The `scene_analysis` pipeline step (`backend/steps/builtin/scene_analysis.py`)
calls `SceneAnalysisClient.analyze()` and writes results into the
`pipeline_data` dict under the keys `scene_detections`, `scene_description`,
`scene_embedding`, `scene_hazards`, and the three `scene_*_available` flags.

See `cognitive-companion/docker-compose.yml` for the commented-out service
block that can be uncommented to run this service alongside the backend.

---

## Graceful degradation

The service is designed to remain fully operational at every level of
capability:

| Inference deps | Behaviour |
| -------------- | --------- |
| None installed | All `Null*` stubs; `/health` returns `detector_available: false`, all result lists empty |
| `ultralytics` only | Detection works; description and embedding return empty |
| All `[inference]` deps | Full pipeline |
| `[onnxruntime]` + ONNX model | YOLO via ORT with configurable EP; Florence-2 and CLIP on CPU |

The `*_available` fields in every response indicate which components are
active, letting callers adapt downstream logic accordingly.
