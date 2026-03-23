# OTX (OpenVINO Training Extensions) - Comprehensive Documentation

**Version:** 2.6.0
**License:** Apache 2.0
**Python:** >= 3.10

OTX is a low-code transfer learning framework for Computer Vision by Intel. It enables training, inference, optimization, and deployment of deep learning models with minimal code. Built on PyTorch Lightning + OpenVINO.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Architecture Overview](#2-architecture-overview)
3. [Installation](#3-installation)
4. [Core Concepts](#4-core-concepts)
5. [Engine System](#5-engine-system)
6. [Data System](#6-data-system)
7. [Models](#7-models)
8. [Recipes (YAML Configs)](#8-recipes-yaml-configs)
9. [CLI Reference](#9-cli-reference)
10. [Python API Reference](#10-python-api-reference)
11. [Export & Deployment](#11-export--deployment)
12. [Backend System](#12-backend-system)
13. [Folder-by-Folder Breakdown](#13-folder-by-folder-breakdown)
14. [Supported Tasks & Models](#14-supported-tasks--models)
15. [Configuration Deep-Dive](#15-configuration-deep-dive)
16. [Metrics & Evaluation](#16-metrics--evaluation)
17. [Advanced Features](#17-advanced-features)
18. [Web UI & REST API (Geti-Tune)](#18-web-ui--rest-api-geti-tune)
19. [CI/CD & Development](#19-cicd--development)
20. [Glossary](#20-glossary)
21. [Local Setup & Offline Models](#21-local-setup--offline-models)
22. [GSoC Project: Object Detection Tracking Support for OTX](#22-gsoc-project-object-detection-tracking-support-for-otx)

---

## 1. Repository Structure

```
otx/
├── lib/                          # Core OTX Python library
│   ├── src/otx/                  # Main package source
│   │   ├── backend/              # Backend implementations
│   │   │   ├── native/           # PyTorch Lightning backend (training + export)
│   │   │   └── openvino/         # OpenVINO backend (inference only)
│   │   ├── cli/                  # Command-line interface
│   │   ├── config/               # Configuration dataclasses
│   │   ├── data/                 # Data loading, transforms, datasets
│   │   ├── engine/               # Abstract Engine base class + factory
│   │   ├── metrics/              # Evaluation metrics
│   │   ├── models/               # Model re-exports (convenience imports)
│   │   ├── recipe/               # 114 YAML config templates
│   │   ├── tools/                # Auto-configurator, converters
│   │   ├── types/                # Enums, type aliases, data entities
│   │   └── utils/                # General utilities
│   ├── tests/                    # Unit and integration tests
│   ├── pyproject.toml            # Package metadata & dependencies
│   └── tox.ini                   # Test automation config
├── backend/                      # Geti-Tune REST API (FastAPI)
│   └── app/                      # API routes, services, workers
├── ui/                           # Web UI (React + TypeScript)
├── docker/                       # Dockerfiles (CPU, CUDA, Geti-Tune)
├── docs/                         # Sphinx documentation source
├── .github/workflows/            # CI/CD pipelines
├── CHANGELOG.md                  # Release notes
├── CONTRIBUTING.md               # Contribution guidelines
├── README.md                     # Project overview
└── LICENSE                       # Apache 2.0
```

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────┐
│                    CLI / Python API               │
│               (otx.cli / otx.engine)              │
├──────────────────────────────────────────────────┤
│                  Engine Layer                      │
│    ┌─────────────────┐  ┌──────────────────┐      │
│    │   OTXEngine      │  │    OVEngine       │     │
│    │ (PyTorch/Light.) │  │ (OpenVINO IR)     │     │
│    └────────┬────────┘  └────────┬─────────┘      │
├─────────────┼────────────────────┼────────────────┤
│             │    Model Layer     │                  │
│    ┌────────┴────────┐  ┌───────┴────────┐        │
│    │   OTXModel       │  │   OVModel       │      │
│    │ (LightningModule)│  │ (OpenVINO wrap) │      │
│    └────────┬────────┘  └────────────────┘        │
│             │                                      │
│    ┌────────┴────────────────────────────┐        │
│    │   Task-Specific Models               │       │
│    │ Detection | Classification | Seg...  │       │
│    └─────────────────────────────────────┘        │
├──────────────────────────────────────────────────┤
│                  Data Layer                        │
│    ┌────────────────────────────────────┐         │
│    │   OTXDataModule (LightningDataMod) │         │
│    │   ├── OTXDataset (per task)        │         │
│    │   ├── Transforms (torchvision)     │         │
│    │   └── Datumaro (format parsing)    │         │
│    └────────────────────────────────────┘         │
├──────────────────────────────────────────────────┤
│                Recipe (YAML Configs)               │
│   Model arch + optimizer + scheduler + data aug   │
└──────────────────────────────────────────────────┘
```

**Key design principles:**

- **Multi-backend:** Native (PyTorch) for training, OpenVINO for optimized inference
- **Recipe-driven:** All training configs defined in YAML files
- **Auto-configuration:** Can auto-detect task type and select defaults from a data directory
- **Lightning-native:** Models are `LightningModule`, data is `LightningDataModule`
- **Factory pattern:** `create_engine()` auto-selects the right engine backend

---

## 3. Installation

### From PyPI

```bash
# With CUDA support (NVIDIA GPU)
pip install otx[cuda]

# With XPU support (Intel GPU)
pip install otx[xpu]

# CPU only
pip install otx
```

### From Source (this repo)

```bash
cd lib
python -m venv .otx
source .otx/bin/activate
pip install -e ".[cuda]"       # or .[xpu] or .[dev]
```

### Key Dependencies

| Package   | Version | Purpose                        |
| --------- | ------- | ------------------------------ |
| PyTorch   | 2.8.0   | Deep learning framework        |
| Lightning | 2.4.0   | Training orchestration         |
| OpenVINO  | 2025.2  | Model optimization & inference |
| Datumaro  | 1.10.0  | Dataset format parsing         |
| OmegaConf | 2.3.0   | YAML config management         |
| Anomalib  | 1.1.3   | Anomaly detection models       |
| NNCF      | 2.17.0  | Neural network compression     |
| timm      | 1.0.3   | Image model library            |

---

## 4. Core Concepts

### Task Types (`otx.types.task.OTXTaskType`)

OTX supports 12 computer vision tasks:

| Task                        | Enum Value               | Description                      |
| --------------------------- | ------------------------ | -------------------------------- |
| Multi-class Classification  | `MULTI_CLASS_CLS`        | Single label per image           |
| Multi-label Classification  | `MULTI_LABEL_CLS`        | Multiple labels per image        |
| Hierarchical Classification | `H_LABEL_CLS`            | Tree-structured labels           |
| Object Detection            | `DETECTION`              | Bounding box prediction          |
| Rotated Detection           | `ROTATED_DETECTION`      | Oriented bounding boxes          |
| Keypoint Detection          | `KEYPOINT_DETECTION`     | Landmark/pose estimation         |
| Instance Segmentation       | `INSTANCE_SEGMENTATION`  | Per-object masks                 |
| Semantic Segmentation       | `SEMANTIC_SEGMENTATION`  | Per-pixel class labels           |
| Anomaly (unified)           | `ANOMALY`                | Anomaly detection (all variants) |
| Anomaly Classification      | `ANOMALY_CLASSIFICATION` | Normal vs anomalous              |
| Anomaly Detection           | `ANOMALY_DETECTION`      | Anomaly localization             |
| Anomaly Segmentation        | `ANOMALY_SEGMENTATION`   | Pixel-level anomaly              |

### Type Aliases (`otx.types.types`)

```python
MODEL = OTXModel | OVModel | PathLike     # A model instance or path to config/checkpoint
DATA = OTXDataModule | PathLike            # A data module or path to data root
METRICS = dict[str, float]                 # Metric name -> value
ANNOTATIONS = list[OTXDataItem]            # List of prediction items
```

---

## 5. Engine System

### `otx.engine.Engine` (Abstract Base)

**File:** `lib/src/otx/engine/engine.py`

The abstract engine defines the core workflow contract:

```python
class Engine(ABC):
    def train(self, **kwargs) -> METRICS           # Train model
    def test(self, **kwargs) -> METRICS            # Evaluate model
    def predict(self, **kwargs) -> ANNOTATIONS     # Run inference
    def export(self, **kwargs) -> Path             # Export model

    @staticmethod
    def is_supported(model, data) -> bool          # Can this engine handle these inputs?

    @property
    def work_dir -> PathLike                       # Working directory
    def model -> MODEL                             # Current model
    def datamodule -> DATA                         # Current data module
```

### `create_engine()` Factory

**File:** `lib/src/otx/engine/__init__.py`

```python
from otx.engine import create_engine

engine = create_engine(model=model_or_path, data=data_or_path, **kwargs)
# Tries OTXEngine first, then OVEngine, then any registered subclasses
```

### `OTXEngine` (Native PyTorch Backend)

**File:** `lib/src/otx/backend/native/engine.py`

The main training engine. Wraps PyTorch Lightning `Trainer`.

#### Constructor

```python
engine = OTXEngine(
    model="path/to/recipe.yaml",         # or OTXModel instance
    data="path/to/dataset",              # or OTXDataModule instance
    work_dir="./otx-workspace",          # output directory
    checkpoint=None,                     # resume from checkpoint
    device=DeviceType.auto,              # auto/gpu/cpu/xpu
    num_devices=1,                       # multi-GPU count
)
```

#### Alternative Constructors

```python
# From a YAML recipe file
engine = OTXEngine.from_config(
    config_path="src/otx/recipe/detection/ssd_mobilenetv2.yaml",
    data_root="/path/to/coco_dataset",
    work_dir="./workspace",
)

# From model name + task type
engine = OTXEngine.from_model_name(
    model_name="ssd_mobilenetv2",
    task=OTXTaskType.DETECTION,
    data_root="/path/to/coco_dataset",
)
```

#### Training

```python
metrics = engine.train(
    max_epochs=200,               # Maximum training epochs
    seed=42,                      # Reproducibility seed
    deterministic=True,           # Deterministic mode
    precision="16",               # Mixed precision: "16", "32", "bf16"
    callbacks=None,               # Additional Lightning callbacks
    logger=None,                  # Custom logger(s)
    resume=False,                 # Resume from last checkpoint
    adaptive_bs="None",           # "None" | "Safe" | "Full" - adaptive batch size
    gradient_clip_val=None,       # Gradient clipping value
    check_val_every_n_epoch=1,    # Validation frequency
)
```

#### Testing

```python
metrics = engine.test(
    checkpoint="path/to/best.ckpt",    # Optional specific checkpoint
    metric=custom_metric_callable,     # Optional custom metric
)
```

#### Prediction

```python
predictions = engine.predict(
    checkpoint="path/to/best.ckpt",
    explain=False,                     # Enable saliency maps (XAI)
    explain_config=None,               # XAI configuration
)
# Returns list of OTXPredItem with bboxes, scores, labels, etc.
```

#### Export

```python
exported_path = engine.export(
    export_format=OTXExportFormatType.OPENVINO,  # OPENVINO or ONNX
    export_precision=OTXPrecisionType.FP32,      # FP32 or FP16
    explain=False,                               # Include XAI in export
)
```

#### Benchmarking

```python
results = engine.benchmark(batch_size=1, n_iters=100)
# Returns: latency, throughput, param count, FLOPs
```

### `OVEngine` (OpenVINO Backend)

**File:** `lib/src/otx/backend/openvino/engine.py`

Inference-only engine for exported OpenVINO IR models.

```python
ov_engine = OVEngine(
    model="path/to/exported_model.xml",
    data=datamodule,
    work_dir="./ov-workspace",
)
metrics = ov_engine.test()
predictions = ov_engine.predict()
# NOTE: ov_engine.train() is NOT supported
```

---

## 6. Data System

### `OTXDataModule`

**File:** `lib/src/otx/data/module.py`

Extends `LightningDataModule`. Handles dataset loading, splitting, and transforms.

```python
from otx.data.module import OTXDataModule

datamodule = OTXDataModule(
    task=OTXTaskType.DETECTION,
    data_format="coco_instances",         # Dataset format
    data_root="/path/to/dataset",         # Root directory
    train_subset=train_config,            # SubsetConfig
    val_subset=val_config,                # SubsetConfig
    test_subset=test_config,              # SubsetConfig
    tile_config=TileConfig(),             # Tiling for large images
    image_color_channel=ImageColorChannel.RGB,
    input_size=(800, 992),                # or "auto"
)
```

### Supported Data Formats

| Format                         | Task                    | Directory Structure        |
| ------------------------------ | ----------------------- | -------------------------- |
| `coco_instances`               | Detection, Instance Seg | COCO JSON annotations      |
| `imagenet_with_subset_dirs`    | Classification          | `train/class_name/img.jpg` |
| `voc`                          | Detection, Segmentation | Pascal VOC XML annotations |
| `common_semantic_segmentation` | Semantic Seg            | Image + mask directories   |
| `mvtec`                        | Anomaly                 | MVTec-style structure      |

### SubsetConfig

**File:** `lib/src/otx/config/data.py`

```python
@dataclass
class SubsetConfig:
    batch_size: int = 1
    subset_name: str = "train"
    transforms: list[Transform] = []
    transform_lib_type: TransformLibType = TransformLibType.TORCHVISION
    num_workers: int = 2
    sampler: SamplerConfig = None
    to_tv_image: bool = False
```

### Available Transforms

All in `lib/src/otx/data/transform_libs/torchvision.py`:

| Transform             | Description                           |
| --------------------- | ------------------------------------- |
| `Resize`              | Resize with optional bbox transform   |
| `RandomResizedCrop`   | Random crop + resize                  |
| `RandomFlip`          | Horizontal/vertical flip              |
| `RandomAffine`        | Rotation, translation, scaling, shear |
| `MinIoURandomCrop`    | IoU-aware random crop (detection)     |
| `CachedMosaic`        | Mosaic augmentation (YOLOX)           |
| `CachedMixUp`         | MixUp augmentation                    |
| `RandomGaussianBlur`  | Gaussian blur                         |
| `RandomGaussianNoise` | Gaussian noise injection              |
| `YOLOXHSVRandomAug`   | HSV color jitter                      |

### Data Entities

**File:** `lib/src/otx/data/entity/`

```python
OTXDataItem    # Single data sample (image + annotations)
OTXDataBatch   # Batch of data items (model input)
OTXPredBatch   # Batch of predictions (model output)
OTXPredItem    # Single prediction
ImageInfo      # Image metadata (shape, padding, scale, normalization)
```

### Dataset Classes

**File:** `lib/src/otx/data/dataset/`

Each task has a dedicated dataset class:

- `OTXMulticlassClsDataset`, `OTXMultilabelClsDataset`, `OTXHlabelClsDataset`
- `OTXDetectionDataset`
- `OTXInstanceSegDataset`
- `OTXSegmentationDataset`
- `OTXKeypointDetectionDataset`
- `OTXAnomalyDataset`
- `OTXTileDatasetFactory` - wraps any dataset for large-image tiling

---

## 7. Models

### Base Model Hierarchy

```
LightningModule
└── OTXModel (lib/src/otx/backend/native/models/base.py)
    ├── OTXDetectionModel (detection/base.py)
    │   ├── SingleStageDetector → SSD, ATSS, YOLOX, RTMDet
    │   └── DETR → RTDETR, DFine, DEIM
    ├── OTXMulticlassClsModel
    ├── OTXMultilabelClsModel
    ├── OTXHlabelClsModel
    ├── OTXInstanceSegModel → MaskRCNN, RTMDetInst
    ├── OTXSegmentationModel → DinoV2Seg, LiteHRNet, SegNext
    ├── OTXKeypointDetectionModel → RTMPose
    └── OTXAnomalyModel → Padim, Stfpm, Uflow
```

### `OTXModel` Base Class

```python
class OTXModel(LightningModule):
    def __init__(
        self,
        label_info: LabelInfoTypes | int,       # Number/info of classes
        data_input_params: DataInputParams,      # input_size, mean, std
        model_name: str = "OTXModel",
        optimizer: OptimizerCallable = ...,      # SGD, AdamW, etc.
        scheduler: LRSchedulerCallable = ...,    # LR schedule
        metric: MetricCallable = ...,            # Evaluation metric
        torch_compile: bool = False,             # torch.compile optimization
        tile_config: TileConfig = ...,           # Large image tiling
    )
```

### DataInputParams

```python
@dataclass
class DataInputParams:
    input_size: tuple[int, int]                  # (H, W) image size
    mean: tuple[float, float, float]             # Normalization mean
    std: tuple[float, float, float]              # Normalization std
```

---

## 8. Recipes (YAML Configs)

**Directory:** `lib/src/otx/recipe/`

Recipes are YAML files that fully define a training configuration: model architecture, optimizer, scheduler, data augmentations, callbacks, and training overrides.

### Recipe Structure

```yaml
# lib/src/otx/recipe/detection/ssd_mobilenetv2.yaml

task: DETECTION # OTXTaskType enum

model:
  class_path: otx.backend.native.models.detection.ssd.SSD
  init_args:
    model_name: ssd_mobilenetv2
    label_info: 80 # COCO classes (overridden by dataset)
    optimizer:
      class_path: torch.optim.SGD
      init_args:
        lr: 0.01
        momentum: 0.9
        weight_decay: 0.0001
    scheduler:
      class_path: otx.backend.native.schedulers.LinearWarmupSchedulerCallable
      init_args:
        num_warmup_steps: 0
        main_scheduler_callable:
          class_path: lightning.pytorch.cli.ReduceLROnPlateau
          init_args:
            mode: max
            factor: 0.1
            patience: 4
            monitor: val/map_50

engine:
  device: auto

data: ../_base_/data/detection.yaml # Inherits base data config

callback_monitor: val/map_50

callbacks:
  - class_path: otx.backend.native.callbacks.adaptive_early_stopping.EarlyStoppingWithWarmup
    init_args:
      patience: 10
      monitor: val/map_50
      warmup_epochs: 3
  - class_path: lightning.pytorch.callbacks.ModelCheckpoint
    init_args:
      monitor: val/map_50
      mode: max
      save_top_k: 1

overrides:
  data:
    input_size: [864, 864]
    train_subset:
      batch_size: 8
      transforms: [...] # Custom augmentation pipeline
```

### Base Data Configs

Located in `lib/src/otx/recipe/_base_/data/`:

| Config                       | Default Format                 | Default Input Size |
| ---------------------------- | ------------------------------ | ------------------ |
| `classification.yaml`        | `imagenet_with_subset_dirs`    | 224x224            |
| `detection.yaml`             | `coco_instances`               | 800x992            |
| `instance_segmentation.yaml` | `coco_instances`               | varies             |
| `semantic_segmentation.yaml` | `common_semantic_segmentation` | varies             |
| `anomaly.yaml`               | `mvtec`                        | varies             |
| `keypoint_detection.yaml`    | `coco_instances`               | varies             |

### YAML Custom Resolvers

```yaml
# Convert string to int tuple
input_size: ${as_int_tuple:500,500}

# Convert string to torch dtype
dtype: ${as_torch_dtype:torch.float32}

# Reference input_size defined elsewhere
scale: $(input_size)
```

---

## 9. CLI Reference

**Entry point:** `otx` (defined in `lib/src/otx/cli/cli.py`)

### Commands

```bash
# List all available model recipes
otx find

# Train a model
otx train --config <recipe.yaml> --data_root <path> [--max_epochs N] [--seed 42]

# Evaluate a model
otx test --config <recipe.yaml> --data_root <path> --checkpoint <path.ckpt>

# Run inference
otx predict --config <recipe.yaml> --data_root <path> --checkpoint <path.ckpt>

# Export to OpenVINO/ONNX
otx export --config <recipe.yaml> --checkpoint <path.ckpt> \
           --export_format openvino --export_precision FP16

# Generate saliency maps (XAI)
otx explain --config <recipe.yaml> --checkpoint <path.ckpt>

# Benchmark inference performance
otx benchmark --config <recipe.yaml> --checkpoint <path.ckpt> --n_iters 100
```

### Example: Train SSD on Custom COCO Dataset

```bash
otx train \
  --config lib/src/otx/recipe/detection/ssd_mobilenetv2.yaml \
  --data_root /data/my_coco_dataset \
  --work_dir ./workspace/ssd_experiment \
  --max_epochs 100 \
  --seed 42
```

---

## 10. Python API Reference

### Minimal Training Example

```python
from otx.engine import create_engine

# Auto-configure from recipe + data path
engine = create_engine(
    model="src/otx/recipe/detection/ssd_mobilenetv2.yaml",
    data="/path/to/coco_dataset",
    work_dir="./workspace",
)

# Train
metrics = engine.train(max_epochs=50, seed=42)
print(f"Training mAP@50: {metrics}")

# Evaluate
test_metrics = engine.test()
print(f"Test mAP@50: {test_metrics}")

# Predict on test set
predictions = engine.predict()
for pred in predictions:
    # pred contains bboxes, scores, labels
    print(pred)

# Export for deployment
exported_path = engine.export(export_format="OPENVINO")
print(f"Exported to: {exported_path}")
```

### Full Control Example

```python
from otx.backend.native.engine import OTXEngine
from otx.backend.native.models.detection.ssd import SSD
from otx.data.module import OTXDataModule
from otx.types.task import OTXTaskType

# Instantiate model directly
model = SSD(
    label_info=20,                              # Number of classes
    data_input_params={
        "input_size": (864, 864),
        "mean": (0.0, 0.0, 0.0),
        "std": (255.0, 255.0, 255.0),
    },
    model_name="ssd_mobilenetv2",
)

# Instantiate data module directly
datamodule = OTXDataModule(
    task=OTXTaskType.DETECTION,
    data_format="coco_instances",
    data_root="/path/to/dataset",
    train_subset=...,                           # SubsetConfig
    val_subset=...,
    test_subset=...,
    input_size=(864, 864),
)

# Create engine
engine = OTXEngine(
    model=model,
    data=datamodule,
    work_dir="./workspace",
    device="auto",
)

# Train with full control
engine.train(
    max_epochs=100,
    precision="16",
    gradient_clip_val=35.0,
    adaptive_bs="Safe",
)
```

### Using `from_config` and `from_model_name`

```python
from otx.backend.native.engine import OTXEngine
from otx.types.task import OTXTaskType

# Method A: From config file
engine = OTXEngine.from_config(
    config_path="src/otx/recipe/detection/yolox_tiny.yaml",
    data_root="/data/my_dataset",
)

# Method B: From model name
engine = OTXEngine.from_model_name(
    model_name="yolox_tiny",
    task=OTXTaskType.DETECTION,
    data_root="/data/my_dataset",
)
```

---

## 11. Export & Deployment

### Export Formats

| Format      | Extension       | Use Case                  |
| ----------- | --------------- | ------------------------- |
| OpenVINO IR | `.xml` + `.bin` | Optimized Intel inference |
| ONNX        | `.onnx`         | Cross-platform inference  |

### Export Precision

| Precision | Description                      |
| --------- | -------------------------------- |
| `FP32`    | Full precision (default)         |
| `FP16`    | Half precision (smaller, faster) |

### Export Workflow

```python
# Export to OpenVINO
ov_path = engine.export(
    export_format=OTXExportFormatType.OPENVINO,
    export_precision=OTXPrecisionType.FP16,
)

# Use exported model for inference via OVEngine
from otx.engine import create_engine
ov_engine = create_engine(model=ov_path, data=engine.datamodule)
ov_predictions = ov_engine.predict()
```

### Export Parameters

Exported models include metadata for post-processing:

- `model_type`, `model_name`, `task_type`
- `label_info` (class names and IDs)
- `confidence_threshold`, `iou_threshold` (for detection)
- `tile_config` (for tiled models)

---

## 12. Backend System

### Native Backend (`lib/src/otx/backend/native/`)

The primary backend for training. Built on PyTorch + Lightning.

```
native/
├── engine.py                 # OTXEngine - main training orchestrator
├── models/                   # All model implementations
│   ├── base.py               # OTXModel base class
│   ├── classification/       # Classification models
│   ├── detection/            # Detection models (ATSS, SSD, YOLOX, RTMDet, RTDETR, DFine, DEIM)
│   ├── segmentation/         # Segmentation models (DinoV2, LiteHRNet, SegNext)
│   ├── instance_segmentation/ # MaskRCNN, RTMDetInst
│   ├── keypoint_detection/   # RTMPose
│   ├── anomaly/              # Padim, Stfpm, Uflow
│   └── common/               # Shared components (backbones, necks, losses, utils)
├── callbacks/                # Training callbacks
│   ├── adaptive_early_stopping.py   # Early stop with warmup
│   ├── adaptive_train_scheduling.py # Adaptive scheduling
│   ├── batchsize_finder.py          # Auto batch size discovery
│   ├── gpu_mem_monitor.py           # GPU memory monitoring
│   ├── iteration_timer.py           # Training speed tracking
│   └── aug_scheduler.py             # Augmentation scheduling
├── exporter/                 # Model export (ONNX, OpenVINO)
│   ├── base.py               # OTXModelExporter base
│   └── native.py             # OTXNativeModelExporter
├── tools/
│   ├── adaptive_bs/          # Adaptive batch size utility
│   ├── explain/              # XAI / saliency map generation
│   └── tile_merge.py         # Tile merging for large-image predictions
├── optimizers/               # Custom optimizer implementations
├── schedulers/               # LR scheduler wrappers (LinearWarmup, etc.)
├── lightning/                # Custom Lightning accelerators (XPU support)
├── cli/                      # Native backend CLI extensions
└── utils/                    # Cache, helpers
```

### OpenVINO Backend (`lib/src/otx/backend/openvino/`)

Inference-only backend using OpenVINO Runtime.

```
openvino/
├── engine.py                 # OVEngine - inference orchestrator
└── models/                   # OpenVINO model wrappers
    ├── base.py               # OVModel base class
    ├── detection.py           # OVDetectionModel
    ├── multiclass_classification.py
    ├── multilabel_classification.py
    ├── hlabel_classification.py
    ├── segmentation.py
    ├── instance_segmentation.py
    ├── keypoint_detection.py
    ├── anomaly.py
    └── utils.py
```

---

## 13. Folder-by-Folder Breakdown

### `lib/src/otx/__init__.py`

Package entry point. Defines `__version__ = "2.6.0"`. Sets environment variables and displays OTX logo.

### `lib/src/otx/engine/`

- `engine.py` - Abstract `Engine` base class defining `train/test/predict/export` contract
- `__init__.py` - `create_engine()` factory that auto-selects OTXEngine or OVEngine

### `lib/src/otx/cli/`

- `cli.py` - `OTXCLI` class. Entry point for `otx` command. Subcommands: train, test, predict, export, explain, find, benchmark

### `lib/src/otx/config/`

- `data.py` - `SubsetConfig`, `TileConfig`, `SamplerConfig` dataclasses
- `device.py` - `DeviceConfig` for device-specific settings
- `explain.py` - `ExplainConfig` for XAI/saliency map settings

### `lib/src/otx/data/`

- `module.py` - `OTXDataModule` (LightningDataModule extension)
- `dataset/` - Task-specific dataset classes, tile dataset factory
- `transform_libs/` - Transform implementations (torchvision-based)
- `samplers/` - `BalancedSampler`, standard samplers
- `factory.py` - `OTXDatasetFactory` for creating datasets from configs
- `entity/` - Data entity definitions (`OTXDataItem`, `OTXPredItem`, `ImageInfo`)
- `utils/` - Helpers for input size adaptation, filtering, worker count

### `lib/src/otx/metrics/`

- `fmeasure.py` - `MeanAveragePrecisionFMeasureCallable` (detection mAP)
- Various metric callables per task type (accuracy, dice, PCK, anomaly)
- `__init__.py` - `MetricCallable` type alias, `NullMetricCallable`

### `lib/src/otx/models/`

Convenience re-exports. Allows `from otx.models import ...` as shorthand.

### `lib/src/otx/recipe/`

114 YAML recipe files organized by task:

- `_base_/data/` - Base data configurations (inherited by model recipes)
- `classification/` - `multi_class_cls/`, `multi_label_cls/`, `h_label_cls/`
- `detection/` - 28 recipes (14 models x regular/tile variants)
- `instance_segmentation/` - MaskRCNN variants, RTMDet-Inst
- `semantic_segmentation/` - DinoV2, LiteHRNet, SegNext
- `anomaly/`, `anomaly_classification/`, `anomaly_detection/`, `anomaly_segmentation/`
- `keypoint_detection/` - RTMPose
- `rotated_detection/` - MaskRCNN variants

### `lib/src/otx/tools/`

- `auto_configurator.py` - `AutoConfigurator` class. Auto-detects task from data directory, selects default model
- `converter/` - Format conversion utilities

### `lib/src/otx/types/`

- `task.py` - `OTXTaskType` enum
- `label.py` - `LabelInfo`, `HLabelInfo`, `SegLabelInfo`, `NullLabelInfo`, `AnomalyLabelInfo`
- `device.py` - `DeviceType` enum (auto, gpu, cpu, xpu, mps, ...)
- `export.py` - `OTXExportFormatType`, `OTXPrecisionType`, `TaskLevelExportParameters`
- `precision.py` - `OTXPrecisionType` enum
- `image.py` - `ImageColorChannel` enum (RGB, BGR)
- `types.py` - Type aliases: `MODEL`, `DATA`, `METRICS`, `ANNOTATIONS`

### `lib/src/otx/utils/`

- `device.py` - Device detection utilities (`get_available_device`, `is_xpu_available`)
- `utils.py` - General helpers (`measure_flops`, etc.)

### `lib/src/otx/backend/native/models/detection/`

Full detection model implementations:

| File           | Model                          | Type                            |
| -------------- | ------------------------------ | ------------------------------- |
| `ssd.py`       | SSD MobileNetV2                | Single-stage, anchor-based      |
| `atss.py`      | ATSS (MobileNetV2, ResNeXt101) | Single-stage, anchor-based      |
| `yolox.py`     | YOLOX (tiny, s, l, x)          | Single-stage, anchor-free       |
| `rtmdet.py`    | RTMDet Tiny                    | Single-stage, anchor-free       |
| `rtdetr.py`    | RT-DETR (18, 50, 101)          | Transformer, no NMS             |
| `d_fine.py`    | D-FINE (n, s, m, l, x)         | Transformer, no NMS             |
| `deim.py`      | DEIM-DFine (n, s, m, l, x)     | Transformer, half training time |
| `base.py`      | `OTXDetectionModel`            | Abstract base for all detectors |
| `detectors.py` | `SingleStageDetector`, `DETR`  | Detector wrappers               |

Supporting submodules:

- `backbones/` - CSPDarknet, CSPNeXt, HGNetv2, PResNet
- `heads/` - ATSSHead, SSDHead, YOLOXHead, RTMDetSepBNHead, Transformer decoders
- `necks/` - FPN, YOLOXPAFPN, CSPNeXtPAFPN, HybridEncoder
- `losses/` - Focal, GIoU, SmoothL1, VFL, quality-focal, DDF
- `utils/` - Anchor generators, bbox coders, assigners (ATSS, SimOTA, etc.)

### `backend/` (Geti-Tune REST API)

FastAPI-based web backend:

- `app/api/` - REST endpoints
- `app/services/` - Business logic
- `app/repositories/` - Data access (SQLAlchemy)
- `app/entities/` - Database models
- `app/workers/` - Background task processing
- `app/webrtc/` - WebRTC streaming

### `ui/` (Web Frontend)

React + TypeScript web interface:

- `src/routes/` - Page routes
- `src/features/` - Feature modules
- `src/components/` - React components
- `src/api/` - API client

### `docker/`

- `Dockerfile` - Multi-stage build for Geti-Tune (nginx + FastAPI + React)
- `Dockerfile.cuda` - PyTorch CUDA training image
- `docker-compose.yaml` - Geti-Tune service + optional MQTT

### `docs/`

Sphinx documentation:

- `source/guide/get_started/` - API tutorial, CLI commands
- `source/guide/tutorials/` - Task-specific tutorials
- `source/guide/explanation/` - Advanced features docs

---

## 14. Supported Tasks & Models

### Object Detection

| Model        | Recipe                  | Input Size | Backbone    | Type         |
| ------------ | ----------------------- | ---------- | ----------- | ------------ |
| SSD          | `ssd_mobilenetv2.yaml`  | 864x864    | MobileNetV2 | Single-stage |
| ATSS         | `atss_mobilenetv2.yaml` | 800x992    | MobileNetV2 | Single-stage |
| ATSS         | `atss_resnext101.yaml`  | 800x992    | ResNeXt101  | Single-stage |
| YOLOX-Tiny   | `yolox_tiny.yaml`       | 640x640    | CSPDarknet  | Anchor-free  |
| YOLOX-S      | `yolox_s.yaml`          | 640x640    | CSPDarknet  | Anchor-free  |
| YOLOX-L      | `yolox_l.yaml`          | 640x640    | CSPDarknet  | Anchor-free  |
| YOLOX-X      | `yolox_x.yaml`          | 640x640    | CSPDarknet  | Anchor-free  |
| RTMDet-Tiny  | `rtmdet_tiny.yaml`      | 640x640    | CSPNeXt     | Anchor-free  |
| RT-DETR-18   | `rtdetr_18.yaml`        | 640x640    | PResNet18   | Transformer  |
| RT-DETR-50   | `rtdetr_50.yaml`        | 640x640    | PResNet50   | Transformer  |
| RT-DETR-101  | `rtdetr_101.yaml`       | 640x640    | PResNet101  | Transformer  |
| D-FINE-X     | `dfine_x.yaml`          | 640x640    | HGNetv2     | Transformer  |
| DEIM-DFine-L | `deim_dfine_l.yaml`     | 640x640    | HGNetv2     | Transformer  |
| DEIM-DFine-M | `deim_dfine_m.yaml`     | 640x640    | HGNetv2     | Transformer  |
| DEIM-DFine-X | `deim_dfine_x.yaml`     | 640x640    | HGNetv2     | Transformer  |

All detection models also have `_tile` variants for large images.

### Classification

| Model                  | Variants                          |
| ---------------------- | --------------------------------- |
| EfficientNet-B0        | multi-class, multi-label, h-label |
| EfficientNet-V2        | multi-class, multi-label, h-label |
| MobileNetV3-Large      | multi-class, multi-label, h-label |
| MobileNetV3-Small (TV) | multi-class, multi-label, h-label |
| MobileNetV4            | multi-class                       |
| DeiT-Tiny              | multi-class, multi-label, h-label |
| DINOv2                 | multi-class, multi-label, h-label |
| EfficientNet-B3 (TV)   | multi-class, multi-label, h-label |
| EfficientNet-V2-L (TV) | multi-class, multi-label, h-label |

### Instance Segmentation

| Model                         | Recipe                          |
| ----------------------------- | ------------------------------- |
| Mask R-CNN (R50)              | `maskrcnn_r50.yaml`             |
| Mask R-CNN (R50 TV)           | `maskrcnn_r50_tv.yaml`          |
| Mask R-CNN (EfficientNet-B2B) | `maskrcnn_efficientnetb2b.yaml` |
| Mask R-CNN (Swin-T)           | `maskrcnn_swint.yaml`           |
| RTMDet-Inst Tiny              | `rtmdet_inst_tiny.yaml`         |

### Semantic Segmentation

| Model        | Recipe              |
| ------------ | ------------------- |
| DINOv2 Seg   | `dino_v2.yaml`      |
| LiteHRNet-18 | `litehrnet_18.yaml` |
| LiteHRNet-S  | `litehrnet_s.yaml`  |
| LiteHRNet-X  | `litehrnet_x.yaml`  |
| SegNext-B    | `segnext_b.yaml`    |
| SegNext-S    | `segnext_s.yaml`    |
| SegNext-T    | `segnext_t.yaml`    |

### Anomaly Detection

| Model  | Variants                                                                 |
| ------ | ------------------------------------------------------------------------ |
| PaDiM  | anomaly, anomaly_classification, anomaly_detection, anomaly_segmentation |
| STFPM  | anomaly, anomaly_classification, anomaly_detection, anomaly_segmentation |
| U-Flow | anomaly, anomaly_classification                                          |

### Keypoint Detection

| Model        | Recipe              |
| ------------ | ------------------- |
| RTMPose-Tiny | `rtmpose_tiny.yaml` |

### Rotated Detection

| Model                         | Recipe                          |
| ----------------------------- | ------------------------------- |
| Mask R-CNN (R50)              | `maskrcnn_r50.yaml`             |
| Mask R-CNN (R50 V2)           | `maskrcnn_r50_v2.yaml`          |
| Mask R-CNN (EfficientNet-B2B) | `maskrcnn_efficientnetb2b.yaml` |

---

## 15. Configuration Deep-Dive

### Auto-Configuration

**File:** `lib/src/otx/tools/auto_configurator.py`

OTX can auto-detect the task type and select a default model:

```python
DEFAULT_CONFIG_PER_TASK = {
    MULTI_CLASS_CLS:      "efficientnet_b0.yaml",
    MULTI_LABEL_CLS:      "efficientnet_b0.yaml",
    H_LABEL_CLS:          "efficientnet_b0.yaml",
    DETECTION:            "atss_mobilenetv2.yaml",
    ROTATED_DETECTION:    "maskrcnn_r50.yaml",
    SEMANTIC_SEGMENTATION: "litehrnet_18.yaml",
    INSTANCE_SEGMENTATION: "maskrcnn_r50.yaml",
    ANOMALY:              "padim.yaml",
    KEYPOINT_DETECTION:   "rtmpose_tiny.yaml",
}
```

When you pass just a `data_root` path, `AutoConfigurator`:

1. Uses Datumaro to detect the dataset format
2. Infers the task type from the format
3. Selects the default recipe
4. Creates the `OTXDataModule` with appropriate transforms

### TileConfig

For processing large images (e.g., satellite imagery):

```python
@dataclass
class TileConfig:
    enable_tiler: bool = False
    tile_size: tuple[int, int] = (400, 400)
    overlap: float = 0.2
    max_num_instances: int = 1500
    object_tile_ratio: float = 0.03
    sampling_ratio: float = 1.0
    with_full_img: bool = False
```

### Callback System

OTX uses Lightning callbacks extensively:

| Callback                        | Purpose                                                 |
| ------------------------------- | ------------------------------------------------------- |
| `EarlyStoppingWithWarmup`       | Stop training when metric plateaus (with warmup period) |
| `AdaptiveTrainScheduling`       | Dynamically adjust training schedule                    |
| `AugmentationSchedulerCallback` | Phase in/out augmentations during training              |
| `GPUMemMonitor`                 | Track GPU memory usage                                  |
| `IterationTimer`                | Track training speed                                    |
| `ModelCheckpoint`               | Save best/last model checkpoints                        |
| `LearningRateMonitor`           | Log learning rate changes                               |
| `RichProgressBar`               | Pretty training progress display                        |

---

## 16. Metrics & Evaluation

**Directory:** `lib/src/otx/metrics/`

| Task            | Default Metric                         | Description                     |
| --------------- | -------------------------------------- | ------------------------------- |
| Detection       | `MeanAveragePrecisionFMeasureCallable` | mAP@50, mAP@75, mAP@50:95       |
| Classification  | `MultiClassClsMetricCallable`          | Accuracy                        |
| Multi-label Cls | `MultiLabelClsMetricCallable`          | Per-label accuracy              |
| Segmentation    | `DiceMetric` / `FMeasure`              | Dice coefficient                |
| Instance Seg    | `MeanAveragePrecisionFMeasureCallable` | Mask mAP                        |
| Keypoint        | `PCKMetric`                            | Percentage of Correct Keypoints |
| Anomaly         | `AnomalyMetric`                        | AUROC, F1                       |

### Custom Metrics

```python
from otx.metrics import MetricCallable

# MetricCallable = Callable[[MetricInput], dict[str, float]]
# Override via engine.train(metric=my_custom_metric)
```

---

## 17. Advanced Features

### Adaptive Batch Size

Automatically finds the largest batch size that fits in GPU memory:

```python
engine.train(adaptive_bs="Safe")   # Conservative estimate
engine.train(adaptive_bs="Full")   # Aggressive (may OOM, then retries lower)
```

### Large Image Tiling

For high-resolution images, OTX splits images into tiles with overlap:

```python
# Use a tile recipe variant
engine = OTXEngine.from_config("recipe/detection/ssd_mobilenetv2_tile.yaml", ...)
```

Or configure manually:

```python
tile_config = TileConfig(
    enable_tiler=True,
    tile_size=(400, 400),
    overlap=0.2,
)
```

### Explainability (XAI)

Generate saliency maps to understand model predictions:

```python
predictions = engine.predict(explain=True, explain_config=ExplainConfig())
# Each prediction includes saliency_map and feature_vector
```

### Mixed Precision Training

```python
engine.train(precision="16")     # FP16 mixed precision
engine.train(precision="bf16")   # BFloat16 (Ampere+ GPUs)
engine.train(precision="32")     # Full precision
```

### Multi-GPU Training

```python
engine = OTXEngine(model=..., data=..., num_devices=4)
engine.train()
```

### Torch Compile

```python
model = SSD(..., torch_compile=True)  # Uses torch.compile for optimization
```

---

## 18. Web UI & REST API (Geti-Tune)

OTX includes a full web application for model training and management.

### Backend (FastAPI)

**Directory:** `backend/app/`

- Tech: FastAPI 0.115, SQLAlchemy 2.0, Uvicorn
- Features: REST API, background workers, WebRTC streaming, MQTT integration
- Port: 80 (via nginx reverse proxy)

### Frontend (React)

**Directory:** `ui/src/`

- Tech: React + TypeScript, Rsbuild bundler
- Features: Model training UI, visualization, management

### Docker Deployment

```bash
cd docker
docker-compose up      # Starts Geti-Tune on port 80

# With MQTT
docker-compose --profile mqtt up
```

---

## 19. CI/CD & Development

### GitHub Actions Workflows

| Workflow                 | Trigger | Purpose                                |
| ------------------------ | ------- | -------------------------------------- |
| `pre_merge.yaml`         | PR      | Linting, unit tests, integration tests |
| `publish.yaml`           | Release | PyPI package publishing                |
| `build.yaml`             | Push    | Build verification                     |
| `docs.yaml`              | Push    | Documentation building                 |
| `daily.yaml`             | Cron    | Daily regression tests                 |
| `perf_benchmark_v2.yaml` | Manual  | Performance benchmarking               |
| `codeql.yaml`            | Push/PR | Static security analysis               |
| `security-scan.yaml`     | Push    | Vulnerability scanning                 |

### Development Setup

```bash
cd lib
pip install -e ".[dev]"
tox -vv -e pre-commit    # Code quality (ruff, mypy)
pytest tests/unit         # Unit tests
pytest tests/integration  # Integration tests
```

### Code Quality Tools

- **Ruff:** Linting + formatting (line-length: 120)
- **mypy:** Static type checking (Python 3.12 target)
- **pytest:** Testing framework with coverage
- **pre-commit:** Git hooks for automated checks

---

## 20. Glossary

| Term           | Definition                                                    |
| -------------- | ------------------------------------------------------------- |
| **OTX**        | OpenVINO Training Extensions                                  |
| **Recipe**     | YAML config file defining a complete training setup           |
| **Engine**     | Orchestrator for train/test/predict/export workflows          |
| **OTXEngine**  | Native PyTorch Lightning-based engine (training)              |
| **OVEngine**   | OpenVINO-based engine (inference only)                        |
| **OTXModel**   | Base class for all trainable models (extends LightningModule) |
| **OVModel**    | Base class for OpenVINO inference model wrappers              |
| **DataModule** | Lightning data handler (dataset + transforms + loaders)       |
| **Datumaro**   | Intel's dataset management library (handles format parsing)   |
| **NNCF**       | Neural Network Compression Framework (quantization, pruning)  |
| **XAI**        | Explainable AI (saliency maps, feature vectors)               |
| **Tile**       | Sub-region of a large image, processed independently          |
| **FPN**        | Feature Pyramid Network (multi-scale feature extraction)      |
| **NMS**        | Non-Maximum Suppression (removes duplicate detections)        |
| **DETR**       | DEtection TRansformer (end-to-end detector, no NMS needed)    |
| **Geti-Tune**  | Web-based training interface (backend + UI)                   |

---

## 21. Local Setup & Offline Models

This section documents the exact setup used on this machine (Mac Pro M5, macOS ARM64).

### 21.1 Environment Setup

```bash
# Prerequisites: uv, Rust (for datumaro build)
# Install Rust if needed:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Create venv
cd /path/to/otx
uv venv .venv --python 3.11
source .venv/bin/activate

# Install PyTorch (CPU/MPS for Mac - no CUDA on Apple Silicon)
uv pip install torch==2.8.0 torchvision==0.23.0

# Install OTX in editable mode (skip decord - no ARM64 wheel)
cd lib
uv pip install -e "." --no-deps

# Install deps manually (excluding decord which has no macOS ARM64 wheel)
uv pip install \
  "datumaro==1.10.0" \
  "omegaconf==2.3.0" \
  "rich==14.0.0" \
  "jsonargparse==4.35.0" \
  "ftfy==6.3.1" \
  "regex==2024.11.6" \
  "importlib_resources==6.5.2" \
  "docstring_parser==0.16" \
  "rich_argparse==1.7.0" \
  "einops==0.8.1" \
  "typeguard>=4.3,<4.5" \
  "lightning==2.4.0" \
  "torchmetrics==1.6.0" \
  "pytorchcv==0.0.67" \
  "timm==1.0.3" \
  "openvino==2025.2" \
  "openvino-model-api==0.3.0.2" \
  "onnx==1.17.0" \
  "onnxconverter-common==1.16.0" \
  "nncf==2.17.0" \
  "anomalib[core]==1.1.3"

# Install extra tools for demo/tracking work
uv pip install \
  gradio \
  opencv-python-headless \
  jupyter \
  ipywidgets \
  lap \
  filterpy \
  scipy
```

### 21.2 Verified Package Versions

| Package     | Version          | Status |
| ----------- | ---------------- | ------ |
| Python      | 3.11.14          | OK     |
| PyTorch     | 2.8.0            | OK     |
| TorchVision | 0.23.0           | OK     |
| Lightning   | 2.4.0            | OK     |
| OTX         | 2.6.0 (editable) | OK     |
| Gradio      | 6.8.0            | OK     |
| OpenCV      | 4.13.0           | OK     |
| timm        | 1.0.3            | OK     |
| MPS backend | Available        | OK     |

### 21.3 macOS ARM64 Caveats

- **`decord`** (video decoding library): No macOS ARM64 wheel. Skipped. Use OpenCV `cv2.VideoCapture` for video reading instead.
- **`datumaro`**: Requires Rust compiler to build from source on ARM64. Install Rust first via `rustup`.
- **`numpy` ≥ 2.0 breaks `imgaug`**: The `anomalib` dependency pulls in `imgaug`, which uses the removed `np.sctypes`. Fix: `uv pip install "numpy<2.0"` (pins to 1.26.x).
- **`setuptools` ≥ 81 removes `pkg_resources`**: The `anomalib` CLIP module imports `pkg_resources`. Fix: `uv pip install "setuptools<81"`.
- **OpenVINO runtime warning**: `openvino.runtime` deprecation warning is cosmetic, safe to ignore.
- **AVF class conflict**: Harmless warning about duplicate `AVFFrameReceiver` between OpenCV and PyAV dylibs.

### 21.4 Pre-Downloaded Models (Offline Use)

All pretrained detection weights are stored in `./models/` and symlinked to PyTorch's cache directory for transparent offline loading.

#### Downloaded Models

| Model       | Local Path               | Size  | Pretrained On     |
| ----------- | ------------------------ | ----- | ----------------- |
| YOLOX-Tiny  | `models/yolox_tiny.pth`  | 19 MB | COCO              |
| RTMDet-Tiny | `models/rtmdet_tiny.pth` | 19 MB | COCO              |
| RT-DETR-18  | `models/rtdetr_18.pth`   | 77 MB | COCO + Objects365 |

#### How Pretrained Weights Work in OTX

Every model class has a `pretrained_weights` class variable mapping model names to URLs:

```python
# Example from lib/src/otx/backend/native/models/detection/yolox.py
class YOLOX(OTXDetectionModel):
    pretrained_weights: ClassVar[dict[str, str]] = {
        "yolox_tiny": "https://storage.openvinotoolkit.org/.../yolox_tiny_8x8.pth",
        "yolox_s": "https://download.openmmlab.com/.../yolox_s_8x8_300e_coco...pth",
        ...
    }
```

During `_create_model()`, the model calls `load_checkpoint(model, self.pretrained_weights[self.model_name])`.

The `load_checkpoint` function (`lib/src/otx/backend/native/models/utils/utils.py:67`) checks:

```python
def load_checkpoint(model, checkpoint, ...):
    if Path(checkpoint).exists():
        # LOCAL FILE: load from disk directly
        torch.load(checkpoint, map_location)
    else:
        # URL: download via torch.utils.model_zoo.load_url()
        # Caches to ~/.cache/torch/hub/checkpoints/<filename>
        load_from_http(checkpoint, map_location)
```

PyTorch's `load_url` caches by **URL basename**. So the cache filename must match exactly.

#### Symlink Setup (For Offline Use)

```bash
mkdir -p ~/.cache/torch/hub/checkpoints/

# YOLOX-Tiny: URL basename is "yolox_tiny_8x8.pth"
ln -sf /Users/omkar/otx/models/yolox_tiny.pth \
       ~/.cache/torch/hub/checkpoints/yolox_tiny_8x8.pth

# RTMDet-Tiny: URL basename is "rtmdet_tiny.pth"
ln -sf /Users/omkar/otx/models/rtmdet_tiny.pth \
       ~/.cache/torch/hub/checkpoints/rtmdet_tiny.pth

# RT-DETR-18: URL basename is "rtdetr_r18vd_5x_coco_objects365_from_paddle.pth"
ln -sf /Users/omkar/otx/models/rtdetr_18.pth \
       ~/.cache/torch/hub/checkpoints/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth
```

With these symlinks, OTX will find the weights locally without any internet connection. No code changes needed.

#### All Available Pretrained Weight URLs

For downloading additional models:

```bash
# SSD MobileNetV2
curl -L -o models/ssd_mobilenetv2.pth \
  "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-992x736.pth"

# ATSS MobileNetV2
curl -L -o models/atss_mobilenetv2.pth \
  "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-atss.pth"

# ATSS ResNeXt101
curl -L -o models/atss_resnext101.pth \
  "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/resnext101_atss_070623.pth"

# YOLOX-S/L/X
curl -L -o models/yolox_s.pth \
  "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
curl -L -o models/yolox_l.pth \
  "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"
curl -L -o models/yolox_x.pth \
  "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"

# RT-DETR-50 / RT-DETR-101
curl -L -o models/rtdetr_50.pth \
  "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth"
curl -L -o models/rtdetr_101.pth \
  "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth"

# D-FINE variants (n/s/m/l/x)
curl -L -o models/dfine_n.pth "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth"
curl -L -o models/dfine_s.pth "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth"
curl -L -o models/dfine_m.pth "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth"
curl -L -o models/dfine_l.pth "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth"
curl -L -o models/dfine_x.pth "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth"

# DEIM-DFine variants (n/s/m/l/x)
curl -L -o models/deim_dfine_n.pth "https://github.com/eugene123tw/DEIM/releases/download/poc/deim_dfine_hgnetv2_n_coco_160e.pth"
curl -L -o models/deim_dfine_s.pth "https://github.com/eugene123tw/DEIM/releases/download/poc/deim_dfine_hgnetv2_s_coco_120e.pth"
curl -L -o models/deim_dfine_m.pth "https://github.com/eugene123tw/DEIM/releases/download/poc/deim_dfine_hgnetv2_m_coco_90e.pth"
curl -L -o models/deim_dfine_l.pth "https://github.com/eugene123tw/DEIM/releases/download/poc/deim_dfine_hgnetv2_l_coco_50e.pth"
curl -L -o models/deim_dfine_x.pth "https://github.com/eugene123tw/DEIM/releases/download/poc/deim_dfine_hgnetv2_x_coco_50e.pth"
```

---

## 22. GSoC Project: Object Detection Tracking Support for OTX

### 22.1 Project Description (As-Is)

**Project:** Object Detection Tracking Support for OTX
**Size:** 350 hours
**Difficulty:** Medium
**Mentors:** Kirill Prokofiev, Leonardo Lai

#### Short Description

Many real-world deployment scenarios require object detection with stable identities over time, commonly referred to as Multi-Object Tracking (MOT). Typical use cases include counting items on conveyor belts, monitoring moving assets, measuring dwell time within specific zones, and enabling reliable downstream logic that depends on consistent object IDs rather than frame-by-frame detections. This project introduces Multi-Object Tracking (MOT) capabilities into the OTX library with the following goals:

- Seamlessly integrate with OTX's design philosophy and align with existing object detection workflows, including APIs, configuration files, entities, and inference patterns.
- Support OTX detection models (e.g., DETR-based, YOLO-based, and other OTX-supported detectors), treating tracking as a post-processing step applied to per-frame detection outputs.
- Deliver a working proof-of-concept, including a comparison of well-established tracking algorithms in realistic scenarios.

As part of the project, the student will implement and integrate a baseline MOT solution (such as ByteTrack or OC-SORT) and explore more advanced, SAM-2-inspired tracking concepts, including:

- **Decoupled detector-tracker design:** the detector processes each frame independently, while the tracker is responsible for maintaining consistent object identities across frames.
- **Memory-based tracking mechanisms:** maintaining a memory bank of object appearance features from previous frames to improve identity stability under occlusions and appearance variations.

#### Expected Outcomes

1. **OTX Tracking API:** a supported tracking component integrated into OTX's detection workflows, producing per-frame results with consistent `track_id` values.
2. **Unified output format:** a stable schema containing `frame_id`, `track_id`, `bbox`, `score`, `label`, plus optional track metadata (`age`, `velocity`, `state`).
3. **Configurable tracking behavior:** expose key knobs (association thresholds, track lifecycle, confidence gating, optional appearance/memory settings) consistent with OTX config patterns.
4. **PoC and comparison report:** evaluate multiple tracking baselines (e.g., ByteTrack / OC-SORT / BoT-SORT / DeepSORT-like) and a memory-bank variant inspired by SAM-2 concepts; document pros/cons and recommended defaults for OTX.
5. **Inference integration:** enable running tracking on videos / image sequences via OTX CLI in a way consistent with existing detection inference commands.
6. **Tracking evaluation support:** provide tooling to evaluate tracking quality on MOT-style annotations (e.g., IDF1, ID switches, MOTA where applicable) and document how to use it in OTX.
7. **Documentation:** user docs covering how to run tracking, tune parameters, and understand limitations.

#### Skills Required/Preferred

- **Python:** Intermediate to advanced Python programming skills. Comfortable working with scientific Python libraries (NumPy, pandas) and reading/understanding existing codebases.
- **Computer Vision fundamentals:** Understanding of object detection concepts (bounding boxes, IoU, confidence scores). Familiarity with tracking basics is helpful.
- **PyTorch basics:** Experience with PyTorch tensors, basic operations, and working with model outputs.
- **Software engineering practices:** Ability to write clean, documented code and basic unit tests. Willingness to learn from code reviews.
- **Bonus:** Prior exposure to tracking algorithms (ByteTrack, DeepSORT, SORT) or video processing.
- **Bonus:** Experience with deep learning frameworks or production ML systems.

---

### 22.2 Project Analysis: How This Fits Into OTX

This section breaks down the project requirements against the actual OTX codebase, explaining where each piece lives, what needs to be built, and how the existing architecture supports (or needs extension for) MOT.

#### 22.2.1 The Core Idea

OTX currently handles **single-frame** computer vision tasks. The detection pipeline is:

```
Input Image -> OTXDetectionModel.forward() -> List[bbox, score, label] per image
```

The tracking project adds a **temporal dimension**:

```
Video/Sequence -> Per-frame detections -> Tracker -> List[bbox, score, label, track_id] per frame
```

The tracker is **detector-agnostic** -- it sits as a post-processing layer on top of any OTX detection model's output. The detector runs independently per frame; the tracker associates detections across frames to assign stable IDs.

#### 22.2.2 What Already Exists in OTX (and What Doesn't)

**EXISTS - Detection models you can build on:**

- All 15+ detection models in `lib/src/otx/backend/native/models/detection/` (SSD, ATSS, YOLOX, RTMDet, RTDETR, DFine, DEIM)
- Each produces per-frame bounding boxes with scores and labels via `predict_step()`
- The `OTXDetectionModel` base class (`detection/base.py`) defines the standard detection output format

**EXISTS - Prediction output format:**

- `OTXPredItem` and `OTXPredBatch` in `lib/src/otx/data/entity/` define how predictions are structured
- Detection predictions include `bboxes` (Nx4 tensor), `scores` (N tensor), `labels` (N tensor)

**EXISTS - Engine predict pipeline:**

- `OTXEngine.predict()` runs inference on a dataset and returns `ANNOTATIONS = list[OTXDataItem]`
- This is the natural hook point for adding tracking as a post-processing step

**EXISTS - Recipe/config system:**

- YAML recipes in `lib/src/otx/recipe/` define full training configs
- `OmegaConf` + custom resolvers handle config parsing
- Tracking config (thresholds, lifecycle params) would follow the same pattern

**EXISTS - CLI infrastructure:**

- `otx predict` already runs detection inference
- Extending it for video input + tracking output is the CLI integration goal

**DOES NOT EXIST - Needs to be built:**

- No video/sequence data loading (OTX only handles single images currently)
- No tracker implementations (ByteTrack, OC-SORT, DeepSORT, etc.)
- No `track_id` field in prediction entities
- No MOT-format output (frame_id, track_id, bbox, score, label)
- No MOT evaluation metrics (MOTA, IDF1, ID switches)
- No tracking-specific configuration classes
- No memory bank / appearance feature extraction for advanced tracking

#### 22.2.3 Where Things Need to Go (Proposed File Layout)

Based on OTX's existing structure, here is where new tracking code would naturally fit:

```
lib/src/otx/
├── backend/native/
│   └── models/
│       └── tracking/                     # NEW: Tracker implementations
│           ├── __init__.py
│           ├── base.py                   # BaseTracker ABC
│           ├── byte_track.py             # ByteTrack implementation
│           ├── oc_sort.py                # OC-SORT implementation
│           ├── bot_sort.py               # BoT-SORT implementation
│           ├── deep_sort.py              # DeepSORT-like (with appearance)
│           └── memory_bank_tracker.py    # SAM-2-inspired memory tracker
├── config/
│   └── tracking.py                       # NEW: TrackingConfig dataclass
├── data/
│   ├── entity/
│   │   └── tracking.py                   # NEW: TrackResult, TrackItem entities
│   └── video.py                          # NEW: Video/sequence data loading
├── metrics/
│   └── tracking.py                       # NEW: MOTA, IDF1, HOTA metrics
├── recipe/
│   └── tracking/                         # NEW: Tracking recipe YAMLs
│       ├── byte_track.yaml
│       ├── oc_sort.yaml
│       └── deep_sort.yaml
└── types/
    └── task.py                           # MODIFY: Add TRACKING task type
```

#### 22.2.4 Understanding the Detection Output (Your Starting Point)

Every OTX detection model's `predict_step()` returns predictions that ultimately contain:

```python
# From OTXDetectionModel (lib/src/otx/backend/native/models/detection/base.py)
# After post-processing (NMS etc.), you get per-image:
#   bboxes: Tensor of shape [N, 4]  (x1, y1, x2, y2)
#   scores: Tensor of shape [N]     (confidence)
#   labels: Tensor of shape [N]     (class index)
```

For DETR-based models (RTDETR, DFine, DEIM), there's no NMS -- the transformer decoder directly outputs a fixed set of predictions, filtered by confidence threshold.

The tracker takes these per-frame detections as input and outputs them with an additional `track_id` field.

#### 22.2.5 Tracker Architecture (Decoupled Design)

The project specifies a **decoupled detector-tracker design**. This maps cleanly to OTX's architecture:

```python
# Conceptual flow:
class BaseTracker(ABC):
    """Base class for all OTX trackers."""

    @abstractmethod
    def update(self, detections: DetectionResult, frame_id: int) -> list[TrackResult]:
        """Associate new detections with existing tracks.

        Args:
            detections: Per-frame detection output (bboxes, scores, labels)
            frame_id: Current frame index

        Returns:
            List of TrackResult with track_id assigned to each detection
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state (e.g., between videos)."""
        ...
```

The tracker **does not modify the detector** -- it receives detection output and adds tracking metadata. This is critical for OTX compatibility.

#### 22.2.6 Tracking Algorithms to Implement

The project requires implementing and comparing these trackers:

| Algorithm                        | Key Idea                                                          | Appearance Features? | Complexity  |
| -------------------------------- | ----------------------------------------------------------------- | -------------------- | ----------- |
| **ByteTrack**                    | Two-stage association: high-confidence first, then low-confidence | No (motion only)     | Simple      |
| **OC-SORT**                      | Observation-centric online smoothing, handles occlusions          | No (motion only)     | Medium      |
| **BoT-SORT**                     | ByteTrack + camera motion compensation + ReID features            | Optional             | Medium-High |
| **DeepSORT-like**                | Kalman filter + deep appearance features (ReID)                   | Yes                  | Medium      |
| **Memory-bank (SAM-2 inspired)** | Maintain feature memory bank for identity stability               | Yes (memory bank)    | High        |

**ByteTrack** and **OC-SORT** are motion-only (IoU + Kalman filter), making them the simplest starting points. **DeepSORT** and **BoT-SORT** add appearance features, requiring a ReID model or feature extractor. The **SAM-2-inspired** variant is the most advanced, using a memory bank of past features.

#### 22.2.7 Unified Output Format

The project requires a stable output schema. Based on OTX's existing entity patterns (`lib/src/otx/data/entity/`):

```python
@dataclass
class TrackResult:
    """Single tracked object in a single frame."""
    frame_id: int                        # Frame index in video
    track_id: int                        # Consistent identity across frames
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    score: float                         # Detection confidence
    label: int                           # Class index

    # Optional track metadata
    age: int = 0                         # Frames since track creation
    velocity: tuple[float, float] = (0, 0)  # Estimated (vx, vy)
    state: str = "active"                # "active" | "lost" | "removed"
```

This extends OTX's prediction format without breaking existing interfaces.

#### 22.2.8 Configuration (OTX Config Pattern)

Following OTX's config pattern (see `lib/src/otx/config/data.py` for reference):

```python
# lib/src/otx/config/tracking.py
@dataclass
class TrackingConfig:
    """Configuration for multi-object tracking."""
    tracker_type: str = "byte_track"         # "byte_track" | "oc_sort" | "bot_sort" | "deep_sort" | "memory_bank"

    # Association thresholds
    match_threshold: float = 0.8             # IoU threshold for matching
    high_score_threshold: float = 0.6        # High-confidence detection threshold
    low_score_threshold: float = 0.1         # Low-confidence threshold (ByteTrack)

    # Track lifecycle
    max_age: int = 30                        # Frames before removing lost track
    min_hits: int = 3                        # Minimum hits to confirm a track
    init_score_threshold: float = 0.5        # Minimum score to initialize a track

    # Appearance / memory (for DeepSORT / memory-bank variants)
    use_appearance: bool = False              # Enable ReID features
    appearance_model: str | None = None       # Path to appearance model
    memory_bank_size: int = 10               # Number of past features to store
    appearance_weight: float = 0.5           # Weight of appearance vs. motion
```

YAML recipe example:

```yaml
# lib/src/otx/recipe/tracking/byte_track.yaml
tracker:
  class_path: otx.backend.native.models.tracking.byte_track.ByteTrack
  init_args:
    match_threshold: 0.8
    high_score_threshold: 0.6
    low_score_threshold: 0.1
    max_age: 30
    min_hits: 3
```

#### 22.2.9 Video/Sequence Data Loading

OTX currently only handles single images. For tracking, you need to add video support:

```python
# lib/src/otx/data/video.py
class VideoReader:
    """Read frames from a video file or image sequence directory."""

    def __init__(self, source: str | Path):
        """
        Args:
            source: Path to video file (.mp4, .avi) or directory of images
        """
        ...

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_id, frame_image) tuples."""
        ...

    def __len__(self) -> int:
        """Total number of frames."""
        ...
```

This integrates with OTX's data pipeline by feeding frames through the same transform pipeline used for detection inference.

#### 22.2.10 CLI Integration

The `otx predict` command (in `lib/src/otx/cli/cli.py`) currently runs single-image inference. For tracking:

```bash
# Proposed CLI extension:
otx track --config recipe/detection/yolox_tiny.yaml \
          --tracker byte_track \
          --checkpoint best_model.ckpt \
          --video_source /path/to/video.mp4 \
          --output_format mot_txt

# Or as part of predict with tracking flag:
otx predict --config recipe/detection/yolox_tiny.yaml \
            --checkpoint best_model.ckpt \
            --data_root /path/to/video_or_sequence \
            --tracking.tracker_type byte_track \
            --tracking.max_age 30
```

Output formats:

- **MOT Challenge TXT:** `<frame_id>,<track_id>,<x>,<y>,<w>,<h>,<score>,<class>,-1,-1`
- **JSON:** List of `TrackResult` objects
- **Visualization:** Video with bounding boxes + track IDs overlaid

#### 22.2.11 MOT Evaluation Metrics

The project requires tracking evaluation support. Standard MOT metrics:

| Metric          | What It Measures                                                  |
| --------------- | ----------------------------------------------------------------- |
| **MOTA**        | Multi-Object Tracking Accuracy (combines FP, FN, ID switches)     |
| **IDF1**        | ID F1-score (measures identity preservation)                      |
| **HOTA**        | Higher-Order Tracking Accuracy (balances detection + association) |
| **ID Switches** | Number of times a track changes identity                          |
| **MT / ML**     | Mostly Tracked / Mostly Lost track ratios                         |

These can be computed using the `trackeval` or `motmetrics` Python libraries. Integration would follow OTX's metric pattern:

```python
# lib/src/otx/metrics/tracking.py
class MOTMetricCallable:
    """Compute MOT metrics (MOTA, IDF1, HOTA)."""

    def __call__(self, predictions: list[TrackResult], ground_truth: list[TrackResult]) -> dict[str, float]:
        return {
            "MOTA": ...,
            "IDF1": ...,
            "HOTA": ...,
            "ID_Switches": ...,
        }
```

#### 22.2.12 SAM-2-Inspired Memory Bank Tracking

The most advanced deliverable. SAM-2 (Segment Anything Model 2) introduced a memory-based approach for video:

- **Memory bank:** Store appearance features of tracked objects from recent frames
- **Memory attention:** When associating new detections, query the memory bank to find the best match
- **Temporal consistency:** Memory provides robustness against occlusions and appearance changes

In the context of this project (bounding-box tracking, not segmentation):

```python
class MemoryBankTracker(BaseTracker):
    """Tracker with SAM-2-inspired feature memory bank."""

    def __init__(self, memory_bank_size: int = 10, feature_dim: int = 256):
        self.memory_bank: dict[int, deque[Tensor]] = {}  # track_id -> recent features
        self.kalman_filters: dict[int, KalmanFilter] = {}

    def update(self, detections, features, frame_id):
        # 1. Predict track positions with Kalman filter
        # 2. Compute appearance similarity using memory bank
        # 3. Combine motion + appearance costs
        # 4. Run Hungarian assignment
        # 5. Update memory bank with matched detection features
        ...
```

This requires extracting features from the detection backbone (e.g., RoI features) or using a separate ReID model.

#### 22.2.13 Implementation Roadmap

Based on the 350-hour project size, a suggested phased approach:

**Phase 1: Foundation (Weeks 1-4, ~100 hours)**

- Study OTX codebase thoroughly (this doc helps)
- Implement `BaseTracker` ABC and `TrackResult` entity
- Implement ByteTrack (simplest, motion-only)
- Add basic video reading capability
- Get end-to-end pipeline working: video -> detector -> ByteTrack -> MOT output

**Phase 2: Core Integration (Weeks 5-8, ~100 hours)**

- Add `TrackingConfig` following OTX config patterns
- Implement OC-SORT and BoT-SORT
- Integrate tracking into OTX CLI (`otx track` or `otx predict --tracking`)
- Add tracking YAML recipes
- Add MOT evaluation metrics (MOTA, IDF1)
- Write unit tests

**Phase 3: Advanced Tracking (Weeks 9-11, ~100 hours)**

- Implement DeepSORT-like tracker with appearance features
- Implement memory-bank tracker (SAM-2 inspired)
- Feature extraction from detection backbone or separate ReID model
- Comparison report: benchmark all trackers on MOT17/MOT20 or similar

**Phase 4: Polish & Documentation (Week 12, ~50 hours)**

- User documentation
- Tune default parameters for recommended configurations
- Edge case handling (empty frames, single-object, very crowded scenes)
- Code review iterations with mentors

#### 22.2.14 Key OTX Files to Study First

Before writing any code, deeply understand these files:

| Priority | File                                                  | Why                                                                  |
| -------- | ----------------------------------------------------- | -------------------------------------------------------------------- |
| 1        | `lib/src/otx/engine/engine.py`                        | Abstract Engine contract -- your tracker wraps around `predict()`    |
| 2        | `lib/src/otx/backend/native/engine.py`                | OTXEngine implementation -- understand the full predict pipeline     |
| 3        | `lib/src/otx/backend/native/models/detection/base.py` | OTXDetectionModel -- how detection output is structured              |
| 4        | `lib/src/otx/data/entity/base.py`                     | OTXDataItem, OTXPredItem -- the prediction data format you'll extend |
| 5        | `lib/src/otx/config/data.py`                          | SubsetConfig, TileConfig -- patterns for your TrackingConfig         |
| 6        | `lib/src/otx/cli/cli.py`                              | CLI structure -- how to add `track` subcommand                       |
| 7        | `lib/src/otx/metrics/fmeasure.py`                     | Metric callable pattern -- how to add MOT metrics                    |
| 8        | `lib/src/otx/recipe/detection/yolox_tiny.yaml`        | Recipe structure -- pattern for tracking recipes                     |
| 9        | `lib/src/otx/tools/auto_configurator.py`              | Auto-config -- may need to register tracking task                    |
| 10       | `lib/src/otx/types/task.py`                           | OTXTaskType enum -- if adding a TRACKING task type                   |

#### 22.2.15 Key Design Decisions to Make Early

1. **New task type or post-processing?** Should tracking be a new `OTXTaskType.TRACKING` or remain a post-processing layer on `DETECTION`? The project description says "post-processing step" -- but CLI integration might benefit from a task type.

2. **Where does the tracker live?** As a model (in `models/tracking/`) or as a tool (in `tools/tracking/`)? Since it's not a trainable model, `tools/` might be more appropriate. But if the memory-bank variant requires gradient-based learning, `models/` makes sense.

3. **Video data loading:** Extend `OTXDataModule` to support video, or create a separate `VideoDataModule`? A separate lightweight reader is probably cleaner since tracking doesn't need the full Lightning data pipeline.

4. **Output format:** Extend `OTXPredItem` with optional `track_id`, or create a new `OTXTrackItem`? Extending is more seamless; a new class is cleaner separation.

5. **Appearance features:** Extract from the detection backbone (free but lower quality) or use a separate ReID model (higher quality but more compute)? Start with backbone features, add ReID as optional.

#### 22.2.16 External Libraries to Consider

| Library                                | Purpose                              | License    |
| -------------------------------------- | ------------------------------------ | ---------- |
| `filterpy`                             | Kalman filter implementation         | MIT        |
| `scipy.optimize.linear_sum_assignment` | Hungarian algorithm                  | BSD        |
| `lap`                                  | Fast linear assignment (C extension) | BSD        |
| `trackeval`                            | MOT evaluation metrics               | MIT        |
| `motmetrics`                           | Alternative MOT metrics              | MIT        |
| `opencv-python`                        | Video I/O, visualization             | Apache 2.0 |
| `supervision` (Roboflow)               | Tracking utilities, visualization    | MIT        |

All are Apache 2.0 compatible.
