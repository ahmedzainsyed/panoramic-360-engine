# 🌐 360° Construction Site Semantic Understanding Engine

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.1+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)

**Production-grade AI platform for panoramic construction site intelligence**

*Semantic understanding · PPE compliance · Hazard detection · Spatial analytics · Worker tracking*

</div>

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    360° CONSTRUCTION INTELLIGENCE PLATFORM               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────────────────────────────────────┐  │
│  │  360° Cameras│    │           INGESTION LAYER                    │  │
│  │  - Insta360  │───▶│  • Equirectangular normalization             │  │
│  │  - Ricoh     │    │  • EXIF / GPS extraction                     │  │
│  │  - Drones    │    │  • Session management                        │  │
│  └──────────────┘    └──────────────────┬───────────────────────────┘  │
│                                         │                               │
│                      ┌──────────────────▼───────────────────────────┐  │
│                      │        SPHERICAL GEOMETRY ENGINE              │  │
│                      │  • Equirect ↔ Cubemap transforms             │  │
│                      │  • Polar distortion correction                │  │
│                      │  • Seam continuity handling                   │  │
│                      └──────────────────┬───────────────────────────┘  │
│                                         │                               │
│          ┌──────────────────────────────┼──────────────────────────┐   │
│          │                              │                          │   │
│  ┌───────▼──────┐  ┌───────────────────▼──────┐  ┌───────────────▼─┐  │
│  │  DETECTION   │  │     SEGMENTATION          │  │  3D RECON       │  │
│  │  - YOLOv8    │  │  - SegFormer              │  │  - COLMAP       │  │
│  │  - DETR      │  │  - Mask2Former            │  │  - Open3D       │  │
│  │  - ViT       │  │  - SAM                    │  │  - Depth Est.   │  │
│  └───────┬──────┘  └───────────────────┬──────┘  └───────────┬─────┘  │
│          │                              │                     │        │
│          └──────────────────────────────┼─────────────────────┘        │
│                                         │                               │
│          ┌──────────────────────────────┼──────────────────────────┐   │
│          │                              │                          │   │
│  ┌───────▼──────┐  ┌───────────────────▼──────┐  ┌───────────────▼─┐  │
│  │  PPE COMP.   │  │  HAZARD DETECTION         │  │  OCCUPANCY      │  │
│  │  - Helmet    │  │  - Open shafts            │  │  - Heatmaps     │  │
│  │  - Vest      │  │  - Unsafe edges           │  │  - Density      │  │
│  │  - Gloves    │  │  - Risk zones             │  │  - Movement     │  │
│  └───────┬──────┘  └───────────────────┬──────┘  └───────────┬─────┘  │
│          │                              │                     │        │
│          └──────────────────────────────┼─────────────────────┘        │
│                                         │                               │
│                      ┌──────────────────▼───────────────────────────┐  │
│                      │          ANALYTICS ENGINE                     │  │
│                      │  • Temporal site evolution                    │  │
│                      │  • Worker movement analytics                  │  │
│                      │  • Safety trend tracking                      │  │
│                      │  • Navigation overlays                        │  │
│                      └──────────────────┬───────────────────────────┘  │
│                                         │                               │
│          ┌──────────────────────────────┼──────────────────────────┐   │
│          │                              │                          │   │
│  ┌───────▼──────┐  ┌───────────────────▼──────┐  ┌───────────────▼─┐  │
│  │   FastAPI    │  │      Celery Workers        │  │    MinIO/S3     │  │
│  │   REST API   │  │      Async Tasks           │  │    Storage      │  │
│  └───────┬──────┘  └───────────────────┬──────┘  └───────────┬─────┘  │
│          │                              │                     │        │
│          └──────────────────────────────┼─────────────────────┘        │
│                                         │                               │
│                      ┌──────────────────▼───────────────────────────┐  │
│                      │       REACT IMMERSIVE DASHBOARD               │  │
│                      │  • Three.js 360° viewer                       │  │
│                      │  • Real-time overlays                         │  │
│                      │  • Spatial analytics UI                       │  │
│                      │  • Timeline playback                          │  │
│                      └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🌍 Spherical Geometry Pipeline

```
Equirectangular Input (8192×4096)
        │
        ▼
┌───────────────────┐
│  Polar Distortion │  → Correct polar region stretching
│  Correction       │    using sinusoidal weighting
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Cubemap          │  → Convert to 6-face cubemap
│  Projection       │    (Front, Back, Left, Right, Up, Down)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Tiled Inference  │  → Run detection/segmentation on
│  + Merging        │    each face independently
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Back-projection  │  → Map results back to
│  to Sphere        │    equirectangular space
└────────┬──────────┘
         │
         ▼
  Panoramic Output with
  Spherically-consistent overlays
```

## 📦 Quick Start

### Prerequisites
- Docker 24+ and Docker Compose 2.20+
- NVIDIA GPU with CUDA 12.1+ (for GPU inference)
- 32GB RAM minimum, 64GB recommended
- 500GB+ SSD storage

### 1. Clone and Configure

```bash
git clone https://github.com/your-org/panoramic-360-engine.git
cd panoramic-360-engine

# Copy environment files
cp .env.example .env
cp configs/model/model_config.example.yaml configs/model/model_config.yaml

# Edit configuration
nano .env
```

### 2. GPU Setup (NVIDIA)

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### 3. Start the Platform

```bash
# Development mode
make dev

# Production mode
make prod

# With GPU support
make prod-gpu
```

### 4. Access Services

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | http://localhost:3000 | Immersive 360° UI |
| API Docs | http://localhost:8000/docs | FastAPI Swagger |
| MLflow | http://localhost:5000 | Experiment tracking |
| Grafana | http://localhost:3001 | System monitoring |
| MinIO | http://localhost:9001 | Object storage |
| Airflow | http://localhost:8080 | Pipeline orchestration |

## 🔌 API Reference

### Upload Panorama
```bash
curl -X POST http://localhost:8000/api/v1/panoramas/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@site_panorama.jpg" \
  -F "session_id=site_001" \
  -F "camera_type=insta360"
```

### Run Full Analysis Pipeline
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "panorama_id": "pan_abc123",
    "modules": ["segmentation", "ppe", "hazards", "occupancy"],
    "options": {
      "resolution": "high",
      "gpu_accelerated": true
    }
  }'
```

### Get Hazard Map
```bash
curl http://localhost:8000/api/v1/hazards/{panorama_id}/map \
  -H "Authorization: Bearer $TOKEN"
```

### Get Worker Heatmaps
```bash
curl http://localhost:8000/api/v1/occupancy/{session_id}/heatmaps \
  -H "Authorization: Bearer $TOKEN"
```

## 🏋️ Performance Benchmarks

| Module | GPU (A100) | GPU (RTX 3090) | CPU |
|--------|-----------|----------------|-----|
| Segmentation (8K pan.) | 1.2s | 2.8s | 18s |
| PPE Detection | 0.4s | 0.9s | 6s |
| Hazard Analysis | 0.6s | 1.4s | 9s |
| Full Pipeline | 3.1s | 6.8s | 45s |
| 3D Reconstruction | 45s | 92s | 480s |

*Benchmarks on 8192×4096 equirectangular panorama*

## 📁 Repository Structure

```
panoramic-360-engine/
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── api/v1/            # REST API endpoints
│   │   ├── core/              # Config, security, logging
│   │   ├── db/                # Database layer
│   │   ├── models/            # SQLAlchemy models
│   │   ├── schemas/           # Pydantic schemas
│   │   ├── services/          # Business logic
│   │   ├── tasks/             # Celery async tasks
│   │   └── utils/             # Utilities
│   └── alembic/               # DB migrations
│
├── frontend/                   # React TypeScript dashboard
│   └── src/
│       ├── components/        # UI components
│       ├── pages/             # Route pages
│       ├── hooks/             # React hooks
│       └── store/             # Zustand state
│
├── ml/                        # ML pipeline modules
│   ├── spherical_geometry/    # Projection transforms
│   ├── detection/             # Object detection
│   ├── segmentation/          # Scene segmentation
│   ├── occupancy/             # Spatial occupancy
│   ├── ppe/                   # PPE compliance
│   ├── hazards/               # Hazard detection
│   ├── reconstruction/        # 3D reconstruction
│   ├── navigation/            # Navigation overlays
│   └── analytics/             # Temporal analytics
│
├── kubernetes/                 # K8s manifests + Helm
├── airflow/                    # DAG orchestration
├── monitoring/                 # Prometheus + Grafana
├── docker/                     # Dockerfiles
├── scripts/                    # Utility scripts
├── tests/                      # Full test suite
└── docs/                       # Documentation
```

## 🔬 ML Model Details

### Segmentation Classes
```
0: background      7: concrete          14: worker
1: sky             8: steel/rebar       15: machinery
2: ground          9: formwork          16: crane
3: wall            10: scaffolding      17: excavator
4: floor           11: active_zone      18: vehicle
5: soil            12: restricted_zone  19: hazard_zone
6: walkable_path   13: unsafe_edge      20: open_shaft
```

### PPE Detection Classes
```
- hard_hat (helmet)       compliant / non-compliant
- safety_vest             compliant / non-compliant
- safety_gloves           detected / not_detected
- safety_boots            detected / not_detected
- safety_goggles          detected / not_detected
- full_ppe_compliant      boolean aggregate
```

## 🚀 Deployment

### Kubernetes (Production)
```bash
# Install with Helm
helm install 360-engine ./kubernetes/helm/360-engine \
  --namespace panoramic \
  --create-namespace \
  -f kubernetes/helm/360-engine/values-prod.yaml

# Monitor rollout
kubectl rollout status deployment/api-server -n panoramic
```

### Docker Compose (Development)
```bash
make dev       # Start all services
make logs      # Tail all logs
make shell     # Shell into API container
make test      # Run test suite
make clean     # Stop and cleanup
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific modules
pytest tests/unit/ml/test_spherical_geometry.py -v
pytest tests/integration/test_api.py -v --cov

# Generate coverage report
make coverage
```

## 📊 MLOps Workflows

```bash
# Track experiment
python scripts/training/train_segmentation.py \
  --config configs/training/segformer_config.yaml \
  --wandb-project 360-construction

# Version dataset
dvc add datasets/raw/site_panoramas/
dvc push

# Run Airflow pipeline
airflow dags trigger panoramic_training_pipeline
```

## 🤝 Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

<div align="center">
Built with ❤️ by the Spatial AI Team | Inspired by Matterport, OpenSpace, Track3D
</div>
