# System Architecture - 360° Construction Intelligence Engine

## Overview

The platform processes equirectangular panoramic imagery through a multi-stage ML pipeline to generate spatial intelligence for construction sites.

## Processing Pipeline

```
Input Panorama (Equirectangular 2:1)
         │
         ▼
┌─────────────────────────────┐
│   Ingestion & Validation    │  ← EXIF/XMP extraction, GPS sync,
│                             │    format normalization, deduplication
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Spherical Geometry Engine │  ← Equirect → Cubemap → Perspective tiles
│                             │    Polar distortion correction
│                             │    Seam continuity handling
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌──────────┐   ┌──────────────┐
│SegFormer │   │  YOLOv8      │   ← Parallel inference
│Tiled Seg.│   │  Detection   │
└────┬─────┘   └──────┬───────┘
     │                │
     └───────┬─────────┘
             │
     ┌───────┴──────────────────────────────────┐
     │                                          │
     ▼           ▼           ▼           ▼      ▼
┌─────────┐ ┌────────┐ ┌─────────┐ ┌────────┐ ┌────────┐
│   PPE   │ │Hazards │ │Occupancy│ │NavMesh │ │Depth3D │
│Compliance│ │Zones   │ │Heatmaps │ │Overlay │ │Recon.  │
└────┬────┘ └───┬────┘ └────┬────┘ └───┬────┘ └───┬────┘
     └──────────┴───────────┴──────────┴──────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Temporal Engine │  ← Cross-session comparison
                   │                 │    Progress tracking
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   API Layer     │  ← FastAPI REST + WebSocket
                   │   (FastAPI)     │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ React Dashboard │  ← Three.js 360° viewer
                   │                 │    Recharts analytics
                   └─────────────────┘
```

## Spherical Geometry Details

### Equirectangular Projection
- Maps sphere to rectangle: longitude → X, latitude → Y
- Width:Height = 2:1 for full 360°×180°
- Pixel (u,v) → angle: φ = 2π·u/W, θ = π·v/H

### Cubemap Projection
- Maps sphere to 6 square faces of unit cube
- Eliminates pole distortion for near-overhead processing
- Implemented via ray tracing: cube face → sphere → equirect sampling

### Perspective Tile Extraction
- Simulates pinhole camera at arbitrary yaw/pitch
- Enables standard CNNs to process spherical content without distortion
- Tiles overlap 20° to ensure seamless detection merging

### Back-Projection
- Maps detections from tile space → equirectangular space
- Uses same rotation matrix used for forward projection (inverse = transpose)
- Handles longitude wraparound at ±180°

## ML Model Architecture

### SegFormer-B5 (Segmentation)
- Hierarchical Transformer encoder (Mix Transformer)
- All-MLP decoder with multi-scale feature fusion
- Input: perspective tile (1024×512) → tiled inference
- Output: 21-class semantic mask
- Mean IoU target: 0.71+ on construction data

### YOLOv8 (Detection + PPE)
- CSP-DarkNet backbone with C2f modules
- Decoupled detection heads
- Construction model: 13 classes
- PPE model: 8 PPE-specific classes
- Processes 640×640 perspective tiles

### SAM (Interactive Segmentation)
- Vision Transformer encoder (ViT-H: 632M params)
- Mask decoder with prompt conditioning
- Used for high-precision instance boundary refinement

## Infrastructure

### Async Processing
```
Client → FastAPI → Redis Queue → Celery Worker
                                      │
                               ┌──────┴──────┐
                               │             │
                        GPU Worker 1   GPU Worker 2
                        (Segmentation) (Detection/PPE)
```

### Storage Layout
```
S3/MinIO:
  panoramas-raw/          → Original uploads
  panoramas-processed/    → Normalized for inference  
  analysis-outputs/       → Results, masks, heatmaps
  model-weights/          → Model checkpoints
```

## Performance Characteristics

| Metric | A100 GPU | RTX 3090 | CPU-only |
|--------|----------|----------|----------|
| Segmentation (4K pan) | 1.2s | 2.8s | 18s |
| PPE Detection | 0.4s | 0.9s | 6s |
| Hazard Analysis | 0.6s | 1.4s | 9s |
| Navigation Gen | 0.3s | 0.7s | 3s |
| Full Pipeline | 3.1s | 6.8s | 45s |
| 3D Reconstruction | 45s | 92s | 480s |

*Benchmarks on 8192×4096 equirectangular panorama*

## Scalability

- Horizontal API scaling: stateless FastAPI → load-balanced behind NGINX
- GPU Worker scaling: Kubernetes HPA based on queue depth
- Storage: MinIO distributed mode or AWS S3
- Database: PostgreSQL with read replicas for analytics queries
- Caching: Redis for hot results (24h TTL)
