# Dataset Structure

## Directory Layout
```
datasets/
├── raw/                    # Original panoramas from cameras
│   ├── site_001/           # Session-organized
│   │   ├── panorama_001.jpg
│   │   ├── panorama_002.jpg
│   │   └── metadata.json
│   └── site_002/
│
├── processed/              # Pre-processed, normalized panoramas
│   ├── train/
│   │   ├── images/         # 2048x1024 equirectangular JPEGs
│   │   └── masks/          # 2048x1024 uint8 PNG segmentation masks
│   ├── val/
│   └── test/
│
├── annotations/            # Label files
│   ├── segmentation/       # Per-pixel class annotations (PNG)
│   ├── detection/          # YOLO format bounding boxes (.txt)
│   ├── ppe/                # PPE label files
│   └── export_formats/     # COCO JSON, Pascal VOC XML
│
└── samples/                # Small sample set for testing/CI
    ├── panorama_sample.jpg  # 512x256 test panorama
    └── annotation_sample.png
```

## Segmentation Classes (21 classes)
See `ml/segmentation/panoramic_segmenter.py:CONSTRUCTION_CLASSES`

## Annotation Guidelines
- Use LabelStudio or CVAT for annotation
- All masks must be uint8 PNG files
- Use class indices (0-20) NOT colors
- Panoramas normalized to 2048x1024 (2:1 ratio)

## Data Versioning
Uses DVC. Run `dvc pull` to fetch dataset from remote storage.
