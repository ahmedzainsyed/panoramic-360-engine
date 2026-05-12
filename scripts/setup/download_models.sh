#!/usr/bin/env bash
# ================================================================
# Download pretrained model weights
# ================================================================
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-./ml/models}"
mkdir -p "$MODELS_DIR"

echo "📦 Downloading model weights to $MODELS_DIR..."

# SAM ViT-H (~2.5GB)
if [ ! -f "$MODELS_DIR/sam_vit_h_4b8939.pth" ]; then
  echo "Downloading SAM ViT-H..."
  wget -q --show-progress -O "$MODELS_DIR/sam_vit_h_4b8939.pth" \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
else
  echo "✓ SAM ViT-H already present"
fi

# YOLOv8x base weights
if [ ! -f "$MODELS_DIR/yolov8x.pt" ]; then
  echo "Downloading YOLOv8x..."
  python3 -c "from ultralytics import YOLO; YOLO('yolov8x.pt')" 2>/dev/null || true
  cp ~/.config/Ultralytics/yolov8x.pt "$MODELS_DIR/" 2>/dev/null || true
fi

echo ""
echo "✅ Model download complete"
echo "   Note: Fine-tuned construction models must be obtained separately."
echo "   Place them in: $MODELS_DIR/"
echo "   Required:"
echo "   - segformer_b5_construction_v2.pth"
echo "   - yolov8x_construction_v3.pt"
echo "   - yolov8m_ppe_v2.pt"
echo "   - dpt_large_construction.pth"
