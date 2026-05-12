"""
Panoramic Semantic Segmentation Pipeline
=========================================
Implements multi-model semantic and instance segmentation for 360° construction imagery.

Models used:
  - SegFormer-B5: primary semantic segmentation
  - Mask2Former: instance segmentation
  - SAM (Segment Anything): interactive/prompted segmentation
  - DeepLabV3+: alternative backbone

Segmentation classes (21 categories):
  0: background       7: concrete          14: worker
  1: sky              8: steel_rebar       15: machinery
  2: ground           9: formwork          16: crane
  3: wall             10: scaffolding      17: excavator
  4: floor            11: active_zone      18: vehicle
  5: soil             12: restricted_zone  19: hazard_zone
  6: walkable_path    13: unsafe_edge      20: open_shaft
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import structlog

logger = structlog.get_logger(__name__)

# ─── Segmentation Constants ───────────────────────────────────

CONSTRUCTION_CLASSES = {
    0:  "background",
    1:  "sky",
    2:  "ground",
    3:  "wall",
    4:  "floor",
    5:  "soil",
    6:  "walkable_path",
    7:  "concrete",
    8:  "steel_rebar",
    9:  "formwork",
    10: "scaffolding",
    11: "active_zone",
    12: "restricted_zone",
    13: "unsafe_edge",
    14: "worker",
    15: "machinery",
    16: "crane",
    17: "excavator",
    18: "vehicle",
    19: "hazard_zone",
    20: "open_shaft",
}

NUM_CLASSES = len(CONSTRUCTION_CLASSES)

# Semantic color palette (RGB) for visualization
CLASS_COLORS = np.array([
    [0,   0,   0  ],  # 0: background    - black
    [135, 206, 235],  # 1: sky           - sky blue
    [139, 119, 101],  # 2: ground        - brown
    [192, 192, 192],  # 3: wall          - silver
    [211, 211, 211],  # 4: floor         - light gray
    [160, 82,  45 ],  # 5: soil          - sienna
    [144, 238, 144],  # 6: walkable_path - light green
    [169, 169, 169],  # 7: concrete      - dark gray
    [184, 134, 11 ],  # 8: steel_rebar   - dark goldenrod
    [210, 180, 140],  # 9: formwork      - tan
    [255, 165, 0  ],  # 10: scaffolding  - orange
    [50,  205, 50 ],  # 11: active_zone  - lime green
    [255, 140, 0  ],  # 12: restricted   - dark orange
    [255, 69,  0  ],  # 13: unsafe_edge  - red-orange
    [0,   0,   255],  # 14: worker       - blue
    [128, 0,   128],  # 15: machinery    - purple
    [255, 215, 0  ],  # 16: crane        - gold
    [0,   128, 0  ],  # 17: excavator    - green
    [0,   255, 255],  # 18: vehicle      - cyan
    [255, 0,   0  ],  # 19: hazard_zone  - red
    [148, 0,   211],  # 20: open_shaft   - dark violet
], dtype=np.uint8)

# High-risk classes requiring immediate attention
HAZARD_CLASS_IDS = {12, 13, 19, 20}  # restricted, unsafe_edge, hazard, shaft

# PPE-relevant region classes (where workers should be detected)
WORK_ZONE_CLASS_IDS = {11, 12, 14}   # active_zone, restricted, worker


@dataclass
class SegmentationResult:
    """Container for segmentation pipeline outputs."""
    semantic_mask: np.ndarray          # (H, W) uint8 class indices
    confidence_map: np.ndarray         # (H, W) float32 [0, 1]
    instance_masks: Optional[List[np.ndarray]] = None  # List of (H, W) bool masks
    instance_classes: Optional[List[int]] = None
    instance_scores: Optional[List[float]] = None
    class_areas: Dict[str, float] = field(default_factory=dict)
    hazard_score: float = 0.0
    inference_time_ms: float = 0.0
    model_name: str = ""

    @property
    def colorized_mask(self) -> np.ndarray:
        """Return RGB colorized segmentation mask."""
        return CLASS_COLORS[self.semantic_mask]

    @property
    def hazard_mask(self) -> np.ndarray:
        """Boolean mask of hazardous regions."""
        return np.isin(self.semantic_mask, list(HAZARD_CLASS_IDS))

    @property
    def walkable_mask(self) -> np.ndarray:
        """Boolean mask of walkable regions."""
        return self.semantic_mask == 6  # walkable_path class


class SegFormerPanoramicSegmenter(nn.Module):
    """
    SegFormer-B5 adapted for panoramic equirectangular segmentation.

    Key modifications for panoramic processing:
    1. Tiled inference with seam-aware blending
    2. Spherical weight map for confidence aggregation
    3. Horizontal wrap-padding for seam continuity
    4. Mixed precision inference support
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = NUM_CLASSES,
        device: str = "cuda",
        use_amp: bool = True,
        tile_size: int = 1024,
        tile_overlap: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        self.model = self._build_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "segformer_initialized",
            device=str(self.device),
            classes=num_classes,
            amp=self.use_amp,
        )

    def _build_model(self, model_path: Optional[str]) -> nn.Module:
        """Load SegFormer model with construction-domain fine-tuning."""
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig

            if model_path and Path(model_path).exists():
                # Load fine-tuned model
                config = SegformerConfig.from_pretrained(
                    "nvidia/segformer-b5-finetuned-ade-640-640",
                    num_labels=self.num_classes,
                    id2label={str(i): v for i, v in CONSTRUCTION_CLASSES.items()},
                    label2id={v: i for i, v in CONSTRUCTION_CLASSES.items()},
                )
                model = SegformerForSemanticSegmentation(config)
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                logger.info("loaded_finetuned_segformer", path=model_path)
            else:
                # Use pretrained ADE20K weights as initialization
                model = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b5-finetuned-ade-640-640",
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                )
                logger.warning("using_pretrained_weights_not_finetuned")

            return model

        except ImportError:
            logger.warning("transformers_not_available_using_mock_model")
            return self._build_mock_model()

    def _build_mock_model(self) -> nn.Module:
        """Lightweight mock for testing without full model weights."""
        class MockSegFormer(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, num_classes, 1),
                )
            def forward(self, pixel_values):
                logits = self.conv(pixel_values)
                from types import SimpleNamespace
                return SimpleNamespace(logits=logits)
        return MockSegFormer(self.num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits tensor."""
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    @torch.inference_mode()
    def segment_panorama(
        self,
        image: np.ndarray,
        return_confidence: bool = True,
    ) -> SegmentationResult:
        """
        Segment a full equirectangular panorama.

        Uses tiled inference:
        1. Divide panorama into overlapping tiles
        2. Run SegFormer on each tile
        3. Back-project probabilities to panorama space
        4. Aggregate with spherical weighting
        5. Argmax to get final class map

        Args:
            image: HxWxC equirectangular image (uint8)
            return_confidence: whether to compute per-pixel confidence

        Returns:
            SegmentationResult with semantic mask and metadata
        """
        t0 = time.perf_counter()
        h, w = image.shape[:2]

        # Step 1: Tiled inference
        logit_accumulator = np.zeros((self.num_classes, h, w), dtype=np.float32)
        weight_accumulator = np.zeros((h, w), dtype=np.float32)

        tiles_info = self._compute_tile_grid(w, h)

        for (x_start, y_start, x_end, y_end) in tiles_info:
            # Extract tile with wrap padding for horizontal seam
            tile = self._extract_tile_with_wrap(image, x_start, y_start, x_end, y_end)

            # Run inference
            logits = self._run_tile_inference(tile)  # (C, th, tw)

            # Back-project and accumulate
            tile_h, tile_w = y_end - y_start, x_end - x_start
            logits_resized = F.interpolate(
                torch.from_numpy(logits).unsqueeze(0),
                size=(tile_h, tile_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).numpy()

            # Feathered weight for tile blending
            tile_weight = self._compute_tile_weight(tile_h, tile_w)

            # Handle horizontal wrap
            x_end_clipped = min(x_end, w)
            slice_w = x_end_clipped - x_start

            logit_accumulator[:, y_start:y_end, x_start:x_end_clipped] += (
                logits_resized[:, :, :slice_w] * tile_weight[np.newaxis, :, :slice_w]
            )
            weight_accumulator[y_start:y_end, x_start:x_end_clipped] += (
                tile_weight[:, :slice_w]
            )

            # Handle seam wrap
            if x_end > w:
                overflow = x_end - w
                logit_accumulator[:, y_start:y_end, :overflow] += (
                    logits_resized[:, :, slice_w:] * tile_weight[np.newaxis, :, slice_w:]
                )
                weight_accumulator[y_start:y_end, :overflow] += tile_weight[:, slice_w:]

        # Normalize accumulated logits
        weight_accumulator = np.maximum(weight_accumulator, 1e-9)
        logit_accumulator /= weight_accumulator[np.newaxis, :, :]

        # Apply spherical weight map (downweight poles)
        sph_weights = self._get_spherical_weights(h, w)
        logit_accumulator *= sph_weights[np.newaxis, :, :]

        # Convert to probabilities and final prediction
        probs = torch.softmax(torch.from_numpy(logit_accumulator), dim=0).numpy()
        semantic_mask = probs.argmax(axis=0).astype(np.uint8)

        confidence_map = probs.max(axis=0) if return_confidence else None

        # Compute class area statistics
        class_areas = self._compute_class_areas(semantic_mask, h * w)

        # Compute hazard score
        hazard_score = self._compute_hazard_score(semantic_mask)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "panorama_segmented",
            shape=f"{h}x{w}",
            hazard_score=f"{hazard_score:.3f}",
            inference_ms=f"{elapsed_ms:.1f}",
            top_class=CONSTRUCTION_CLASSES[int(semantic_mask.max())],
        )

        return SegmentationResult(
            semantic_mask=semantic_mask,
            confidence_map=confidence_map,
            class_areas=class_areas,
            hazard_score=hazard_score,
            inference_time_ms=elapsed_ms,
            model_name="segformer-b5-construction",
        )

    def _run_tile_inference(self, tile: np.ndarray) -> np.ndarray:
        """Run SegFormer inference on a single tile. Returns (C, H, W) logits."""
        # Normalize to [0, 1] and convert to tensor
        tile_float = tile.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        tile_norm = (tile_float - mean) / std

        # (H, W, C) → (1, C, H, W)
        tensor = torch.from_numpy(tile_norm.transpose(2, 0, 1)).float().unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            logits = self.forward(tensor)  # (1, C, h, w)

        return logits.squeeze(0).float().cpu().numpy()

    def _compute_tile_grid(
        self,
        width: int,
        height: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Compute overlapping tile grid covering the full panorama."""
        step = int(self.tile_size * (1 - self.tile_overlap))
        tiles = []

        y = 0
        while y < height:
            x = 0
            while x < width:
                tiles.append((x, y, x + self.tile_size, y + self.tile_size))
                x += step
                if x + self.tile_size > width and x < width:
                    tiles.append((width - self.tile_size, y, width, y + self.tile_size))
                    break
            y += step
            if y + self.tile_size > height and y < height:
                y = height - self.tile_size

        return tiles

    def _extract_tile_with_wrap(
        self,
        image: np.ndarray,
        x_start: int,
        y_start: int,
        x_end: int,
        y_end: int,
    ) -> np.ndarray:
        """Extract image tile with horizontal wrap-around for seam handling."""
        h, w = image.shape[:2]
        tile_h = min(y_end, h) - y_start
        tile_w = min(x_end, w) - x_start

        tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)

        # Main region
        src_y1 = max(0, y_start)
        src_y2 = min(h, y_end)
        tile[:tile_h, :tile_w] = image[src_y1:src_y2, x_start:min(w, x_end)]

        # Handle wrap
        if x_end > w:
            overflow = x_end - w
            tile[:tile_h, tile_w:tile_w+overflow] = image[src_y1:src_y2, :overflow]

        # Resize to tile_size if needed
        if tile.shape[:2] != (self.tile_size, self.tile_size):
            tile = cv2.resize(tile, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR)

        return tile

    def _compute_tile_weight(self, h: int, w: int) -> np.ndarray:
        """Cosine feathering weight map for smooth tile blending."""
        margin = int(min(h, w) * self.tile_overlap)
        weight = np.ones((h, w), dtype=np.float32)

        # Feather all 4 edges
        for i in range(margin):
            val = (1 - np.cos(np.pi * i / margin)) / 2
            weight[i, :] *= val
            weight[h-1-i, :] *= val
            weight[:, i] *= val
            weight[:, w-1-i] *= val

        return weight

    def _get_spherical_weights(self, h: int, w: int) -> np.ndarray:
        """Per-row cosine weight map (poles → less weight)."""
        lat = (np.arange(h, dtype=np.float32) / h - 0.5) * np.pi
        weights = np.cos(lat)
        return np.tile(weights[:, np.newaxis], (1, w))

    def _compute_class_areas(
        self,
        mask: np.ndarray,
        total_pixels: int,
    ) -> Dict[str, float]:
        """Compute percentage area for each class."""
        areas = {}
        for class_id, class_name in CONSTRUCTION_CLASSES.items():
            count = int((mask == class_id).sum())
            areas[class_name] = count / total_pixels * 100.0
        return areas

    def _compute_hazard_score(self, mask: np.ndarray) -> float:
        """
        Compute overall site hazard score [0, 1].
        Weighted sum of hazard class pixels.
        """
        total = mask.size
        if total == 0:
            return 0.0

        hazard_weights = {
            20: 1.0,   # open_shaft - most dangerous
            13: 0.85,  # unsafe_edge
            19: 0.75,  # hazard_zone
            12: 0.50,  # restricted_zone
        }

        score = sum(
            weight * (mask == cls_id).sum() / total
            for cls_id, weight in hazard_weights.items()
        )
        return min(float(score * 10), 1.0)  # Scale to 0-1


class SAMInteractiveSegmenter:
    """
    Segment Anything Model (SAM) for interactive and prompted segmentation.
    Used for:
    - Precise worker instance segmentation
    - Equipment boundary delineation
    - Hazard zone refinement
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "vit_h",
        device: str = "cuda",
    ):
        self.device = device
        self.predictor = None
        self._load_model(model_path, model_type)

    def _load_model(self, model_path: Optional[str], model_type: str):
        """Load SAM predictor."""
        try:
            from segment_anything import sam_model_registry, SamPredictor

            if model_path and Path(model_path).exists():
                sam = sam_model_registry[model_type](checkpoint=model_path)
                sam.to(self.device)
                self.predictor = SamPredictor(sam)
                logger.info("sam_loaded", type=model_type, path=model_path)
            else:
                logger.warning("sam_model_not_found", path=model_path)
        except ImportError:
            logger.warning("segment_anything_not_installed")

    def segment_with_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,  # (N, 4) [x1, y1, x2, y2]
    ) -> List[np.ndarray]:
        """
        Generate precise instance masks for detected bounding boxes.
        Used to get pixel-perfect worker/object boundaries.
        """
        if self.predictor is None:
            return [self._box_to_rect_mask(box, image.shape[:2]) for box in boxes]

        self.predictor.set_image(image)
        masks = []
        for box in boxes:
            m, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=True,
            )
            # Select highest-scoring mask
            best = m[np.argmax(scores)]
            masks.append(best.astype(bool))

        return masks

    def _box_to_rect_mask(
        self,
        box: np.ndarray,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """Fallback rectangular mask when SAM unavailable."""
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        mask[y1:y2, x1:x2] = True
        return mask


def create_segmentation_overlay(
    image: np.ndarray,
    result: SegmentationResult,
    alpha: float = 0.5,
    highlight_hazards: bool = True,
) -> np.ndarray:
    """
    Create visualization overlay of segmentation on original image.

    Args:
        image: Original panorama (HxWxC uint8)
        result: SegmentationResult from pipeline
        alpha: Transparency of segmentation overlay
        highlight_hazards: Whether to emphasize hazard regions

    Returns:
        Blended visualization image
    """
    overlay = result.colorized_mask.astype(np.float32)
    img_float = image.astype(np.float32)

    blended = img_float * (1 - alpha) + overlay * alpha

    if highlight_hazards:
        hazard_mask = result.hazard_mask
        # Pulsing red overlay for hazards
        hazard_color = np.array([255, 0, 0], dtype=np.float32)
        blended[hazard_mask] = (
            blended[hazard_mask] * 0.3 + hazard_color * 0.7
        )

    return np.clip(blended, 0, 255).astype(np.uint8)


def compute_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """
    Compute per-class IoU and mean IoU for segmentation evaluation.

    Args:
        pred_mask: (H, W) predicted class indices
        gt_mask: (H, W) ground truth class indices
        num_classes: number of segmentation classes

    Returns:
        Dict with per-class IoU and 'mean_iou'
    """
    iou_per_class = {}
    ious = []

    for cls_id in range(num_classes):
        pred_cls = pred_mask == cls_id
        gt_cls = gt_mask == cls_id

        intersection = (pred_cls & gt_cls).sum()
        union = (pred_cls | gt_cls).sum()

        if union == 0:
            iou = float("nan")
        else:
            iou = intersection / union

        class_name = CONSTRUCTION_CLASSES.get(cls_id, f"class_{cls_id}")
        iou_per_class[class_name] = iou
        if not np.isnan(iou):
            ious.append(iou)

    iou_per_class["mean_iou"] = float(np.mean(ious)) if ious else 0.0
    return iou_per_class
