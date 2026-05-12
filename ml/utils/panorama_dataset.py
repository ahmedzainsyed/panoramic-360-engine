"""
PyTorch Dataset for Panoramic Construction Imagery.
Supports segmentation, detection, and PPE label formats.
"""
from __future__ import annotations
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PanoramaSegmentationDataset:
    """
    Dataset for panoramic semantic segmentation.

    Directory structure:
        root/
          train/
            images/   ← JPEG equirectangular panoramas
            masks/    ← uint8 PNG segmentation masks (0-20 class indices)
          val/
          test/

    Augmentations:
      - Random equirectangular rotation (yaw shift)
      - Random horizontal flip (with mask)
      - Color jitter
      - Polar-aware cropping
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: Tuple[int, int] = (1024, 512),  # (W, H)
        augment: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.img_w, self.img_h = img_size
        self.augment = augment and split == "train"

        img_dir = self.root / split / "images"
        mask_dir = self.root / split / "masks"

        if not img_dir.exists():
            logger.warning("dataset_dir_not_found", path=str(img_dir))
            self.samples = []
            return

        img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        self.samples = []
        for img_path in img_files:
            mask_path = mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                self.samples.append((str(img_path), str(mask_path)))

        if max_samples:
            self.samples = self.samples[:max_samples]

        logger.info("dataset_loaded", split=split, samples=len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, mask_path = self.samples[idx]

        # Load image and mask
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LANCZOS4)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        # Normalize image
        img_float = img.astype(np.float32) / 255.0
        img_float = (img_float - np.array(self.MEAN)) / np.array(self.STD)

        result = {
            "image_path": img_path,
            "pixel_values": img_float.transpose(2, 0, 1).astype(np.float32),  # (C, H, W)
            "labels": mask.astype(np.int64),
        }

        if HAS_TORCH:
            import torch
            result["pixel_values"] = torch.from_numpy(result["pixel_values"])
            result["labels"] = torch.from_numpy(result["labels"])

        return result

    def _augment(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply panorama-aware augmentations."""
        # Equirectangular yaw rotation (horizontal circular shift)
        if random.random() < 0.8:
            shift = random.randint(0, self.img_w - 1)
            img  = np.roll(img,  shift, axis=1)
            mask = np.roll(mask, shift, axis=1)

        # Horizontal flip
        if random.random() < 0.5:
            img  = np.fliplr(img)
            mask = np.fliplr(mask)

        # Color jitter
        if random.random() < 0.7:
            img = self._color_jitter(img)

        # Polar region blur (simulate distortion)
        if random.random() < 0.3:
            img = self._polar_blur(img)

        return img, mask

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        img_float = img.astype(np.float32)
        # Brightness
        img_float *= random.uniform(0.7, 1.3)
        # Contrast
        mean = img_float.mean()
        img_float = (img_float - mean) * random.uniform(0.7, 1.3) + mean
        # Saturation (HSV)
        hsv = cv2.cvtColor(np.clip(img_float, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.7, 1.3)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img

    def _polar_blur(self, img: np.ndarray) -> np.ndarray:
        h = img.shape[0]
        blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
        # Blend: poles blurred, equator original
        lat = (np.arange(h, dtype=np.float32) / h - 0.5) * np.pi
        alpha = ((1 - np.cos(lat)) / 2 * 0.5)[:, np.newaxis, np.newaxis]
        return (img * (1 - alpha) + blurred * alpha).astype(np.uint8)

    def get_dataloader(
        self,
        batch_size: int = 4,
        num_workers: int = 4,
        shuffle: Optional[bool] = None,
    ):
        """Return a configured DataLoader."""
        if not HAS_TORCH:
            raise ImportError("torch required for DataLoader")
        from torch.utils.data import DataLoader
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle if shuffle is not None else (self.split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(self.split == "train"),
            persistent_workers=num_workers > 0,
        )
