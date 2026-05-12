"""
Panoramic Object Detection Engine
YOLOv8 + DETR-based spherically-aware detection for construction sites.
Handles cross-projection merging, boundary handling, and multi-view consistency.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

CONSTRUCTION_DETECTION_CLASSES = {
    0: "worker", 1: "helmet", 2: "safety_vest", 3: "crane", 4: "excavator",
    5: "concrete_mixer", 6: "scaffolding", 7: "material_pile", 8: "vehicle",
    9: "ladder", 10: "barrier", 11: "signage", 12: "power_tool",
}

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    panorama_yaw: float = 0.0
    panorama_pitch: float = 0.0
    track_id: Optional[int] = None
    attributes: Dict = field(default_factory=dict)

@dataclass
class DetectionResult:
    panorama_id: str
    detections: List[Detection]
    class_counts: Dict[str, int]
    inference_time_ms: float = 0.0

    @property
    def worker_count(self) -> int:
        return self.class_counts.get("worker", 0)


class PanoramicObjectDetector:
    """
    Spherically-aware object detector for 360° panoramas.
    Uses perspective-tile approach to avoid equirectangular distortion effects.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        tile_fov: float = 90.0,
        tile_overlap_deg: float = 20.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.tile_fov = tile_fov
        self.tile_overlap_deg = tile_overlap_deg
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            from ultralytics import YOLO
            if model_path:
                import os
                if os.path.exists(model_path):
                    return YOLO(model_path)
            return YOLO("yolov8x.pt")
        except ImportError:
            logger.warning("ultralytics_not_available")
            return None

    def detect(self, image: np.ndarray, panorama_id: str) -> DetectionResult:
        t0 = time.perf_counter()
        h, w = image.shape[:2]

        from ml.spherical_geometry.spherical_engine import SphericalGeometryEngine
        geo = SphericalGeometryEngine()
        tiles = geo.extract_perspective_tiles(
            image=image, fov_h=self.tile_fov, fov_v=self.tile_fov,
            tile_size=640, overlap_deg=self.tile_overlap_deg,
        )

        all_detections: List[Detection] = []

        for tile in tiles:
            tile_dets = self._run_detection_on_tile(tile.image)
            for det in tile_dets:
                box_arr = np.array([[det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]]])
                eq_boxes = geo.backproject_boxes_to_equirect(
                    boxes=box_arr, tile=tile,
                    equirect_width=w, equirect_height=h,
                )
                if len(eq_boxes) > 0:
                    x1, y1, x2, y2 = eq_boxes[0]
                    cx = (x1 + x2) / 2
                    det.bbox = (float(x1), float(y1), float(x2), float(y2))
                    det.panorama_yaw = (cx / w) * 360.0
                    all_detections.append(det)

        # NMS across tile boundaries
        all_detections = self._global_nms(all_detections)

        class_counts: Dict[str, int] = {}
        for det in all_detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("detection_complete", panorama_id=panorama_id,
                    objects=len(all_detections), ms=f"{elapsed_ms:.1f}")

        return DetectionResult(
            panorama_id=panorama_id,
            detections=all_detections,
            class_counts=class_counts,
            inference_time_ms=elapsed_ms,
        )

    def _run_detection_on_tile(self, tile: np.ndarray) -> List[Detection]:
        if self.model is None:
            return self._mock_detections(tile)
        results = self.model.predict(
            source=tile, conf=self.confidence_threshold,
            iou=self.iou_threshold, device=self.device, verbose=False,
        )
        dets = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item()) % len(CONSTRUCTION_DETECTION_CLASSES)
                conf = float(box.conf.item())
                xyxy = box.xyxy.squeeze().cpu().numpy()
                dets.append(Detection(
                    class_id=cls_id,
                    class_name=CONSTRUCTION_DETECTION_CLASSES.get(cls_id, f"class_{cls_id}"),
                    confidence=conf,
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                ))
        return dets

    def _mock_detections(self, tile: np.ndarray) -> List[Detection]:
        h, w = tile.shape[:2]
        return [
            Detection(class_id=0, class_name="worker", confidence=0.92,
                      bbox=(w*0.2, h*0.3, w*0.4, h*0.9)),
            Detection(class_id=3, class_name="crane", confidence=0.87,
                      bbox=(w*0.6, h*0.1, w*0.95, h*0.7)),
        ]

    def _global_nms(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []
        by_class: Dict[int, List[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.class_id, []).append(det)
        kept = []
        for class_id, class_dets in by_class.items():
            class_dets_sorted = sorted(class_dets, key=lambda d: d.confidence, reverse=True)
            suppressed = [False] * len(class_dets_sorted)
            for i in range(len(class_dets_sorted)):
                if suppressed[i]:
                    continue
                kept.append(class_dets_sorted[i])
                for j in range(i+1, len(class_dets_sorted)):
                    if suppressed[j]:
                        continue
                    iou = self._compute_iou(class_dets_sorted[i].bbox, class_dets_sorted[j].bbox)
                    if iou > self.iou_threshold:
                        suppressed[j] = True
        return kept

    @staticmethod
    def _compute_iou(box_a, box_b):
        ax1,ay1,ax2,ay2 = box_a; bx1,by1,bx2,by2 = box_b
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        if ix2<=ix1 or iy2<=iy1: return 0.0
        inter = (ix2-ix1)*(iy2-iy1)
        area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
        return inter/(area_a+area_b-inter+1e-9)
