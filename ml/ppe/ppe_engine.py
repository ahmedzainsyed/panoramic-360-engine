"""
PPE Compliance Detection Engine
================================
Real-time Personal Protective Equipment detection and worker safety tracking.

Detects:
  - Hard hat / helmet presence
  - Safety vest (high-visibility)
  - Safety gloves
  - Safety boots
  - Safety goggles / face shield

Compliance logic:
  - Per-worker PPE status assessment
  - Site-level compliance scoring
  - Non-compliance alerts with localization
  - Temporal tracking (DeepSORT)

Built on YOLOv8 fine-tuned on construction PPE datasets.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import structlog

logger = structlog.get_logger(__name__)


# ─── PPE Classes & Compliance Rules ──────────────────────────

class PPEItem(str, Enum):
    HARD_HAT       = "hard_hat"
    NO_HARD_HAT    = "no_hard_hat"
    SAFETY_VEST    = "safety_vest"
    NO_SAFETY_VEST = "no_safety_vest"
    SAFETY_GLOVES  = "safety_gloves"
    SAFETY_BOOTS   = "safety_boots"
    SAFETY_GOGGLES = "safety_goggles"
    PERSON         = "person"


# YOLOv8 class mapping for PPE model
PPE_CLASS_MAP = {
    0:  PPEItem.PERSON,
    1:  PPEItem.HARD_HAT,
    2:  PPEItem.NO_HARD_HAT,
    3:  PPEItem.SAFETY_VEST,
    4:  PPEItem.NO_SAFETY_VEST,
    5:  PPEItem.SAFETY_GLOVES,
    6:  PPEItem.SAFETY_BOOTS,
    7:  PPEItem.SAFETY_GOGGLES,
}

# Mandatory PPE requirements (site-level configurable)
MANDATORY_PPE_DEFAULT = {
    PPEItem.HARD_HAT,
    PPEItem.SAFETY_VEST,
}

# Color coding for visualization
PPE_COLORS = {
    PPEItem.HARD_HAT:       (0, 255, 0),    # Green
    PPEItem.NO_HARD_HAT:    (0, 0, 255),    # Red
    PPEItem.SAFETY_VEST:    (0, 255, 0),    # Green
    PPEItem.NO_SAFETY_VEST: (0, 0, 255),    # Red
    PPEItem.SAFETY_GLOVES:  (255, 165, 0),  # Orange
    PPEItem.SAFETY_BOOTS:   (255, 165, 0),  # Orange
    PPEItem.SAFETY_GOGGLES: (255, 165, 0),  # Orange
    PPEItem.PERSON:         (255, 255, 255), # White
}


@dataclass
class PPEDetection:
    """Single PPE item detection."""
    item: PPEItem
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    worker_id: Optional[str] = None


@dataclass
class WorkerPPEStatus:
    """Aggregated PPE compliance status for a single worker."""
    worker_id: str
    bbox: Tuple[float, float, float, float]   # worker bounding box
    detected_ppe: List[PPEItem] = field(default_factory=list)
    missing_ppe: List[PPEItem] = field(default_factory=list)
    violation_items: List[PPEItem] = field(default_factory=list)  # explicitly detected violations
    is_compliant: bool = False
    compliance_score: float = 0.0
    confidence: float = 0.0
    track_id: Optional[int] = None
    frame_timestamp: float = 0.0

    # Panorama-specific fields
    panorama_yaw: float = 0.0   # angular position in panorama
    panorama_id: Optional[str] = None


@dataclass
class PPEComplianceReport:
    """Site-level PPE compliance analysis report."""
    panorama_id: str
    total_workers: int
    compliant_workers: int
    non_compliant_workers: int
    compliance_rate: float           # 0.0 - 1.0
    worker_statuses: List[WorkerPPEStatus] = field(default_factory=list)
    violation_summary: Dict[str, int] = field(default_factory=dict)
    risk_level: str = "low"         # low | medium | high | critical
    alerts: List[str] = field(default_factory=list)
    inference_time_ms: float = 0.0
    timestamp: float = 0.0

    @property
    def non_compliance_rate(self) -> float:
        return 1.0 - self.compliance_rate


class PPEDetectionEngine:
    """
    Production PPE detection engine using YOLOv8.

    Architecture:
    1. Person detection across panorama tiles
    2. Per-worker PPE item detection
    3. Compliance assessment per mandatory rule set
    4. Temporal tracking with DeepSORT
    5. Panorama-space localization
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        detection_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        mandatory_ppe: Optional[set] = None,
        use_tracker: bool = True,
    ):
        self.detection_threshold = detection_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.mandatory_ppe = mandatory_ppe or MANDATORY_PPE_DEFAULT
        self.use_tracker = use_tracker

        self.model = self._load_yolo_model(model_path)
        self.tracker = self._init_tracker() if use_tracker else None

        logger.info(
            "ppe_engine_initialized",
            device=device,
            threshold=detection_threshold,
            mandatory_ppe=[p.value for p in self.mandatory_ppe],
        )

    def _load_yolo_model(self, model_path: Optional[str]):
        """Load YOLOv8 PPE detection model."""
        try:
            from ultralytics import YOLO

            if model_path and Path(model_path).exists():
                model = YOLO(model_path)
            else:
                # Use YOLOv8m as base (would be fine-tuned in production)
                model = YOLO("yolov8m.pt")
                logger.warning(
                    "ppe_using_base_yolo_not_finetuned",
                    message="Use fine-tuned PPE model for production",
                )
            return model

        except ImportError:
            logger.warning("ultralytics_not_installed_using_mock")
            return None

    def _init_tracker(self):
        """Initialize DeepSORT tracker for worker ID persistence."""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            return DeepSort(
                max_age=30,
                n_init=3,
                nms_max_overlap=1.0,
                max_cosine_distance=0.4,
                nn_budget=None,
            )
        except ImportError:
            logger.warning("deepsort_not_available_tracking_disabled")
            return None

    def analyze_panorama(
        self,
        image: np.ndarray,
        panorama_id: str,
        tiles_mode: bool = True,
    ) -> PPEComplianceReport:
        """
        Full PPE analysis pipeline for a 360° panorama.

        Process:
        1. Tile the panorama into perspective-corrected views
        2. Detect workers and PPE in each tile
        3. Merge detections back to panorama space
        4. Assess compliance per worker
        5. Generate site-level report

        Args:
            image: HxWxC equirectangular panorama
            panorama_id: unique panorama identifier
            tiles_mode: process via perspective tiles (recommended for accuracy)

        Returns:
            PPEComplianceReport with per-worker and site-level analysis
        """
        t0 = time.perf_counter()

        if tiles_mode:
            all_detections = self._detect_via_tiles(image)
        else:
            all_detections = self._detect_direct(image)

        # Group PPE detections by worker
        worker_statuses = self._associate_ppe_to_workers(all_detections, image.shape)

        # Apply temporal tracking
        if self.tracker:
            worker_statuses = self._apply_tracking(worker_statuses, image)

        # Assess compliance
        for worker in worker_statuses:
            self._assess_compliance(worker)

        # Build site report
        report = self._build_report(panorama_id, worker_statuses)
        report.inference_time_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "ppe_analysis_complete",
            panorama_id=panorama_id,
            workers=report.total_workers,
            compliant=report.compliant_workers,
            compliance_rate=f"{report.compliance_rate:.2%}",
            risk_level=report.risk_level,
            inference_ms=f"{report.inference_time_ms:.1f}",
        )

        return report

    def _detect_via_tiles(self, image: np.ndarray) -> List[PPEDetection]:
        """
        Run detection on perspective tiles of the panorama.
        Ensures accurate results near poles and avoids equirectangular distortion.
        """
        from ml.spherical_geometry.spherical_engine import SphericalGeometryEngine

        geo = SphericalGeometryEngine()
        tiles = geo.extract_perspective_tiles(
            image=image,
            fov_h=90.0,
            fov_v=90.0,
            tile_size=640,
            overlap_deg=20.0,
        )

        h, w = image.shape[:2]
        all_detections = []

        for tile in tiles:
            tile_dets = self._run_yolo_on_tile(tile.image)

            # Back-project detections to panorama coordinates
            for det in tile_dets:
                box_array = np.array([[det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]]])
                eq_boxes = geo.backproject_boxes_to_equirect(
                    boxes=box_array,
                    tile=tile,
                    equirect_width=w,
                    equirect_height=h,
                )
                if len(eq_boxes) > 0:
                    det.bbox = tuple(eq_boxes[0].tolist())
                    all_detections.append(det)

        return all_detections

    def _detect_direct(self, image: np.ndarray) -> List[PPEDetection]:
        """Direct detection on full equirectangular image (faster, less accurate)."""
        # Resize if too large
        h, w = image.shape[:2]
        target_w = min(w, 2048)
        if w > target_w:
            scale = target_w / w
            image = cv2.resize(image, (target_w, int(h * scale)))

        return self._run_yolo_on_tile(image, scale=target_w / w)

    def _run_yolo_on_tile(
        self,
        tile: np.ndarray,
        scale: float = 1.0,
    ) -> List[PPEDetection]:
        """Run YOLOv8 inference on a single image tile."""
        if self.model is None:
            return self._mock_detections(tile)

        results = self.model.predict(
            source=tile,
            conf=self.detection_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            stream=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                if cls_id not in PPE_CLASS_MAP:
                    continue
                conf = float(box.conf.item())
                xyxy = box.xyxy.squeeze().cpu().numpy()
                x1, y1, x2, y2 = (
                    float(xyxy[0]) / scale,
                    float(xyxy[1]) / scale,
                    float(xyxy[2]) / scale,
                    float(xyxy[3]) / scale,
                )
                detections.append(
                    PPEDetection(
                        item=PPE_CLASS_MAP[cls_id],
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return detections

    def _mock_detections(self, image: np.ndarray) -> List[PPEDetection]:
        """Generate mock detections for testing."""
        h, w = image.shape[:2]
        return [
            PPEDetection(
                item=PPEItem.PERSON,
                confidence=0.95,
                bbox=(w * 0.3, h * 0.4, w * 0.45, h * 0.8),
            ),
            PPEDetection(
                item=PPEItem.HARD_HAT,
                confidence=0.88,
                bbox=(w * 0.32, h * 0.38, w * 0.42, h * 0.45),
            ),
        ]

    def _associate_ppe_to_workers(
        self,
        detections: List[PPEDetection],
        image_shape: Tuple[int, int, int],
    ) -> List[WorkerPPEStatus]:
        """
        Associate PPE detections with detected workers via IoU-based matching.

        For each detected person, find overlapping PPE item detections.
        Uses centroid containment: PPE bbox centroid must be inside worker bbox.
        """
        # Separate persons from PPE items
        persons = [d for d in detections if d.item == PPEItem.PERSON]
        ppe_items = [d for d in detections if d.item != PPEItem.PERSON]

        worker_statuses = []

        for i, person in enumerate(persons):
            worker_id = str(uuid.uuid4())[:8]
            px1, py1, px2, py2 = person.bbox

            # Expand search region upward (PPE on head/torso)
            search_x1 = px1
            search_y1 = py1 - (py2 - py1) * 0.2  # expand 20% upward
            search_x2 = px2
            search_y2 = py2

            # Find PPE items whose center is within worker bounding box
            associated_ppe = []
            associated_violations = []

            for ppe in ppe_items:
                cx = (ppe.bbox[0] + ppe.bbox[2]) / 2
                cy = (ppe.bbox[1] + ppe.bbox[3]) / 2

                if search_x1 <= cx <= search_x2 and search_y1 <= cy <= search_y2:
                    if ppe.item in (PPEItem.NO_HARD_HAT, PPEItem.NO_SAFETY_VEST):
                        associated_violations.append(ppe.item)
                    else:
                        associated_ppe.append(ppe.item)

            # Compute angular position in panorama
            w = image_shape[1]
            center_x = (px1 + px2) / 2
            yaw = (center_x / w) * 360.0

            worker_statuses.append(
                WorkerPPEStatus(
                    worker_id=worker_id,
                    bbox=person.bbox,
                    detected_ppe=associated_ppe,
                    violation_items=associated_violations,
                    confidence=person.confidence,
                    panorama_yaw=yaw,
                )
            )

        return worker_statuses

    def _apply_tracking(
        self,
        workers: List[WorkerPPEStatus],
        image: np.ndarray,
    ) -> List[WorkerPPEStatus]:
        """Apply DeepSORT tracking to maintain consistent worker IDs."""
        if self.tracker is None:
            return workers

        try:
            # Prepare detections for tracker
            detections_for_tracker = []
            for w in workers:
                x1, y1, x2, y2 = w.bbox
                detections_for_tracker.append(
                    ([x1, y1, x2 - x1, y2 - y1], w.confidence, "person")
                )

            tracks = self.tracker.update_tracks(
                detections_for_tracker,
                frame=image,
            )

            # Match tracks to workers by bbox IoU
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_bbox = track.to_ltwh()
                track_x1 = track_bbox[0]
                track_y1 = track_bbox[1]
                track_x2 = track_bbox[0] + track_bbox[2]
                track_y2 = track_bbox[1] + track_bbox[3]

                best_iou = 0.0
                best_worker = None
                for worker in workers:
                    iou = self._compute_iou(
                        worker.bbox,
                        (track_x1, track_y1, track_x2, track_y2),
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_worker = worker

                if best_worker and best_iou > 0.3:
                    best_worker.track_id = track.track_id

        except Exception as e:
            logger.warning("tracking_failed", error=str(e))

        return workers

    def _assess_compliance(self, worker: WorkerPPEStatus) -> None:
        """Assess PPE compliance for a single worker."""
        # Check mandatory PPE items
        detected_mandatory = set(worker.detected_ppe) & self.mandatory_ppe
        violated_mandatory = set(worker.violation_items) & {
            PPEItem.NO_HARD_HAT, PPEItem.NO_SAFETY_VEST
        }

        # Missing = mandatory items not detected and no explicit violation detected
        missing = set()
        for req in self.mandatory_ppe:
            has_item = req in worker.detected_ppe
            # Check for explicit negative detection
            neg_map = {
                PPEItem.HARD_HAT: PPEItem.NO_HARD_HAT,
                PPEItem.SAFETY_VEST: PPEItem.NO_SAFETY_VEST,
            }
            has_violation = neg_map.get(req) in worker.violation_items
            if not has_item and not has_violation:
                missing.add(req)

        worker.missing_ppe = list(missing)
        worker.is_compliant = (
            len(violated_mandatory) == 0
            and len(missing) == 0
        )

        # Compliance score: fraction of mandatory items present
        n_mandatory = len(self.mandatory_ppe)
        n_present = sum(
            1 for req in self.mandatory_ppe
            if req in worker.detected_ppe
            and req not in {
                PPEItem.NO_HARD_HAT, PPEItem.NO_SAFETY_VEST
            }
        )
        worker.compliance_score = n_present / n_mandatory if n_mandatory > 0 else 1.0

    def _build_report(
        self,
        panorama_id: str,
        workers: List[WorkerPPEStatus],
    ) -> PPEComplianceReport:
        """Build site-level compliance report."""
        total = len(workers)
        compliant = sum(1 for w in workers if w.is_compliant)
        compliance_rate = compliant / total if total > 0 else 1.0

        # Violation summary
        violation_summary: Dict[str, int] = {}
        for worker in workers:
            for viol in worker.violation_items + worker.missing_ppe:
                key = viol.value if hasattr(viol, 'value') else str(viol)
                violation_summary[key] = violation_summary.get(key, 0) + 1

        # Risk level
        if compliance_rate >= 0.95:
            risk_level = "low"
        elif compliance_rate >= 0.75:
            risk_level = "medium"
        elif compliance_rate >= 0.50:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Alerts
        alerts = []
        if compliance_rate < 0.75:
            alerts.append(
                f"⚠️ Site PPE compliance below threshold: {compliance_rate:.0%}"
            )
        if "no_hard_hat" in violation_summary:
            n = violation_summary["no_hard_hat"]
            alerts.append(f"🚨 {n} worker(s) detected without hard hat")
        if "no_safety_vest" in violation_summary:
            n = violation_summary["no_safety_vest"]
            alerts.append(f"🚨 {n} worker(s) detected without safety vest")

        return PPEComplianceReport(
            panorama_id=panorama_id,
            total_workers=total,
            compliant_workers=compliant,
            non_compliant_workers=total - compliant,
            compliance_rate=compliance_rate,
            worker_statuses=workers,
            violation_summary=violation_summary,
            risk_level=risk_level,
            alerts=alerts,
            timestamp=time.time(),
        )

    @staticmethod
    def _compute_iou(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        """Compute Intersection-over-Union for two boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


def draw_ppe_annotations(
    image: np.ndarray,
    report: PPEComplianceReport,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw PPE detection annotations on the panorama image.

    Color coding:
    - Green box: fully compliant worker
    - Red box: non-compliant worker
    - Yellow text: PPE violation details
    """
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for worker in report.worker_statuses:
        x1, y1, x2, y2 = [int(v) for v in worker.bbox]
        color = (0, 255, 0) if worker.is_compliant else (0, 0, 255)
        thickness = 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label
        label = f"ID:{worker.track_id or worker.worker_id[:4]}"
        label += f" {worker.compliance_score:.0%}"
        cv2.putText(
            annotated, label,
            (x1, y1 - 8),
            font, font_scale, color, thickness,
        )

        # Violation text
        if worker.violation_items:
            violation_text = "Missing: " + ", ".join(
                v.value if hasattr(v, 'value') else str(v)
                for v in worker.violation_items[:2]
            )
            cv2.putText(
                annotated, violation_text,
                (x1, y2 + 20),
                font, font_scale * 0.8, (0, 165, 255), 1,
            )

    # Site compliance banner
    compliance_text = (
        f"Site Compliance: {report.compliance_rate:.0%} | "
        f"Workers: {report.total_workers} | "
        f"Risk: {report.risk_level.upper()}"
    )
    bg_color = {
        "low": (0, 200, 0),
        "medium": (0, 165, 255),
        "high": (0, 69, 255),
        "critical": (0, 0, 200),
    }.get(report.risk_level, (128, 128, 128))

    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), bg_color, -1)
    cv2.putText(
        annotated, compliance_text,
        (10, 28), font, 0.7, (255, 255, 255), 2,
    )

    return annotated


# Add missing import
from pathlib import Path
