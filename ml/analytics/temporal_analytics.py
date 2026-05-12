"""
Temporal Site Analytics Engine
Tracks construction progress, safety trends, worker patterns,
and site evolution across panorama sequences over time.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TemporalFrame:
    panorama_id: str
    timestamp: float
    semantic_mask: Optional[np.ndarray] = None
    ppe_compliance_rate: float = 0.0
    hazard_risk_score: float = 0.0
    worker_count: int = 0
    worker_heatmap: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class TemporalAnalysisResult:
    session_id: str
    frames: List[TemporalFrame]
    change_detection_map: np.ndarray       # (H, W) float32 - areas of change
    progress_map: np.ndarray               # (H, W) construction progress
    compliance_trend: List[float]          # PPE compliance over time
    hazard_trend: List[float]              # risk score over time
    worker_count_trend: List[int]
    activity_evolution: List[Dict]
    site_change_summary: Dict
    timeline_annotations: List[Dict]       # key change events
    inference_time_ms: float = 0.0


class TemporalAnalyticsEngine:
    """
    Analyzes temporal evolution of construction sites across multiple panoramas.

    Capabilities:
    - Semantic change detection between sessions
    - Construction progress estimation
    - Safety trend analysis
    - Worker activity timeline
    - Anomaly detection (sudden risk increases)
    """

    def __init__(
        self,
        change_threshold: float = 0.15,
        progress_classes: Optional[List[int]] = None,
        device: str = "cpu",
    ):
        self.change_threshold = change_threshold
        # Classes that indicate construction progress
        self.progress_classes = progress_classes or [7, 8, 9, 10]  # concrete, rebar, formwork, scaffold
        self.device = device

    def analyze_temporal_sequence(
        self,
        frames: List[TemporalFrame],
        session_id: str,
        reference_shape: Tuple[int, int] = (1024, 2048),
    ) -> TemporalAnalysisResult:
        """
        Analyze full temporal sequence.

        Args:
            frames: ordered list of TemporalFrame objects (chronological)
            session_id: unique session identifier
            reference_shape: (H, W) for all heatmaps/maps

        Returns:
            TemporalAnalysisResult with evolution maps and trend data
        """
        t0 = time.perf_counter()
        h, w = reference_shape

        if len(frames) < 2:
            logger.warning("temporal_needs_min_2_frames", count=len(frames))
            frames = frames + [frames[0]] if frames else []

        # Step 1: Change detection between consecutive frames
        change_maps = self._compute_change_maps(frames, h, w)
        accumulated_change = self._accumulate_changes(change_maps, h, w)

        # Step 2: Construction progress map
        progress_map = self._compute_progress_map(frames, h, w)

        # Step 3: Trend analysis
        compliance_trend = [f.ppe_compliance_rate for f in frames]
        hazard_trend = [f.hazard_risk_score for f in frames]
        worker_trend = [f.worker_count for f in frames]

        # Step 4: Activity evolution timeline
        activity_evolution = self._compute_activity_evolution(frames, h, w)

        # Step 5: Timeline annotations (key events)
        annotations = self._detect_timeline_events(frames, compliance_trend, hazard_trend)

        # Step 6: Site change summary
        summary = self._build_change_summary(frames, accumulated_change, compliance_trend, hazard_trend)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("temporal_analysis_complete", session=session_id,
                    frames=len(frames), events=len(annotations), ms=f"{elapsed_ms:.1f}")

        return TemporalAnalysisResult(
            session_id=session_id,
            frames=frames,
            change_detection_map=accumulated_change,
            progress_map=progress_map,
            compliance_trend=compliance_trend,
            hazard_trend=hazard_trend,
            worker_count_trend=worker_trend,
            activity_evolution=activity_evolution,
            site_change_summary=summary,
            timeline_annotations=annotations,
            inference_time_ms=elapsed_ms,
        )

    def _compute_change_maps(self, frames, h, w):
        change_maps = []
        for i in range(1, len(frames)):
            prev, curr = frames[i-1], frames[i]
            if prev.semantic_mask is None or curr.semantic_mask is None:
                change_maps.append(np.zeros((h, w), dtype=np.float32))
                continue
            prev_resized = cv2.resize(prev.semantic_mask.astype(np.float32), (w, h))
            curr_resized = cv2.resize(curr.semantic_mask.astype(np.float32), (w, h))
            diff = np.abs(curr_resized - prev_resized) / 20.0  # normalize by num classes
            change_maps.append(diff.astype(np.float32))
        return change_maps

    def _accumulate_changes(self, change_maps, h, w):
        if not change_maps:
            return np.zeros((h, w), dtype=np.float32)
        accumulated = np.stack(change_maps, axis=0).mean(axis=0)
        # Smooth and normalize
        accumulated = cv2.GaussianBlur(accumulated, (21, 21), sigmaX=8)
        if accumulated.max() > 0:
            accumulated /= accumulated.max()
        return accumulated

    def _compute_progress_map(self, frames, h, w):
        """Build map showing where construction materials appeared over time."""
        first_frame = frames[0]
        last_frame = frames[-1]
        progress_map = np.zeros((h, w), dtype=np.float32)
        if first_frame.semantic_mask is None or last_frame.semantic_mask is None:
            return progress_map
        first_resized = cv2.resize(first_frame.semantic_mask.astype(np.float32), (w, h))
        last_resized = cv2.resize(last_frame.semantic_mask.astype(np.float32), (w, h))
        # Where progress classes appeared
        for cls_id in self.progress_classes:
            appeared = (last_resized == cls_id) & (first_resized != cls_id)
            progress_map[appeared] = 1.0
        progress_map = cv2.GaussianBlur(progress_map, (21, 21), sigmaX=8)
        return progress_map

    def _compute_activity_evolution(self, frames, h, w):
        evolution = []
        for i, frame in enumerate(frames):
            entry = {
                "frame_index": i,
                "timestamp": frame.timestamp,
                "panorama_id": frame.panorama_id,
                "worker_count": frame.worker_count,
                "compliance_rate": frame.ppe_compliance_rate,
                "risk_score": frame.hazard_risk_score,
                "activity_level": self._classify_activity(frame),
            }
            evolution.append(entry)
        return evolution

    def _classify_activity(self, frame):
        if frame.worker_count == 0:
            return "inactive"
        elif frame.worker_count <= 3:
            return "low"
        elif frame.worker_count <= 10:
            return "medium"
        return "high"

    def _detect_timeline_events(self, frames, compliance_trend, hazard_trend):
        annotations = []
        for i in range(1, len(frames)):
            # Compliance drop
            delta_comp = compliance_trend[i-1] - compliance_trend[i]
            if delta_comp > 0.2:
                annotations.append({
                    "frame_index": i,
                    "timestamp": frames[i].timestamp,
                    "event_type": "compliance_drop",
                    "severity": "high",
                    "description": f"PPE compliance dropped by {delta_comp:.0%}",
                    "panorama_id": frames[i].panorama_id,
                })
            # Risk spike
            delta_risk = hazard_trend[i] - hazard_trend[i-1]
            if delta_risk > 0.2:
                annotations.append({
                    "frame_index": i,
                    "timestamp": frames[i].timestamp,
                    "event_type": "risk_spike",
                    "severity": "critical" if hazard_trend[i] > 0.7 else "medium",
                    "description": f"Site risk increased by {delta_risk:.2f}",
                    "panorama_id": frames[i].panorama_id,
                })
            # Worker count spike
            if frames[i].worker_count > frames[i-1].worker_count * 2 and frames[i].worker_count > 5:
                annotations.append({
                    "frame_index": i,
                    "timestamp": frames[i].timestamp,
                    "event_type": "worker_surge",
                    "severity": "info",
                    "description": f"Worker count increased from {frames[i-1].worker_count} to {frames[i].worker_count}",
                    "panorama_id": frames[i].panorama_id,
                })
        return annotations

    def _build_change_summary(self, frames, change_map, compliance_trend, hazard_trend):
        n = len(frames)
        if n == 0:
            return {}
        change_area = float((change_map > self.change_threshold).sum()) / change_map.size
        return {
            "total_frames": n,
            "duration_seconds": frames[-1].timestamp - frames[0].timestamp if n > 1 else 0,
            "change_area_percent": change_area * 100.0,
            "avg_compliance_rate": float(np.mean(compliance_trend)),
            "min_compliance_rate": float(np.min(compliance_trend)),
            "max_risk_score": float(np.max(hazard_trend)),
            "avg_worker_count": float(np.mean([f.worker_count for f in frames])),
            "peak_worker_count": int(max(f.worker_count for f in frames)),
            "compliance_trend": "improving" if compliance_trend[-1] > compliance_trend[0] else "declining",
            "risk_trend": "increasing" if hazard_trend[-1] > hazard_trend[0] else "stable",
        }

    def compare_panoramas(
        self,
        panorama_a: np.ndarray,
        panorama_b: np.ndarray,
    ) -> np.ndarray:
        """
        Compute visual change map between two panoramas.
        Returns (H, W) float32 normalized difference map.
        """
        if panorama_a.shape != panorama_b.shape:
            panorama_b = cv2.resize(panorama_b, (panorama_a.shape[1], panorama_a.shape[0]))
        # Convert to LAB color space for perceptual difference
        lab_a = cv2.cvtColor(panorama_a, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_b = cv2.cvtColor(panorama_b, cv2.COLOR_RGB2LAB).astype(np.float32)
        diff = np.sqrt(np.sum((lab_a - lab_b)**2, axis=2))
        diff = cv2.GaussianBlur(diff, (21, 21), sigmaX=5)
        if diff.max() > 0:
            diff /= diff.max()
        return diff.astype(np.float32)
