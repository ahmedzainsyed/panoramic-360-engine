"""
Spatial Occupancy Engine
Generates worker density heatmaps, movement zones, congestion analysis,
and spatial utilization analytics from panoramic imagery sequences.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class OccupancyFrame:
    """Single-frame occupancy snapshot."""
    panorama_id: str
    timestamp: float
    worker_positions: np.ndarray          # (N, 2) x,y in panorama coords
    worker_ids: List[str]
    worker_boxes: np.ndarray              # (N, 4) bounding boxes
    frame_index: int = 0


@dataclass
class OccupancyResult:
    """Full occupancy analysis result."""
    session_id: str
    density_heatmap: np.ndarray           # (H, W) float32 density
    temporal_heatmap: np.ndarray          # (H, W) accumulated over time
    activity_zones: List[Dict]            # clustered activity regions
    congestion_map: np.ndarray            # (H, W) congestion score
    spatial_utilization: float            # 0-1 fraction of site used
    worker_count_stats: Dict[str, float]  # min, max, mean worker counts
    idle_zones: List[Dict]                # zones with low movement
    high_traffic_zones: List[Dict]        # frequently visited zones
    inference_time_ms: float = 0.0


class SpatialOccupancyEngine:
    """
    Analyzes spatial occupancy patterns across panoramic imagery sequences.

    Key capabilities:
    - Gaussian KDE-based density estimation
    - Temporal accumulation for long-term heatmaps
    - K-means clustering for activity zone identification
    - Congestion detection via density thresholding
    - Idle zone analysis via temporal variance
    """

    def __init__(
        self,
        heatmap_sigma: float = 30.0,
        congestion_threshold: float = 0.6,
        min_zone_area_fraction: float = 0.005,
        temporal_decay: float = 0.95,
        device: str = "cpu",
    ):
        self.heatmap_sigma = heatmap_sigma
        self.congestion_threshold = congestion_threshold
        self.min_zone_area_fraction = min_zone_area_fraction
        self.temporal_decay = temporal_decay
        self.device = device

        # Persistent temporal accumulator (reset per session)
        self._temporal_accumulator: Optional[np.ndarray] = None
        self._frame_count = 0

    def analyze_frame(
        self,
        frame: OccupancyFrame,
        panorama_shape: Tuple[int, int],
        session_id: str,
    ) -> OccupancyResult:
        """Analyze a single frame's occupancy."""
        t0 = time.perf_counter()
        h, w = panorama_shape

        # Initialize temporal accumulator on first frame
        if self._temporal_accumulator is None or self._temporal_accumulator.shape != (h, w):
            self._temporal_accumulator = np.zeros((h, w), dtype=np.float32)
            self._frame_count = 0

        # Build per-frame density heatmap
        density_heatmap = self._build_density_heatmap(
            frame.worker_positions, h, w
        )

        # Update temporal accumulator with exponential decay
        self._temporal_accumulator = (
            self._temporal_accumulator * self.temporal_decay + density_heatmap
        )
        self._frame_count += 1

        # Normalize temporal heatmap
        temporal_heatmap = self._temporal_accumulator.copy()
        if temporal_heatmap.max() > 0:
            temporal_heatmap /= temporal_heatmap.max()

        # Congestion map: density above threshold
        congestion_map = (density_heatmap > self.congestion_threshold).astype(np.float32)
        congestion_map = gaussian_filter(congestion_map, sigma=10)

        # Activity zones via clustering
        activity_zones = self._identify_activity_zones(
            temporal_heatmap, h, w, frame.worker_ids
        )

        # High-traffic and idle zones
        high_traffic = self._find_high_traffic_zones(temporal_heatmap, h, w)
        idle_zones = self._find_idle_zones(temporal_heatmap, h, w)

        # Spatial utilization
        used_area = (temporal_heatmap > 0.05).sum()
        spatial_utilization = float(used_area / (h * w))

        worker_count = len(frame.worker_positions)
        worker_count_stats = {
            "current": float(worker_count),
            "density_peak": float(density_heatmap.max()),
            "mean_density": float(density_heatmap.mean()),
        }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("occupancy_analyzed", session=session_id,
                    workers=worker_count, utilization=f"{spatial_utilization:.2%}",
                    ms=f"{elapsed_ms:.1f}")

        return OccupancyResult(
            session_id=session_id,
            density_heatmap=density_heatmap,
            temporal_heatmap=temporal_heatmap,
            activity_zones=activity_zones,
            congestion_map=congestion_map,
            spatial_utilization=spatial_utilization,
            worker_count_stats=worker_count_stats,
            idle_zones=idle_zones,
            high_traffic_zones=high_traffic,
            inference_time_ms=elapsed_ms,
        )

    def analyze_session(
        self,
        frames: List[OccupancyFrame],
        panorama_shape: Tuple[int, int],
        session_id: str,
    ) -> OccupancyResult:
        """Analyze full session across multiple frames."""
        self.reset_temporal_state()
        result = None
        for frame in frames:
            result = self.analyze_frame(frame, panorama_shape, session_id)
        return result

    def _build_density_heatmap(
        self,
        positions: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Build Gaussian KDE density heatmap from worker positions.
        Each worker contributes a Gaussian blob of sigma=heatmap_sigma pixels.
        """
        heatmap = np.zeros((height, width), dtype=np.float32)
        if len(positions) == 0:
            return heatmap

        for pos in positions:
            x = int(np.clip(pos[0], 0, width - 1))
            y = int(np.clip(pos[1], 0, height - 1))
            heatmap[y, x] += 1.0

        # Apply Gaussian smoothing
        heatmap = gaussian_filter(heatmap, sigma=self.heatmap_sigma)

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    def _identify_activity_zones(
        self,
        heatmap: np.ndarray,
        height: int,
        width: int,
        worker_ids: List[str],
    ) -> List[Dict]:
        """Identify distinct activity clusters using connected component analysis."""
        threshold = np.percentile(heatmap[heatmap > 0], 75) if heatmap.max() > 0 else 0.5
        binary = (heatmap > threshold).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        min_area = int(height * width * self.min_zone_area_fraction)

        zones = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x1 = int(stats[label_id, cv2.CC_STAT_LEFT])
            y1 = int(stats[label_id, cv2.CC_STAT_TOP])
            bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[label_id, 0]), float(centroids[label_id, 1])
            zone_mask = (labels == label_id)
            intensity = float(heatmap[zone_mask].mean())
            yaw = (cx / width) * 360.0

            zones.append({
                "zone_id": f"activity_{label_id}",
                "bbox": [x1, y1, x1 + bw, y1 + bh],
                "centroid": [cx, cy],
                "area_pixels": area,
                "area_percent": area / (height * width) * 100.0,
                "mean_intensity": intensity,
                "panorama_yaw": yaw,
                "activity_level": "high" if intensity > 0.7 else "medium" if intensity > 0.4 else "low",
            })

        return sorted(zones, key=lambda z: z["mean_intensity"], reverse=True)

    def _find_high_traffic_zones(self, heatmap, height, width):
        high_mask = (heatmap > 0.7).astype(np.uint8)
        if high_mask.sum() == 0:
            return []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high_mask, 8)
        zones = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < 100:
                continue
            zones.append({
                "type": "high_traffic",
                "centroid": [float(centroids[label_id, 0]), float(centroids[label_id, 1])],
                "area": area,
                "yaw": (float(centroids[label_id, 0]) / width) * 360.0,
            })
        return zones

    def _find_idle_zones(self, heatmap, height, width):
        idle_mask = (heatmap < 0.02).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(idle_mask, 8)
        min_area = int(height * width * 0.02)
        zones = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            zones.append({
                "type": "idle",
                "centroid": [float(centroids[label_id, 0]), float(centroids[label_id, 1])],
                "area": area,
                "yaw": (float(centroids[label_id, 0]) / width) * 360.0,
            })
        return zones[:5]  # top 5 idle zones

    def reset_temporal_state(self):
        """Reset temporal accumulator for new session."""
        self._temporal_accumulator = None
        self._frame_count = 0

    @staticmethod
    def positions_from_boxes(boxes: np.ndarray) -> np.ndarray:
        """Extract centroid positions from bounding boxes."""
        if len(boxes) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        return np.stack([cx, cy], axis=1)


def blend_heatmap_on_panorama(
    panorama: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_HOT,
) -> np.ndarray:
    """Overlay density heatmap on panorama image."""
    heatmap_resized = cv2.resize(heatmap, (panorama.shape[1], panorama.shape[0]))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    mask = heatmap_resized > 0.05
    result = panorama.copy().astype(np.float32)
    result[mask] = result[mask] * (1 - alpha) + colored_rgb[mask].astype(np.float32) * alpha
    return result.astype(np.uint8)
