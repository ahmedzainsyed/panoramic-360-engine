"""
Spatial Navigation Overlay Engine
Generates walkable path estimation, navigation overlays, route planning,
and site accessibility maps from segmentation and depth data.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class NavigationResult:
    """Navigation overlay outputs."""
    panorama_id: str
    walkable_mask: np.ndarray          # (H, W) bool
    walkable_distance_map: np.ndarray  # (H, W) EDT distance to obstacles
    navigation_overlay: np.ndarray     # (H, W, 3) RGB visualization
    route_graph_nodes: List[Dict]      # skeleton waypoints
    accessibility_score: float         # 0-1 site accessibility
    blocked_zones: List[Dict]
    recommended_paths: List[Dict]
    inference_time_ms: float = 0.0


class NavigationOverlayEngine:
    """
    Generates navigable route overlays and spatial navigation maps.

    Uses segmentation results to identify:
    - Walkable floor regions
    - Obstacle boundaries
    - Safe passage corridors
    - Emergency exit paths
    """

    def __init__(
        self,
        walkable_class_ids: Optional[List[int]] = None,
        obstacle_class_ids: Optional[List[int]] = None,
        min_path_width_pixels: int = 40,
        device: str = "cpu",
    ):
        # Default walkable classes from construction segmentation
        self.walkable_class_ids = walkable_class_ids or [4, 6, 2]  # floor, walkable_path, ground
        self.obstacle_class_ids = obstacle_class_ids or [3, 10, 12, 13, 15, 16, 17, 19, 20]
        self.min_path_width_pixels = min_path_width_pixels
        self.device = device

    def generate_navigation_overlay(
        self,
        image: np.ndarray,
        panorama_id: str,
        seg_result=None,
        depth_map: Optional[np.ndarray] = None,
    ) -> NavigationResult:
        """
        Generate full navigation overlay for a panorama.

        Pipeline:
        1. Extract walkable mask from segmentation
        2. Apply depth-based refinement if available
        3. Compute distance transform for corridor widths
        4. Skeletonize walkable region for route graph
        5. Generate visual overlay
        """
        t0 = time.perf_counter()
        h, w = image.shape[:2]

        # Step 1: Build walkable mask
        if seg_result is not None:
            walkable_mask = self._build_walkable_mask_from_seg(seg_result, h, w)
        else:
            walkable_mask = self._estimate_walkable_from_image(image)

        # Step 2: Depth refinement
        if depth_map is not None:
            walkable_mask = self._refine_with_depth(walkable_mask, depth_map)

        # Step 3: Morphological cleanup
        walkable_mask = self._clean_walkable_mask(walkable_mask)

        # Step 4: Distance transform (EDT)
        dist_map = distance_transform_edt(walkable_mask).astype(np.float32)
        if dist_map.max() > 0:
            dist_map_norm = dist_map / dist_map.max()
        else:
            dist_map_norm = dist_map

        # Step 5: Route skeleton extraction
        skeleton_nodes = self._extract_route_skeleton(walkable_mask, dist_map)

        # Step 6: Find recommended paths (widest corridors)
        recommended_paths = self._find_recommended_paths(dist_map, walkable_mask)

        # Step 7: Blocked zones
        obstacle_mask = ~walkable_mask
        blocked_zones = self._find_blocked_zones(obstacle_mask, h, w)

        # Step 8: Accessibility score
        walkable_fraction = float(walkable_mask.sum()) / (h * w)
        min_path_width = float(dist_map[walkable_mask].mean()) if walkable_mask.any() else 0
        accessibility_score = min(1.0, walkable_fraction * 2.0 * (min_path_width / 50.0 + 0.5))

        # Step 9: Navigation visualization overlay
        overlay = self._build_navigation_overlay(image, walkable_mask, dist_map, skeleton_nodes, recommended_paths)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("navigation_generated", panorama_id=panorama_id,
                    walkable_pct=f"{walkable_fraction:.1%}", nodes=len(skeleton_nodes),
                    access_score=f"{accessibility_score:.2f}", ms=f"{elapsed_ms:.1f}")

        return NavigationResult(
            panorama_id=panorama_id,
            walkable_mask=walkable_mask,
            walkable_distance_map=dist_map_norm,
            navigation_overlay=overlay,
            route_graph_nodes=skeleton_nodes,
            accessibility_score=accessibility_score,
            blocked_zones=blocked_zones,
            recommended_paths=recommended_paths,
            inference_time_ms=elapsed_ms,
        )

    def _build_walkable_mask_from_seg(self, seg_result, height, width):
        walkable = np.zeros((height, width), dtype=bool)
        for cls_id in self.walkable_class_ids:
            walkable |= (seg_result.semantic_mask == cls_id)
        # Exclude hazards
        for cls_id in [12, 13, 19, 20]:
            walkable &= ~(seg_result.semantic_mask == cls_id)
        return walkable

    def _estimate_walkable_from_image(self, image):
        """Heuristic: lower half of image tends to be floor/ground."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h//2:, :] = True
        return mask

    def _refine_with_depth(self, walkable_mask, depth_map):
        """Remove walkable pixels that have high depth variance (obstacles)."""
        if depth_map.shape != walkable_mask.shape:
            depth_map = cv2.resize(depth_map.astype(np.float32), 
                                   (walkable_mask.shape[1], walkable_mask.shape[0]))
        # Compute local depth variance
        depth_norm = depth_map / (depth_map.max() + 1e-9)
        depth_variance = cv2.Laplacian(depth_norm.astype(np.float32), cv2.CV_32F)
        depth_variance = np.abs(depth_variance)
        # High variance = obstacle
        obstacle_from_depth = depth_variance > 0.1
        return walkable_mask & ~obstacle_from_depth

    def _clean_walkable_mask(self, mask):
        mask_u8 = mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        return mask_u8.astype(bool)

    def _extract_route_skeleton(self, walkable_mask, dist_map):
        """Extract skeleton waypoints as navigation graph nodes."""
        from skimage.morphology import skeletonize
        skeleton = skeletonize(walkable_mask)
        ys, xs = np.where(skeleton)
        if len(xs) == 0:
            return []
        # Subsample skeleton points
        step = max(1, len(xs) // 50)
        nodes = []
        for i in range(0, len(xs), step):
            x, y = int(xs[i]), int(ys[i])
            width_here = float(dist_map[y, x])
            nodes.append({
                "id": f"node_{i}",
                "x": x, "y": y,
                "path_width": width_here,
                "yaw": (x / walkable_mask.shape[1]) * 360.0,
                "passable": width_here >= self.min_path_width_pixels * 0.3,
            })
        return nodes

    def _find_recommended_paths(self, dist_map, walkable_mask):
        """Find widest corridor paths (safest routes)."""
        # High-distance pixels = corridor centers
        wide_mask = (dist_map > np.percentile(dist_map[walkable_mask], 80)
                     if walkable_mask.any() else np.zeros_like(dist_map, dtype=bool))
        wide_mask = wide_mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wide_mask, 8)
        h, w = dist_map.shape
        paths = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < 200:
                continue
            cx = float(centroids[label_id, 0])
            cy = float(centroids[label_id, 1])
            paths.append({
                "path_id": f"path_{label_id}",
                "centroid": [cx, cy],
                "area": area,
                "mean_width": float(dist_map[labels == label_id].mean()),
                "yaw": (cx / w) * 360.0,
                "quality": "primary" if area > 1000 else "secondary",
            })
        return sorted(paths, key=lambda p: p["mean_width"], reverse=True)[:5]

    def _find_blocked_zones(self, obstacle_mask, height, width):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        dilated = cv2.dilate(obstacle_mask.astype(np.uint8), kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, 8)
        min_area = int(height * width * 0.01)
        zones = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x1 = int(stats[label_id, cv2.CC_STAT_LEFT])
            y1 = int(stats[label_id, cv2.CC_STAT_TOP])
            bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            cx = float(centroids[label_id, 0])
            zones.append({"bbox": [x1, y1, x1+bw, y1+bh], "centroid": [cx, float(centroids[label_id, 1])],
                          "area": area, "yaw": (cx/width)*360.0})
        return zones

    def _build_navigation_overlay(self, image, walkable_mask, dist_map, nodes, paths):
        overlay = image.copy().astype(np.float32)
        # Walkable region: green tint
        walkable_color = np.array([0, 220, 0], dtype=np.float32)
        overlay[walkable_mask] = overlay[walkable_mask] * 0.6 + walkable_color * 0.4
        # Distance map gradient on walkable region
        if dist_map.max() > 0:
            dist_norm = (dist_map / dist_map.max() * 255).astype(np.uint8)
            dist_colored = cv2.applyColorMap(dist_norm, cv2.COLORMAP_COOL)
            dist_colored_rgb = cv2.cvtColor(dist_colored, cv2.COLOR_BGR2RGB).astype(np.float32)
            overlay[walkable_mask] = overlay[walkable_mask] * 0.7 + dist_colored_rgb[walkable_mask] * 0.3
        result = np.clip(overlay, 0, 255).astype(np.uint8)
        # Draw skeleton nodes
        for node in nodes:
            color = (0, 255, 100) if node.get("passable", True) else (255, 100, 0)
            cv2.circle(result, (node["x"], node["y"]), 3, color, -1)
        return result
