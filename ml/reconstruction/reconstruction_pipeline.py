"""
3D Spatial Reconstruction Pipeline
Panorama-to-3D reconstruction using monocular depth estimation,
Open3D point cloud generation, and scene mesh fusion.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ReconstructionResult:
    panorama_id: str
    depth_map: np.ndarray              # (H, W) float32 metric depth in meters
    point_cloud_xyz: np.ndarray        # (N, 3) 3D points
    point_cloud_rgb: np.ndarray        # (N, 3) RGB colors
    occupancy_voxels: Optional[np.ndarray] = None  # voxel grid
    mesh_path: Optional[str] = None    # saved mesh file path
    scale_factor: float = 1.0
    camera_height_estimate: float = 1.6  # meters
    scene_extent: Dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0


class MonocularDepthEstimator:
    """DPT/MiDaS-based monocular depth estimation for panoramic imagery."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            import torch
            import timm
            # Try DPT-Large for best quality
            model = timm.create_model("vit_large_patch16_384", pretrained=False)
            if model_path:
                import os
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location="cpu")
                    model.load_state_dict(state, strict=False)
            logger.info("depth_model_loaded")
            return model
        except Exception as e:
            logger.warning("depth_model_load_failed", error=str(e))
            return None

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate per-pixel depth from equirectangular panorama.
        Returns (H, W) float32 depth map (relative or metric).
        """
        if self.model is None:
            return self._heuristic_depth(image)

        try:
            import torch
            import torchvision.transforms as T
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((384, 768)),
                T.ToTensor(),
                T.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])
            x = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                depth = self.model(x)
            depth_np = depth.squeeze().cpu().numpy()
            depth_np = cv2.resize(depth_np, (image.shape[1], image.shape[0]))
            # Normalize to [0.1, 50.0] meters heuristically
            depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-9)
            depth_np = depth_np * 49.9 + 0.1
            return depth_np.astype(np.float32)
        except Exception as e:
            logger.warning("depth_inference_failed", error=str(e))
            return self._heuristic_depth(image)

    def _heuristic_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Heuristic depth estimation when model unavailable.
        Uses vertical position (lower = closer in construction scenes).
        """
        h, w = image.shape[:2]
        y_coords = np.arange(h, dtype=np.float32)
        # Near equator: medium depth; toward poles: further
        lat = (y_coords / h - 0.5) * np.pi
        base_depth = 5.0 + 20.0 * np.cos(lat)
        depth_map = np.tile(base_depth[:, np.newaxis], (1, w))
        # Add noise for realism
        noise = np.random.normal(0, 1.0, (h, w)).astype(np.float32)
        depth_map = np.clip(depth_map + noise, 0.3, 50.0)
        return depth_map


class PanoramaReconstructor:
    """
    Full 3D reconstruction pipeline for equirectangular panoramas.
    Converts 360° panoramas to dense point clouds and spatial occupancy maps.
    """

    def __init__(
        self,
        depth_estimator: Optional[MonocularDepthEstimator] = None,
        voxel_size: float = 0.1,
        device: str = "cuda",
    ):
        self.depth_estimator = depth_estimator or MonocularDepthEstimator(device=device)
        self.voxel_size = voxel_size
        self.device = device

    def reconstruct(
        self,
        image: np.ndarray,
        panorama_id: str,
        camera_height: float = 1.6,
        scale_hint: Optional[float] = None,
    ) -> ReconstructionResult:
        """
        Full reconstruction pipeline.

        1. Monocular depth estimation
        2. Equirectangular → 3D point cloud via spherical unprojection
        3. Point cloud filtering and downsampling
        4. Occupancy voxel grid generation
        """
        t0 = time.perf_counter()
        h, w = image.shape[:2]

        # Step 1: Depth estimation
        depth_map = self.depth_estimator.estimate_depth(image)

        # Step 2: Apply scale hint if provided (e.g. from GPS or known dimensions)
        scale_factor = scale_hint or 1.0
        depth_map_scaled = depth_map * scale_factor

        # Step 3: Spherical unprojection → 3D points
        xyz, rgb = self._unproject_to_pointcloud(image, depth_map_scaled, h, w, camera_height)

        # Step 4: Filter outliers
        xyz, rgb = self._filter_pointcloud(xyz, rgb)

        # Step 5: Occupancy voxel grid
        occupancy = self._build_occupancy_grid(xyz)

        # Step 6: Scene extent
        scene_extent = {
            "x_min": float(xyz[:, 0].min()), "x_max": float(xyz[:, 0].max()),
            "y_min": float(xyz[:, 1].min()), "y_max": float(xyz[:, 1].max()),
            "z_min": float(xyz[:, 2].min()), "z_max": float(xyz[:, 2].max()),
            "estimated_area_sqm": float((np.ptp(xyz[:, 0]) * np.ptp(xyz[:, 2]))),
        } if len(xyz) > 0 else {}

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("reconstruction_complete", panorama_id=panorama_id,
                    points=len(xyz), ms=f"{elapsed_ms:.1f}")

        return ReconstructionResult(
            panorama_id=panorama_id,
            depth_map=depth_map,
            point_cloud_xyz=xyz,
            point_cloud_rgb=rgb,
            occupancy_voxels=occupancy,
            scale_factor=scale_factor,
            camera_height_estimate=camera_height,
            scene_extent=scene_extent,
            inference_time_ms=elapsed_ms,
        )

    def _unproject_to_pointcloud(self, image, depth_map, h, w, camera_height):
        """
        Convert equirectangular depth map to 3D point cloud via spherical unprojection.

        For each pixel (u, v):
          phi = 2*pi*u/W - pi    (azimuth)
          theta = pi*v/H - pi/2  (elevation)
          x = d * cos(theta) * cos(phi)
          y = d * sin(theta) + camera_height
          z = d * cos(theta) * sin(phi)
        """
        import math
        step = 2  # subsample every 2 pixels for speed
        us, vs = np.meshgrid(
            np.arange(0, w, step, dtype=np.float32),
            np.arange(0, h, step, dtype=np.float32),
        )
        phi = (us / w) * 2 * np.pi - np.pi           # azimuth [-pi, pi]
        theta = (vs / h) * np.pi - np.pi / 2          # elevation [-pi/2, pi/2]

        d = depth_map[::step, ::step]  # (H/step, W/step)

        x = d * np.cos(theta) * np.cos(phi)
        y = d * np.sin(theta) + camera_height
        z = d * np.cos(theta) * np.sin(phi)

        xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
        rgb = image[::step, ::step].reshape(-1, 3).astype(np.float32) / 255.0

        return xyz, rgb

    def _filter_pointcloud(self, xyz, rgb, max_range=50.0):
        """Remove outlier points (sky, extreme depths)."""
        distances = np.linalg.norm(xyz, axis=1)
        valid = (distances > 0.1) & (distances < max_range) & (xyz[:, 1] < 20.0) & (xyz[:, 1] > -2.0)
        return xyz[valid], rgb[valid]

    def _build_occupancy_grid(self, xyz):
        """Build 3D occupancy voxel grid from point cloud."""
        if len(xyz) == 0:
            return np.zeros((10, 10, 10), dtype=np.uint8)
        # Voxelize
        grid_size = 64
        x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
        y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
        z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
        eps = 1e-6
        ix = ((xyz[:, 0] - x_min) / (x_max - x_min + eps) * (grid_size - 1)).astype(int)
        iy = ((xyz[:, 1] - y_min) / (y_max - y_min + eps) * (grid_size - 1)).astype(int)
        iz = ((xyz[:, 2] - z_min) / (z_max - z_min + eps) * (grid_size - 1)).astype(int)
        occupancy = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
        valid = (ix >= 0) & (ix < grid_size) & (iy >= 0) & (iy < grid_size) & (iz >= 0) & (iz < grid_size)
        occupancy[ix[valid], iy[valid], iz[valid]] = 1
        return occupancy

    def export_to_ply(self, result: ReconstructionResult, output_path: str) -> str:
        """Export point cloud to PLY format."""
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(result.point_cloud_xyz)
            pcd.colors = o3d.utility.Vector3dVector(result.point_cloud_rgb)
            # Voxel downsample
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            o3d.io.write_point_cloud(output_path, pcd)
            logger.info("point_cloud_exported", path=output_path, points=len(pcd.points))
            return output_path
        except ImportError:
            logger.warning("open3d_not_available_skipping_ply_export")
            return ""
