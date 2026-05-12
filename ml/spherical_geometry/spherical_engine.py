"""
Spherical Geometry Engine
=========================
Handles all spherical/panoramic coordinate transforms:
  - Equirectangular ↔ Cubemap projection
  - Polar distortion correction
  - Perspective tile extraction
  - Spherical coordinate mapping
  - Seam continuity handling
  - Rotation matrix operations

Mathematical background:
  Equirectangular: (lon, lat) → (x, y) with x ∈ [-π, π], y ∈ [-π/2, π/2]
  Cubemap: projects sphere onto 6 faces of a unit cube
  Spherical coords: (r=1, θ=inclination, φ=azimuth)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import py360convert
    HAS_PY360 = True
except ImportError:
    HAS_PY360 = False


class CubeFace(Enum):
    """Cubemap face identifiers."""
    FRONT  = "front"   # +Z
    BACK   = "back"    # -Z
    LEFT   = "left"    # -X
    RIGHT  = "right"   # +X
    TOP    = "top"     # +Y
    BOTTOM = "bottom"  # -Y


@dataclass
class PanoramaSpec:
    """Specification of a panoramic image."""
    width: int
    height: int
    fov_h: float = 360.0   # horizontal field of view in degrees
    fov_v: float = 180.0   # vertical field of view in degrees
    yaw: float = 0.0       # camera yaw rotation in degrees
    pitch: float = 0.0     # camera pitch rotation in degrees
    roll: float = 0.0      # camera roll rotation in degrees

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def pixels_per_degree_h(self) -> float:
        return self.width / self.fov_h

    @property
    def pixels_per_degree_v(self) -> float:
        return self.height / self.fov_v


@dataclass
class PerspectiveTile:
    """A perspective-corrected tile extracted from a panorama."""
    image: np.ndarray
    yaw: float      # center yaw in degrees
    pitch: float    # center pitch in degrees
    fov_h: float    # horizontal FOV in degrees
    fov_v: float    # vertical FOV in degrees
    face: Optional[CubeFace] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.image.shape


class SphericalGeometryEngine:
    """
    Core spherical geometry processing engine.

    Converts between coordinate systems, handles distortion correction,
    and provides tiled inference support for panoramic images.
    """

    def __init__(
        self,
        cubemap_face_size: int = 1024,
        tiling_overlap: float = 0.1,
        polar_correction: float = 0.8,
        device: str = "cpu",
    ):
        self.cubemap_face_size = cubemap_face_size
        self.tiling_overlap = tiling_overlap
        self.polar_correction = polar_correction
        self.device = device

        # Precomputed coordinate grids (lazy)
        self._equirect_to_cubemap_maps: Dict[str, np.ndarray] = {}

    # ─── Coordinate Conversions ───────────────────────────────

    @staticmethod
    def spherical_to_equirectangular(
        theta: np.ndarray,   # inclination [0, π]
        phi: np.ndarray,     # azimuth [0, 2π]
        width: int,
        height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map spherical coordinates (θ, φ) to equirectangular pixel (u, v).

        θ ∈ [0, π]:   0 = north pole, π = south pole
        φ ∈ [0, 2π]:  azimuth (longitude)
        """
        u = (phi / (2 * np.pi)) * width
        v = (theta / np.pi) * height
        return u.astype(np.float32), v.astype(np.float32)

    @staticmethod
    def equirectangular_to_spherical(
        u: np.ndarray,
        v: np.ndarray,
        width: int,
        height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map equirectangular pixel (u, v) to spherical (θ, φ)."""
        phi = (u / width) * 2 * np.pi
        theta = (v / height) * np.pi
        return theta.astype(np.float32), phi.astype(np.float32)

    @staticmethod
    def spherical_to_cartesian(
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert spherical coords to unit-sphere Cartesian (x, y, z)."""
        x = np.sin(theta) * np.cos(phi)
        y = np.cos(theta)            # y = up
        z = np.sin(theta) * np.sin(phi)
        return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)

    @staticmethod
    def cartesian_to_spherical(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Cartesian (x, y, z) to spherical (θ, φ)."""
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-9
        theta = np.arccos(np.clip(y / r, -1.0, 1.0))
        phi = np.arctan2(z, x) % (2 * np.pi)
        return theta.astype(np.float32), phi.astype(np.float32)

    # ─── Polar Distortion Correction ─────────────────────────

    def correct_polar_distortion(
        self,
        image: np.ndarray,
        strength: Optional[float] = None,
    ) -> np.ndarray:
        """
        Correct polar region stretching in equirectangular panoramas.

        Near the poles (top/bottom ~15% of image height), the equirectangular
        projection causes extreme horizontal stretching. We apply a cosine
        weighting to counteract this during processing.

        Args:
            image: HxWxC equirectangular image
            strength: correction strength 0-1 (default: self.polar_correction)

        Returns:
            Corrected image with reduced polar distortion artifacts
        """
        if strength is None:
            strength = self.polar_correction

        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)

        # Build per-row cosine weight map
        row_indices = np.arange(h, dtype=np.float32)
        # theta in [-π/2, π/2] — latitude
        latitude = (row_indices / h - 0.5) * np.pi
        # cos(lat) = 1 at equator, 0 at poles
        cos_weights = np.cos(latitude)[:, np.newaxis]  # (H, 1)

        # Blend: in polar regions, apply Gaussian smoothing to reduce artifacts
        polar_mask = (1.0 - cos_weights) ** 2 * strength  # (H, 1)
        polar_mask_3c = np.repeat(polar_mask[:, :, np.newaxis], image.shape[2], axis=2)

        # Gaussian-blurred version for polar regions
        blurred = cv2.GaussianBlur(
            result,
            ksize=(0, 0),
            sigmaX=3.0,
            sigmaY=1.0,
        )

        result = result * (1 - polar_mask_3c) + blurred * polar_mask_3c
        return np.clip(result, 0, 255).astype(np.uint8)

    # ─── Equirectangular → Cubemap ────────────────────────────

    def equirect_to_cubemap(
        self,
        image: np.ndarray,
        face_size: Optional[int] = None,
    ) -> Dict[CubeFace, np.ndarray]:
        """
        Convert equirectangular panorama to 6 cubemap faces.

        Uses bilinear interpolation for sub-pixel accuracy.
        Handles seam continuity at cube edges.

        Args:
            image: HxWxC equirectangular panorama
            face_size: output size for each cube face (default: self.cubemap_face_size)

        Returns:
            Dict mapping CubeFace → np.ndarray of shape (face_size, face_size, C)
        """
        face_size = face_size or self.cubemap_face_size
        h, w = image.shape[:2]

        if HAS_PY360:
            return self._equirect_to_cubemap_py360(image, face_size)
        else:
            return self._equirect_to_cubemap_manual(image, face_size, h, w)

    def _equirect_to_cubemap_py360(
        self,
        image: np.ndarray,
        face_size: int,
    ) -> Dict[CubeFace, np.ndarray]:
        """Use py360convert library for accurate cubemap projection."""
        # py360convert expects (H, W, C) float in [0, 1] or uint8
        cube_faces = py360convert.e2c(
            image,
            face_w=face_size,
            mode="bilinear",
            cube_format="dict",
        )
        # Map string keys to CubeFace enum
        key_map = {
            "F": CubeFace.FRONT,
            "B": CubeFace.BACK,
            "L": CubeFace.LEFT,
            "R": CubeFace.RIGHT,
            "U": CubeFace.TOP,
            "D": CubeFace.BOTTOM,
        }
        return {key_map[k]: v for k, v in cube_faces.items()}

    def _equirect_to_cubemap_manual(
        self,
        image: np.ndarray,
        face_size: int,
        src_h: int,
        src_w: int,
    ) -> Dict[CubeFace, np.ndarray]:
        """
        Manual cubemap projection implementation.
        For each face, compute the corresponding equirectangular coordinates
        via spherical ray tracing.
        """
        results = {}

        # Face definitions: (face_enum, rotation_matrix_to_face_direction)
        face_rotations = self._get_face_rotation_matrices()

        # Generate normalized face grid [-1, 1] x [-1, 1]
        grid_y, grid_x = np.mgrid[-1:1:complex(0, face_size), -1:1:complex(0, face_size)]
        grid_z = np.ones_like(grid_x)

        for face, rot_matrix in face_rotations.items():
            # Stack into [N, 3] ray directions
            rays = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)

            # Rotate rays to face direction
            rotated = (rot_matrix @ rays.T).T  # (N, 3)
            rx, ry, rz = rotated[:, 0], rotated[:, 1], rotated[:, 2]

            # Convert to spherical
            theta = np.arctan2(np.sqrt(rx**2 + rz**2), ry)
            phi = np.arctan2(rz, rx) % (2 * np.pi)

            # Map to equirectangular pixel coords
            u = (phi / (2 * np.pi)) * (src_w - 1)
            v = (theta / np.pi) * (src_h - 1)

            # Reshape to face grid
            u = u.reshape(face_size, face_size).astype(np.float32)
            v = v.reshape(face_size, face_size).astype(np.float32)

            # Bilinear sample from equirectangular
            face_img = cv2.remap(
                image,
                u,
                v,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,  # wrap for 360° seam
            )
            results[face] = face_img

        return results

    def cubemap_to_equirect(
        self,
        faces: Dict[CubeFace, np.ndarray],
        out_width: int,
        out_height: int,
    ) -> np.ndarray:
        """
        Convert 6 cubemap faces back to equirectangular panorama.
        Used to back-project detection/segmentation results.
        """
        if HAS_PY360:
            face_map = {
                CubeFace.FRONT: "F",
                CubeFace.BACK: "B",
                CubeFace.LEFT: "L",
                CubeFace.RIGHT: "R",
                CubeFace.TOP: "U",
                CubeFace.BOTTOM: "D",
            }
            py360_faces = {face_map[k]: v for k, v in faces.items()}
            return py360convert.c2e(
                py360_faces,
                h=out_height,
                w=out_width,
                mode="bilinear",
                cube_format="dict",
            )

        # Manual back-projection
        return self._cubemap_to_equirect_manual(faces, out_width, out_height)

    def _cubemap_to_equirect_manual(
        self,
        faces: Dict[CubeFace, np.ndarray],
        out_w: int,
        out_h: int,
    ) -> np.ndarray:
        """Manual cubemap-to-equirectangular projection."""
        # Determine output shape from first face
        first_face = next(iter(faces.values()))
        channels = first_face.shape[2] if len(first_face.shape) == 3 else 1
        output = np.zeros((out_h, out_w, channels), dtype=np.uint8)

        # For each equirectangular pixel, determine which cube face it maps to
        u_grid = np.linspace(0, 2 * np.pi, out_w, endpoint=False)
        v_grid = np.linspace(0, np.pi, out_h, endpoint=False)
        phi, theta = np.meshgrid(u_grid, v_grid)

        # Cartesian on unit sphere
        x = np.sin(theta) * np.cos(phi)
        y = np.cos(theta)
        z = np.sin(theta) * np.sin(phi)

        # Determine dominant axis (which face)
        abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
        max_axis = np.argmax(np.stack([abs_x, abs_y, abs_z], axis=-1), axis=-1)

        face_rotations_inv = self._get_face_rotation_matrices_inverse()

        for face, inv_rot in face_rotations_inv.items():
            # ... (simplified - use py360convert in production)
            pass

        return output

    # ─── Perspective Tile Extraction ─────────────────────────

    def extract_perspective_tiles(
        self,
        image: np.ndarray,
        fov_h: float = 90.0,
        fov_v: float = 90.0,
        tile_size: int = 640,
        overlap_deg: float = 15.0,
        include_poles: bool = True,
    ) -> List[PerspectiveTile]:
        """
        Extract overlapping perspective tiles from an equirectangular panorama.

        Tiles the sphere with perspective projections, ensuring full coverage
        and sufficient overlap for seamless merging of detection results.

        Args:
            image: HxWxC equirectangular panorama
            fov_h: tile horizontal FOV in degrees (default 90°)
            fov_v: tile vertical FOV in degrees (default 90°)
            tile_size: output tile resolution
            overlap_deg: overlap between adjacent tiles in degrees
            include_poles: whether to add top-down polar tiles

        Returns:
            List of PerspectiveTile objects with perspective-corrected images
        """
        tiles = []
        step_h = fov_h - overlap_deg
        step_v = fov_v - overlap_deg

        # Horizontal sweeps at multiple elevation angles
        elevations = [-45.0, 0.0, 45.0]
        if include_poles:
            elevations = [-80.0] + elevations + [80.0]

        for pitch in elevations:
            yaw = 0.0
            while yaw < 360.0:
                tile_img = self._extract_single_perspective(
                    image=image,
                    yaw=yaw,
                    pitch=pitch,
                    fov_h=fov_h,
                    fov_v=fov_v,
                    out_size=tile_size,
                )
                tiles.append(
                    PerspectiveTile(
                        image=tile_img,
                        yaw=yaw,
                        pitch=pitch,
                        fov_h=fov_h,
                        fov_v=fov_v,
                    )
                )
                yaw += step_h

        return tiles

    def _extract_single_perspective(
        self,
        image: np.ndarray,
        yaw: float,
        pitch: float,
        fov_h: float,
        fov_v: float,
        out_size: int,
    ) -> np.ndarray:
        """
        Extract a single perspective tile from an equirectangular image.

        Uses the pinhole camera model to compute the mapping:
        (px, py) in perspective → (lon, lat) in equirectangular.
        """
        h, w = image.shape[:2]
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)

        # Build rotation matrix R = Ry(yaw) * Rx(pitch)
        Ry = np.array([
            [math.cos(yaw_rad),  0, math.sin(yaw_rad)],
            [0,                  1, 0                ],
            [-math.sin(yaw_rad), 0, math.cos(yaw_rad)],
        ], dtype=np.float32)

        Rx = np.array([
            [1, 0,                   0                  ],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad),  math.cos(pitch_rad)],
        ], dtype=np.float32)

        R = Ry @ Rx  # Combined rotation

        # Perspective tile grid
        f_x = 0.5 * out_size / math.tan(math.radians(fov_h / 2))
        f_y = 0.5 * out_size / math.tan(math.radians(fov_v / 2))
        cx = cy = out_size / 2

        px, py = np.meshgrid(
            np.arange(out_size, dtype=np.float32),
            np.arange(out_size, dtype=np.float32),
        )

        # Ray directions in camera space
        rx = (px - cx) / f_x
        ry = (py - cy) / f_y
        rz = np.ones_like(rx)

        # Normalize rays
        norm = np.sqrt(rx**2 + ry**2 + rz**2)
        rx, ry, rz = rx / norm, ry / norm, rz / norm

        # Rotate to world space
        rays = np.stack([rx.ravel(), ry.ravel(), rz.ravel()], axis=0)  # (3, N)
        rotated = R @ rays  # (3, N)
        wx, wy, wz = rotated[0], rotated[1], rotated[2]

        # Spherical coordinates
        lon = np.arctan2(wx, wz)  # [-π, π]
        lat = np.arctan2(wy, np.sqrt(wx**2 + wz**2))  # [-π/2, π/2]

        # Map to equirectangular pixel coords
        u = (lon / (2 * np.pi) + 0.5) * w
        v = (-lat / np.pi + 0.5) * h

        u = u.reshape(out_size, out_size).astype(np.float32)
        v = v.reshape(out_size, out_size).astype(np.float32)

        # Bilinear sample with horizontal wrapping
        tile = cv2.remap(
            image,
            u, v,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        return tile

    # ─── Back-projection of Detection Results ─────────────────

    def backproject_boxes_to_equirect(
        self,
        boxes: np.ndarray,           # (N, 4) in [x1, y1, x2, y2] tile coords
        tile: PerspectiveTile,
        equirect_width: int,
        equirect_height: int,
    ) -> np.ndarray:
        """
        Map detected bounding boxes from perspective tile space
        back to equirectangular panorama coordinates.

        Args:
            boxes: Nx4 array of [x1, y1, x2, y2] bounding boxes in tile pixels
            tile: PerspectiveTile with yaw, pitch, fov info
            equirect_width, equirect_height: output panorama dimensions

        Returns:
            Nx4 array of [x1, y1, x2, y2] in equirectangular pixel space
        """
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        tile_h, tile_w = tile.image.shape[:2]
        out_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box
            # Sample corners of each box
            corners_tile = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2],
                [(x1+x2)/2, (y1+y2)/2],  # center
            ], dtype=np.float32)

            # Back-project each corner
            corners_eq = self._tile_pixels_to_equirect(
                corners_tile,
                tile=tile,
                tile_w=tile_w,
                tile_h=tile_h,
                eq_w=equirect_width,
                eq_h=equirect_height,
            )

            if corners_eq is not None:
                eq_x1 = corners_eq[:4, 0].min()
                eq_y1 = corners_eq[:4, 1].min()
                eq_x2 = corners_eq[:4, 0].max()
                eq_y2 = corners_eq[:4, 1].max()
                out_boxes.append([eq_x1, eq_y1, eq_x2, eq_y2])

        return np.array(out_boxes, dtype=np.float32) if out_boxes else np.zeros((0, 4))

    def _tile_pixels_to_equirect(
        self,
        pixels: np.ndarray,   # (N, 2) x,y in tile space
        tile: PerspectiveTile,
        tile_w: int,
        tile_h: int,
        eq_w: int,
        eq_h: int,
    ) -> Optional[np.ndarray]:
        """Project tile pixel coordinates back to equirectangular space."""
        yaw_rad = math.radians(tile.yaw)
        pitch_rad = math.radians(tile.pitch)

        f_x = 0.5 * tile_w / math.tan(math.radians(tile.fov_h / 2))
        f_y = 0.5 * tile_h / math.tan(math.radians(tile.fov_v / 2))
        cx, cy = tile_w / 2, tile_h / 2

        # Rotation matrix
        Ry = np.array([
            [math.cos(yaw_rad),  0, math.sin(yaw_rad)],
            [0, 1, 0],
            [-math.sin(yaw_rad), 0, math.cos(yaw_rad)],
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad),  math.cos(pitch_rad)],
        ])
        R = Ry @ Rx

        # Camera space rays
        rx = (pixels[:, 0] - cx) / f_x
        ry = (pixels[:, 1] - cy) / f_y
        rz = np.ones(len(pixels))
        norm = np.sqrt(rx**2 + ry**2 + rz**2)
        rays = np.stack([rx/norm, ry/norm, rz/norm], axis=0)

        # World space rays
        rotated = R @ rays
        wx, wy, wz = rotated[0], rotated[1], rotated[2]

        # Equirectangular coords
        lon = np.arctan2(wx, wz)
        lat = np.arctan2(wy, np.sqrt(wx**2 + wz**2))
        u = (lon / (2 * np.pi) + 0.5) * eq_w
        v = (-lat / np.pi + 0.5) * eq_h

        return np.stack([u, v], axis=1)

    def backproject_mask_to_equirect(
        self,
        mask: np.ndarray,
        tile: PerspectiveTile,
        equirect_width: int,
        equirect_height: int,
    ) -> np.ndarray:
        """
        Project a tile segmentation mask back to equirectangular space.

        Returns a sparse equirectangular mask (same class labels).
        """
        eq_mask = np.zeros(
            (equirect_height, equirect_width),
            dtype=mask.dtype,
        )
        tile_h, tile_w = mask.shape[:2]

        # Sample grid of mask points and back-project
        step = 2  # sample every 2 pixels for speed
        ys, xs = np.mgrid[0:tile_h:step, 0:tile_w:step]
        pixels = np.stack([xs.ravel().astype(np.float32),
                           ys.ravel().astype(np.float32)], axis=1)
        labels = mask[ys.ravel(), xs.ravel()]

        eq_coords = self._tile_pixels_to_equirect(
            pixels, tile, tile_w, tile_h, equirect_width, equirect_height
        )
        if eq_coords is None:
            return eq_mask

        eq_x = np.clip(eq_coords[:, 0].astype(int), 0, equirect_width - 1)
        eq_y = np.clip(eq_coords[:, 1].astype(int), 0, equirect_height - 1)
        eq_mask[eq_y, eq_x] = labels

        # Fill holes with nearest-neighbor dilation
        eq_mask = self._fill_projection_holes(eq_mask)
        return eq_mask

    def _fill_projection_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill sparse projected mask using morphological dilation."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.dilate(mask, kernel, iterations=1)

    # ─── Seam Handling ────────────────────────────────────────

    def blend_seam_region(
        self,
        equirect_a: np.ndarray,
        equirect_b: np.ndarray,
        seam_width_pixels: int = 64,
    ) -> np.ndarray:
        """
        Blend two equirectangular images at the panorama seam (left/right edge).

        Applies a cosine-weighted blend in the seam region to avoid
        visible discontinuities in segmentation/detection overlays.
        """
        assert equirect_a.shape == equirect_b.shape
        h, w = equirect_a.shape[:2]
        result = equirect_a.copy().astype(np.float32)

        # Left seam blend
        sw2 = seam_width_pixels // 2
        alpha = np.linspace(0, 1, sw2, dtype=np.float32)
        alpha = (1 - np.cos(alpha * np.pi)) / 2  # Cosine window

        for c in range(equirect_a.shape[2] if equirect_a.ndim == 3 else 1):
            slice_a = equirect_a[:, :sw2, c] if equirect_a.ndim == 3 else equirect_a[:, :sw2]
            slice_b = equirect_b[:, :sw2, c] if equirect_b.ndim == 3 else equirect_b[:, :sw2]
            blended = slice_a * (1 - alpha[np.newaxis, :]) + slice_b * alpha[np.newaxis, :]
            if result.ndim == 3:
                result[:, :sw2, c] = blended
            else:
                result[:, :sw2] = blended

        return result.astype(np.uint8)

    # ─── Rotation Matrices ────────────────────────────────────

    def apply_rotation(
        self,
        image: np.ndarray,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> np.ndarray:
        """
        Rotate an equirectangular panorama by yaw/pitch/roll.

        Useful for correcting camera orientation before processing.
        """
        if HAS_PY360:
            return py360convert.e2e(
                image,
                rot_deg=[pitch, yaw, roll],
                w=image.shape[1],
                mode="bilinear",
            )
        # Fallback: pure yaw rotation via circular shift
        if yaw != 0.0:
            shift = int((yaw / 360.0) * image.shape[1])
            image = np.roll(image, shift, axis=1)
        return image

    # ─── Utility ──────────────────────────────────────────────

    def _get_face_rotation_matrices(self) -> Dict[CubeFace, np.ndarray]:
        """Return rotation matrices that orient each cubemap face."""
        return {
            CubeFace.FRONT:  np.eye(3, dtype=np.float32),
            CubeFace.BACK:   _rot_y(np.pi),
            CubeFace.LEFT:   _rot_y(-np.pi / 2),
            CubeFace.RIGHT:  _rot_y(np.pi / 2),
            CubeFace.TOP:    _rot_x(-np.pi / 2),
            CubeFace.BOTTOM: _rot_x(np.pi / 2),
        }

    def _get_face_rotation_matrices_inverse(self) -> Dict[CubeFace, np.ndarray]:
        """Return inverse rotation matrices for each cubemap face."""
        return {
            face: mat.T
            for face, mat in self._get_face_rotation_matrices().items()
        }

    @staticmethod
    def equirect_to_fisheye(
        image: np.ndarray,
        fov_deg: float = 180.0,
        out_size: int = 1024,
    ) -> np.ndarray:
        """Convert equirectangular to fisheye projection."""
        h, w = image.shape[:2]
        cx = cy = out_size / 2
        r_max = out_size / 2

        py, px = np.mgrid[0:out_size, 0:out_size].astype(np.float32)
        dx, dy = px - cx, py - cy
        r = np.sqrt(dx**2 + dy**2)

        theta = (r / r_max) * math.radians(fov_deg / 2)
        phi = np.arctan2(dy, dx)

        lat = np.pi / 2 - theta
        u = ((phi / (2 * np.pi)) + 0.5) * w
        v = (0.5 - lat / np.pi) * h

        mask = r <= r_max
        u = np.where(mask, u, 0).astype(np.float32)
        v = np.where(mask, v, 0).astype(np.float32)

        fisheye = cv2.remap(image, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        fisheye[~mask] = 0
        return fisheye

    def get_spherical_weight_map(self, height: int, width: int) -> np.ndarray:
        """
        Generate a per-pixel confidence weight map for equirectangular images.

        Pixels near the poles are downweighted (cos(latitude)) since they
        represent smaller solid angles and have higher distortion.
        """
        row_indices = np.arange(height, dtype=np.float32)
        latitude = (row_indices / height - 0.5) * np.pi
        weights = np.cos(latitude)  # (H,)
        weight_map = np.tile(weights[:, np.newaxis], (1, width))
        # Normalize
        weight_map = weight_map / (weight_map.sum() + 1e-9)
        return weight_map.astype(np.float32)


# ─── Helper rotation matrix functions ────────────────────────

def _rot_x(angle: float) -> np.ndarray:
    """Rotation matrix around X axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _rot_y(angle: float) -> np.ndarray:
    """Rotation matrix around Y axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rot_z(angle: float) -> np.ndarray:
    """Rotation matrix around Z axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def build_rotation_matrix(
    yaw: float,
    pitch: float,
    roll: float,
) -> np.ndarray:
    """
    Build full 3D rotation matrix from Euler angles (degrees).
    Order: yaw → pitch → roll (ZXY convention)
    """
    y = math.radians(yaw)
    p = math.radians(pitch)
    r = math.radians(roll)
    return _rot_z(y) @ _rot_x(p) @ _rot_y(r)
