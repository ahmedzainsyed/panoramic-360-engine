"""
EXIF and Panorama Metadata Extractor
Extracts GPS, camera orientation, spherical metadata from panorama files.
"""
from __future__ import annotations
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PanoramaMetadata:
    """Extracted panorama metadata."""
    width: int = 0
    height: int = 0
    camera_make: str = ""
    camera_model: str = ""
    software: str = ""
    # GPS
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    gps_timestamp: Optional[str] = None
    # Orientation
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    # Spherical
    projection_type: str = "equirectangular"
    full_pano_width: Optional[int] = None
    full_pano_height: Optional[int] = None
    # Camera exposure
    exposure_time: Optional[str] = None
    iso: Optional[int] = None
    focal_length: Optional[float] = None
    # Extra
    raw_exif: Dict[str, Any] = field(default_factory=dict)


class MetadataExtractor:
    """Extracts rich metadata from 360° panorama files."""

    def extract(self, file_path: str) -> PanoramaMetadata:
        """Main extraction entry point."""
        meta = PanoramaMetadata()
        try:
            # Try pillow first
            self._extract_with_pillow(file_path, meta)
        except Exception as e:
            logger.warning("pillow_extraction_failed", error=str(e))
        try:
            # Try piexif for detailed EXIF
            self._extract_with_piexif(file_path, meta)
        except Exception as e:
            logger.debug("piexif_extraction_skipped", error=str(e))
        try:
            # Try XMP for spherical/Google Street View metadata
            self._extract_xmp(file_path, meta)
        except Exception as e:
            logger.debug("xmp_extraction_skipped", error=str(e))
        return meta

    def _extract_with_pillow(self, file_path: str, meta: PanoramaMetadata):
        from PIL import Image
        with Image.open(file_path) as img:
            meta.width, meta.height = img.size
            exif_data = img._getexif() if hasattr(img, '_getexif') and img._getexif() else {}
            if exif_data:
                from PIL.ExifTags import TAGS, GPSTAGS
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, str(tag_id))
                    if tag == "Make": meta.camera_make = str(value)
                    elif tag == "Model": meta.camera_model = str(value)
                    elif tag == "Software": meta.software = str(value)
                    elif tag == "ExposureTime":
                        meta.exposure_time = str(value)
                    elif tag == "ISOSpeedRatings": meta.iso = int(value)
                    elif tag == "FocalLength":
                        meta.focal_length = float(value) if not isinstance(value, tuple) else value[0]/value[1]
                    elif tag == "GPSInfo":
                        gps = {GPSTAGS.get(k, k): v for k, v in value.items()}
                        meta.gps_latitude = self._parse_gps_coord(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
                        meta.gps_longitude = self._parse_gps_coord(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
                        if "GPSAltitude" in gps:
                            alt = gps["GPSAltitude"]
                            meta.gps_altitude = float(alt[0])/float(alt[1]) if isinstance(alt, tuple) else float(alt)
                    meta.raw_exif[tag] = str(value)[:200]

    def _extract_with_piexif(self, file_path: str, meta: PanoramaMetadata):
        import piexif
        try:
            exif = piexif.load(file_path)
            # Additional extraction if needed
        except Exception:
            pass

    def _extract_xmp(self, file_path: str, meta: PanoramaMetadata):
        """Parse XMP metadata for panoramic/spherical information."""
        try:
            with open(file_path, "rb") as f:
                content = f.read(65536)
            xmp_start = content.find(b"<x:xmpmeta")
            xmp_end = content.find(b"</x:xmpmeta>")
            if xmp_start == -1 or xmp_end == -1:
                return
            xmp_str = content[xmp_start:xmp_end + 12].decode("utf-8", errors="ignore")
            if "ProjectionType" in xmp_str:
                if "equirectangular" in xmp_str.lower():
                    meta.projection_type = "equirectangular"
            import re
            for tag, attr in [("FullPanoWidthPixels", "full_pano_width"),
                               ("FullPanoHeightPixels", "full_pano_height")]:
                m = re.search(rf'{tag}="(\d+)"', xmp_str)
                if m:
                    setattr(meta, attr, int(m.group(1)))
            for tag, attr in [("PoseHeadingDegrees", "yaw"),
                               ("PosePitchDegrees", "pitch"),
                               ("PoseRollDegrees", "roll")]:
                m = re.search(rf'{tag}="([\d.-]+)"', xmp_str)
                if m:
                    setattr(meta, attr, float(m.group(1)))
        except Exception as e:
            logger.debug("xmp_parse_failed", error=str(e))

    @staticmethod
    def _parse_gps_coord(coord_tuple, ref) -> Optional[float]:
        if not coord_tuple:
            return None
        try:
            d = coord_tuple[0]; m = coord_tuple[1]; s = coord_tuple[2]
            deg = (float(d[0])/d[1] if isinstance(d, tuple) else float(d) +
                   float(m[0])/m[1]/60 if isinstance(m, tuple) else float(m)/60 +
                   float(s[0])/s[1]/3600 if isinstance(s, tuple) else float(s)/3600)
            if ref in ("S", "W"):
                deg = -deg
            return round(deg, 7)
        except Exception:
            return None
