"""Panorama business logic service."""
from __future__ import annotations
import uuid
from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.panorama import Panorama
from app.schemas.panorama import PanoramaCreate, PanoramaResponse


class PanoramaService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, schema: PanoramaCreate) -> Panorama:
        panorama = Panorama(**schema.model_dump())
        self.db.add(panorama)
        await self.db.flush()
        return panorama

    async def get_by_id(self, panorama_id: str) -> Optional[Panorama]:
        result = await self.db.execute(
            select(Panorama).where(Panorama.id == uuid.UUID(panorama_id), Panorama.is_deleted == False)
        )
        return result.scalar_one_or_none()

    async def get_by_hash(self, file_hash: str) -> Optional[Panorama]:
        result = await self.db.execute(
            select(Panorama).where(Panorama.file_hash == file_hash, Panorama.is_deleted == False)
        )
        return result.scalar_one_or_none()

    async def list_panoramas(
        self,
        session_id: Optional[str] = None,
        camera_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
        user_id: Optional[str] = None,
    ) -> Tuple[List[Panorama], int]:
        query = select(Panorama).where(Panorama.is_deleted == False)
        if session_id:
            query = query.where(Panorama.session_id == session_id)
        if camera_type:
            query = query.where(Panorama.camera_type == camera_type)
        count_result = await self.db.execute(select(func.count()).select_from(query.subquery()))
        total = count_result.scalar() or 0
        result = await self.db.execute(query.offset(skip).limit(limit))
        return result.scalars().all(), total

    async def soft_delete(self, panorama_id: str, user_id: str) -> bool:
        panorama = await self.get_by_id(panorama_id)
        if not panorama:
            return False
        panorama.is_deleted = True
        await self.db.flush()
        return True

    async def get_metadata(self, panorama_id: str) -> Optional[dict]:
        panorama = await self.get_by_id(panorama_id)
        if not panorama:
            return None
        return {
            "panorama_id": panorama_id,
            "exif": panorama.exif_data,
            "gps": {"lat": panorama.gps_latitude, "lon": panorama.gps_longitude, "alt": panorama.gps_altitude},
            "camera_settings": {"type": panorama.camera_type},
            "spherical_metadata": {"width": panorama.width, "height": panorama.height},
            "processing_status": panorama.status,
        }

    async def generate_thumbnail(self, panorama_id: str, size: int) -> bytes:
        """Generate JPEG thumbnail bytes for a panorama."""
        import numpy as np
        import cv2
        # Return placeholder in production - would fetch from storage
        dummy = np.zeros((size // 2, size, 3), dtype=np.uint8)
        dummy[:] = (40, 40, 40)
        cv2.putText(dummy, "Panorama Thumbnail", (10, size//4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        _, buffer = cv2.imencode(".jpg", dummy)
        return buffer.tobytes()
