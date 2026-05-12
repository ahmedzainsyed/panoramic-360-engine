"""Pydantic schemas for panorama API."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PanoramaCreate(BaseModel):
    id: uuid.UUID
    session_id: str
    storage_key: str
    original_filename: str
    file_size_bytes: int
    file_hash: str
    camera_type: str = "unknown"
    format: str = "jpg"
    location_name: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    capture_timestamp: Optional[str] = None
    floor_level: Optional[int] = None
    notes: Optional[str] = None
    uploaded_by: Optional[str] = None


class PanoramaUploadResponse(BaseModel):
    panorama_id: str
    status: str
    message: str
    storage_key: str
    file_size_bytes: Optional[int] = None
    processing_task_id: Optional[str] = None


class PanoramaResponse(BaseModel):
    id: str
    session_id: str
    storage_key: str
    original_filename: str
    file_size_bytes: int
    camera_type: str
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    location_name: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    floor_level: Optional[int] = None
    status: str
    analysis_results: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PanoramaListResponse(BaseModel):
    panoramas: List[PanoramaResponse]
    total: int
    skip: int
    limit: int


class PanoramaMetadata(BaseModel):
    panorama_id: str
    exif: Optional[Dict[str, Any]] = None
    gps: Optional[Dict[str, float]] = None
    camera_settings: Optional[Dict[str, Any]] = None
    spherical_metadata: Optional[Dict[str, Any]] = None
    processing_status: str


class AnalysisRequest(BaseModel):
    panorama_id: str
    modules: List[str] = Field(default=["segmentation", "ppe", "hazards", "occupancy"])
    options: Dict[str, Any] = Field(default_factory=dict)


class AnalysisStatusResponse(BaseModel):
    panorama_id: str
    task_id: str
    status: str
    progress: int
    stage: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
