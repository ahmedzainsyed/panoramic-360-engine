"""
Panorama Ingestion API Endpoints
Handles 360° image upload, validation, metadata extraction, and storage.
"""
from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path
from typing import List, Optional

import structlog
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import get_current_user
from app.core.rate_limit import rate_limiter
from app.db.session import get_db
from app.models.user import User
from app.schemas.panorama import (
    PanoramaCreate,
    PanoramaResponse,
    PanoramaListResponse,
    PanoramaUploadResponse,
    PanoramaMetadata,
)
from app.services.panorama_service import PanoramaService
from app.services.storage_service import StorageService
from app.tasks.ingestion_tasks import process_uploaded_panorama

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post(
    "/upload",
    response_model=PanoramaUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a 360° panorama image",
    description=(
        "Upload equirectangular panoramas from Insta360, Ricoh Theta, "
        "drone cameras, or any 360° source. Triggers automatic metadata "
        "extraction and validation pipeline."
    ),
)
async def upload_panorama(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="360° panorama image file"),
    session_id: str = Form(..., description="Construction site session identifier"),
    camera_type: str = Form(
        "unknown",
        description="Camera type: insta360, ricoh_theta, drone, matterport, unknown",
    ),
    location_name: Optional[str] = Form(None, description="Site location name"),
    gps_latitude: Optional[float] = Form(None, description="GPS latitude"),
    gps_longitude: Optional[float] = Form(None, description="GPS longitude"),
    gps_altitude: Optional[float] = Form(None, description="GPS altitude in meters"),
    capture_timestamp: Optional[str] = Form(
        None, description="ISO 8601 capture timestamp"
    ),
    floor_level: Optional[int] = Form(None, description="Building floor level"),
    notes: Optional[str] = Form(None, description="Additional notes"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _: None = Depends(rate_limiter),
):
    """
    Upload and ingest a 360° panoramic image.

    Supported formats: JPG, JPEG, PNG, TIFF, EXR, HDR
    Max file size: 500MB (configurable)

    The upload triggers:
    1. Format and resolution validation
    2. EXIF metadata extraction
    3. GPS data parsing (if not provided explicitly)
    4. Equirectangular detection and normalization
    5. Thumbnail generation
    6. Background processing queue entry
    """
    # ── Validate file extension ───────────────────────────────
    if file.filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext not in settings.ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported format '{ext}'. Allowed: {settings.ALLOWED_IMAGE_FORMATS}",
        )

    # ── Read and size-check file ──────────────────────────────
    contents = await file.read()
    file_size = len(contents)

    if file_size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {file_size / 1e6:.1f}MB exceeds limit "
                f"{settings.MAX_UPLOAD_SIZE_MB}MB"
            ),
        )

    if file_size < 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too small to be a valid panorama",
        )

    # ── Compute hash for deduplication ───────────────────────
    file_hash = hashlib.sha256(contents).hexdigest()

    panorama_service = PanoramaService(db)
    storage_service = StorageService()

    # Check for duplicate
    existing = await panorama_service.get_by_hash(file_hash)
    if existing:
        logger.info("duplicate_panorama_detected", hash=file_hash, id=existing.id)
        return PanoramaUploadResponse(
            panorama_id=str(existing.id),
            status="duplicate",
            message="Identical panorama already exists",
            storage_key=existing.storage_key,
        )

    # ── Generate panorama ID and storage key ─────────────────
    panorama_id = str(uuid.uuid4())
    storage_key = f"sessions/{session_id}/panoramas/{panorama_id}.{ext}"

    # ── Upload to object storage ──────────────────────────────
    try:
        await storage_service.upload_bytes(
            bucket=settings.S3_BUCKET_PANORAMAS,
            key=storage_key,
            data=contents,
            content_type=file.content_type or f"image/{ext}",
            metadata={
                "panorama_id": panorama_id,
                "session_id": session_id,
                "camera_type": camera_type,
                "original_filename": file.filename,
                "uploader_id": str(current_user.id),
            },
        )
    except Exception as e:
        logger.error("storage_upload_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage upload failed. Please retry.",
        )

    # ── Create database record ────────────────────────────────
    create_schema = PanoramaCreate(
        id=uuid.UUID(panorama_id),
        session_id=session_id,
        storage_key=storage_key,
        original_filename=file.filename,
        file_size_bytes=file_size,
        file_hash=file_hash,
        camera_type=camera_type,
        format=ext,
        location_name=location_name,
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude,
        gps_altitude=gps_altitude,
        capture_timestamp=capture_timestamp,
        floor_level=floor_level,
        notes=notes,
        uploaded_by=current_user.id,
    )

    panorama_record = await panorama_service.create(create_schema)

    # ── Trigger background processing ─────────────────────────
    background_tasks.add_task(
        _trigger_processing,
        panorama_id=panorama_id,
        storage_key=storage_key,
        camera_type=camera_type,
    )

    logger.info(
        "panorama_uploaded",
        panorama_id=panorama_id,
        session_id=session_id,
        size_mb=f"{file_size / 1e6:.2f}",
        camera_type=camera_type,
    )

    return PanoramaUploadResponse(
        panorama_id=panorama_id,
        status="uploaded",
        message="Panorama uploaded successfully. Processing queued.",
        storage_key=storage_key,
        file_size_bytes=file_size,
        processing_task_id=f"task_{panorama_id[:8]}",
    )


@router.post(
    "/upload-batch",
    response_model=List[PanoramaUploadResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Batch upload multiple panoramas",
)
async def upload_panoramas_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple panorama files"),
    session_id: str = Form(...),
    camera_type: str = Form("unknown"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload multiple panoramas at once (max 20 files per batch)."""
    if len(files) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 20 files per batch upload",
        )

    results = []
    for file in files:
        try:
            # Reuse single upload logic for each file
            result = await upload_panorama(
                background_tasks=background_tasks,
                file=file,
                session_id=session_id,
                camera_type=camera_type,
                location_name=None,
                gps_latitude=None,
                gps_longitude=None,
                gps_altitude=None,
                capture_timestamp=None,
                floor_level=None,
                notes=None,
                db=db,
                current_user=current_user,
                _=None,
            )
            results.append(result)
        except HTTPException as e:
            results.append(
                PanoramaUploadResponse(
                    panorama_id="",
                    status="error",
                    message=f"{file.filename}: {e.detail}",
                    storage_key="",
                )
            )

    return results


@router.get(
    "/{panorama_id}",
    response_model=PanoramaResponse,
    summary="Get panorama details",
)
async def get_panorama(
    panorama_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve full panorama metadata and analysis status."""
    panorama_service = PanoramaService(db)
    panorama = await panorama_service.get_by_id(panorama_id)

    if not panorama:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Panorama '{panorama_id}' not found",
        )

    return panorama


@router.get(
    "/",
    response_model=PanoramaListResponse,
    summary="List panoramas",
)
async def list_panoramas(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    camera_type: Optional[str] = Query(None, description="Filter by camera type"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List panoramas with optional filtering and pagination."""
    panorama_service = PanoramaService(db)
    panoramas, total = await panorama_service.list_panoramas(
        session_id=session_id,
        camera_type=camera_type,
        skip=skip,
        limit=limit,
        user_id=current_user.id,
    )
    return PanoramaListResponse(
        panoramas=panoramas,
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{panorama_id}/thumbnail",
    summary="Get panorama thumbnail",
    response_class=StreamingResponse,
)
async def get_thumbnail(
    panorama_id: str,
    size: int = Query(512, ge=128, le=2048, description="Thumbnail width in pixels"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return a scaled-down thumbnail of the panorama."""
    panorama_service = PanoramaService(db)
    thumbnail_bytes = await panorama_service.generate_thumbnail(panorama_id, size)

    return StreamingResponse(
        io.BytesIO(thumbnail_bytes),
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=3600"},
    )


@router.delete(
    "/{panorama_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a panorama",
)
async def delete_panorama(
    panorama_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Soft-delete a panorama and its associated analysis results."""
    panorama_service = PanoramaService(db)
    deleted = await panorama_service.soft_delete(panorama_id, current_user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Panorama not found or access denied",
        )


@router.get(
    "/{panorama_id}/metadata",
    response_model=PanoramaMetadata,
    summary="Get extracted panorama metadata",
)
async def get_panorama_metadata(
    panorama_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return extracted EXIF metadata, GPS coordinates, camera orientation,
    and spherical geometry parameters.
    """
    panorama_service = PanoramaService(db)
    metadata = await panorama_service.get_metadata(panorama_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Metadata not found")
    return metadata


@router.post(
    "/{panorama_id}/reprocess",
    summary="Re-trigger processing for a panorama",
)
async def reprocess_panorama(
    panorama_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Force re-processing of a panorama through the full pipeline."""
    panorama_service = PanoramaService(db)
    panorama = await panorama_service.get_by_id(panorama_id)
    if not panorama:
        raise HTTPException(status_code=404, detail="Panorama not found")

    background_tasks.add_task(
        _trigger_processing,
        panorama_id=panorama_id,
        storage_key=panorama.storage_key,
        camera_type=panorama.camera_type,
    )

    return {"message": "Reprocessing queued", "panorama_id": panorama_id}


async def _trigger_processing(
    panorama_id: str,
    storage_key: str,
    camera_type: str,
) -> None:
    """Helper: dispatch async Celery processing task."""
    process_uploaded_panorama.apply_async(
        kwargs={
            "panorama_id": panorama_id,
            "storage_key": storage_key,
            "camera_type": camera_type,
        },
        task_id=f"ingest_{panorama_id}",
        countdown=0,
    )
    logger.info(
        "processing_task_dispatched",
        panorama_id=panorama_id,
        camera_type=camera_type,
    )
