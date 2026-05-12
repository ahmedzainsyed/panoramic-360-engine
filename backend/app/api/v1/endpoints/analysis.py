"""Analysis pipeline orchestration endpoint."""
from __future__ import annotations
from typing import Any, Dict, Optional
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.panorama import User
from app.schemas.panorama import AnalysisRequest, AnalysisStatusResponse
from app.tasks.ingestion_tasks import process_uploaded_panorama

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/", response_model=AnalysisStatusResponse, summary="Run full analysis pipeline")
async def run_analysis(
    request: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Dispatch async analysis pipeline for a panorama."""
    task = process_uploaded_panorama.apply_async(
        kwargs={
            "panorama_id": request.panorama_id,
            "storage_key": f"panoramas/{request.panorama_id}.jpg",
            "run_modules": request.modules,
        },
        task_id=f"analysis_{request.panorama_id}",
    )
    return AnalysisStatusResponse(
        panorama_id=request.panorama_id,
        task_id=task.id,
        status="queued",
        progress=0,
        stage="queued",
    )


@router.get("/{panorama_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(
    panorama_id: str,
    current_user: User = Depends(get_current_user),
):
    """Get current analysis status and progress."""
    from celery.result import AsyncResult
    from app.core.celery_app import celery_app
    task_id = f"analysis_{panorama_id}"
    result = AsyncResult(task_id, app=celery_app)
    state = result.state
    meta = result.info or {}
    return AnalysisStatusResponse(
        panorama_id=panorama_id,
        task_id=task_id,
        status=state.lower(),
        progress=meta.get("progress", 0) if isinstance(meta, dict) else 0,
        stage=meta.get("stage") if isinstance(meta, dict) else None,
        results=meta if state == "SUCCESS" and isinstance(meta, dict) else None,
        error=str(meta) if state == "FAILURE" else None,
    )


@router.get("/{panorama_id}/results", summary="Get completed analysis results")
async def get_analysis_results(
    panorama_id: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return full analysis results for a completed panorama."""
    import json
    import redis
    from app.core.config import settings
    try:
        r = redis.from_url(settings.REDIS_URL)
        data = r.get(f"results:{panorama_id}")
        if data:
            return json.loads(data)
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Analysis results not found or still processing")
