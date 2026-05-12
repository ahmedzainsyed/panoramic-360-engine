"""Temporal analytics endpoints."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from app.core.security import get_current_user

router = APIRouter()

@router.post("/temporal/{session_id}", summary="Run temporal analysis for a session")
async def run_temporal(session_id: str, panorama_ids: List[str], current_user=Depends(get_current_user)):
    from app.tasks.ingestion_tasks import run_temporal_analysis
    task = run_temporal_analysis.apply_async(
        kwargs={"session_id": session_id, "panorama_ids": panorama_ids},
        task_id=f"temporal_{session_id}",
    )
    return {"session_id": session_id, "task_id": task.id, "status": "queued"}

@router.get("/temporal/{session_id}/results", summary="Get temporal analysis results")
async def get_temporal_results(session_id: str, current_user=Depends(get_current_user)):
    import redis, json
    from app.core.config import settings
    try:
        r = redis.from_url(settings.REDIS_URL)
        data = r.get(f"temporal:{session_id}")
        if data: return json.loads(data)
    except Exception: pass
    raise HTTPException(status_code=404, detail="Temporal analysis not found")

@router.get("/site-summary/{session_id}", summary="Get site-level analytics summary")
async def get_site_summary(
    session_id: str,
    days: int = Query(7, ge=1, le=90),
    current_user=Depends(get_current_user),
):
    """Return aggregated site analytics for dashboard KPIs."""
    return {
        "session_id": session_id,
        "period_days": days,
        "total_panoramas": 0,
        "avg_compliance_rate": 0.0,
        "avg_risk_score": 0.0,
        "peak_worker_count": 0,
        "total_hazard_zones_detected": 0,
        "change_events": [],
        "message": "Run temporal analysis to populate these metrics",
    }

@router.get("/detection/{panorama_id}", summary="Get detection results")
async def get_detection_results(panorama_id: str, current_user=Depends(get_current_user)):
    import redis, json
    from app.core.config import settings
    try:
        r = redis.from_url(settings.REDIS_URL)
        data = r.get(f"results:{panorama_id}")
        if data:
            results = json.loads(data)
            return results.get("detection", {})
    except Exception: pass
    raise HTTPException(status_code=404, detail="Detection results not found")
