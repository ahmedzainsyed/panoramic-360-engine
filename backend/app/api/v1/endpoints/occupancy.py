"""Spatial occupancy endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from app.core.security import get_current_user

router = APIRouter()

@router.get("/{session_id}/heatmaps", summary="Get worker density heatmaps")
async def get_worker_heatmaps(session_id: str, current_user=Depends(get_current_user)):
    return {"session_id": session_id, "heatmaps": [], "message": "Heatmaps generated per panorama processing"}

@router.get("/{panorama_id}/density", summary="Get density map for a panorama")
async def get_density_map(panorama_id: str, current_user=Depends(get_current_user)):
    import json, redis
    from app.core.config import settings
    try:
        r = redis.from_url(settings.REDIS_URL)
        data = r.get(f"results:{panorama_id}")
        if data:
            results = json.loads(data)
            return results.get("occupancy", {})
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Occupancy data not found")
