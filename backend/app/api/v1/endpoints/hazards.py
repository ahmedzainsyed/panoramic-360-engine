"""Hazard detection endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from app.core.security import get_current_user

router = APIRouter()

@router.get("/{panorama_id}/map", summary="Get hazard risk map")
async def get_hazard_map(panorama_id: str, current_user=Depends(get_current_user)):
    import json, redis
    from app.core.config import settings
    try:
        r = redis.from_url(settings.REDIS_URL)
        data = r.get(f"results:{panorama_id}")
        if data:
            results = json.loads(data)
            return results.get("hazards", {"message": "hazard analysis not yet run"})
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Hazard map not found")

@router.get("/{panorama_id}/zones", summary="List detected hazard zones")
async def get_hazard_zones(panorama_id: str, current_user=Depends(get_current_user)):
    return {"panorama_id": panorama_id, "zones": [], "message": "Use /analyze to run hazard detection first"}
