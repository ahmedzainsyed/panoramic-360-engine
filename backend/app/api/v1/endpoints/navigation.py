"""Navigation overlay endpoints."""
from fastapi import APIRouter, Depends
from app.core.security import get_current_user

router = APIRouter()

@router.get("/{panorama_id}/overlays", summary="Get navigation overlays")
async def get_navigation_overlays(panorama_id: str, current_user=Depends(get_current_user)):
    import json, redis
    from app.core.config import settings
    try:
        r = redis.from_url(settings.REDIS_URL)
        data = r.get(f"results:{panorama_id}")
        if data:
            import json
            results = json.loads(data)
            return results.get("navigation", {})
    except Exception:
        pass
    return {"panorama_id": panorama_id, "message": "Run analysis to generate navigation overlays"}

@router.get("/{panorama_id}/walkable-paths", summary="Get walkable path map")
async def get_walkable_paths(panorama_id: str, current_user=Depends(get_current_user)):
    return {"panorama_id": panorama_id, "walkable_paths": []}
