"""PPE compliance endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from app.core.security import get_current_user

router = APIRouter()

@router.get("/{panorama_id}/report", summary="Get PPE compliance report")
async def get_ppe_report(panorama_id: str, current_user=Depends(get_current_user)):
    import json, redis
    from app.core.config import settings
    try:
        r = redis.from_url(settings.REDIS_URL)
        data = r.get(f"results:{panorama_id}")
        if data:
            results = json.loads(data)
            return results.get("ppe", {"message": "PPE analysis not yet run"})
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="PPE report not found")

@router.get("/{session_id}/compliance-trend", summary="PPE compliance trend for a session")
async def get_compliance_trend(session_id: str, current_user=Depends(get_current_user)):
    return {"session_id": session_id, "trend": [], "message": "Run temporal analysis for trend data"}
