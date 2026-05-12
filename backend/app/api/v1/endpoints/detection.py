"""Object detection endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from app.core.security import get_current_user

router = APIRouter()

@router.post("/{panorama_id}/run", summary="Run object detection")
async def run_detection(panorama_id: str, current_user=Depends(get_current_user)):
    from app.tasks.ingestion_tasks import process_uploaded_panorama
    task = process_uploaded_panorama.apply_async(
        kwargs={"panorama_id": panorama_id,
                "storage_key": f"panoramas/{panorama_id}.jpg",
                "run_modules": ["detection"]},
        task_id=f"det_{panorama_id}",
    )
    return {"panorama_id": panorama_id, "task_id": task.id, "status": "queued"}

@router.get("/{panorama_id}/results", summary="Get detection results")
async def get_detection(panorama_id: str, current_user=Depends(get_current_user)):
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
