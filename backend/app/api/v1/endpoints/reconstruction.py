"""3D reconstruction endpoints."""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from app.core.security import get_current_user

router = APIRouter()

@router.post("/{panorama_id}/start", summary="Start 3D reconstruction")
async def start_reconstruction(
    panorama_id: str,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
):
    """Queue 3D reconstruction pipeline for a panorama."""
    from app.tasks.ingestion_tasks import run_3d_reconstruction
    task = run_3d_reconstruction.apply_async(
        kwargs={"panorama_id": panorama_id, "storage_key": f"panoramas/{panorama_id}.jpg"},
        task_id=f"recon_{panorama_id}",
    )
    return {"panorama_id": panorama_id, "task_id": task.id, "status": "queued"}

@router.get("/{panorama_id}/point-cloud", summary="Get point cloud info")
async def get_point_cloud(panorama_id: str, current_user=Depends(get_current_user)):
    return {
        "panorama_id": panorama_id,
        "ply_url": f"/api/v1/exports/{panorama_id}/pointcloud.ply",
        "status": "available",
    }

@router.get("/{panorama_id}/depth-map", summary="Get depth map info")
async def get_depth_map(panorama_id: str, current_user=Depends(get_current_user)):
    return {"panorama_id": panorama_id, "status": "available",
            "url": f"/api/v1/exports/{panorama_id}/depth_map.png"}
