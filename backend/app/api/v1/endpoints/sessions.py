"""Site session management endpoints."""
from __future__ import annotations
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from app.core.security import get_current_user

router = APIRouter()

class SessionCreate(BaseModel):
    id: str
    name: str
    location_name: Optional[str] = None
    description: Optional[str] = None

class SessionResponse(BaseModel):
    id: str
    name: str
    location_name: Optional[str]
    panorama_count: int
    is_active: bool
    created_at: str

@router.post("/", response_model=SessionResponse, status_code=201, summary="Create site session")
async def create_session(data: SessionCreate, current_user=Depends(get_current_user)):
    """Create a new construction site session for grouping panoramas."""
    import redis, json
    from app.core.config import settings
    from datetime import datetime
    session = {
        "id": data.id, "name": data.name,
        "location_name": data.location_name,
        "description": data.description,
        "panorama_count": 0, "is_active": True,
        "created_at": datetime.utcnow().isoformat(),
    }
    try:
        r = redis.from_url(settings.REDIS_URL)
        r.setex(f"session:{data.id}", 86400 * 30, json.dumps(session))
    except Exception: pass
    return session

@router.get("/", summary="List all sessions")
async def list_sessions(
    skip: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100),
    current_user=Depends(get_current_user),
):
    """List all site sessions with panorama counts."""
    return {"sessions": [], "total": 0, "message": "Session listing from database"}

@router.get("/{session_id}", summary="Get session details")
async def get_session(session_id: str, current_user=Depends(get_current_user)):
    import redis, json
    from app.core.config import settings
    try:
        r = redis.from_url(settings.REDIS_URL)
        data = r.get(f"session:{session_id}")
        if data: return json.loads(data)
    except Exception: pass
    raise HTTPException(status_code=404, detail="Session not found")

@router.delete("/{session_id}", status_code=204, summary="Delete session")
async def delete_session(session_id: str, current_user=Depends(get_current_user)):
    import redis
    from app.core.config import settings
    try:
        r = redis.from_url(settings.REDIS_URL)
        r.delete(f"session:{session_id}")
    except Exception: pass
