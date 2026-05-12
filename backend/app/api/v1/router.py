"""
API v1 Router - aggregates all endpoint modules
"""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    panoramas,
    analysis,
    segmentation,
    detection,
    ppe,
    hazards,
    occupancy,
    navigation,
    reconstruction,
    analytics,
    sessions,
    exports,
)

api_router = APIRouter()

# ─── Authentication ───────────────────────────────────────────
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

# ─── Sessions ─────────────────────────────────────────────────
api_router.include_router(
    sessions.router,
    prefix="/sessions",
    tags=["Sessions"],
)

# ─── Panorama Ingestion ───────────────────────────────────────
api_router.include_router(
    panoramas.router,
    prefix="/panoramas",
    tags=["Panoramas"],
)

# ─── Analysis Pipeline ────────────────────────────────────────
api_router.include_router(
    analysis.router,
    prefix="/analyze",
    tags=["Analysis Pipeline"],
)

# ─── Segmentation ─────────────────────────────────────────────
api_router.include_router(
    segmentation.router,
    prefix="/segmentation",
    tags=["Segmentation"],
)

# ─── Object Detection ─────────────────────────────────────────
api_router.include_router(
    detection.router,
    prefix="/detection",
    tags=["Object Detection"],
)

# ─── PPE Compliance ───────────────────────────────────────────
api_router.include_router(
    ppe.router,
    prefix="/ppe",
    tags=["PPE Compliance"],
)

# ─── Hazard Detection ─────────────────────────────────────────
api_router.include_router(
    hazards.router,
    prefix="/hazards",
    tags=["Hazard Detection"],
)

# ─── Occupancy Analysis ───────────────────────────────────────
api_router.include_router(
    occupancy.router,
    prefix="/occupancy",
    tags=["Spatial Occupancy"],
)

# ─── Navigation Overlays ──────────────────────────────────────
api_router.include_router(
    navigation.router,
    prefix="/navigation",
    tags=["Navigation Overlays"],
)

# ─── 3D Reconstruction ────────────────────────────────────────
api_router.include_router(
    reconstruction.router,
    prefix="/reconstruction",
    tags=["3D Reconstruction"],
)

# ─── Temporal Analytics ───────────────────────────────────────
api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["Temporal Analytics"],
)

# ─── Exports ──────────────────────────────────────────────────
api_router.include_router(
    exports.router,
    prefix="/exports",
    tags=["Exports"],
)
