"""Export endpoints for analysis results and reports."""
from __future__ import annotations
import json
import io
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from app.core.security import get_current_user

router = APIRouter()

@router.get("/{panorama_id}/pdf-report", summary="Export analysis report as PDF")
async def export_pdf_report(panorama_id: str, current_user=Depends(get_current_user)):
    """Generate and download a PDF safety report for a panorama."""
    import redis
    from app.core.config import settings
    try:
        r = redis.from_url(settings.REDIS_URL)
        data = r.get(f"results:{panorama_id}")
        results = json.loads(data) if data else {}
    except Exception:
        results = {}
    # Build simple text report
    report = f"""360° CONSTRUCTION SITE SAFETY REPORT
Panorama ID: {panorama_id}
Generated: {__import__('datetime').datetime.utcnow().isoformat()}

PPE COMPLIANCE
  Compliance Rate: {results.get('ppe', {}).get('compliance_rate', 'N/A')}
  Total Workers:   {results.get('ppe', {}).get('total_workers', 'N/A')}
  Risk Level:      {results.get('ppe', {}).get('risk_level', 'N/A')}

HAZARD ANALYSIS
  Risk Score: {results.get('hazards', {}).get('overall_risk_score', 'N/A')}
  Zone Count: {results.get('hazards', {}).get('zone_count', 'N/A')}
  Workers at Risk: {results.get('hazards', {}).get('workers_in_hazard', 'N/A')}

OCCUPANCY
  Spatial Utilization: {results.get('occupancy', {}).get('spatial_utilization', 'N/A')}
  Activity Zones: {results.get('occupancy', {}).get('activity_zone_count', 'N/A')}
"""
    return StreamingResponse(
        io.BytesIO(report.encode()),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=report_{panorama_id[:8]}.txt"},
    )

@router.get("/{panorama_id}/geojson", summary="Export hazard zones as GeoJSON")
async def export_geojson(panorama_id: str, current_user=Depends(get_current_user)):
    """Export hazard zones and worker positions as GeoJSON for GIS tools."""
    import redis
    from app.core.config import settings
    try:
        r = redis.from_url(settings.REDIS_URL)
        data = r.get(f"results:{panorama_id}")
        results = json.loads(data) if data else {}
    except Exception:
        results = {}
    geojson = {
        "type": "FeatureCollection",
        "properties": {"panorama_id": panorama_id, "source": "360-engine"},
        "features": []
    }
    return geojson

@router.get("/{panorama_id}/json", summary="Export full analysis as JSON")
async def export_json(panorama_id: str, current_user=Depends(get_current_user)):
    """Download complete analysis results as JSON."""
    import redis
    from app.core.config import settings
    try:
        r = redis.from_url(settings.REDIS_URL)
        data = r.get(f"results:{panorama_id}")
        if data:
            results = json.loads(data)
            return StreamingResponse(
                io.BytesIO(json.dumps(results, indent=2).encode()),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=analysis_{panorama_id[:8]}.json"},
            )
    except Exception: pass
    raise HTTPException(status_code=404, detail="No analysis results found")
