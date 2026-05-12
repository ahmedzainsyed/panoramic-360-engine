"""Deep health check utilities."""
import asyncio
from typing import Dict, Any

async def check_all_dependencies() -> Dict[str, Any]:
    results = {"ready": True, "checks": {}}
    # DB check
    try:
        from app.db.session import engine
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        results["checks"]["database"] = "ok"
    except Exception as e:
        results["checks"]["database"] = f"error: {e}"
        results["ready"] = False
    # Redis check
    try:
        import redis.asyncio as aioredis
        from app.core.config import settings
        r = aioredis.from_url(settings.REDIS_URL)
        await r.ping()
        results["checks"]["redis"] = "ok"
        await r.aclose()
    except Exception as e:
        results["checks"]["redis"] = f"error: {e}"
        results["ready"] = False
    return results
