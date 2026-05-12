"""Rate limiting dependency."""
import time
from collections import defaultdict
from fastapi import HTTPException, Request, status
from app.core.config import settings

_request_counts: dict = defaultdict(list)

async def rate_limiter(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60.0
    _request_counts[client_ip] = [t for t in _request_counts[client_ip] if now - t < window]
    if len(_request_counts[client_ip]) >= settings.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Rate limit exceeded. Try again in 60 seconds.")
    _request_counts[client_ip].append(now)
