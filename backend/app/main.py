"""
360° Construction Site Intelligence Engine
FastAPI Application Entry Point
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.middleware.sessions import SessionMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.session import engine, create_all_tables
from app.core.events import startup_event, shutdown_event

# ─── Structured Logging ──────────────────────────────────────
configure_logging()
logger = structlog.get_logger(__name__)

# ─── Prometheus Metrics ──────────────────────────────────────
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP request count",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
INFERENCE_COUNT = Counter(
    "ml_inference_total",
    "Total ML inference count",
    ["module", "model", "status"],
)
PANORAMA_PROCESSED = Counter(
    "panoramas_processed_total",
    "Total panoramas processed",
    ["camera_type", "analysis_type"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger.info(
        "starting_application",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )
    await startup_event()
    yield
    logger.info("shutting_down_application")
    await shutdown_event()


# ─── Application Factory ─────────────────────────────────────
def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "Production-grade AI platform for 360° construction site "
            "semantic understanding, PPE compliance, hazard detection, "
            "and spatial analytics."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_CREDENTIALS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

    if not settings.DEBUG:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS,
        )

    # ── Request ID + Metrics Middleware ───────────────────────
    @app.middleware("http")
    async def request_middleware(request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start_time
        endpoint = request.url.path
        method = request.method
        status_code = str(response.status_code)

        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.4f}s"

        logger.info(
            "http_request",
            method=method,
            path=endpoint,
            status_code=response.status_code,
            duration=f"{duration:.4f}s",
            request_id=request_id,
        )
        return response

    # ── API Routes ────────────────────────────────────────────
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # ── Prometheus metrics endpoint ───────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── Health check ─────────────────────────────────────────
    @app.get("/health", tags=["health"], include_in_schema=False)
    async def health_check():
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        }

    @app.get("/ready", tags=["health"], include_in_schema=False)
    async def readiness_check():
        """Deep readiness check - verifies DB, Redis, and storage."""
        from app.core.health import check_all_dependencies

        result = await check_all_dependencies()
        if not result["ready"]:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail=result)
        return result

    return app


app = create_application()
