"""Celery application configuration."""
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "panoramic_360",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.ingestion_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "tasks.process_uploaded_panorama": {"queue": "high_priority"},
        "tasks.run_temporal_analysis": {"queue": "analytics"},
        "tasks.run_3d_reconstruction": {"queue": "gpu_heavy"},
    },
    beat_schedule={
        "cleanup-old-tasks": {
            "task": "tasks.cleanup_old_results",
            "schedule": 3600.0,
        },
    },
)
