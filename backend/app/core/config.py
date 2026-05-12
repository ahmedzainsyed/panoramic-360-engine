"""
Application Configuration - Pydantic Settings
Loads from environment variables / .env file
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

from pydantic import AnyHttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ─── Application ──────────────────────────────────────────
    APP_NAME: str = "360° Construction Intelligence Engine"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    API_V1_PREFIX: str = "/api/v1"
    ALLOWED_HOSTS: List[str] = ["*"]

    # ─── Server ───────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    # ─── Security ─────────────────────────────────────────────
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ─── Database ─────────────────────────────────────────────
    DATABASE_URL: str = (
        "postgresql+asyncpg://panoramic:password@localhost:5432/panoramic360"
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 0
    DATABASE_POOL_RECYCLE: int = 3600

    # ─── Redis ────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ─── Storage ──────────────────────────────────────────────
    STORAGE_PROVIDER: str = "minio"
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY_ID: str = "minioadmin"
    S3_SECRET_ACCESS_KEY: str = "minioadmin"
    S3_REGION: str = "us-east-1"
    S3_BUCKET_PANORAMAS: str = "panoramas-raw"
    S3_BUCKET_PROCESSED: str = "panoramas-processed"
    S3_BUCKET_OUTPUTS: str = "analysis-outputs"
    S3_BUCKET_MODELS: str = "model-weights"

    # ─── ML / GPU ─────────────────────────────────────────────
    CUDA_VISIBLE_DEVICES: str = "0"
    GPU_MEMORY_FRACTION: float = 0.85
    USE_MIXED_PRECISION: bool = True
    INFERENCE_BATCH_SIZE: int = 4

    # Model paths
    ML_MODELS_DIR: Path = Path("/app/ml/models")
    SEGMENTATION_MODEL_PATH: str = "segformer_b5_construction_v2.pth"
    DETECTION_MODEL_PATH: str = "yolov8x_construction_v3.pt"
    PPE_MODEL_PATH: str = "yolov8m_ppe_v2.pt"
    DEPTH_MODEL_PATH: str = "dpt_large_construction.pth"
    SAM_MODEL_PATH: str = "sam_vit_h_4b8939.pth"
    MASK2FORMER_MODEL_PATH: str = "mask2former_swin_l_construction.pth"

    # Triton
    USE_TRITON: bool = False
    TRITON_HOST: str = "localhost"
    TRITON_HTTP_PORT: int = 8000
    TRITON_GRPC_PORT: int = 8001

    # ─── Spherical Geometry ───────────────────────────────────
    PANORAMA_MAX_WIDTH: int = 8192
    PANORAMA_MAX_HEIGHT: int = 4096
    CUBEMAP_FACE_SIZE: int = 1024
    TILING_OVERLAP: float = 0.1
    POLAR_CORRECTION_STRENGTH: float = 0.8

    # ─── MLOps ────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "panoramic-360-engine"
    WANDB_PROJECT: str = "360-construction-intelligence"
    WANDB_ENTITY: str = "your-team"

    # ─── CORS ─────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    CORS_CREDENTIALS: bool = True

    # ─── Rate Limiting ────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 20

    # ─── File Uploads ─────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_IMAGE_FORMATS: List[str] = [
        "jpg", "jpeg", "png", "tiff", "tif", "exr", "hdr"
    ]
    TEMP_UPLOAD_DIR: Path = Path("/tmp/panoramic_uploads")

    # ─── Monitoring ───────────────────────────────────────────
    ENABLE_TELEMETRY: bool = True
    JAEGER_HOST: str = "jaeger"
    JAEGER_PORT: int = 6831

    # ─── Validators ───────────────────────────────────────────
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("ALLOWED_IMAGE_FORMATS", mode="before")
    @classmethod
    def parse_formats(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [fmt.strip().lower() for fmt in v.split(",")]
        return v

    @model_validator(mode="after")
    def ensure_temp_dir(self) -> "Settings":
        self.TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def full_segmentation_model_path(self) -> Path:
        return self.ML_MODELS_DIR / self.SEGMENTATION_MODEL_PATH

    @property
    def full_detection_model_path(self) -> Path:
        return self.ML_MODELS_DIR / self.DETECTION_MODEL_PATH

    @property
    def full_ppe_model_path(self) -> Path:
        return self.ML_MODELS_DIR / self.PPE_MODEL_PATH


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


# Module-level singleton
settings = get_settings()
