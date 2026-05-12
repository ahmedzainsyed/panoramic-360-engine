"""Object storage service (MinIO/S3 compatible)."""
from __future__ import annotations
import io
from typing import Optional
import aiobotocore.session
import structlog
from app.core.config import settings

logger = structlog.get_logger(__name__)


class StorageService:
    """Async S3-compatible object storage client."""

    def __init__(self):
        self._session = aiobotocore.session.get_session()

    def _client_kwargs(self):
        kwargs = dict(
            service_name="s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.S3_ACCESS_KEY_ID,
            aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
            region_name=settings.S3_REGION,
        )
        return kwargs

    async def upload_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict] = None,
    ) -> str:
        async with self._session.create_client(**self._client_kwargs()) as client:
            await client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                Metadata={k: str(v) for k, v in (metadata or {}).items()},
            )
        logger.debug("storage_upload_ok", bucket=bucket, key=key, size=len(data))
        return f"{settings.S3_ENDPOINT_URL}/{bucket}/{key}"

    async def download_bytes(self, bucket: str, key: str) -> bytes:
        async with self._session.create_client(**self._client_kwargs()) as client:
            response = await client.get_object(Bucket=bucket, Key=key)
            async with response["Body"] as stream:
                data = await stream.read()
        return data

    async def generate_presigned_url(self, bucket: str, key: str, expires_in: int = 3600) -> str:
        async with self._session.create_client(**self._client_kwargs()) as client:
            url = await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        return url

    async def delete_object(self, bucket: str, key: str) -> None:
        async with self._session.create_client(**self._client_kwargs()) as client:
            await client.delete_object(Bucket=bucket, Key=key)

    async def ensure_buckets_exist(self) -> None:
        buckets = [
            settings.S3_BUCKET_PANORAMAS,
            settings.S3_BUCKET_PROCESSED,
            settings.S3_BUCKET_OUTPUTS,
            settings.S3_BUCKET_MODELS,
        ]
        async with self._session.create_client(**self._client_kwargs()) as client:
            existing = {b["Name"] for b in (await client.list_buckets()).get("Buckets", [])}
            for bucket in buckets:
                if bucket not in existing:
                    await client.create_bucket(Bucket=bucket)
                    logger.info("storage_bucket_created", bucket=bucket)
