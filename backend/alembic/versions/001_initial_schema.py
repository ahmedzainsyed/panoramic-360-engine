"""Initial schema - panoramas, sessions, users, analysis results.

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("users",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(512), nullable=False),
        sa.Column("full_name", sa.String(512)),
        sa.Column("role", sa.String(32), default="viewer"),
        sa.Column("organization", sa.String(512)),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("is_superuser", sa.Boolean, default=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("last_login_at", sa.DateTime),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table("site_sessions",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("location_name", sa.String(512)),
        sa.Column("description", sa.Text),
        sa.Column("panorama_count", sa.Integer, default=0),
        sa.Column("created_by", sa.String(255)),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("metadata", postgresql.JSONB),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table("panoramas",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("storage_key", sa.String(1024), nullable=False),
        sa.Column("original_filename", sa.String(512)),
        sa.Column("file_size_bytes", sa.Integer, default=0),
        sa.Column("file_hash", sa.String(64)),
        sa.Column("camera_type", sa.String(64), default="unknown"),
        sa.Column("format", sa.String(16), default="jpg"),
        sa.Column("width", sa.Integer),
        sa.Column("height", sa.Integer),
        sa.Column("location_name", sa.String(512)),
        sa.Column("gps_latitude", sa.Float),
        sa.Column("gps_longitude", sa.Float),
        sa.Column("gps_altitude", sa.Float),
        sa.Column("capture_timestamp", sa.DateTime),
        sa.Column("floor_level", sa.Integer),
        sa.Column("notes", sa.Text),
        sa.Column("status", sa.String(32), default="uploaded"),
        sa.Column("processing_started_at", sa.DateTime),
        sa.Column("processing_completed_at", sa.DateTime),
        sa.Column("exif_data", postgresql.JSONB),
        sa.Column("analysis_results", postgresql.JSONB),
        sa.Column("uploaded_by", sa.String(255)),
        sa.Column("is_deleted", sa.Boolean, default=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_panoramas_session_id", "panoramas", ["session_id"])
    op.create_index("ix_panoramas_file_hash", "panoramas", ["file_hash"])
    op.create_index("ix_panoramas_status", "panoramas", ["status"])
    op.create_index("ix_panoramas_session_created", "panoramas", ["session_id", "created_at"])

    op.create_table("analysis_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("panorama_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("panoramas.id"), nullable=False),
        sa.Column("module", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), default="pending"),
        sa.Column("result_data", postgresql.JSONB),
        sa.Column("artifact_storage_key", sa.String(1024)),
        sa.Column("inference_time_ms", sa.Float),
        sa.Column("model_version", sa.String(64)),
        sa.Column("error_message", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_analysis_panorama_id", "analysis_results", ["panorama_id"])
    op.create_index("ix_analysis_module", "analysis_results", ["module"])


def downgrade() -> None:
    op.drop_table("analysis_results")
    op.drop_table("panoramas")
    op.drop_table("site_sessions")
    op.drop_table("users")
