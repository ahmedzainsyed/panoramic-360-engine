"""Application startup and shutdown events."""
import structlog
logger = structlog.get_logger(__name__)

async def startup_event():
    """Initialize connections, models, and resources on startup."""
    from app.db.session import create_all_tables
    await create_all_tables()
    logger.info("database_tables_initialized")

async def shutdown_event():
    """Clean up connections on shutdown."""
    from app.db.session import engine
    await engine.dispose()
    logger.info("database_connections_closed")
