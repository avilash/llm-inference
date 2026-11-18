import logging
from typing import Dict

from src.storage.base import Base
from src.storage.models import APIKey
from src.storage.postgres import PostgresDatabase
from src.storage.redis import RedisClient

logger = logging.getLogger(__name__)


async def init_storage() -> None:
    """Initialize both PostgreSQL and Redis connections."""
    logger.info("Initializing storage connectors...")
    await PostgresDatabase().init()
    await RedisClient().init()
    logger.info("All storage connectors initialized successfully.")


async def create_db_tables() -> None:
    """Create all registered database tables in PostgreSQL."""
    logger.info("Verifying and creating database tables...")
    async with PostgresDatabase().engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified successfully.")


async def close_storage() -> None:
    """Gracefully close all storage connections."""
    logger.info("Closing all storage connections...")
    await PostgresDatabase().close()
    await RedisClient().close()
    logger.info("All storage connections closed.")


async def check_storage_health() -> Dict[str, bool]:
    """Check health across all storage layers."""
    return {
        "postgres": await PostgresDatabase().check_health(),
        "redis": await RedisClient().check_health(),
    }


__all__ = [
    "Base",
    "APIKey",
    "PostgresDatabase",
    "RedisClient",
    "init_storage",
    "create_db_tables",
    "close_storage",
    "check_storage_health",
]
