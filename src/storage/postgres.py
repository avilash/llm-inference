import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class PostgresDatabase:
    """Manages the PostgreSQL engine, session lifecycle, and connection pool as a Singleton."""

    _instance: Optional["PostgresDatabase"] = None

    def __new__(cls, *args, **kwargs) -> "PostgresDatabase":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, settings: Optional[Settings] = None) -> None:
        if getattr(self, "_initialized", False):
            return
        self._settings = settings or get_settings()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialized = True

    @classmethod
    async def reset(cls) -> None:
        """Dispose of connections and reset the singleton instance."""
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None

    @property
    def engine(self) -> AsyncEngine:
        """Get or lazily create the AsyncEngine."""
        if self._engine is None:
            self._engine = create_async_engine(
                self._settings.database_url,
                pool_size=self._settings.db_pool_size,
                max_overflow=self._settings.db_max_overflow,
                pool_timeout=self._settings.db_pool_timeout,
                pool_pre_ping=True,
                echo=self._settings.db_echo,
            )
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get or lazily create the session factory."""
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
        return self._session_factory

    async def init(self) -> None:
        """Initialize engine and verify database connectivity."""
        logger.info("Initializing PostgreSQL async engine...")
        async with self.engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connection established successfully.")

    async def close(self) -> None:
        """Dispose of connection pool gracefully."""
        if self._engine is not None:
            logger.info("Closing PostgreSQL connection pool...")
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("PostgreSQL engine disposed.")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Context manager for managing transactions and sessions."""
        async with self.session_factory() as sess:
            try:
                yield sess
            except Exception:
                await sess.rollback()
                raise
            finally:
                await sess.close()

    async def check_health(self) -> bool:
        """Execute a lightweight ping to verify PostgreSQL health."""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.warning(f"PostgreSQL health check failed: {exc}")
            return False
