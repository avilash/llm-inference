import logging
from typing import Optional
from redis.asyncio import ConnectionPool, Redis

from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Manages Redis connection pool and async client lifecycle as a Singleton."""

    _instance: Optional["RedisClient"] = None

    def __new__(cls, *args, **kwargs) -> "RedisClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, settings: Optional[Settings] = None) -> None:
        if getattr(self, "_initialized", False):
            return
        self._settings = settings or get_settings()
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = None
        self._initialized = True

    @classmethod
    async def reset(cls) -> None:
        """Close connections and reset the singleton instance."""
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None

    @property
    def pool(self) -> ConnectionPool:
        """Get the active Redis connection pool or create one."""
        if self._pool is None:
            self._pool = ConnectionPool.from_url(
                self._settings.redis_url,
                max_connections=self._settings.redis_max_connections,
                socket_timeout=self._settings.redis_socket_timeout,
                decode_responses=True,
            )
        return self._pool

    @property
    def client(self) -> Redis:
        """Get the active async Redis client instance."""
        if self._client is None:
            self._client = Redis(connection_pool=self.pool)
        return self._client

    async def init(self) -> None:
        """Initialize Redis pool and verify connectivity with a ping."""
        logger.info("Initializing Redis connection...")
        pong = await self.client.ping()
        if not pong:
            raise ConnectionError("Redis ping failed during initialization.")
        logger.info("Redis connection established successfully.")

    async def close(self) -> None:
        """Close the Redis client and connection pool gracefully."""
        if self._client is not None:
            logger.info("Closing Redis client...")
            await self._client.aclose()
            self._client = None
        if self._pool is not None:
            logger.info("Disconnecting Redis connection pool...")
            await self._pool.disconnect()
            self._pool = None
            logger.info("Redis pool disconnected.")

    async def check_health(self) -> bool:
        """Execute a ping to verify Redis health."""
        try:
            return bool(await self.client.ping())
        except Exception as exc:
            logger.warning(f"Redis health check failed: {exc}")
            return False
