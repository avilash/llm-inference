from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "LLM Inference Engine"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", alias="ENV")
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # PostgreSQL Configuration
    postgres_user: str = "postgres"
    postgres_password: str = "mysecretpassword"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "llm-inference"
    postgres_url_override: Optional[str] = Field(default=None, alias="DATABASE_URL")
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_echo: bool = False

    @property
    def database_url(self) -> str:
        if self.postgres_url_override:
            # Normalize driver for asyncpg if standard postgresql:// was provided
            if self.postgres_url_override.startswith("postgresql://"):
                return self.postgres_url_override.replace("postgresql://", "postgresql+asyncpg://", 1)
            return self.postgres_url_override
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = "mysecretpassword"
    redis_db: int = 0
    redis_url_override: Optional[str] = Field(default=None, alias="REDIS_URL")
    redis_max_connections: int = 50
    redis_socket_timeout: float = 5.0

    @property
    def redis_url(self) -> str:
        if self.redis_url_override:
            return self.redis_url_override
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # Authentication & Rate Limiting Settings
    auth_enabled: bool = True
    auth_cache_ttl_seconds: int = 300  # 5 minutes
    auth_cache_prefix: str = "auth:session:"
    auth_key_prefix: str = "inf_live_"
    rate_limit_default_rpm: int = 600
    rate_limit_default_tpm: int = 100000

    # Model & Inference Settings
    model_name: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct", alias="MODEL_NAME")
    model_device: Optional[str] = Field(default=None, alias="MODEL_DEVICE")  # None = auto-detect cuda/cpu
    model_dtype: str = Field(default="float16", alias="MODEL_DTYPE")  # float16, bfloat16, float32
    model_max_length: int = Field(default=2048, alias="MODEL_MAX_LENGTH")


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
