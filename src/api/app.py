from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI

from src.api.routes import router
from src.core.config import get_settings
from src.engine import get_async_engine
from src.storage import close_storage, create_db_tables, init_storage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycles."""
    # 1. Initialize storage connectors (PostgreSQL & Redis connection pools)
    await init_storage()
    # 2. Verify and create database tables
    await create_db_tables()
    # 3. Start continuous batching inference engine (preloads model weights via runner)
    engine = get_async_engine()
    engine.start()
    yield
    # Gracefully stop engine and dispose of connection pools on shutdown
    await engine.stop()
    await close_storage()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="High-throughput, low-latency LLM inference server",
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
