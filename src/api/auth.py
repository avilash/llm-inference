import hashlib
import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from src.api.schemas import (
    AuthenticatedClient,
    KeyCreateRequest,
    KeyCreateResponse,
    KeyGetResponse,
)
from src.core.config import get_settings
from src.storage import PostgresDatabase, RedisClient
from src.storage.models import APIKey

logger = logging.getLogger(__name__)

# Security scheme for OpenAPI / Swagger UI documentation
bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Bearer token authentication. Enter your API key.",
)


def hash_token(token: str) -> str:
    """Compute the SHA-256 hex digest for an API key token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def get_cached_session(token_hash: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached session metadata from Redis by token hash."""
    try:
        redis = RedisClient().client
        settings = get_settings()
        cached = await redis.get(f"{settings.auth_cache_prefix}{token_hash}")
        if cached:
            return json.loads(cached)
    except Exception as exc:
        logger.warning(f"Redis get failed for session cache: {exc}")
    return None


async def set_cached_session(
    token_hash: str,
    session_data: Dict[str, Any],
    ttl: Optional[int] = None,
) -> None:
    """Store session metadata in Redis with TTL."""
    try:
        redis = RedisClient().client
        settings = get_settings()
        expire = ttl if ttl is not None else settings.auth_cache_ttl_seconds
        await redis.set(
            f"{settings.auth_cache_prefix}{token_hash}",
            json.dumps(session_data),
            ex=expire,
        )
    except Exception as exc:
        logger.warning(f"Redis set failed for session cache: {exc}")


async def invalidate_cached_session(token_hash: str) -> None:
    """Evict cached session metadata from Redis."""
    try:
        redis = RedisClient().client
        settings = get_settings()
        await redis.delete(f"{settings.auth_cache_prefix}{token_hash}")
    except Exception as exc:
        logger.warning(f"Redis delete failed for session cache: {exc}")


def generate_api_key_credentials() -> Tuple[str, str, str]:
    """
    Generate public key_id, raw plaintext token, and its SHA-256 hash.

    Format: <prefix><key_id>_<secret>
    Returns:
        (key_id, raw_key, hashed_secret)
    """
    settings = get_settings()
    key_id = f"{settings.auth_key_prefix}{secrets.token_hex(6)}"
    secret = secrets.token_urlsafe(32)
    raw_key = f"{key_id}_{secret}"
    hashed_secret = hash_token(raw_key)
    return key_id, raw_key, hashed_secret


async def create_api_key_service(
    payload: KeyCreateRequest,
) -> KeyCreateResponse:
    """
    Business service to generate and persist a new API key record in PostgreSQL.
    Manages its own database session and transaction.
    Returns the response with the plaintext key disclosed only once.
    """
    async with PostgresDatabase().session() as session:
        key_id, raw_key, hashed_secret = generate_api_key_credentials()

        api_key_record = APIKey(
            key_id=key_id,
            hashed_secret=hashed_secret,
            name=payload.name,
            tenant_id=payload.tenant_id,
            rate_limit_rpm=payload.rate_limit_rpm,
            rate_limit_tpm=payload.rate_limit_tpm,
            is_active=True,
        )
        session.add(api_key_record)
        await session.commit()
        await session.refresh(api_key_record)

        return KeyCreateResponse(
            key_id=api_key_record.key_id,
            name=api_key_record.name,
            tenant_id=api_key_record.tenant_id,
            api_key=raw_key,
            rate_limit_rpm=api_key_record.rate_limit_rpm,
            rate_limit_tpm=api_key_record.rate_limit_tpm,
            created_at=api_key_record.created_at,
        )


async def get_api_keys_service(
    tenant_id: Optional[str] = None,
) -> List[KeyGetResponse]:
    """
    Business service to retrieve all API key records from PostgreSQL.
    Optionally filters by tenant_id.
    """
    async with PostgresDatabase().session() as session:
        stmt = select(APIKey).order_by(APIKey.created_at.desc())
        if tenant_id:
            stmt = stmt.where(APIKey.tenant_id == tenant_id)
        result = await session.execute(stmt)
        records = result.scalars().all()
        return [KeyGetResponse.model_validate(record) for record in records]


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> AuthenticatedClient:
    """
    Central FastAPI authentication dependency with Redis fast-path session caching.

    Extracts Bearer token from 'Authorization' header, checks Redis session cache,
    and falls back to PostgreSQL if cache misses.

    Raises:
        HTTPException(401): If token is missing, expired, or invalid.

    Returns:
        AuthenticatedClient: Authenticated caller details for downstream use.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header or invalid scheme. Expected 'Authorization: Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()
    token_hash = hash_token(token)

    # 1. Fast Path: Check Redis Session Cache
    cached_session = await get_cached_session(token_hash)
    if cached_session is not None:
        if not cached_session.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or inactive API key provided.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        expires_at_str = cached_session.get("expires_at")
        if expires_at_str:
            expires_at = datetime.fromisoformat(expires_at_str)
            if expires_at < datetime.now(timezone.utc):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key has expired.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        return AuthenticatedClient(
            key_id=cached_session["key_id"],
            tenant_id=cached_session["tenant_id"],
            is_active=cached_session.get("is_active", True),
        )

    # 2. Slow Path: Cache Miss -> Fallback to PostgreSQL
    async with PostgresDatabase().session() as session:
        stmt = select(APIKey).where(APIKey.hashed_secret == token_hash)
        result = await session.execute(stmt)
        key_record = result.scalar_one_or_none()

        if key_record is None or not key_record.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or inactive API key provided.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if key_record.expires_at is not None and key_record.expires_at < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key has expired.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Track usage timestamp on PostgreSQL during cache miss
        key_record.last_used_at = datetime.now(timezone.utc)
        await session.commit()

        # 3. Populate Redis Session Cache for future requests
        session_data = {
            "key_id": key_record.key_id,
            "tenant_id": key_record.tenant_id,
            "is_active": key_record.is_active,
            "expires_at": key_record.expires_at.isoformat() if key_record.expires_at else None,
            "rate_limit_rpm": key_record.rate_limit_rpm,
            "rate_limit_tpm": key_record.rate_limit_tpm,
        }
        await set_cached_session(token_hash, session_data)

        return AuthenticatedClient(
            key_id=key_record.key_id,
            tenant_id=key_record.tenant_id,
            is_active=key_record.is_active,
        )
