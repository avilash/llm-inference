import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class KeyCreateRequest(BaseModel):
    """Request payload for generating a new API key."""

    name: str = Field(..., description="Human-readable name or label for the API key", min_length=1, max_length=255)
    tenant_id: str = Field(default="default", description="Associated tenant or organization ID", max_length=64)
    rate_limit_rpm: int = Field(default=600, ge=1, description="Requests per minute limit")
    rate_limit_tpm: int = Field(default=100000, ge=1, description="Tokens per minute limit")


class KeyCreateResponse(BaseModel):
    """
    Response returned when an API key is created.
    The full raw api_key is displayed ONLY ONCE in this response.
    """

    model_config = ConfigDict(from_attributes=True)

    key_id: str
    name: str
    tenant_id: str
    api_key: str = Field(..., description="Full secret API key. Store this securely; it will not be shown again.")
    rate_limit_rpm: int
    rate_limit_tpm: int
    created_at: datetime


class KeyGetResponse(BaseModel):
    """
    Response schema representing an API key record (excluding hashed secret).
    """

    model_config = ConfigDict(from_attributes=True)

    id: Optional[uuid.UUID] = None
    key_id: str
    name: str
    tenant_id: str
    is_active: bool
    rate_limit_rpm: int
    rate_limit_tpm: int
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class AuthenticatedClient(BaseModel):
    """
    Context model representing the authenticated client/tenant.
    Provides caller metadata (key ID, tenant ID) to route handlers.
    """

    key_id: str
    tenant_id: str
    is_active: bool = True


class ChatMessage(BaseModel):
    """Message item in a chat conversation."""

    model_config = ConfigDict(extra="ignore")

    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    """Request payload for /v1/chat/completions."""

    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = Field(default=None, description="Model ID to route the inference request to")
    messages: list[ChatMessage] = Field(..., min_length=1, description="List of conversation turns")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum number of tokens to generate")
    max_new_tokens: Optional[int] = Field(default=128, ge=1, description="Alias for max_tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    stream: bool = Field(default=False, description="Whether to stream back partial progress via SSE")


class CompletionRequest(BaseModel):
    """Request payload for /v1/completions."""

    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = Field(default=None, description="Model ID to route the inference request to")
    prompt: str = Field(..., min_length=1, description="Input text prompt for completion")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum number of tokens to generate")
    max_new_tokens: Optional[int] = Field(default=128, ge=1, description="Alias for max_tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    stream: bool = Field(default=False, description="Whether to stream back partial progress via SSE")


