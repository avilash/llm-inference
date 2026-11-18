import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional
from fastapi import APIRouter, Depends, status
from sse_starlette.sse import EventSourceResponse

from src.api.auth import (
    create_api_key_service,
    get_api_keys_service,
    verify_api_key,
)
from src.api.schemas import (
    AuthenticatedClient,
    ChatCompletionRequest,
    CompletionRequest,
    KeyCreateRequest,
    KeyCreateResponse,
    KeyGetResponse,
)
from src.engine import SamplingParams, get_async_engine

# Public router: accessible without authentication (e.g. liveness/readiness probes, key generation)
public_router = APIRouter()

# Protected router: central authentication dependency applied to ALL enclosed routes
protected_router = APIRouter(dependencies=[Depends(verify_api_key)])


@public_router.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
    }


@public_router.post(
    "/v1/keys",
    response_model=KeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
)
async def create_api_key(payload: KeyCreateRequest) -> KeyCreateResponse:
    """Generate and store a new API key in PostgreSQL, returning the plaintext key once."""
    return await create_api_key_service(payload)


@public_router.get(
    "/v1/keys",
    response_model=List[KeyGetResponse],
    tags=["Authentication"],
)
async def get_keys(tenant_id: Optional[str] = None) -> List[KeyGetResponse]:
    """Retrieve API keys (unauthenticated for development/testing)."""
    return await get_api_keys_service(tenant_id=tenant_id)


@protected_router.get("/v1/models", tags=["Inference"])
async def list_models(auth: AuthenticatedClient = Depends(verify_api_key)) -> Dict[str, Any]:
    engine = get_async_engine()
    return {
        "object": "list",
        "data": [
            {
                "id": engine.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llm-inference",
            }
        ],
    }


@protected_router.post("/v1/chat/completions", tags=["Inference"])
async def create_chat_completion(
    payload: ChatCompletionRequest,
    auth: AuthenticatedClient = Depends(verify_api_key),
) -> Any:
    engine = get_async_engine()
    model_name = payload.model or engine.model_name
    max_new_tokens = payload.max_tokens if payload.max_tokens is not None else (payload.max_new_tokens or 128)
    formatted_messages = [msg.model_dump() for msg in payload.messages]

    sampling_params = SamplingParams(
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=max_new_tokens,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if payload.stream:
        async def event_generator() -> AsyncGenerator[Dict[str, str], None]:
            async for chunk in engine.chat(messages=formatted_messages, sampling_params=sampling_params):
                delta_text = chunk["text"]
                finish_reason = chunk.get("finish_reason")
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_text} if delta_text else {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield {"data": json.dumps(data)}
            yield {"data": "[DONE]"}

        return EventSourceResponse(event_generator())

    # Non-streaming: accumulate output from the continuous batching engine
    collected_text: List[str] = []
    finish_reason = "stop"
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in engine.chat(messages=formatted_messages, sampling_params=sampling_params):
        if chunk["text"]:
            collected_text.append(chunk["text"])
        if chunk.get("finish_reason"):
            finish_reason = chunk["finish_reason"]
        prompt_tokens = chunk.get("num_prompt_tokens", prompt_tokens)
        completion_tokens = chunk.get("num_output_tokens", completion_tokens)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(collected_text),
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@protected_router.post("/v1/completions", tags=["Inference"])
async def create_completion(
    payload: CompletionRequest,
    auth: AuthenticatedClient = Depends(verify_api_key),
) -> Any:
    engine = get_async_engine()
    model_name = payload.model or engine.model_name
    max_new_tokens = payload.max_tokens if payload.max_tokens is not None else (payload.max_new_tokens or 128)

    sampling_params = SamplingParams(
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=max_new_tokens,
    )

    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if payload.stream:
        async def event_generator() -> AsyncGenerator[Dict[str, str], None]:
            async for chunk in engine.generate(prompt=payload.prompt, sampling_params=sampling_params):
                delta_text = chunk["text"]
                finish_reason = chunk.get("finish_reason")
                data = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": delta_text,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield {"data": json.dumps(data)}
            yield {"data": "[DONE]"}

        return EventSourceResponse(event_generator())

    # Non-streaming: accumulate output from the continuous batching engine
    collected_text: List[str] = []
    finish_reason = "stop"
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in engine.generate(prompt=payload.prompt, sampling_params=sampling_params):
        if chunk["text"]:
            collected_text.append(chunk["text"])
        if chunk.get("finish_reason"):
            finish_reason = chunk["finish_reason"]
        prompt_tokens = chunk.get("num_prompt_tokens", prompt_tokens)
        completion_tokens = chunk.get("num_output_tokens", completion_tokens)

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created_ts,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": "".join(collected_text),
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# Aggregated router exported to FastAPI app (maintains backwards compatibility with app.py)
router = APIRouter()
router.include_router(public_router)
router.include_router(protected_router)

