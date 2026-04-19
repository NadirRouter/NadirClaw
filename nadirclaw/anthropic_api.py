"""Anthropic Messages API compatibility layer for NadirClaw.

Provides /v1/messages endpoint that accepts Anthropic SDK format requests,
routes them through NadirClaw's smart routing, and returns responses in
Anthropic format.

For providers with Anthropic-compatible endpoints (e.g. third-party proxies),
the original Anthropic request body is forwarded directly — no format
conversion needed. This avoids the double-conversion overhead and
format-compatibility issues that arise from Anthropic→OpenAI→Anthropic.

For non-Anthropic providers (vLLM, Ollama, OpenAI), the request is converted
to OpenAI format and sent through LiteLLM.

Uses the proven "fake streaming" approach: waits for complete response from
the upstream model, then emits Anthropic SSE events.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from nadirclaw.settings import settings

logger = logging.getLogger("nadirclaw.anthropic_api")

router = APIRouter()

# Same limits as /v1/chat/completions (server.py)
_MAX_CONTENT_LENGTH = 1_000_000  # 1 MB
_MAX_ANTHROPIC_TOKENS = 128_000  # Cap for max_tokens parameter

# Provider env-var mapping for Anthropic-compatible endpoints.
# Format: provider_name → (api_base_env_var, api_key_env_var)
_ANTHROPIC_COMPAT_PROVIDERS: Dict[str, Tuple[str, str]] = {
    "zai": ("ZAI_API_BASE", "ZAI_API_KEY"),
    "kimi": ("KIMI_API_BASE", "KIMI_API_KEY"),
    "minimax": ("MINIMAX_API_BASE", "MINIMAX_API_KEY"),
    "anthropic": ("ANTHROPIC_API_BASE", "ANTHROPIC_API_KEY"),
}

# Providers that serve Anthropic-native endpoints — no format conversion needed.
_ANTHROPIC_NATIVE_PROVIDERS = set(_ANTHROPIC_COMPAT_PROVIDERS)


def get_anthropic_compat_endpoint(
    provider: str,
) -> Optional[Tuple[str, str]]:
    """Return (api_base, api_key) if the provider has an Anthropic-compatible endpoint.

    Reads credentials from .env file directly to avoid override by process
    environment variables (e.g., Claude Code injects ANTHROPIC_API_KEY=local).
    """
    if provider not in _ANTHROPIC_COMPAT_PROVIDERS:
        return None
    base_env, key_env = _ANTHROPIC_COMPAT_PROVIDERS[provider]

    # Read from .env file first (avoids process env overrides like "local")
    from dotenv import dotenv_values
    from pathlib import Path
    _env_file = Path.home() / ".nadirclaw" / ".env"
    file_vals = dotenv_values(_env_file) if _env_file.exists() else {}

    api_base = file_vals.get(base_env, "") or os.getenv(base_env, "")
    api_key = file_vals.get(key_env, "") or os.getenv(key_env, "")

    # Sanity check: if api_key looks like a placeholder, skip direct path
    if api_key in ("local", "dummy", "sk-placeholder", ""):
        return None

    if not api_base or not api_key:
        return None
    api_base = api_base.rstrip("/")
    return api_base, api_key


async def call_anthropic_direct(
    api_base: str,
    api_key: str,
    model: str,
    body: Dict[str, Any],
    timeout: float = 600.0,
) -> Dict[str, Any]:
    """Call an Anthropic-compatible endpoint directly, no format conversion.

    Forwards the original Anthropic request body verbatim and returns the
    raw Anthropic response.  This avoids the Anthropic→OpenAI→Anthropic
    double-conversion that can cause format issues on some providers.
    """
    url = f"{api_base}/v1/messages"

    # Build the Anthropic request body from the original
    ant_body: Dict[str, Any] = {"model": model, "max_tokens": body.get("max_tokens", 4096)}
    if body.get("messages"):
        ant_body["messages"] = body["messages"]
    if body.get("system"):
        ant_body["system"] = body["system"]
    if body.get("tools"):
        ant_body["tools"] = body["tools"]
    if body.get("tool_choice"):
        ant_body["tool_choice"] = body["tool_choice"]
    if body.get("thinking"):
        ant_body["thinking"] = body["thinking"]
    if body.get("temperature") is not None:
        ant_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        ant_body["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        ant_body["stop_sequences"] = body["stop_sequences"]
    if body.get("metadata"):
        ant_body["metadata"] = body["metadata"]

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    # Forward beta headers if present
    beta = body.get("anthropic_beta", "")
    if beta:
        headers["anthropic-beta"] = beta

    logger.debug("Direct Anthropic call: model=%s url=%s", model, url)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=ant_body)

    if resp.status_code >= 400:
        error_text = resp.text[:500]
        logger.warning(
            "Direct Anthropic call failed (%s): %s", resp.status_code, error_text,
        )
        from litellm.exceptions import (
            AuthenticationError as LiteLLMAuthError,
            BadRequestError as LiteLLMBadRequestError,
            InternalServerError as LiteLLMInternalServerError,
            RateLimitError as LiteLLMRateLimitError,
        )
        if resp.status_code == 400:
            raise LiteLLMBadRequestError(
                message=error_text, model=model, llm_provider="anthropic",
            )
        if resp.status_code == 401:
            raise LiteLLMAuthError(
                message=error_text, model=model, llm_provider="anthropic",
            )
        if resp.status_code == 429:
            raise LiteLLMRateLimitError(
                message=error_text, model=model, llm_provider="anthropic",
            )
        raise LiteLLMInternalServerError(
            message=error_text, model=model, llm_provider="anthropic",
        )

    return resp.json()


def anthropic_response_to_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stats from a raw Anthropic response for logging/telemetry."""
    usage = data.get("usage", {})
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "stop_reason": data.get("stop_reason", ""),
        "model": data.get("model", ""),
    }


# ---------------------------------------------------------------------------
# Format conversion: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def anthropic_to_openai_messages(
    messages: List[Dict[str, Any]],
    system: Optional[Union[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Convert Anthropic Messages API format to OpenAI chat completion format."""
    result: List[Dict[str, Any]] = []

    if system:
        if isinstance(system, str):
            result.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = [
                b.get("text", "")
                for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if text_parts:
                result.append({"role": "system", "content": "\n".join(text_parts)})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "assistant":
            if isinstance(content, list):
                text_parts = []
                tool_calls = []
                for i, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                entry: Dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                result.append(entry)
            else:
                result.append({"role": "assistant", "content": content})

        elif role == "user":
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, list):
                            tc_texts = [
                                tc.get("text", "")
                                for tc in tool_content
                                if isinstance(tc, dict) and tc.get("type") == "text"
                            ]
                            tool_content = "\n".join(tc_texts)
                        result.append({
                            "role": "tool",
                            "content": str(tool_content),
                            "tool_call_id": block.get("tool_use_id", ""),
                        })
                if text_parts:
                    result.append({"role": "user", "content": "\n".join(text_parts)})
            else:
                result.append({"role": "user", "content": content})
        else:
            result.append({"role": role, "content": str(content) if content else ""})

    return result


def anthropic_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function format.

    Note: Anthropic built-in tools (bash_20250124, text_editor_20250124, etc.)
    have no ``input_schema`` and are silently dropped. Only tools with
    ``type="custom"`` or an explicit ``input_schema`` are forwarded.
    """
    result = []
    dropped = []
    for tool in tools:
        if tool.get("type") == "custom" or "input_schema" in tool:
            result.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        else:
            dropped.append(tool.get("name", tool.get("type", "?")))
    if dropped:
        logger.warning(
            "Dropped %d Anthropic built-in tools (no input_schema): %s",
            len(dropped), dropped,
        )
    return result


# ---------------------------------------------------------------------------
# Format conversion: OpenAI → Anthropic
# ---------------------------------------------------------------------------

def openai_response_to_anthropic(
    response_data: Dict[str, Any],
    model: str,
    request_id: str,
) -> Dict[str, Any]:
    """Convert internal OpenAI-style response to Anthropic Messages API format."""
    content_blocks = []

    text = response_data.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tc in response_data.get("tool_calls", []):
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            input_data = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", str(uuid.uuid4())),
            "name": func.get("name", ""),
            "input": input_data,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    finish = response_data.get("finish_reason", "stop")
    if finish == "tool_calls":
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    return {
        "id": f"msg_{request_id}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": response_data.get("prompt_tokens", 0),
            "output_tokens": response_data.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Fake streaming: build SSE from complete response
# ---------------------------------------------------------------------------

def _build_anthropic_streaming_response(
    request_id: str,
    model: str,
    response_data: Dict[str, Any],
):
    """Build Anthropic-compatible SSE stream from a completed response.

    This is the proven "fake streaming" approach from 0.11.0:
    wait for the complete response, then emit all SSE events at once.
    """
    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        content = response_data.get("content", "") or ""
        tool_calls = response_data.get("tool_calls", [])
        input_tokens = response_data.get("prompt_tokens", 0)
        output_tokens = response_data.get("completion_tokens", 0)
        finish = response_data.get("finish_reason", "stop")

        if finish == "tool_calls":
            stop_reason = "tool_use"
        elif finish == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        msg_id = f"msg_{request_id}"

        # Event: message_start
        yield {
            "event": "message_start",
            "data": json.dumps({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            }),
        }

        block_index = 0

        # Text content block
        if content:
            yield {
                "event": "content_block_start",
                "data": json.dumps({
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                }),
            }
            yield {
                "event": "content_block_delta",
                "data": json.dumps({
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": content},
                }),
            }
            yield {
                "event": "content_block_stop",
                "data": json.dumps({
                    "type": "content_block_stop",
                    "index": block_index,
                }),
            }
            block_index += 1

        # Tool use blocks
        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                input_data = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                input_data = {}

            yield {
                "event": "content_block_start",
                "data": json.dumps({
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": func.get("name", ""),
                        "input": {},
                    },
                }),
            }
            yield {
                "event": "content_block_delta",
                "data": json.dumps({
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(input_data),
                    },
                }),
            }
            yield {
                "event": "content_block_stop",
                "data": json.dumps({
                    "type": "content_block_stop",
                    "index": block_index,
                }),
            }
            block_index += 1

        # Event: message_delta (stop reason + final usage)
        yield {
            "event": "message_delta",
            "data": json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            }),
        }

        # Event: message_stop
        yield {
            "event": "message_stop",
            "data": json.dumps({"type": "message_stop"}),
        }

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


def build_anthropic_sse_events(
    request_id: str,
    model: str,
    response_data: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build Anthropic SSE event list from a completed OpenAI-format response.

    Public helper for testing and diagnostics. Returns a list of
    ``{"event": ..., "data": ...}`` dicts following Anthropic's SSE protocol.
    """
    content = response_data.get("content", "") or ""
    tool_calls = response_data.get("tool_calls", [])
    input_tokens = response_data.get("prompt_tokens", 0)
    output_tokens = response_data.get("completion_tokens", 0)
    finish = response_data.get("finish_reason", "stop")

    stop_reason = (
        "tool_use" if finish == "tool_calls"
        else ("max_tokens" if finish == "length" else "end_turn")
    )
    msg_id = f"msg_{request_id}"
    events: List[Dict[str, str]] = []

    events.append({
        "event": "message_start",
        "data": json.dumps({
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }),
    })

    block_idx = 0

    if content:
        events.append({"event": "content_block_start", "data": json.dumps({
            "type": "content_block_start", "index": block_idx,
            "content_block": {"type": "text", "text": ""},
        })})
        events.append({"event": "content_block_delta", "data": json.dumps({
            "type": "content_block_delta", "index": block_idx,
            "delta": {"type": "text_delta", "text": content},
        })})
        events.append({"event": "content_block_stop", "data": json.dumps({
            "type": "content_block_stop", "index": block_idx,
        })})
        block_idx += 1

    for tc in tool_calls:
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            input_data = {}

        events.append({"event": "content_block_start", "data": json.dumps({
            "type": "content_block_start", "index": block_idx,
            "content_block": {
                "type": "tool_use", "id": tc.get("id", str(uuid.uuid4())),
                "name": func.get("name", ""), "input": {},
            },
        })})
        events.append({"event": "content_block_delta", "data": json.dumps({
            "type": "content_block_delta", "index": block_idx,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(input_data)},
        })})
        events.append({"event": "content_block_stop", "data": json.dumps({
            "type": "content_block_stop", "index": block_idx,
        })})
        block_idx += 1

    events.append({"event": "message_delta", "data": json.dumps({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })})
    events.append({"event": "message_stop", "data": json.dumps({
        "type": "message_stop",
    })})

    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_last_user_text(messages: List[Dict[str, Any]]) -> str:
    """Extract text from the last user message in Anthropic format."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
            ]
            if parts:
                return "\n".join(parts)
    return ""


def _extract_anthropic_text(data: Dict[str, Any]) -> str:
    """Extract concatenated text from an Anthropic response."""
    parts = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _extract_anthropic_tool_calls(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool_calls in OpenAI-compatible format from an Anthropic response."""
    result = []
    for block in data.get("content", []):
        if block.get("type") == "tool_use":
            result.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
    return result


def _build_anthropic_streaming_response_from_raw(
    request_id: str,
    model: str,
    raw_response: Dict[str, Any],
):
    """Build SSE stream from a raw Anthropic response (direct path)."""
    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        usage = raw_response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        msg_id = raw_response.get("id", f"msg_{request_id}")

        yield {
            "event": "message_start",
            "data": json.dumps({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            }),
        }

        block_index = 0
        for block in raw_response.get("content", []):
            block_type = block.get("type", "text")
            yield {
                "event": "content_block_start",
                "data": json.dumps({
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": block,
                }),
            }
            if block_type == "text":
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": block.get("text", "")},
                    }),
                }
            elif block_type == "tool_use":
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(block.get("input", {})),
                        },
                    }),
                }
            yield {
                "event": "content_block_stop",
                "data": json.dumps({"type": "content_block_stop", "index": block_index}),
            }
            block_index += 1

        yield {
            "event": "message_delta",
            "data": json.dumps({
                "type": "message_delta",
                "delta": {
                    "stop_reason": raw_response.get("stop_reason", "end_turn"),
                    "stop_sequence": raw_response.get("stop_sequence"),
                },
                "usage": {"output_tokens": output_tokens},
            }),
        }
        yield {
            "event": "message_stop",
            "data": json.dumps({"type": "message_stop"}),
        }

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/v1/messages")
async def anthropic_messages(raw_request: Request):
    """Anthropic Messages API compatibility endpoint.

    Two call paths based on target provider:
    - Path A (Anthropic-compatible): Direct httpx call, original body forwarded verbatim.
      No format conversion needed — avoids Anthropic→OpenAI→Anthropic double-conversion.
    - Path B (non-Anthropic): Convert to OpenAI format, call through LiteLLM.

    Both paths share the same routing pipeline for model selection.
    """
    from nadirclaw.server import (
        _call_with_fallback,
        _dispatch_model,
        _extract_request_metadata,
        _log_request,
        _smart_route_full,
        ChatMessage,
        ChatCompletionRequest,
        UserSession,
    )

    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        body = await raw_request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Validate auth — support both Bearer token and x-api-key
    auth_header = raw_request.headers.get("authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    x_api_key = raw_request.headers.get("x-api-key", "")
    effective_token = token or x_api_key
    if settings.AUTH_TOKEN and effective_token != settings.AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # --- Rate limiting (parity with /v1/chat/completions) ---
    from nadirclaw.server import _rate_limiter
    user_id = effective_token or "anonymous"
    retry_after = _rate_limiter.check(user_id)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    # Extract Anthropic fields
    ant_model = body.get("model", "")
    # ALL claude-* models → "auto" for smart routing (proven approach from 0.11.0)
    if ant_model.startswith("claude-"):
        ant_model = "auto"
    ant_stream = body.get("stream", False)
    ant_max_tokens = body.get("max_tokens", 4096)
    ant_tools = body.get("tools", [])

    # Cap max_tokens to prevent abuse
    if ant_max_tokens > _MAX_ANTHROPIC_TOKENS:
        logger.warning(
            "max_tokens=%d exceeds cap, clamping to %d",
            ant_max_tokens, _MAX_ANTHROPIC_TOKENS,
        )
        ant_max_tokens = _MAX_ANTHROPIC_TOKENS

    prompt_text = _extract_last_user_text(body.get("messages", []))

    # Convert to OpenAI format
    openai_messages = anthropic_to_openai_messages(
        body.get("messages", []),
        body.get("system"),
    )
    openai_tools = anthropic_tools_to_openai(ant_tools) if ant_tools else []

    # --- Input size validation (parity with /v1/chat/completions) ---
    total_content_len = sum(
        len(m.get("content", "")) if isinstance(m.get("content"), str)
        else len(json.dumps(m.get("content", "")))
        for m in openai_messages
    )
    if total_content_len > _MAX_CONTENT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Request content too large ({total_content_len:,} chars). "
                   f"Maximum is {_MAX_CONTENT_LENGTH:,} chars.",
        )

    # Build ChatCompletionRequest
    chat_messages = []
    for m in openai_messages:
        kwargs: Dict[str, Any] = {"role": m["role"], "content": m.get("content")}
        if m["role"] == "tool":
            kwargs["tool_call_id"] = m.get("tool_call_id", "")
            kwargs["content"] = m.get("content", "")
        if m["role"] == "assistant" and "tool_calls" in m:
            kwargs["tool_calls"] = m["tool_calls"]
        chat_messages.append(ChatMessage(**kwargs))

    req_data: Dict[str, Any] = {
        "messages": [{"role": m.role, "content": m.content, **(m.model_extra or {})} for m in chat_messages],
        "model": ant_model,
        "max_tokens": ant_max_tokens,
        "stream": False,  # Always non-streaming internally
    }
    if openai_tools:
        req_data["tools"] = openai_tools

    request = ChatCompletionRequest(**req_data)
    req_meta = _extract_request_metadata(request)

    # --- Full routing pipeline (same as /v1/chat/completions) ---
    from nadirclaw.routing import (
        apply_routing_modifiers,
        get_session_cache,
        resolve_alias,
        resolve_profile,
        get_pool_for_model,
        select_from_pool,
    )

    profile = resolve_profile(request.model)

    if profile == "eco":
        selected_model = settings.SIMPLE_MODEL
        analysis_info: Dict[str, Any] = {
            "strategy": "profile:eco", "selected_model": selected_model,
            "tier": "simple", "confidence": 1.0, "complexity_score": 0,
        }
    elif profile == "premium":
        selected_model = settings.COMPLEX_MODEL
        analysis_info = {
            "strategy": "profile:premium", "selected_model": selected_model,
            "tier": "complex", "confidence": 1.0, "complexity_score": 0,
        }
    elif profile == "free":
        selected_model = settings.FREE_MODEL
        analysis_info = {
            "strategy": "profile:free", "selected_model": selected_model,
            "tier": "free", "confidence": 1.0, "complexity_score": 0,
        }
    elif profile == "reasoning":
        selected_model = settings.REASONING_MODEL
        analysis_info = {
            "strategy": "profile:reasoning", "selected_model": selected_model,
            "tier": "reasoning", "confidence": 1.0, "complexity_score": 0,
        }
    elif request.model and request.model != "auto" and profile is None:
        resolved = resolve_alias(request.model)
        if resolved:
            selected_model = resolved
            analysis_info = {
                "strategy": "alias", "selected_model": selected_model,
                "alias_from": request.model, "tier": "direct",
                "confidence": 1.0, "complexity_score": 0,
            }
        else:
            selected_model = request.model
            analysis_info = {
                "strategy": "direct", "selected_model": selected_model,
                "tier": "direct", "confidence": 1.0, "complexity_score": 0,
            }
    else:
        # Smart routing (auto / claude-* → auto)
        session_cache = get_session_cache()
        cached = session_cache.get(request.messages)
        if cached:
            cached_model, cached_tier = cached
            selected_model = cached_model
            analysis_info = {
                "strategy": "session-cache", "selected_model": selected_model,
                "tier": cached_tier, "confidence": 1.0, "complexity_score": 0,
            }
            selected_model, final_tier, routing_info = apply_routing_modifiers(
                base_model=selected_model, base_tier=cached_tier,
                request_meta=req_meta, messages=request.messages,
                simple_model=settings.SIMPLE_MODEL, complex_model=settings.COMPLEX_MODEL,
                reasoning_model=settings.REASONING_MODEL, free_model=settings.FREE_MODEL,
                sonnet_model=settings.SONNET_MODEL,
                explore_model=settings.EXPLORE_MODEL, subagent_model=settings.SUBAGENT_MODEL,
                review_model=settings.REVIEW_MODEL,
            )
            if final_tier != cached_tier:
                analysis_info["tier"] = final_tier
                analysis_info["selected_model"] = selected_model
                analysis_info["routing_modifiers"] = routing_info
        else:
            selected_model, analysis_info = await _smart_route_full(
                request.messages, UserSession({"id": "anthropic_api"})
            )
            selected_model, final_tier, routing_info = apply_routing_modifiers(
                base_model=selected_model, base_tier=analysis_info.get("tier", "simple"),
                request_meta=req_meta, messages=request.messages,
                simple_model=settings.SIMPLE_MODEL, complex_model=settings.COMPLEX_MODEL,
                reasoning_model=settings.REASONING_MODEL, free_model=settings.FREE_MODEL,
                sonnet_model=settings.SONNET_MODEL,
                explore_model=settings.EXPLORE_MODEL, subagent_model=settings.SUBAGENT_MODEL,
                review_model=settings.REVIEW_MODEL,
            )
            analysis_info["tier"] = final_tier
            analysis_info["selected_model"] = selected_model
            analysis_info["routing_modifiers"] = routing_info
            session_cache.put(request.messages, selected_model, final_tier)

    # Pool selection — skip for capability-critical tiers
    _tier = analysis_info.get("tier", "")
    if _tier not in ("sonnet", "reasoning", "review", "long_context"):
        pool_name = get_pool_for_model(selected_model)
        if pool_name:
            pool_model = select_from_pool(pool_name)
            if pool_model:
                logger.info("Pool %s: %s → %s", pool_name, selected_model, pool_model)
                selected_model = pool_model

    # Call model — two paths based on provider type
    from nadirclaw.credentials import detect_provider
    provider = detect_provider(selected_model)

    # Build fallback chain for this tier
    tier = analysis_info.get("tier", "simple")
    fallback_chain = settings.get_tier_fallback_chain(tier)
    # Remove the primary model from the chain
    fallback_chain = [m for m in fallback_chain if m != selected_model]

    try:
        from nadirclaw.telemetry import record_llm_call, trace_span

        with trace_span("anthropic_messages", {"nadirclaw.tier": tier}) as span:
            # Try direct Anthropic call first, then fallback chain
            raw_response = None
            fallback_from = None
            final_model = selected_model

            # Build candidate list: primary + fallback chain
            # When tools are present, prefer direct-Anthropic candidates (better
            # tool-call schema adherence) over LiteLLM candidates like gpt-5.4
            # which may mangle parameter names (e.g., taskId vs task_id).
            has_tools = bool(openai_tools)
            candidates = [selected_model] + fallback_chain
            if has_tools:
                anthropic_candidates = []
                litellm_candidates = []
                for c in candidates:
                    cp = detect_provider(c)
                    ce = get_anthropic_compat_endpoint(cp) if cp else None
                    if ce:
                        anthropic_candidates.append(c)
                    else:
                        litellm_candidates.append(c)
                candidates = anthropic_candidates + litellm_candidates

            for candidate_model in candidates:
                candidate_provider = detect_provider(candidate_model)
                candidate_endpoint = (
                    get_anthropic_compat_endpoint(candidate_provider)
                    if candidate_provider else None
                )

                if candidate_endpoint:
                    # Path A: Direct Anthropic call
                    try:
                        api_base, api_key = candidate_endpoint
                        raw_response = await call_anthropic_direct(
                            api_base=api_base,
                            api_key=api_key,
                            model=candidate_model,
                            body=body,
                        )
                        final_model = candidate_model
                        if candidate_model != selected_model:
                            fallback_from = selected_model
                        break
                    except Exception as e:
                        logger.warning(
                            "Direct Anthropic call failed for %s: %s — trying next",
                            candidate_model, str(e)[:200],
                        )
                        continue
                else:
                    # Path B: Convert to OpenAI and use LiteLLM (single try).
                    # Use _dispatch_model directly — one try per candidate.
                    # Do NOT use _call_with_fallback here; it has its own
                    # internal chain that would consume all LiteLLM candidates
                    # and bypass the outer loop's direct-Anthropic path.
                    logger.info(
                        "LiteLLM call for %s (provider=%s)",
                        candidate_model, candidate_provider,
                    )
                    response_data = await _dispatch_model(
                        candidate_model, request, candidate_provider,
                    )
                    final_model = candidate_model
                    if final_model != selected_model:
                        fallback_from = selected_model
                    # response_data is in OpenAI format — convert to Anthropic for return
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    stats = {
                        "prompt_tokens": response_data.get("prompt_tokens", 0),
                        "completion_tokens": response_data.get("completion_tokens", 0),
                    }
                    record_llm_call(
                        span, model=final_model, provider=candidate_provider,
                        prompt_tokens=stats["prompt_tokens"],
                        completion_tokens=stats["completion_tokens"],
                        tier=tier, latency_ms=elapsed_ms,
                    )
                    _log_request({
                        "type": "anthropic_messages",
                        "request_id": request_id,
                        "prompt": prompt_text[:2000],
                        "selected_model": final_model,
                        "tier": tier,
                        "fallback_used": fallback_from,
                        "total_latency_ms": elapsed_ms,
                        **stats,
                        "status": "ok",
                        "call_path": "litellm_fallback",
                        **req_meta,
                    })
                    display_model = body.get("model", "") or final_model
                    if ant_stream:
                        return _build_anthropic_streaming_response(
                            request_id, display_model, response_data,
                        )
                    return JSONResponse(
                        content=openai_response_to_anthropic(
                            response_data, display_model, request_id,
                        ),
                        headers={"content-type": "application/json"},
                    )

            if raw_response is None:
                raise RuntimeError(f"All models failed in fallback chain for tier={tier}")

            # --- Success via direct Anthropic path ---
            stats = anthropic_response_to_stats(raw_response)
            elapsed_ms = int((time.time() - start_time) * 1000)

            # Some providers (GLM, Kimi) report inaccurate input_tokens.
            # Use our own estimate when upstream reports suspiciously low values.
            reported_pt = stats["prompt_tokens"]
            # Estimate from raw body: system + messages + tools
            est_chars = len(json.dumps(body.get("system", "")))
            est_chars += len(json.dumps(body.get("messages", [])))
            est_chars += len(json.dumps(body.get("tools", [])))
            estimated_pt = est_chars // 4
            prompt_tokens = max(reported_pt, estimated_pt)

            record_llm_call(
                span, model=final_model, provider=detect_provider(final_model),
                prompt_tokens=prompt_tokens,
                completion_tokens=stats["completion_tokens"],
                tier=tier, latency_ms=elapsed_ms,
            )

            _log_request({
                "type": "anthropic_messages",
                "request_id": request_id,
                "prompt": prompt_text[:2000],
                "selected_model": final_model,
                "tier": tier,
                "fallback_used": fallback_from,
                "total_latency_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": stats["completion_tokens"],
                "status": "ok",
                "call_path": "direct_anthropic",
                **req_meta,
            })

            display_model = body.get("model", "") or final_model
            if ant_stream:
                return _build_anthropic_streaming_response_from_raw(
                    request_id, display_model, raw_response,
                )
            return JSONResponse(
                content=raw_response, headers={"content-type": "application/json"},
            )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error("Anthropic messages error: %s", e, exc_info=True)
        _log_request({
            "type": "anthropic_messages", "request_id": request_id,
            "status": "error", "error": str(e), "total_latency_ms": elapsed_ms,
        })
        raise HTTPException(status_code=500, detail=f"Internal error. Request ID: {request_id}")
