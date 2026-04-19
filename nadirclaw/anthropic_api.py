"""Anthropic Messages API compatibility layer for NadirClaw.

Provides /v1/messages endpoint that accepts Anthropic SDK format requests,
converts them to internal OpenAI format, routes through NadirClaw's smart
routing, and returns responses in Anthropic format.

This enables tools that use the Anthropic SDK (e.g., Claude Code) to work
with NadirClaw as a transparent proxy.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from nadirclaw.auth import UserSession, validate_local_auth
from nadirclaw.settings import settings

logger = logging.getLogger("nadirclaw.anthropic_api")

router = APIRouter()


# ---------------------------------------------------------------------------
# Format conversion: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def anthropic_to_openai_messages(
    messages: List[Dict[str, Any]],
    system: Optional[Union[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Convert Anthropic Messages API format to OpenAI chat completion format."""
    result: List[Dict[str, Any]] = []

    # System prompt
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
    """Convert Anthropic tool definitions to OpenAI function format."""
    result = []
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


def build_anthropic_sse_events(
    request_id: str,
    model: str,
    response_data: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build Anthropic SSE event list from a completed response."""
    content = response_data.get("content", "") or ""
    tool_calls = response_data.get("tool_calls", [])
    input_tokens = response_data.get("prompt_tokens", 0)
    output_tokens = response_data.get("completion_tokens", 0)
    finish = response_data.get("finish_reason", "stop")

    stop_reason = "tool_use" if finish == "tool_calls" else ("max_tokens" if finish == "length" else "end_turn")
    msg_id = f"msg_{request_id}"
    events = []

    # message_start
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

    # Text block
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

    # Tool use blocks
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

    # message_delta + message_stop
    events.append({"event": "message_delta", "data": json.dumps({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })})
    events.append({"event": "message_stop", "data": json.dumps({"type": "message_stop"})})

    return events


# ---------------------------------------------------------------------------
# Endpoint
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


@router.post("/v1/messages")
async def anthropic_messages(
    raw_request: Request,
    current_user: UserSession = Depends(validate_local_auth),
):
    """Anthropic Messages API compatibility endpoint.

    Accepts requests in Anthropic format, routes them through NadirClaw's
    smart routing, and returns responses in Anthropic format.

    Auth is handled by validate_local_auth (supports x-api-key,
    X-API-Key, and Authorization: Bearer headers).
    """
    from nadirclaw.server import (
        _call_with_fallback,
        _extract_request_metadata,
        _log_request,
        _rate_limiter,
        ChatMessage,
        ChatCompletionRequest,
    )
    from sse_starlette.sse import EventSourceResponse

    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        body = await raw_request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Extract Anthropic fields
    ant_model = body.get("model", "")
    ant_messages = body.get("messages", [])
    ant_system = body.get("system")
    ant_max_tokens = body.get("max_tokens", 4096)
    ant_stream = body.get("stream", False)
    ant_tools = body.get("tools", [])

    prompt_text = _extract_last_user_text(ant_messages)

    # Convert to OpenAI format
    openai_messages = anthropic_to_openai_messages(ant_messages, ant_system)
    openai_tools = anthropic_tools_to_openai(ant_tools) if ant_tools else []

    # Build a simple request dict for internal routing
    from nadirclaw.routing import (
        apply_routing_modifiers,
        resolve_alias,
        resolve_profile,
    )

    # Resolve model
    profile = resolve_profile(ant_model)
    if profile == "eco":
        selected_model = settings.SIMPLE_MODEL
        base_tier = "simple"
    elif profile == "premium":
        selected_model = settings.COMPLEX_MODEL
        base_tier = "complex"
    elif profile == "reasoning":
        selected_model = settings.REASONING_MODEL
        base_tier = "reasoning"
    elif ant_model and ant_model != "auto" and profile is None:
        resolved = resolve_alias(ant_model)
        selected_model = resolved or ant_model
        base_tier = "complex"
    else:
        selected_model = settings.SIMPLE_MODEL
        base_tier = "simple"

    # Convert messages to ChatMessage objects for routing
    from nadirclaw.server import ChatMessage, ChatCompletionRequest
    chat_messages = []
    for m in openai_messages:
        kwargs: Dict[str, Any] = {"role": m["role"], "content": m.get("content")}
        if m["role"] == "tool":
            kwargs["tool_call_id"] = m.get("tool_call_id", "")
            kwargs["content"] = m.get("content", "")
        if m["role"] == "assistant" and "tool_calls" in m:
            kwargs["tool_calls"] = m["tool_calls"]
        chat_messages.append(ChatMessage(**kwargs))

    req_data = {
        "messages": [{"role": m.role, "content": m.content, **(m.model_extra or {})} for m in chat_messages],
        "model": "auto",
        "max_tokens": ant_max_tokens,
        "stream": False,
    }
    if openai_tools:
        req_data["tools"] = openai_tools

    request = ChatCompletionRequest(**req_data)

    req_meta = _extract_request_metadata(request)
    selected_model, final_tier, routing_info = apply_routing_modifiers(
        base_model=selected_model, base_tier=base_tier,
        request_meta=req_meta, messages=request.messages,
        simple_model=settings.SIMPLE_MODEL, complex_model=settings.COMPLEX_MODEL,
        reasoning_model=settings.REASONING_MODEL, free_model=settings.FREE_MODEL,
    )

    from nadirclaw.credentials import detect_provider
    provider = detect_provider(selected_model)
    analysis_info = {
        "strategy": "anthropic_api",
        "selected_model": selected_model,
        "tier": final_tier,
        "routing_modifiers": routing_info,
    }

    response_data, selected_model, analysis_info = await _call_with_fallback(
        selected_model, request, provider, analysis_info,
    )

    # Extract from LiteLLM response shape (choices[0].message)
    # _call_with_fallback returns OpenAI-format data
    if "choices" in response_data:
        choice = response_data["choices"][0] if response_data["choices"] else {}
        msg = choice.get("message", {})
        response_data = {
            "content": msg.get("content", ""),
            "tool_calls": msg.get("tool_calls", []),
            "finish_reason": choice.get("finish_reason", "stop"),
            "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": response_data.get("usage", {}).get("completion_tokens", 0),
        }

    elapsed_ms = int((time.time() - start_time) * 1000)

    _log_request({
        "type": "anthropic_messages",
        "request_id": request_id,
        "prompt": prompt_text[:2000],
        "selected_model": selected_model,
        "tier": final_tier,
        "total_latency_ms": elapsed_ms,
        "prompt_tokens": response_data.get("prompt_tokens", 0),
        "completion_tokens": response_data.get("completion_tokens", 0),
        "cost_usd": analysis_info.get("cost_usd"),
        "cached_tokens": analysis_info.get("cached_tokens", 0),
        "status": "ok",
        "message_count": len(ant_messages),
        "has_system_prompt": ant_system is not None,
        "system_prompt_length": len(ant_system) if isinstance(ant_system, str) else 0,
        "tool_count": len(ant_tools),
        "has_tools": len(ant_tools) > 0,
        "requested_model": ant_model,
        "stream": ant_stream,
    })

    if ant_stream:
        events = build_anthropic_sse_events(request_id, selected_model, response_data)

        async def event_gen():
            for evt in events:
                yield evt

        return EventSourceResponse(event_gen(), media_type="text/event-stream")

    return JSONResponse(
        content=openai_response_to_anthropic(response_data, selected_model, request_id),
        headers={"content-type": "application/json"},
    )
