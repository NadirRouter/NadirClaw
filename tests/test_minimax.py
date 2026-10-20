"""Current MiniMax choices and real LiteLLM HTTP-boundary regressions."""

import json

import httpx
import pytest

from nadirclaw.routing import (
    BUILTIN_MODEL_REGISTRY, MINIMAX_MODELS, MODEL_REGISTRY,
    estimate_cost, get_context_window, has_vision, resolve_alias,
)
from nadirclaw.setup import get_available_models_for_providers, select_default_model


def test_current_choices_and_legacy_metadata(monkeypatch):
    assert [m for m in BUILTIN_MODEL_REGISTRY if m.startswith("minimax/")] == list(MINIMAX_MODELS)
    assert resolve_alias("minimax") == MINIMAX_MODELS[0]
    assert get_context_window(MINIMAX_MODELS[0]) == 1_000_000
    assert has_vision(MINIMAX_MODELS[0])
    assert estimate_cost(MINIMAX_MODELS[0], 1_000_000, 1_000_000) == pytest.approx(0.75)
    monkeypatch.setitem(MODEL_REGISTRY, "minimax/saved-custom", {"context_window": 12345})
    tiers = get_available_models_for_providers(
        ["minimax"], {"minimax": ["minimax/MiniMax-M2.7", "minimax/saved-custom"]},
    )
    assert tiers["simple"] == [{"model": MINIMAX_MODELS[0], "provider": "minimax"}]
    assert tiers["complex"] == [{"model": MINIMAX_MODELS[1], "provider": "minimax"}]
    assert select_default_model("simple", ["minimax"], tiers["simple"]) == MINIMAX_MODELS[0]
    assert get_context_window("minimax/saved-custom") == 12345
    assert resolve_alias("minimax-m2.7") == "minimax/MiniMax-M2.7"
    assert get_context_window("minimax/MiniMax-M2.7") == 204_800
    assert resolve_alias("sonnet") == "claude-sonnet-4-5-20250929"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("base", [None, "https://api.minimaxi.com/v1", "https://custom.example/v1"])
@pytest.mark.parametrize("model", ["minimax", "minimax-m3", "minimax-m2.7", "minimax/saved-custom"])
async def test_real_sdk_http_payload(monkeypatch, stream, base, model):
    """Mock only HTTP: exercise LiteLLM's provider and SDK serialization."""
    import litellm
    from nadirclaw import credentials, server
    from nadirclaw.server import ChatCompletionRequest, _call_litellm, _stream_litellm

    monkeypatch.setattr(credentials, "get_credential", lambda provider: "offline-fixture")
    monkeypatch.setenv("NADIRCLAW_API_BASE", base or "")
    monkeypatch.setattr(litellm, "telemetry", False, raising=False)
    captured = []

    async def send(client, request, **kwargs):
        body = json.loads(request.content)
        captured.append((str(request.url), body))
        message = {"role": "assistant", "content": "Done", "reasoning_content": "Considered"}
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if stream:
            chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 0,
                     "model": body["model"], "choices": [{"index": 0, "delta": message, "finish_reason": "stop"}], "usage": usage}
            return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"},
                                  content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")
        return httpx.Response(200, request=request, json={
            "id": "fixture", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "usage": usage,
        })

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Describe"},
         {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]},
        {"role": "assistant", "content": None, "reasoning_content": "Keep this thought",
         "tool_calls": [{"id": "call_fixture", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_fixture", "content": "Found"},
    ]
    request = ChatCompletionRequest(model=model, messages=messages, tools=[tool],
                                    reasoning_effort="high", thinking={"type": "adaptive"}, max_tokens=512)
    selected = resolve_alias(request.model) or request.model
    if stream:
        chunks = [chunk async for chunk in _stream_litellm(selected, request, "minimax")]
        assert any(chunk[0].get("reasoning_content") == "Considered" for chunk in chunks)
    else:
        result = await _call_litellm(selected, request, "minimax")
        assert result["reasoning_content"] == "Considered"
    assert len(captured) == 1
    url, body = captured[0]
    assert url == (base or "https://api.minimax.io/v1") + "/chat/completions"
    assert body["model"] == selected.removeprefix("minimax/")
    assert body["messages"][0]["content"] == messages[0]["content"]
    assert body["messages"][1]["reasoning_content"] == "Keep this thought"
    assert body["messages"][1]["tool_calls"] == messages[1]["tool_calls"]
    assert body["messages"][2]["tool_call_id"] == "call_fixture"
    assert body["tools"] == [tool]
    assert body["reasoning_effort"] == "high"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["max_tokens"] == 512
    assert "service_tier" not in body
    assert body.get("reasoning_split") is not False


@pytest.mark.parametrize("model", ["minimax", "minimax/MiniMax-M3", "minimax/MiniMax-M2.7"])
def test_chat_endpoint_resolves_default_and_preserves_saved_models(monkeypatch, model):
    from fastapi.testclient import TestClient
    from nadirclaw import credentials, server

    monkeypatch.setattr(credentials, "get_credential", lambda provider: "offline-fixture")
    monkeypatch.setenv("NADIRCLAW_API_BASE", "")
    monkeypatch.setenv("NADIRCLAW_CACHE_ENABLED", "0")
    captured = []

    async def send(client, request, **kwargs):
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, request=request, json={
            "id": "fixture", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    client = TestClient(server.app, base_url="http://localhost", client=("127.0.0.1", 50000))
    response = client.post("/v1/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": "Hello " + model}],
    })
    assert response.status_code == 200, response.text
    selected = resolve_alias(model) or model
    assert response.headers["x-routed-model"] == selected
    assert len(captured) == 1
    assert captured[0]["model"] == selected.removeprefix("minimax/")
    assert "thinking" not in captured[0]
    assert "reasoning_effort" not in captured[0]
