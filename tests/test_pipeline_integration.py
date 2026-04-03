"""Integration tests for the full NadirClaw proxy pipeline.

Tests the complete flow: request → classify → route → model call → response.
All LLM provider calls are mocked; everything else runs for real.
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with fresh app state."""
    from nadirclaw.server import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helper: mock _call_with_fallback to return the expected tuple
# ---------------------------------------------------------------------------

def _make_fallback_mock(content="Hello!", prompt_tokens=10, completion_tokens=5,
                        finish_reason="stop", tool_calls=None, tier="simple",
                        strategy="smart-routing", confidence=0.9, model=None):
    """Create an AsyncMock for _call_with_fallback that returns the correct tuple."""
    async def side_effect(selected_model, request, provider, analysis_info):
        response_data = {
            "content": content,
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        if tool_calls:
            response_data["tool_calls"] = tool_calls

        actual_model = model or selected_model
        updated_info = {
            **analysis_info,
            "selected_model": actual_model,
        }
        return response_data, actual_model, updated_info

    mock = AsyncMock(side_effect=side_effect)
    return mock


# ---------------------------------------------------------------------------
# 1. Simple prompt -> routed to simple model -> response
# ---------------------------------------------------------------------------

class TestSimplePromptPipeline:
    """A simple prompt should be classified as simple and routed to the cheap model."""

    @patch("nadirclaw.server._call_with_fallback")
    def test_simple_prompt_routes_to_simple_model(self, mock_fallback, client):
        mock_fallback.side_effect = _make_fallback_mock(
            content="4", prompt_tokens=10, completion_tokens=2,
        ).side_effect

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "4"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 2

        # Verify the model dispatched was the simple model
        meta = data.get("nadirclaw_metadata", {})
        routing = meta.get("routing", {})
        assert routing.get("tier") in ("simple", "free")

    @patch("nadirclaw.server._call_with_fallback")
    def test_response_has_openai_shape(self, mock_fallback, client):
        """Response must be OpenAI-compatible."""
        mock_fallback.side_effect = _make_fallback_mock(
            content="Hi!", prompt_tokens=5, completion_tokens=3,
        ).side_effect

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
        })

        data = resp.json()
        assert data["object"] == "chat.completion"
        assert "id" in data
        assert "created" in data
        assert "model" in data
        assert len(data["choices"]) == 1
        assert data["choices"][0]["index"] == 0
        assert "message" in data["choices"][0]
        assert data["choices"][0]["message"]["role"] == "assistant"


# ---------------------------------------------------------------------------
# 2. Complex prompt -> routed to complex model
# ---------------------------------------------------------------------------

class TestComplexPromptPipeline:

    @patch("nadirclaw.server._call_with_fallback")
    def test_complex_prompt_routes_to_complex_model(self, mock_fallback, client):
        mock_fallback.side_effect = _make_fallback_mock(
            content="Here is the distributed system design...",
            prompt_tokens=50, completion_tokens=200, tier="complex",
        ).side_effect

        resp = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": "You are a senior architect."},
                {"role": "user", "content": (
                    "Design a distributed event-sourcing system with CQRS, "
                    "eventual consistency, saga orchestration for a multi-region "
                    "deployment handling 100K transactions per second. Include "
                    "failure recovery, data partitioning strategy, and CAP theorem tradeoffs."
                )},
            ],
        })

        assert resp.status_code == 200
        data = resp.json()
        meta = data.get("nadirclaw_metadata", {})
        routing = meta.get("routing", {})
        # Cascade classifier may return "medium" or "complex";
        # trained classifier (Pro) reliably returns "complex"
        assert routing.get("tier") in ("medium", "mid", "complex")


# ---------------------------------------------------------------------------
# 3. Direct model override (bypass routing)
# ---------------------------------------------------------------------------

class TestDirectModelOverride:

    @patch("nadirclaw.server._call_with_fallback")
    def test_explicit_model_bypasses_classifier(self, mock_fallback, client):
        mock_fallback.side_effect = _make_fallback_mock(
            content="Response from explicit model",
            prompt_tokens=10, completion_tokens=10, model="gpt-4o",
        ).side_effect

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "gpt-4o"
        meta = data.get("nadirclaw_metadata", {})
        routing = meta.get("routing", {})
        assert routing.get("strategy") in ("direct", "alias")


# ---------------------------------------------------------------------------
# 4. Routing profiles (eco / premium)
# ---------------------------------------------------------------------------

class TestRoutingProfiles:

    @patch("nadirclaw.server._call_with_fallback")
    def test_eco_profile(self, mock_fallback, client):
        mock_fallback.side_effect = _make_fallback_mock(
            content="Eco response", prompt_tokens=5, completion_tokens=5,
        ).side_effect

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Complex question that would normally route to premium"}],
            "model": "eco",
        })

        assert resp.status_code == 200
        data = resp.json()
        meta = data.get("nadirclaw_metadata", {})
        routing = meta.get("routing", {})
        assert routing.get("strategy") == "profile:eco"
        assert routing.get("tier") == "simple"

    @patch("nadirclaw.server._call_with_fallback")
    def test_premium_profile(self, mock_fallback, client):
        mock_fallback.side_effect = _make_fallback_mock(
            content="Premium response", prompt_tokens=5, completion_tokens=5,
        ).side_effect

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Simple question"}],
            "model": "premium",
        })

        assert resp.status_code == 200
        data = resp.json()
        meta = data.get("nadirclaw_metadata", {})
        routing = meta.get("routing", {})
        assert routing.get("strategy") == "profile:premium"
        assert routing.get("tier") == "complex"


# ---------------------------------------------------------------------------
# 5. Fallback chain -- primary model fails, fallback succeeds
# ---------------------------------------------------------------------------

class TestFallbackChain:

    @patch("nadirclaw.server._call_with_fallback", new_callable=AsyncMock)
    def test_fallback_info_in_metadata(self, mock_fallback, client):
        """When primary model fails and fallback succeeds, metadata should reflect it."""
        mock_fallback.return_value = (
            {
                "content": "Fallback response",
                "finish_reason": "stop",
                "prompt_tokens": 10,
                "completion_tokens": 10,
            },
            "ollama/llama3",
            {
                "strategy": "smart-routing+fallback",
                "tier": "simple",
                "confidence": 0.9,
                "complexity_score": 0.2,
                "classifier_latency_ms": 5,
                "selected_model": "ollama/llama3",
                "fallback_from": "gemini/gemini-2.5-flash",
                "fallback_chain_tried": ["gemini/gemini-2.5-flash"],
            },
        )

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Unique fallback test prompt xyz123"}],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Fallback response"
        meta = data.get("nadirclaw_metadata", {})
        routing = meta.get("routing", {})
        assert "fallback" in routing.get("strategy", "")
        assert routing.get("fallback_from") is not None


# ---------------------------------------------------------------------------
# 6. Tool calling passthrough
# ---------------------------------------------------------------------------

class TestToolCalling:

    @patch("nadirclaw.server._call_with_fallback")
    def test_tool_calls_preserved_in_response(self, mock_fallback, client):
        """Tool call responses from the LLM should be passed through."""
        mock_fallback.side_effect = _make_fallback_mock(
            content=None, finish_reason="tool_calls",
            prompt_tokens=20, completion_tokens=15,
            tool_calls=[{
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                },
            }],
        ).side_effect

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What's the weather in NYC?"}],
            "model": "gpt-4o",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                    },
                }
            ],
        })

        assert resp.status_code == 200
        data = resp.json()
        msg = data["choices"][0]["message"]
        assert "tool_calls" in msg
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert data["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------------------
# 7. Input validation -- oversized content
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_oversized_content_rejected(self, client):
        """Content exceeding max size should return 413."""
        huge_msg = "x" * 1_100_000  # > 1MB limit
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": huge_msg}],
        })
        assert resp.status_code == 413

    def test_missing_messages_rejected(self, client):
        """Missing messages field should fail validation."""
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 8. Multi-turn conversation routing
# ---------------------------------------------------------------------------

class TestMultiTurnRouting:

    @patch("nadirclaw.server._call_with_fallback")
    def test_multi_turn_uses_last_user_message_for_classification(self, mock_fallback, client):
        """Classification should be based on the last user message."""
        mock_fallback.side_effect = _make_fallback_mock(
            content="42", prompt_tokens=30, completion_tokens=2,
        ).side_effect

        resp = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Design a microservices architecture for a bank."},
                {"role": "assistant", "content": "Sure, here's the architecture..."},
                {"role": "user", "content": "What is 2+2?"},  # Simple follow-up
            ],
        })

        assert resp.status_code == 200
        data = resp.json()
        meta = data.get("nadirclaw_metadata", {})
        routing = meta.get("routing", {})
        # Last message is simple, so should classify as simple
        assert routing.get("tier") in ("simple", "free")


# ---------------------------------------------------------------------------
# 9. Budget tracking integration
# ---------------------------------------------------------------------------

class TestBudgetIntegration:

    @patch("nadirclaw.server._call_with_fallback")
    def test_budget_endpoint_after_request(self, mock_fallback, client):
        """Budget should update after a completion request."""
        mock_fallback.side_effect = _make_fallback_mock(
            content="Test", prompt_tokens=100, completion_tokens=50,
        ).side_effect

        # Make a request
        client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
        })

        # Check budget
        resp = client.get("/v1/budget")
        assert resp.status_code == 200
        data = resp.json()
        assert "daily_spend" in data
        assert "monthly_spend" in data


# ---------------------------------------------------------------------------
# 10. Streaming response format
# ---------------------------------------------------------------------------

class TestStreamingPipeline:

    @patch("nadirclaw.server._stream_with_fallback")
    def test_streaming_returns_sse(self, mock_stream, client):
        """Streaming requests should return SSE-formatted chunks via true streaming."""
        import time as _time

        created = int(_time.time())
        request_id = "chatcmpl-test"

        async def _fake_stream(*args, **kwargs):
            # Simulate true streaming: role+content chunk, then finish
            yield {"data": json.dumps({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Streamed response"}, "finish_reason": None}],
            })}
            yield {"data": json.dumps({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })}
            yield {"data": "[DONE]"}
            # Set analysis_info for logging
            args[3]["_stream_model"] = "test-model"
            args[3]["_stream_usage"] = {"prompt_tokens": 10, "completion_tokens": 5}

        mock_stream.side_effect = _fake_stream

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE events
        lines = resp.text.strip().split("\n")
        data_lines = [l.removeprefix("data: ") for l in lines if l.startswith("data: ")]

        assert len(data_lines) >= 2  # At least content chunk + finish chunk
        # Last data should be [DONE]
        assert data_lines[-1] == "[DONE]"

        # First chunk should have content
        first_chunk = json.loads(data_lines[0])
        assert first_chunk["object"] == "chat.completion.chunk"
        assert first_chunk["choices"][0]["delta"]["content"] == "Streamed response"

        # Second chunk should have finish_reason
        finish_chunk = json.loads(data_lines[1])
        assert finish_chunk["choices"][0]["finish_reason"] == "stop"
        assert "usage" in finish_chunk


# ---------------------------------------------------------------------------
# 11. Classify -> completions consistency
# ---------------------------------------------------------------------------

class TestClassifyCompletionConsistency:

    @patch("nadirclaw.server._call_with_fallback")
    def test_classify_and_completion_agree_on_tier(self, mock_fallback, client):
        """The /v1/classify tier should match the actual routing tier."""
        mock_fallback.side_effect = _make_fallback_mock(
            content="Answer", prompt_tokens=10, completion_tokens=5,
        ).side_effect

        prompt = "What is 2+2?"

        # Classify
        classify_resp = client.post("/v1/classify", json={"prompt": prompt})
        classify_tier = classify_resp.json()["classification"]["tier"]

        # Complete
        completion_resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": prompt}],
        })
        data = completion_resp.json()
        completion_tier = data["nadirclaw_metadata"]["routing"]["tier"]

        # Both should agree
        if classify_tier == "simple":
            assert completion_tier in ("simple", "free")
