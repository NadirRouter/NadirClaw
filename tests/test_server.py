"""Tests for nadirclaw.server — health endpoint and basic API contract."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the NadirClaw FastAPI app."""
    from nadirclaw.server import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "simple_model" in data
        assert "complex_model" in data

    def test_root_returns_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "NadirClaw"
        assert data["status"] == "ok"
        assert "version" in data

    def test_provider_health_hidden_by_default(self, client):
        resp = client.get("/internal/provider_health")
        assert resp.status_code == 404

    def test_provider_health_returns_snapshot_when_enabled(self, client):
        with patch("nadirclaw.server.settings") as mock_settings:
            mock_settings.PROVIDER_HEALTH = True
            resp = client.get("/internal/provider_health")

        assert resp.status_code == 200
        assert "models" in resp.json()


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1
        # Each model should have an id
        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"


class TestClassifyEndpoint:
    def test_classify_returns_classification(self, client):
        resp = client.post("/v1/classify", json={"prompt": "What is 2+2?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "classification" in data
        assert data["classification"]["tier"] in ("simple", "complex")
        assert "confidence" in data["classification"]
        assert "selected_model" in data["classification"]

    def test_classify_batch(self, client):
        resp = client.post(
            "/v1/classify/batch",
            json={"prompts": ["Hello", "Design a distributed system"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2


class TestMessagesEndpointHelpers:
    """Pure helpers behind the Anthropic-compatible /v1/messages endpoint."""

    def test_strip_provider_prefix(self):
        from nadirclaw.server import _strip_provider_prefix
        assert _strip_provider_prefix("anthropic/claude-opus-4-7") == "claude-opus-4-7"
        assert _strip_provider_prefix("claude/claude-haiku-4-5") == "claude-haiku-4-5"
        assert _strip_provider_prefix("claude-sonnet-4-6") == "claude-sonnet-4-6"
        assert _strip_provider_prefix("") == ""

    def test_anthropic_messages_to_chat_string_content(self):
        from nadirclaw.server import _anthropic_messages_to_chat
        chat = _anthropic_messages_to_chat([
            {"role": "user", "content": "hello world"},
        ])
        assert len(chat) == 1
        assert chat[0].role == "user"
        assert chat[0].text_content() == "hello world"

    def test_anthropic_messages_to_chat_block_content(self):
        from nadirclaw.server import _anthropic_messages_to_chat
        chat = _anthropic_messages_to_chat([
            {"role": "user", "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
                {"type": "image", "source": {}},  # ignored for routing
            ]},
        ])
        assert chat[0].text_content() == "first\nsecond"

    def test_anthropic_messages_to_chat_tool_result(self):
        from nadirclaw.server import _anthropic_messages_to_chat
        chat = _anthropic_messages_to_chat([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": "result text"},
            ]},
        ])
        assert "result text" in chat[0].text_content()

    def test_extract_text_from_anthropic_response(self):
        from nadirclaw.server import _extract_text_from_anthropic_response
        payload = {"content": [
            {"type": "text", "text": "hello "},
            {"type": "thinking", "thinking": "ignored"},
            {"type": "text", "text": "world"},
        ]}
        assert _extract_text_from_anthropic_response(payload) == "hello world"


class TestMessagesEndpoint:
    """The /v1/messages Anthropic-compatible proxy endpoint."""

    def test_missing_credential_returns_401(self, client):
        with patch("nadirclaw.credentials.get_credential", return_value=None):
            resp = client.post("/v1/messages", json={
                "model": "nadir-eco",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status_code == 401
        assert "setup-token" in resp.json()["detail"]

    def test_invalid_body_returns_400(self, client):
        resp = client.post(
            "/v1/messages",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_eco_profile_rewrites_model_and_forwards(self, client):
        """nadir-eco → SIMPLE_MODEL, body forwarded with rewritten model."""
        import httpx
        from nadirclaw.settings import settings

        captured = {}

        class _FakeResponse:
            status_code = 200
            headers = {"content-type": "application/json"}
            def json(self):
                return {
                    "id": "msg_1",
                    "model": captured.get("model"),
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                }

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, headers=None, json=None):
                captured["url"] = url
                captured["model"] = json.get("model")
                captured["auth"] = headers.get("Authorization") or headers.get("x-api-key")
                return _FakeResponse()

        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient):
            resp = client.post("/v1/messages", json={
                "model": "nadir-eco",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            })

        assert resp.status_code == 200
        # nadir-eco must resolve to the configured simple model
        assert captured["model"] == settings.SIMPLE_MODEL
        assert captured["url"].endswith("/v1/messages")
        # OAuth token → Bearer header
        assert captured["auth"] == "Bearer sk-ant-oat01-test"

    def test_upstream_error_is_passed_through(self, client):
        import httpx

        class _FakeResponse:
            status_code = 429
            text = '{"type":"error","error":{"type":"rate_limit_error"}}'
            content = text.encode()
            headers = {"content-type": "application/json"}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, headers=None, json=None):
                return _FakeResponse()

        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-api-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient):
            resp = client.post("/v1/messages", json={
                "model": "claude-opus-4-7",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            })

        # Upstream 429 surfaced to the caller as-is
        assert resp.status_code == 429

    def test_streaming_pipes_sse_bytes_through(self, client):
        """stream:true → upstream SSE bytes are forwarded verbatim."""
        import httpx
        from nadirclaw.settings import settings

        captured = {}
        sse_chunks = [
            b'event: message_start\ndata: {"type":"message_start"}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]

        class _FakeStream:
            status_code = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def aiter_bytes(self):
                for c in sse_chunks:
                    yield c
            async def aread(self):
                return b""

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            def stream(self, method, url, headers=None, json=None):
                captured["model"] = json.get("model")
                captured["url"] = url
                return _FakeStream()

        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient):
            resp = client.post("/v1/messages", json={
                "model": "nadir-premium",
                "max_tokens": 10,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            })

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.content
        # every upstream chunk made it through, in order
        for chunk in sse_chunks:
            assert chunk in body
        assert body.index(sse_chunks[0]) < body.index(sse_chunks[-1])
        # nadir-premium resolved to the complex model before forwarding
        assert captured["model"] == settings.COMPLEX_MODEL

    def test_streaming_upstream_error_emits_sse_error_event(self, client):
        """A non-200 upstream status in streaming mode → an SSE error event."""
        import httpx

        class _FakeStream:
            status_code = 500
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def aiter_bytes(self):
                if False:
                    yield b""  # pragma: no cover
            async def aread(self):
                return b'{"type":"error","error":{"type":"overloaded_error"}}'

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            def stream(self, method, url, headers=None, json=None):
                return _FakeStream()

        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient):
            resp = client.post("/v1/messages", json={
                "model": "nadir-eco",
                "max_tokens": 10,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            })

        assert resp.status_code == 200  # SSE stream opened
        assert b"event: error" in resp.content
        assert b"overloaded_error" in resp.content


# ---------------------------------------------------------------------------
# X-Routed-* response headers
# ---------------------------------------------------------------------------

def _mock_fallback(content="OK", prompt_tokens=10, completion_tokens=5, model=None):
    """Build a side_effect callable for patching _call_with_fallback."""
    async def _side_effect(selected_model, request, provider, analysis_info):
        actual_model = model or selected_model
        return (
            {
                "content": content,
                "finish_reason": "stop",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            actual_model,
            {**analysis_info, "selected_model": actual_model},
        )
    return _side_effect


class TestRoutingHeaders:
    """X-Routed-Model, X-Routed-Tier, X-Complexity-Score headers."""

    @patch("nadirclaw.server._call_with_fallback")
    def test_non_streaming_response_has_routing_headers(self, mock_fb, client):
        mock_fb.side_effect = _mock_fallback(content="hi")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "routing header test 8x2q"}],
        })
        assert resp.status_code == 200
        assert "X-Routed-Model" in resp.headers
        assert resp.headers["X-Routed-Model"] != ""
        assert "X-Routed-Tier" in resp.headers
        assert resp.headers["X-Routed-Tier"] in ("simple", "mid", "complex", "reasoning", "direct", "free")
        assert "X-Complexity-Score" in resp.headers

    @patch("nadirclaw.server._call_with_fallback")
    def test_direct_model_has_routing_headers(self, mock_fb, client):
        mock_fb.side_effect = _mock_fallback(content="hi", model="gpt-4o")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "direct model header test 3v7w"}],
            "model": "gpt-4o",
        })
        assert resp.status_code == 200
        assert resp.headers["X-Routed-Model"] == "gpt-4o"
        assert resp.headers["X-Routed-Tier"] == "direct"

    @patch("nadirclaw.server._stream_with_fallback")
    def test_streaming_response_has_routing_headers(self, mock_stream, client):
        async def _fake_stream(*args, **kwargs):
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield "data: [DONE]\n\n"
        mock_stream.return_value = _fake_stream()
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "streaming header test 5k9z"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert "X-Routed-Model" in resp.headers
        assert "X-Routed-Tier" in resp.headers
        assert "X-Complexity-Score" in resp.headers
