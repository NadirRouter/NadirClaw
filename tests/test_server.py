"""Tests for nadirclaw.server — health endpoint and basic API contract."""

import pytest
from copy import deepcopy
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


class TestClaudeCodeIdentityInjection:
    """The opt-in Claude Code identity system block injection (#74)."""

    IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

    def test_inject_into_body_without_system(self):
        from nadirclaw.server import _inject_claude_code_identity
        body = {"model": "claude-opus-4-7", "messages": []}
        assert _inject_claude_code_identity(body) is True
        assert body["system"] == [{"type": "text", "text": self.IDENTITY}]

    def test_inject_prepends_before_string_system(self):
        from nadirclaw.server import _inject_claude_code_identity
        body = {"system": "Be terse."}
        assert _inject_claude_code_identity(body) is True
        assert body["system"] == [
            {"type": "text", "text": self.IDENTITY},
            {"type": "text", "text": "Be terse."},
        ]

    def test_inject_prepends_before_block_array_system(self):
        from nadirclaw.server import _inject_claude_code_identity
        body = {"system": [{"type": "text", "text": "Be terse."}]}
        assert _inject_claude_code_identity(body) is True
        assert body["system"][0] == {"type": "text", "text": self.IDENTITY}
        assert body["system"][1] == {"type": "text", "text": "Be terse."}

    def test_inject_is_noop_when_identity_already_first(self):
        from nadirclaw.server import _inject_claude_code_identity
        body = {"system": [{"type": "text", "text": self.IDENTITY + " extra"}]}
        assert _inject_claude_code_identity(body) is False
        assert len(body["system"]) == 1

    def test_inject_is_noop_when_string_system_already_identity(self):
        from nadirclaw.server import _inject_claude_code_identity
        body = {"system": self.IDENTITY}
        assert _inject_claude_code_identity(body) is False
        assert body["system"] == self.IDENTITY


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

    @staticmethod
    def _capturing_client():
        """Return (FakeClient, captured) recording the forwarded JSON body."""
        import httpx
        captured = {}

        class _FakeResponse:
            status_code = 200
            headers = {"content-type": "application/json"}
            def json(self):
                return {"id": "msg_1", "model": captured.get("model"),
                        "content": [{"type": "text", "text": "ok"}],
                        "usage": {"input_tokens": 3, "output_tokens": 1}}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, headers=None, json=None):
                captured["model"] = json.get("model")
                captured["system"] = json.get("system")
                captured["auth"] = headers.get("Authorization") or headers.get("x-api-key")
                return _FakeResponse()

        return httpx, _FakeClient, captured

    def test_identity_injected_for_oauth_when_enabled(self, client, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_CLAUDE_CODE_IDENTITY", "1")
        httpx, _FakeClient, captured = self._capturing_client()
        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient):
            resp = client.post("/v1/messages", json={
                "model": "claude-opus-4-7", "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status_code == 200
        assert captured["auth"] == "Bearer sk-ant-oat01-test"
        assert captured["system"][0]["text"].startswith("You are Claude Code")

    def test_identity_not_injected_when_disabled(self, client, monkeypatch):
        monkeypatch.delenv("NADIRCLAW_CLAUDE_CODE_IDENTITY", raising=False)
        httpx, _FakeClient, captured = self._capturing_client()
        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient):
            resp = client.post("/v1/messages", json={
                "model": "claude-opus-4-7", "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status_code == 200
        assert captured["system"] is None

    def test_identity_not_injected_for_api_key_token(self, client, monkeypatch):
        """Even with the flag on, an sk-ant-api key uses x-api-key — no injection."""
        monkeypatch.setenv("NADIRCLAW_CLAUDE_CODE_IDENTITY", "1")
        httpx, _FakeClient, captured = self._capturing_client()
        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-api-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient):
            resp = client.post("/v1/messages", json={
                "model": "claude-opus-4-7", "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status_code == 200
        assert captured["auth"] == "sk-ant-api-test"  # x-api-key path
        assert captured["system"] is None

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


# ---------------------------------------------------------------------------
# /v1/messages: usage recording (#71), count_tokens (#72), max_tokens clamp (#73)
# ---------------------------------------------------------------------------

class TestMessagesUsageHelpers:
    """Pure helpers behind the enriched /v1/messages path."""

    def test_parse_max_tokens_ceiling(self):
        from nadirclaw.server import _parse_max_tokens_ceiling
        assert _parse_max_tokens_ceiling(
            "max_tokens: 100000 > 64000, which is the maximum allowed"
        ) == 64000
        assert _parse_max_tokens_ceiling("some other 400 error") is None
        assert _parse_max_tokens_ceiling("") is None

    def test_parse_anthropic_sse_usage(self):
        from nadirclaw.server import _parse_anthropic_sse_usage
        text = (
            'event: message_start\n'
            'data: {"type":"message_start","message":{"usage":{"input_tokens":12,"output_tokens":1}}}\n\n'
            'event: message_delta\n'
            'data: {"type":"message_delta","usage":{"output_tokens":34}}\n\n'
        )
        assert _parse_anthropic_sse_usage(text) == (12, 34)

    def test_parse_anthropic_sse_usage_ignores_garbage(self):
        from nadirclaw.server import _parse_anthropic_sse_usage
        assert _parse_anthropic_sse_usage("data: [DONE]\n\nnot-data\n") == (None, None)

    def test_record_messages_usage_populates_recordable_fields(self):
        """The recorded entry must carry the fields record_request()/budget read."""
        from nadirclaw.server import _record_messages_usage
        captured = {}
        with patch("nadirclaw.server._log_request", side_effect=lambda e: captured.update(e)):
            _record_messages_usage(
                {"type": "messages", "selected_model": "claude-haiku-4-5"},
                "claude-haiku-4-5", 100, 50, status="ok", latency_ms=42,
            )
        assert captured["type"] == "messages"
        assert captured["prompt_tokens"] == 100
        assert captured["completion_tokens"] == 50
        assert captured["total_tokens"] == 150
        assert captured["status"] == "ok"
        assert captured["total_latency_ms"] == 42
        assert "cost" in captured  # budget tracker stamped a cost


class TestMessagesUsageRecording:
    """#71 — /v1/messages requests must reach metrics + budget."""

    def test_messages_type_is_recorded_by_metrics(self):
        """record_request must not drop type='messages' entries."""
        from nadirclaw import metrics as metrics_mod
        key = ("claude-haiku-4-5", "simple", "ok")
        before = dict(metrics_mod.requests_total.items()).get(key, 0.0)
        metrics_mod.record_request({
            "type": "messages",
            "selected_model": "claude-haiku-4-5",
            "tier": "simple",
            "status": "ok",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost": 0.001,
        })
        after = dict(metrics_mod.requests_total.items()).get(key, 0.0)
        assert after == before + 1

    def test_non_streaming_records_usage(self, client):
        import httpx

        class _FakeResponse:
            status_code = 200
            headers = {"content-type": "application/json"}
            def json(self):
                return {
                    "id": "msg_1",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, headers=None, json=None):
                return _FakeResponse()

        recorded = {}
        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient), \
             patch("nadirclaw.server._log_request", side_effect=lambda e: recorded.update(e)):
            resp = client.post("/v1/messages", json={
                "model": "nadir-eco",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            })

        assert resp.status_code == 200
        assert recorded["type"] == "messages"
        assert recorded["prompt_tokens"] == 11
        assert recorded["completion_tokens"] == 7
        assert recorded["status"] == "ok"
        assert "cost" in recorded

    def test_streaming_records_usage_from_sse(self, client):
        import httpx

        sse_chunks = [
            b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":21,"output_tokens":1}}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n',
            b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":9}}\n\n',
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
                return _FakeStream()

        recorded = {}
        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient), \
             patch("nadirclaw.server._log_request", side_effect=lambda e: recorded.update(e)):
            resp = client.post("/v1/messages", json={
                "model": "nadir-premium",
                "max_tokens": 10,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            })
            body = resp.content  # drain the stream so the generator records usage

        assert resp.status_code == 200
        for chunk in sse_chunks:
            assert chunk in body
        assert recorded["type"] == "messages"
        assert recorded["prompt_tokens"] == 21
        assert recorded["completion_tokens"] == 9
        assert recorded["status"] == "ok"


class TestMaxTokensClamp:
    """#73 — clamp max_tokens against the routed model's output ceiling."""

    def test_non_streaming_clamps_and_retries_on_400(self, client):
        import httpx

        calls = []

        class _Resp400:
            status_code = 400
            text = '{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: 100000 > 64000, which is the maximum allowed number of output tokens"}}'
            content = text.encode()
            headers = {"content-type": "application/json"}

        class _Resp200:
            status_code = 200
            headers = {"content-type": "application/json"}
            def json(self):
                return {"id": "msg_1", "content": [{"type": "text", "text": "ok"}],
                        "usage": {"input_tokens": 3, "output_tokens": 2}}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, headers=None, json=None):
                calls.append(json.get("max_tokens"))
                return _Resp400() if len(calls) == 1 else _Resp200()

        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient):
            resp = client.post("/v1/messages", json={
                "model": "nadir-eco",
                "max_tokens": 100000,
                "messages": [{"role": "user", "content": "hi"}],
            })

        assert resp.status_code == 200
        # First attempt sent the original ceiling, retry sent the clamped value.
        assert calls == [100000, 64000]
        assert resp.headers.get("X-NadirClaw-MaxTokens-Clamped") == "true"

    def test_non_max_tokens_400_is_not_retried(self, client):
        import httpx

        calls = []

        class _Resp400:
            status_code = 400
            text = '{"error":{"message":"some unrelated bad request"}}'
            content = text.encode()
            headers = {"content-type": "application/json"}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, headers=None, json=None):
                calls.append(json.get("max_tokens"))
                return _Resp400()

        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient):
            resp = client.post("/v1/messages", json={
                "model": "nadir-eco",
                "max_tokens": 100000,
                "messages": [{"role": "user", "content": "hi"}],
            })

        assert resp.status_code == 400
        assert len(calls) == 1  # no retry
        assert "X-NadirClaw-MaxTokens-Clamped" not in resp.headers


class TestParamReconcile:
    """#83 — reconcile request parameters the routed model does not support."""

    @staticmethod
    def _clear_cache():
        from nadirclaw.server import _MODEL_PARAM_FIXES
        _MODEL_PARAM_FIXES.clear()

    # --- individual fixers ---------------------------------------------------

    def test_rewrites_adaptive_to_enabled_below_max_tokens(self):
        from nadirclaw.server import _downgrade_adaptive_thinking
        body = {"max_tokens": 8192, "thinking": {"type": "adaptive", "display": "omitted"}}
        modifier = _downgrade_adaptive_thinking(body)
        assert body["thinking"] == {
            "type": "enabled", "budget_tokens": 8191, "display": "omitted",
        }
        assert modifier == "thinking_downgrade(adaptive\u2192enabled, budget=8191)"

    def test_thinking_budget_is_capped(self):
        from nadirclaw.server import _downgrade_adaptive_thinking, _MAX_THINKING_BUDGET
        body = {"max_tokens": 200000, "thinking": {"type": "adaptive"}}
        _downgrade_adaptive_thinking(body)
        assert body["thinking"]["budget_tokens"] == _MAX_THINKING_BUDGET
        assert "display" not in body["thinking"]

    def test_thinking_is_dropped_when_no_budget_fits(self):
        """budget_tokens must be >= 1024 and < max_tokens, so tiny caps leave no room."""
        from nadirclaw.server import _downgrade_adaptive_thinking
        body = {"max_tokens": 512, "thinking": {"type": "adaptive"}}
        assert _downgrade_adaptive_thinking(body) == "thinking_downgrade(adaptive\u2192off)"
        assert "thinking" not in body

    def test_other_thinking_shapes_are_untouched(self):
        from nadirclaw.server import _downgrade_adaptive_thinking
        for thinking in ({"type": "enabled", "budget_tokens": 2000}, {"type": "disabled"}):
            body = {"max_tokens": 8192, "thinking": dict(thinking)}
            assert _downgrade_adaptive_thinking(body) is None
            assert body["thinking"] == thinking
        assert _downgrade_adaptive_thinking({"max_tokens": 8192}) is None

    def test_drops_effort_from_output_config(self):
        from nadirclaw.server import _drop_effort
        body = {"output_config": {"effort": "high"}}
        assert _drop_effort(body) == "effort_drop(high)"
        assert "output_config" not in body  # emptied, so removed entirely

        body = {"output_config": {"effort": "high", "other": 1}}
        assert _drop_effort(body) == "effort_drop(high)"
        assert body["output_config"] == {"other": 1}

        body = {"effort": "low"}
        assert _drop_effort(body) == "effort_drop(low)"
        assert body == {}

        assert _drop_effort({"output_config": {"other": 1}}) is None

    def test_folds_system_messages_into_the_system_field(self):
        from nadirclaw.server import _fold_system_messages
        body = {
            "system": [{"type": "text", "text": "first"}],
            "messages": [
                {"role": "system", "content": "from a message"},
                {"role": "user", "content": "hi"},
            ],
        }
        assert _fold_system_messages(body) == "system_role_fold(1)"
        assert body["system"] == [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "from a message"},
        ]
        assert body["messages"] == [{"role": "user", "content": "hi"}]

    def test_fold_leaves_a_body_with_no_other_message(self):
        """Anthropic needs >= 1 message, so folding the only turn would be invalid."""
        from nadirclaw.server import _fold_system_messages
        body = {"messages": [{"role": "system", "content": "only"}]}
        assert _fold_system_messages(body) is None
        assert body["messages"] == [{"role": "system", "content": "only"}]

    def test_fold_is_a_noop_without_system_turns(self):
        from nadirclaw.server import _fold_system_messages
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert _fold_system_messages(body) is None

    # --- error matching and caching ------------------------------------------

    def test_each_400_selects_its_own_fixer(self):
        from nadirclaw.server import _reconcile_from_error, _MODEL_PARAM_FIXES
        self._clear_cache()
        body = {"max_tokens": 8192, "thinking": {"type": "adaptive"}}
        assert _reconcile_from_error(body, "m", "adaptive thinking is not supported on this model")
        assert body["thinking"]["type"] == "enabled"

        body = {"output_config": {"effort": "high"}}
        assert _reconcile_from_error(body, "m", "This model does not support the effort parameter.")
        assert "output_config" not in body

        assert _MODEL_PARAM_FIXES["m"] == {"thinking", "effort"}
        self._clear_cache()

    def test_unknown_400_matches_nothing(self):
        from nadirclaw.server import _reconcile_from_error, _MODEL_PARAM_FIXES
        self._clear_cache()
        body = {"max_tokens": 8192, "thinking": {"type": "adaptive"}}
        assert _reconcile_from_error(body, "m", "some unrelated bad request") == []
        assert _reconcile_from_error(body, "m", "") == []
        assert body["thinking"] == {"type": "adaptive"}  # untouched
        assert not _MODEL_PARAM_FIXES

    def test_known_fixes_are_preapplied(self):
        from nadirclaw.server import _preapply_known_fixes, _MODEL_PARAM_FIXES
        self._clear_cache()
        assert _preapply_known_fixes({"max_tokens": 8192}, "m") == []
        _MODEL_PARAM_FIXES["m"] = {"thinking", "effort"}
        body = {"max_tokens": 8192, "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"}}
        modifiers = _preapply_known_fixes(body, "m")
        assert set(modifiers) == {
            "thinking_downgrade(adaptive\u2192enabled, budget=8191)", "effort_drop(high)",
        }
        assert body["thinking"]["type"] == "enabled" and "output_config" not in body
        self._clear_cache()

    def test_modifiers_are_recorded_in_the_request_log(self):
        from nadirclaw.server import _record_messages_usage
        captured = {}
        with patch("nadirclaw.server._log_request", side_effect=lambda e: captured.update(e)):
            _record_messages_usage(
                {"type": "messages", "modifiers_applied": ["agent_role[SUBAGENT]"]},
                "claude-haiku-4-5", 10, 5,
                extra_modifiers=["thinking_downgrade(adaptive\u2192enabled, budget=8191)"],
            )
        assert captured["modifiers_applied"] == [
            "agent_role[SUBAGENT]",
            "thinking_downgrade(adaptive\u2192enabled, budget=8191)",
        ]

    # --- end to end through the endpoint ------------------------------------

    @staticmethod
    def _fake_client(error_messages):
        """Fake httpx client that 400s with each message in turn, then succeeds."""
        import httpx
        sent = []
        remaining = list(error_messages)

        class _Resp:
            def __init__(self, message=None):
                self._message = message
                self.status_code = 400 if message else 200
                self.text = ('{"type":"error","error":{"type":"invalid_request_error",'
                             f'"message":"{message}"}}}}') if message else "{}"
                self.content = self.text.encode()
                self.headers = {"content-type": "application/json"}
            def json(self):
                return {"id": "msg_1", "content": [{"type": "text", "text": "ok"}],
                        "usage": {"input_tokens": 3, "output_tokens": 2}}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, headers=None, json=None):
                # deep copy: the fixers mutate these structures in place
                sent.append(deepcopy(
                    {k: json.get(k) for k in ("thinking", "output_config", "messages")}
                ))
                return _Resp(remaining.pop(0) if remaining else None)

        return httpx, _FakeClient, sent

    def test_converges_over_successive_400s(self, client):
        """One request absorbs every unsupported-parameter 400 and still returns 200."""
        self._clear_cache()
        httpx, _FakeClient, sent = self._fake_client([
            "adaptive thinking is not supported on this model",
            "This model does not support the effort parameter.",
        ])
        try:
            with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
                 patch.object(httpx, "AsyncClient", _FakeClient):
                resp = client.post("/v1/messages", json={
                    "model": "nadir-eco", "max_tokens": 8192,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                    "messages": [{"role": "user", "content": "hi"}],
                })
        finally:
            self._clear_cache()

        assert resp.status_code == 200
        assert len(sent) == 3
        assert sent[0]["thinking"] == {"type": "adaptive"}
        assert sent[1]["thinking"] == {"type": "enabled", "budget_tokens": 8191}
        assert sent[1]["output_config"] == {"effort": "high"}
        assert sent[2]["output_config"] is None  # effort dropped on the last attempt
        assert resp.headers.get("X-NadirClaw-Params-Reconciled") == "true"

    def test_second_request_needs_no_failed_attempt(self, client):
        """The cache makes the fixes pre-emptive, so the retry cost is once per model."""
        from nadirclaw.server import _MODEL_PARAM_FIXES
        from nadirclaw.settings import settings
        self._clear_cache()
        _MODEL_PARAM_FIXES[settings.SIMPLE_MODEL] = {"thinking", "effort"}
        httpx, _FakeClient, sent = self._fake_client([])
        try:
            with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
                 patch.object(httpx, "AsyncClient", _FakeClient):
                resp = client.post("/v1/messages", json={
                    "model": "nadir-eco", "max_tokens": 8192,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                    "messages": [{"role": "user", "content": "hi"}],
                })
        finally:
            self._clear_cache()

        assert resp.status_code == 200
        assert len(sent) == 1  # no wasted round trip
        assert sent[0]["thinking"] == {"type": "enabled", "budget_tokens": 8191}
        assert sent[0]["output_config"] is None

    def test_unknown_400_is_returned_without_retrying(self, client):
        from nadirclaw.server import _MODEL_PARAM_FIXES
        self._clear_cache()
        httpx, _FakeClient, sent = self._fake_client(["some unrelated bad request"])
        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient):
            resp = client.post("/v1/messages", json={
                "model": "nadir-eco", "max_tokens": 8192,
                "thinking": {"type": "adaptive"},
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status_code == 400
        assert len(sent) == 1
        assert sent[0]["thinking"] == {"type": "adaptive"}
        assert "X-NadirClaw-Params-Reconciled" not in resp.headers
        assert not _MODEL_PARAM_FIXES

    def test_streaming_reconciles_before_any_byte_is_sent(self, client):
        import httpx
        self._clear_cache()
        sent = []
        remaining = ["adaptive thinking is not supported on this model"]

        class _Stream:
            def __init__(self, message=None):
                self._message = message
                self.status_code = 400 if message else 200
            async def aread(self):
                return (f'{{"type":"error","error":{{"message":"{self._message}"}}}}'.encode()
                        if self._message else b"")
            async def aiter_bytes(self):
                yield (b'event: message_start\ndata: {"type":"message_start","message":'
                       b'{"usage":{"input_tokens":3,"output_tokens":1}}}\n\n')

        class _Ctx:
            def __init__(self, resp): self._resp = resp
            async def __aenter__(self): return self._resp
            async def __aexit__(self, *a): return False

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            def stream(self, method, url, headers=None, json=None):
                sent.append(json.get("thinking"))
                return _Ctx(_Stream(remaining.pop(0) if remaining else None))

        try:
            with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
                 patch.object(httpx, "AsyncClient", _FakeClient):
                resp = client.post("/v1/messages", json={
                    "model": "nadir-eco", "max_tokens": 8192, "stream": True,
                    "thinking": {"type": "adaptive"},
                    "messages": [{"role": "user", "content": "hi"}],
                })
                text = resp.text
        finally:
            self._clear_cache()

        assert resp.status_code == 200
        assert sent == [{"type": "adaptive"}, {"type": "enabled", "budget_tokens": 8191}]
        assert "message_start" in text
        assert "upstream_error" not in text  # the 400 never reached the client


class TestCountTokensEndpoint:
    """#72 — /v1/messages/count_tokens proxy."""

    def test_missing_credential_returns_401(self, client):
        with patch("nadirclaw.credentials.get_credential", return_value=None):
            resp = client.post("/v1/messages/count_tokens", json={
                "model": "nadir-eco",
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status_code == 401
        assert "setup-token" in resp.json()["detail"]

    def test_resolves_model_and_forwards_verbatim(self, client):
        import httpx
        from nadirclaw.settings import settings

        captured = {}

        class _FakeResponse:
            status_code = 200
            content = b'{"input_tokens": 42}'
            headers = {"content-type": "application/json"}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, headers=None, json=None):
                captured["url"] = url
                captured["model"] = json.get("model")
                return _FakeResponse()

        with patch("nadirclaw.credentials.get_credential", return_value="sk-ant-oat01-test"), \
             patch.object(httpx, "AsyncClient", _FakeClient):
            resp = client.post("/v1/messages/count_tokens", json={
                "model": "nadir-eco",
                "messages": [{"role": "user", "content": "hi"}],
            })

        assert resp.status_code == 200
        assert resp.json() == {"input_tokens": 42}
        assert captured["url"].endswith("/v1/messages/count_tokens")
        # routed through the same router as /v1/messages
        assert captured["model"] == settings.SIMPLE_MODEL
