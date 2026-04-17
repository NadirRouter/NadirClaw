"""Tests for nadirclaw.server — health endpoint and basic API contract."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock
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


class TestModelsEndpoint:
    @patch("nadirclaw.server.get_openai_codex_runtime")
    @patch("nadirclaw.credentials.get_credential")
    def test_list_models(self, mock_get_cred, mock_runtime_factory, client):
        mock_get_cred.return_value = "tok"
        runtime = mock_runtime_factory.return_value
        runtime.refresh_if_stale = AsyncMock(return_value=["gpt-5.4", "gpt-5.3-codex"])
        runtime.get_discovered_display_models.return_value = [
            "openai-codex/gpt-5.4",
            "openai-codex/gpt-5.3-codex",
        ]
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1
        model_ids = [m["id"] for m in data["data"]]
        assert "auto" in model_ids
        assert "eco" in model_ids
        assert "premium" in model_ids
        assert "openai-codex/gpt-5.4" in model_ids
        # deduplicated: appears only once even if also configured statically
        assert model_ids.count("openai-codex/gpt-5.3-codex") == 1
        runtime.refresh_if_stale.assert_called_once()
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


class TestDispatchOpenAICodex:
    @patch("nadirclaw.server._call_openai_codex")
    def test_dispatch_model_routes_openai_codex(self, mock_call):
        from nadirclaw.server import ChatCompletionRequest, _dispatch_model

        mock_call.return_value = {
            "content": "ok",
            "finish_reason": "stop",
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
        out = asyncio.run(_dispatch_model("openai-codex/gpt-5.4", req, "openai-codex"))
        assert out["content"] == "ok"
        mock_call.assert_awaited_once()

    @patch("nadirclaw.server._stream_openai_codex")
    def test_dispatch_stream_routes_openai_codex(self, mock_stream):
        from nadirclaw.server import ChatCompletionRequest, _dispatch_model_stream

        async def _fake_stream(*args, **kwargs):
            yield {"content": "ok"}, {"prompt_tokens": 1, "completion_tokens": 1}, "stop"

        mock_stream.return_value = _fake_stream()

        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
        async def _collect():
            got = []
            async for item in _dispatch_model_stream("openai-codex/gpt-5.4", req, "openai-codex"):
                got.append(item)
            return got

        got = asyncio.run(_collect())
        assert got and got[0][0]["content"] == "ok"
        mock_stream.assert_called_once()


class TestOpenAICodexFallbackBehavior:
    @patch("nadirclaw.server.settings")
    @patch("nadirclaw.server._dispatch_model")
    def test_fallback_on_retryable_upstream_error(self, mock_dispatch, mock_settings):
        from nadirclaw.server import ChatCompletionRequest, UpstreamModelError, _call_with_fallback

        mock_settings.FALLBACK_CHAIN = ["openai-codex/gpt-5.4", "gemini-2.5-flash"]
        mock_settings.get_tier_fallback_chain.side_effect = lambda tier: mock_settings.FALLBACK_CHAIN

        async def _dispatch_side_effect(model, request, provider):
            if model == "openai-codex/gpt-5.4":
                raise UpstreamModelError(model=model, provider="openai-codex", message="timeout")
            return {
                "content": "fallback-ok",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
            }

        mock_dispatch.side_effect = _dispatch_side_effect

        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
        analysis = {"tier": "", "strategy": "smart-routing", "selected_model": "openai-codex/gpt-5.4"}
        response, model_used, updated = asyncio.run(
            _call_with_fallback("openai-codex/gpt-5.4", req, "openai-codex", analysis)
        )
        assert response["content"] == "fallback-ok"
        assert model_used == "gemini-2.5-flash"
        assert updated["fallback_from"] == "openai-codex/gpt-5.4"

    @patch("nadirclaw.server.settings")
    @patch("nadirclaw.server._dispatch_model")
    def test_no_fallback_on_local_http_exception(self, mock_dispatch, mock_settings):
        from fastapi import HTTPException
        from nadirclaw.server import ChatCompletionRequest, _call_with_fallback

        mock_settings.FALLBACK_CHAIN = ["openai-codex/gpt-5.4", "gemini-2.5-flash"]
        mock_settings.get_tier_fallback_chain.side_effect = lambda tier: mock_settings.FALLBACK_CHAIN
        mock_dispatch.side_effect = HTTPException(status_code=500, detail="No credential")

        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
        analysis = {"tier": "", "strategy": "smart-routing", "selected_model": "openai-codex/gpt-5.4"}
        with pytest.raises(HTTPException):
            asyncio.run(_call_with_fallback("openai-codex/gpt-5.4", req, "openai-codex", analysis))


class _FakeHttpxResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    async def aread(self):
        return self.text.encode()


class _FakeHttpxStreamResponse:
    def __init__(self, status_code=200, lines=None, text=""):
        self.status_code = status_code
        self._lines = lines or []
        self.text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aread(self):
        return self.text.encode()

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeHttpxClient:
    def __init__(self, posts=None, streams=None):
        self.posts = list(posts or [])
        self.streams = list(streams or [])
        self.post_calls = []
        self.stream_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None):
        self.post_calls.append((url, json))
        return self.posts.pop(0)

    def stream(self, method, url, headers=None, json=None):
        self.stream_calls.append((method, url, json))
        return self.streams.pop(0)


class TestCodexTransportDecisions:
    @patch("nadirclaw.credentials.get_credential_source", return_value="oauth")
    @patch("nadirclaw.credentials.get_credential", return_value="tok")
    @patch("nadirclaw.server.get_openai_codex_runtime")
    def test_non_streaming_responses_success(self, mock_runtime_factory, *_):
        from nadirclaw.server import ChatCompletionRequest, _call_openai_codex

        runtime = mock_runtime_factory.return_value
        runtime.responses_url = "https://resp"
        runtime.chat_completions_url = "https://chat"
        runtime.refresh_if_stale = AsyncMock(return_value=[])
        runtime.resolve_runtime_model.return_value = ("gpt-5.4", "discovery")
        fake_client = _FakeHttpxClient(posts=[_FakeHttpxResponse(200, payload={"status": "completed", "output_text": "ok", "usage": {}})])
        with patch("httpx.AsyncClient", return_value=fake_client):
            req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
            out = asyncio.run(_call_openai_codex("openai-codex/gpt-5.4", req, "openai-codex"))
        assert out["content"] == "ok"
        assert fake_client.post_calls[0][0] == "https://resp"

    @patch("nadirclaw.credentials.get_credential_source", return_value="oauth")
    @patch("nadirclaw.credentials.get_credential", return_value="tok")
    @patch("nadirclaw.server.get_openai_codex_runtime")
    def test_non_streaming_400_falls_back_to_chat(self, mock_runtime_factory, *_):
        from nadirclaw.server import ChatCompletionRequest, _call_openai_codex

        runtime = mock_runtime_factory.return_value
        runtime.responses_url = "https://resp"
        runtime.chat_completions_url = "https://chat"
        runtime.refresh_if_stale = AsyncMock(return_value=[])
        runtime.resolve_runtime_model.return_value = ("gpt-5.4", "configured")
        fake_client = _FakeHttpxClient(
            posts=[
                _FakeHttpxResponse(400, text="bad request"),
                _FakeHttpxResponse(200, payload={"choices": [{"message": {"content": "chat-ok"}, "finish_reason": "stop"}], "usage": {}}),
            ]
        )
        with patch("httpx.AsyncClient", return_value=fake_client):
            req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
            out = asyncio.run(_call_openai_codex("openai-codex/gpt-5.4", req, "openai-codex"))
        assert out["content"] == "chat-ok"
        assert [c[0] for c in fake_client.post_calls] == ["https://resp", "https://chat"]

    @patch("nadirclaw.credentials.get_credential_source", return_value="oauth")
    @patch("nadirclaw.credentials.get_credential", return_value="tok")
    @patch("nadirclaw.server.get_openai_codex_runtime")
    def test_non_streaming_401_has_clear_error(self, mock_runtime_factory, *_):
        from fastapi import HTTPException
        from nadirclaw.server import ChatCompletionRequest, _call_openai_codex

        runtime = mock_runtime_factory.return_value
        runtime.responses_url = "https://resp"
        runtime.chat_completions_url = "https://chat"
        runtime.refresh_if_stale = AsyncMock(return_value=[])
        runtime.resolve_runtime_model.return_value = ("gpt-5.4", "configured")
        fake_client = _FakeHttpxClient(posts=[_FakeHttpxResponse(401, text="unauthorized")])
        with patch("httpx.AsyncClient", return_value=fake_client):
            req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
            with pytest.raises(HTTPException) as err:
                asyncio.run(_call_openai_codex("openai-codex/gpt-5.4", req, "openai-codex"))
        assert err.value.status_code == 401

    @patch("nadirclaw.credentials.get_credential_source", return_value="oauth")
    @patch("nadirclaw.credentials.get_credential", return_value="tok")
    @patch("nadirclaw.server.get_openai_codex_runtime")
    def test_non_streaming_5xx_is_upstream_error(self, mock_runtime_factory, *_):
        from nadirclaw.server import ChatCompletionRequest, UpstreamModelError, _call_openai_codex

        runtime = mock_runtime_factory.return_value
        runtime.responses_url = "https://resp"
        runtime.chat_completions_url = "https://chat"
        runtime.refresh_if_stale = AsyncMock(return_value=[])
        runtime.resolve_runtime_model.return_value = ("gpt-5.4", "configured")
        fake_client = _FakeHttpxClient(posts=[_FakeHttpxResponse(503, text="outage")])
        with patch("httpx.AsyncClient", return_value=fake_client):
            req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
            with pytest.raises(UpstreamModelError):
                asyncio.run(_call_openai_codex("openai-codex/gpt-5.4", req, "openai-codex"))

    @patch("nadirclaw.credentials.get_credential_source", return_value="oauth")
    @patch("nadirclaw.credentials.get_credential", return_value="tok")
    @patch("nadirclaw.server.get_openai_codex_runtime")
    def test_tool_history_bypasses_responses_path(self, mock_runtime_factory, *_):
        from nadirclaw.server import ChatCompletionRequest, _call_openai_codex

        runtime = mock_runtime_factory.return_value
        runtime.responses_url = "https://resp"
        runtime.chat_completions_url = "https://chat"
        runtime.refresh_if_stale = AsyncMock(return_value=[])
        runtime.resolve_runtime_model.return_value = ("gpt-5.4", "configured")
        fake_client = _FakeHttpxClient(posts=[_FakeHttpxResponse(200, payload={"choices": [{"message": {"content": "chat-tool"}, "finish_reason": "stop"}], "usage": {}})])
        with patch("httpx.AsyncClient", return_value=fake_client):
            req = ChatCompletionRequest(messages=[{"role": "tool", "content": "tool result"}])
            out = asyncio.run(_call_openai_codex("openai-codex/gpt-5.4", req, "openai-codex"))
        assert out["content"] == "chat-tool"
        assert fake_client.post_calls[0][0] == "https://chat"

    @patch("nadirclaw.credentials.get_credential_source", return_value="oauth")
    @patch("nadirclaw.credentials.get_credential", return_value="tok")
    @patch("nadirclaw.server.get_openai_codex_runtime")
    def test_streaming_400_switches_to_chat_stream(self, mock_runtime_factory, *_):
        from nadirclaw.server import ChatCompletionRequest, _stream_openai_codex

        runtime = mock_runtime_factory.return_value
        runtime.responses_url = "https://resp"
        runtime.chat_completions_url = "https://chat"
        runtime.refresh_if_stale = AsyncMock(return_value=[])
        runtime.resolve_runtime_model.return_value = ("gpt-5.4", "configured")
        lines = ['data: ' + json.dumps({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]})]
        fake_client = _FakeHttpxClient(streams=[_FakeHttpxStreamResponse(status_code=400, text="bad"), _FakeHttpxStreamResponse(status_code=200, lines=lines)])
        with patch("httpx.AsyncClient", return_value=fake_client):
            req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

            async def _collect():
                out = []
                async for item in _stream_openai_codex("openai-codex/gpt-5.4", req, "openai-codex"):
                    out.append(item)
                return out

            out = asyncio.run(_collect())
        assert out and out[0][0]["content"] == "hi"
        assert fake_client.stream_calls[0][1] == "https://resp"
        assert fake_client.stream_calls[1][1] == "https://chat"

    @patch("nadirclaw.credentials.get_credential_source", return_value="oauth")
    @patch("nadirclaw.credentials.get_credential", return_value="tok")
    @patch("nadirclaw.server.get_openai_codex_runtime")
    def test_streaming_5xx_raises_upstream_error(self, mock_runtime_factory, *_):
        from nadirclaw.server import ChatCompletionRequest, UpstreamModelError, _stream_openai_codex

        runtime = mock_runtime_factory.return_value
        runtime.responses_url = "https://resp"
        runtime.chat_completions_url = "https://chat"
        runtime.refresh_if_stale = AsyncMock(return_value=[])
        runtime.resolve_runtime_model.return_value = ("gpt-5.4", "configured")
        fake_client = _FakeHttpxClient(streams=[_FakeHttpxStreamResponse(status_code=503, text="outage")])
        with patch("httpx.AsyncClient", return_value=fake_client):
            req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

            async def _consume():
                async for _ in _stream_openai_codex("openai-codex/gpt-5.4", req, "openai-codex"):
                    pass

            with pytest.raises(UpstreamModelError):
                asyncio.run(_consume())

    @patch("nadirclaw.credentials.get_credential_source", return_value="oauth")
    @patch("nadirclaw.credentials.get_credential", return_value="tok")
    @patch("nadirclaw.server.get_openai_codex_runtime")
    def test_streaming_tool_history_uses_chat_first(self, mock_runtime_factory, *_):
        from nadirclaw.server import ChatCompletionRequest, _stream_openai_codex

        runtime = mock_runtime_factory.return_value
        runtime.responses_url = "https://resp"
        runtime.chat_completions_url = "https://chat"
        runtime.refresh_if_stale = AsyncMock(return_value=[])
        runtime.resolve_runtime_model.return_value = ("gpt-5.4", "configured")
        lines = ['data: ' + json.dumps({"choices": [{"delta": {"content": "tool"}, "finish_reason": None}]})]
        fake_client = _FakeHttpxClient(streams=[_FakeHttpxStreamResponse(status_code=200, lines=lines)])
        with patch("httpx.AsyncClient", return_value=fake_client):
            req = ChatCompletionRequest(messages=[{"role": "tool", "content": "res"}])

            async def _collect():
                out = []
                async for item in _stream_openai_codex("openai-codex/gpt-5.4", req, "openai-codex"):
                    out.append(item)
                return out

            out = asyncio.run(_collect())
        assert out and out[0][0]["content"] == "tool"
        assert fake_client.stream_calls[0][1] == "https://chat"

    @patch("nadirclaw.credentials.get_credential_source", return_value="oauth")
    @patch("nadirclaw.credentials.get_credential", return_value="tok")
    @patch("nadirclaw.server.get_openai_codex_runtime")
    def test_streaming_responses_completed_maps_finish_reason(self, mock_runtime_factory, *_):
        from nadirclaw.server import ChatCompletionRequest, _stream_openai_codex

        runtime = mock_runtime_factory.return_value
        runtime.responses_url = "https://resp"
        runtime.chat_completions_url = "https://chat"
        runtime.refresh_if_stale = AsyncMock(return_value=[])
        runtime.resolve_runtime_model.return_value = ("gpt-5.4", "configured")
        lines = [
            'data: ' + json.dumps({"type": "response.output_text.delta", "delta": "hi"}),
            'data: ' + json.dumps({"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 1, "output_tokens": 2}}}),
        ]
        fake_client = _FakeHttpxClient(streams=[_FakeHttpxStreamResponse(status_code=200, lines=lines)])
        with patch("httpx.AsyncClient", return_value=fake_client):
            req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

            async def _collect():
                out = []
                async for item in _stream_openai_codex("openai-codex/gpt-5.4", req, "openai-codex"):
                    out.append(item)
                return out

            out = asyncio.run(_collect())
        assert out[0][0]["content"] == "hi"
        assert out[-1][2] == "stop"


class TestStreamingMidStreamAbort:
    @patch("nadirclaw.server.settings")
    @patch("nadirclaw.server._dispatch_model_stream")
    def test_mid_stream_error_does_not_restart_chain(self, mock_dispatch_stream, mock_settings):
        from nadirclaw.server import ChatCompletionRequest, UpstreamModelError, _stream_with_fallback

        mock_settings.FALLBACK_CHAIN = ["openai-codex/gpt-5.4", "gemini-2.5-flash"]
        mock_settings.get_tier_fallback_chain.side_effect = lambda tier: mock_settings.FALLBACK_CHAIN

        async def _generator(*args, **kwargs):
            yield {"content": "partial"}, None, None
            raise UpstreamModelError("openai-codex/gpt-5.4", "openai-codex", "stream broke")

        mock_dispatch_stream.return_value = _generator()

        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
        analysis = {"tier": "", "strategy": "smart-routing", "selected_model": "openai-codex/gpt-5.4"}

        async def _collect():
            events = []
            async for item in _stream_with_fallback("openai-codex/gpt-5.4", req, "openai-codex", analysis, "req-1"):
                events.append(item)
            return events

        events = asyncio.run(_collect())
        payloads = [e["data"] for e in events]
        joined = "\n".join(payloads)
        assert "partial" in joined
        assert "Stream interrupted" in joined
        assert "gemini-2.5-flash" not in joined
