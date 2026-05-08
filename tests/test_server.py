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
