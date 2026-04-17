"""Tests for OpenAI Codex runtime discovery filtering and caching."""

import asyncio
import json

from nadirclaw.openai_codex import OpenAICodexRuntime


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, responses):
        self._responses = list(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, headers=None):
        if not self._responses:
            return _FakeResponse(500, text="no response configured")
        return self._responses.pop(0)


def test_filter_discovered_models_pragmatic_default():
    runtime = OpenAICodexRuntime()
    entries = [
        {"id": "gpt-5.4"},
        {"id": "gpt-5.4-mini"},
        {"id": "gpt-5.3-codex"},
        {"id": "gpt-4.1"},
        {"id": "text-embedding-3-large"},
        {"id": "omni-moderation-latest"},
    ]
    models = runtime.filter_discovered_models(entries)
    assert "gpt-5.4" in models
    assert "gpt-5.4-mini" in models
    assert "gpt-5.3-codex" in models
    assert "gpt-4.1" not in models
    assert "text-embedding-3-large" not in models


def test_resolve_runtime_model_uses_prefix_mapping():
    runtime = OpenAICodexRuntime()
    model, source = runtime.resolve_runtime_model("openai-codex/gpt-5.4")
    assert model == "gpt-5.4"
    assert source == "configured"


def test_discovery_first_endpoint_fails_second_succeeds(monkeypatch):
    runtime = OpenAICodexRuntime()
    fake = _FakeAsyncClient([
        _FakeResponse(500, text="boom"),
        _FakeResponse(200, payload={"data": [{"id": "gpt-5.4"}, {"id": "text-embedding-3-large"}]}),
    ])
    monkeypatch.setattr("nadirclaw.openai_codex.httpx.AsyncClient", lambda timeout=20: fake)

    models = asyncio.run(runtime.refresh_if_stale("tok", force=True))
    assert models == ["gpt-5.4"]


def test_discovery_handles_unexpected_schema(monkeypatch):
    runtime = OpenAICodexRuntime()
    fake = _FakeAsyncClient([_FakeResponse(200, payload={"unexpected": {"foo": "bar"}})])
    monkeypatch.setattr("nadirclaw.openai_codex.httpx.AsyncClient", lambda timeout=20: fake)

    models = asyncio.run(runtime.refresh_if_stale("tok", force=True))
    assert models == []


def test_corrupt_cache_file_is_ignored(tmp_path, monkeypatch):
    cache_file = tmp_path / "openai_codex_models.json"
    cache_file.write_text("{this is not valid json")
    monkeypatch.setenv("NADIRCLAW_OPENAI_CODEX_MODEL_CACHE", str(cache_file))

    runtime = OpenAICodexRuntime()
    # Should not raise on cache read
    assert runtime.get_discovered_display_models() == []


def test_cache_raw_payload_is_optional(tmp_path, monkeypatch):
    cache_file = tmp_path / "openai_codex_models.json"
    monkeypatch.setenv("NADIRCLAW_OPENAI_CODEX_MODEL_CACHE", str(cache_file))
    monkeypatch.delenv("NADIRCLAW_OPENAI_CODEX_CACHE_INCLUDE_RAW", raising=False)

    runtime = OpenAICodexRuntime()
    runtime._write_cache(["gpt-5.4"], {"data": [{"id": "gpt-5.4"}]})
    payload = json.loads(cache_file.read_text())
    assert payload["models"] == ["gpt-5.4"]
    assert "raw" not in payload
