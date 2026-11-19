"""A stalled provider must fail fast and hand over to the fallback chain.

Before NADIRCLAW_PROVIDER_TIMEOUT, LiteLLM calls ran with LiteLLM's 600 s
default plus two OpenAI SDK retries, so a provider that accepted the
connection and then went silent held the request for up to ~30 minutes
before the fallback chain ever ran. These tests drive the real LiteLLM call
against a local stub provider (no network, no spend).
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

os.environ.setdefault("NADIRCLAW_SIMPLE_MODEL", "claude-haiku-4-20250514")
os.environ.setdefault("NADIRCLAW_COMPLEX_MODEL", "claude-opus-4-20250514")

from nadirclaw.server import (  # noqa: E402
    ChatCompletionRequest,
    _call_with_fallback,
    _provider_timeout_kwargs,
    _stream_with_fallback,
)

STALL_S = 8  # far longer than the 1 s timeout used below
HITS: dict = {}


class _Provider(BaseHTTPRequestHandler):
    """OpenAI-compatible stub: model "stall" goes silent, anything else answers."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        HITS[body["model"]] = HITS.get(body["model"], 0) + 1
        if body["model"] == "stall":
            time.sleep(STALL_S)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            # Three chunks 0.6 s apart: 1.2 s total, longer than the 1 s
            # timeout, but no single gap is. A flowing stream must survive.
            for i, (text, finish) in enumerate([("he", None), ("ll", None), ("o", "stop")]):
                if i:
                    time.sleep(0.6)
                chunk = {
                    "id": "x", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        payload = json.dumps({
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def provider(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("NADIRCLAW_API_BASE", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-stub")
    monkeypatch.setenv("NADIRCLAW_PROVIDER_TIMEOUT", "1")
    monkeypatch.setenv("NADIRCLAW_FALLBACK_CHAIN", "openai/ok")
    HITS.clear()
    yield
    server.shutdown()


def _request(stream=False):
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], stream=stream)


@pytest.mark.asyncio
async def test_stalled_provider_falls_back_within_timeout(provider):
    start = time.monotonic()
    data, used, info = await _call_with_fallback("openai/stall", _request(), "openai", {})
    elapsed = time.monotonic() - start

    assert used == "openai/ok"
    assert data["content"] == "hello"
    assert info["fallback_reasons"][0]["error_type"] == "Timeout"
    assert HITS["stall"] == 1  # no SDK retries of the timed-out call
    assert elapsed < STALL_S, elapsed  # did not wait out the stall


@pytest.mark.asyncio
async def test_stalled_stream_falls_back_and_flowing_stream_survives(provider):
    start = time.monotonic()
    info: dict = {}
    events = [e async for e in _stream_with_fallback("openai/stall", _request(True), "openai", info, "req-1")]
    elapsed = time.monotonic() - start

    text = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in (json.loads(e["data"]) for e in events if e["data"] != "[DONE]")
    )
    assert text == "hello"  # fallback streamed all 1.2 s of chunks under a 1 s timeout
    assert info["fallback_from"] == "openai/stall"
    assert info["_stream_model"] == "openai/ok"
    assert HITS["stall"] == 1
    assert elapsed < STALL_S, elapsed


def test_timeout_kwargs(monkeypatch):
    monkeypatch.delenv("NADIRCLAW_PROVIDER_TIMEOUT", raising=False)
    assert _provider_timeout_kwargs("openai/gpt-5") == {"timeout": 120.0, "max_retries": 0}
    # Local Ollama keeps LiteLLM defaults unless the user opts in: cold loads are slow.
    assert _provider_timeout_kwargs("ollama/llama3") == {}
    monkeypatch.setenv("NADIRCLAW_PROVIDER_TIMEOUT", "300")
    assert _provider_timeout_kwargs("ollama_chat/llama3") == {"timeout": 300.0, "max_retries": 0}
    monkeypatch.setenv("NADIRCLAW_PROVIDER_TIMEOUT", "0")
    assert _provider_timeout_kwargs("openai/gpt-5") == {}
    monkeypatch.setenv("NADIRCLAW_PROVIDER_TIMEOUT", "junk")
    assert _provider_timeout_kwargs("openai/gpt-5") == {"timeout": 120.0, "max_retries": 0}
