"""NadirClaw client — OpenAI SDK-compatible interface for the NadirClaw router.

Supports sync and async usage, streaming, and all NadirClaw-specific endpoints
(classify, explain, feedback, quality, budget).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional

import httpx


# ---------------------------------------------------------------------------
# Response models (lightweight dataclasses, no pydantic dependency)
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list] = None


@dataclass
class Choice:
    index: int = 0
    message: Optional[Message] = None
    delta: Optional[Message] = None
    finish_reason: Optional[str] = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class NadirMetadata:
    request_id: str = ""
    response_time_ms: int = 0
    routing: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatCompletion:
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[Choice] = field(default_factory=list)
    usage: Optional[Usage] = None
    nadirclaw_metadata: Optional[NadirMetadata] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ChatCompletion":
        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            choices.append(Choice(
                index=c.get("index", 0),
                message=Message(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content"),
                    tool_calls=msg.get("tool_calls"),
                ),
                finish_reason=c.get("finish_reason"),
            ))

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        ) if usage_data else None

        meta = data.get("nadirclaw_metadata")
        metadata = NadirMetadata(
            request_id=meta.get("request_id", ""),
            response_time_ms=meta.get("response_time_ms", 0),
            routing=meta.get("routing", {}),
        ) if meta else None

        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            nadirclaw_metadata=metadata,
        )


@dataclass
class ChatCompletionChunk:
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[Choice] = field(default_factory=list)
    usage: Optional[Usage] = None


# ---------------------------------------------------------------------------
# Completions namespace (mirrors openai.chat.completions)
# ---------------------------------------------------------------------------

class _Completions:
    """Handles /v1/chat/completions calls."""

    def __init__(self, client: "NadirClient"):
        self._client = client

    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        tools: Optional[list] = None,
        tool_choice: Optional[Any] = None,
        **kwargs,
    ):
        """Create a chat completion via NadirClaw.

        When stream=True, returns an iterator of ChatCompletionChunk.
        When stream=False, returns a ChatCompletion.
        """
        body: Dict[str, Any] = {"messages": messages}
        if model is not None:
            body["model"] = model
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if stream:
            body["stream"] = True
        body.update(kwargs)

        if stream:
            return self._stream(body)
        else:
            resp = self._client._post("/v1/chat/completions", json=body)
            return ChatCompletion.from_dict(resp)

    def _stream(self, body: dict) -> Iterator[ChatCompletionChunk]:
        """Stream SSE chunks from the server."""
        with self._client._client.stream(
            "POST",
            f"{self._client._base_url}/v1/chat/completions",
            json=body,
            headers=self._client._headers(),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                data = json.loads(data_str)
                choices = []
                for c in data.get("choices", []):
                    delta = c.get("delta", {})
                    choices.append(Choice(
                        index=c.get("index", 0),
                        delta=Message(
                            role=delta.get("role"),
                            content=delta.get("content"),
                            tool_calls=delta.get("tool_calls"),
                        ),
                        finish_reason=c.get("finish_reason"),
                    ))
                yield ChatCompletionChunk(
                    id=data.get("id", ""),
                    model=data.get("model", ""),
                    created=data.get("created", 0),
                    choices=choices,
                )


class _Chat:
    """Namespace: client.chat.completions"""

    def __init__(self, client: "NadirClient"):
        self.completions = _Completions(client)


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class NadirClient:
    """NadirClaw Python client.

    Auto-discovers the NadirClaw server via (in order):
      1. Explicit base_url parameter
      2. NADIRCLAW_API_URL environment variable
      3. Default: http://localhost:8856

    Usage::

        client = NadirClient()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "hello"}],
        )
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self._base_url = (
            base_url
            or os.getenv("NADIRCLAW_API_URL")
            or "http://localhost:8856"
        ).rstrip("/")

        self._api_key = api_key or os.getenv("NADIRCLAW_AUTH_TOKEN", "")
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

        # OpenAI-compatible namespace
        self.chat = _Chat(self)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _post(self, path: str, **kwargs) -> dict:
        resp = self._client.post(
            f"{self._base_url}{path}",
            headers=self._headers(),
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, **kwargs) -> dict:
        resp = self._client.get(
            f"{self._base_url}{path}",
            headers=self._headers(),
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    # --- NadirClaw-specific endpoints ---

    def classify(self, prompt: str, system_message: str = "") -> dict:
        """Classify a prompt without making an LLM call."""
        return self._post("/v1/classify", json={
            "prompt": prompt,
            "system_message": system_message,
        })

    def explain(self, messages: List[Dict[str, Any]], model: Optional[str] = None) -> dict:
        """Explain how a request would be routed (dry-run)."""
        body: dict = {"messages": messages}
        if model:
            body["model"] = model
        return self._post("/v1/routing-explain", json=body)

    def feedback(
        self,
        request_id: str,
        rating: Optional[int] = None,
        reason: Optional[str] = None,
        correct_tier: Optional[str] = None,
    ) -> dict:
        """Submit feedback on a routing decision."""
        body: Dict[str, Any] = {"request_id": request_id}
        if rating is not None:
            body["rating"] = rating
        if reason:
            body["reason"] = reason
        if correct_tier:
            body["correct_tier"] = correct_tier
        return self._post("/v1/feedback", json=body)

    def quality(self) -> dict:
        """Get quality scoring statistics."""
        return self._get("/v1/quality")

    def budget(self) -> dict:
        """Get budget and spend status."""
        return self._get("/v1/budget")

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return self._get("/v1/cache")

    def models(self) -> dict:
        """List available models."""
        return self._get("/v1/models")

    def health(self, deep: bool = False) -> dict:
        """Check server health."""
        path = "/health/deep" if deep else "/health"
        return self._get(path)

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
