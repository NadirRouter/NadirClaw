"""Tests for tool-calling passthrough in NadirClaw.

Verifies that tool definitions, tool-role messages, and tool_calls in
LLM responses are correctly preserved when routing through _call_litellm
and returned in both streaming and non-streaming response formats.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nadirclaw.server import (
    ChatCompletionRequest,
    ChatMessage,
    _build_streaming_response,
    _call_litellm,
    _extract_request_metadata,
    app,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app)


def _make_request(messages, tools=None, tool_choice=None, stream=False, model="auto"):
    """Build a ChatCompletionRequest with optional tools."""

    data = {"messages": messages, "model": model, "stream": stream}
    if tools is not None:
        data["tools"] = tools
    if tool_choice is not None:
        data["tool_choice"] = tool_choice
    return ChatCompletionRequest(**data)


# Sample tool definition (OpenAI format)
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    },
}

# Sample tool_calls from an LLM response
SAMPLE_TOOL_CALL = {
    "id": "call_abc123",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": '{"location": "San Francisco"}',
    },
}

# Model name constants
# Placeholder used in tests where the model identity is irrelevant
TEST_MODEL = "ollama/test-model"
# Real model name used in tests asserting ollama→ollama_chat upgrade behaviour
OLLAMA_MODEL = "ollama/qwen3:4b"
OLLAMA_PROVIDER = "ollama"


# ---------------------------------------------------------------------------
# _call_litellm: message preservation
# ---------------------------------------------------------------------------


class TestCallLitellmMessages:
    """Verify _call_litellm builds correct messages for LiteLLM."""

    def _mock_response(self, content="Hello", tool_calls=None):
        """Build a fake litellm response."""
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "stop" if not tool_calls else "tool_calls"
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage
        return resp

    @pytest.mark.asyncio
    async def test_plain_messages_preserved(self):
        """Simple user/assistant messages should pass through."""

        request = _make_request(
            [
                {"role": "user", "content": "Hello"},
            ]
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = self._mock_response("Hi there")
            result = await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        call_kwargs = mock_comp.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["content"] == "Hi there"

    @pytest.mark.asyncio
    async def test_ollama_upgraded_to_ollama_chat_with_tools(self):
        """ollama/ prefix should auto-upgrade to ollama_chat/ when tools are present."""

        request = _make_request(
            [{"role": "user", "content": "What's the weather?"}],
            tools=[WEATHER_TOOL],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = self._mock_response()
            await _call_litellm(OLLAMA_MODEL, request, OLLAMA_PROVIDER)

        call_kwargs = mock_comp.call_args[1]
        assert call_kwargs["model"] == "ollama_chat/qwen3:4b"

    @pytest.mark.asyncio
    async def test_ollama_not_upgraded_without_tools(self):
        """ollama/ prefix should stay as-is when no tools are present."""

        request = _make_request(
            [{"role": "user", "content": "Hello"}],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = self._mock_response()
            await _call_litellm(OLLAMA_MODEL, request, OLLAMA_PROVIDER)

        call_kwargs = mock_comp.call_args[1]
        assert call_kwargs["model"] == OLLAMA_MODEL

    @pytest.mark.asyncio
    async def test_tools_passed_to_litellm(self):
        """Tool definitions should be forwarded to litellm.acompletion."""

        request = _make_request(
            [{"role": "user", "content": "What's the weather?"}],
            tools=[WEATHER_TOOL],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = self._mock_response()
            await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        call_kwargs = mock_comp.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == [WEATHER_TOOL]

    @pytest.mark.asyncio
    async def test_tool_choice_passed_to_litellm(self):
        """tool_choice should be forwarded to litellm.acompletion."""

        request = _make_request(
            [{"role": "user", "content": "Call get_weather"}],
            tools=[WEATHER_TOOL],
            tool_choice="required",
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = self._mock_response()
            await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        call_kwargs = mock_comp.call_args[1]
        assert call_kwargs["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_no_tools_when_absent(self):
        """When no tools are provided, tools/tool_choice should not be in kwargs."""

        request = _make_request([{"role": "user", "content": "Hello"}])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = self._mock_response()
            await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        call_kwargs = mock_comp.call_args[1]
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    @pytest.mark.asyncio
    async def test_tool_calls_in_assistant_message_preserved(self):
        """Assistant messages with tool_calls should preserve the field."""

        request = _make_request(
            [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [SAMPLE_TOOL_CALL],
                },
                {
                    "role": "tool",
                    "content": "72F sunny",
                    "tool_call_id": "call_abc123",
                    "name": "get_weather",
                },
                {"role": "user", "content": "Thanks!"},
            ],
            tools=[WEATHER_TOOL],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = self._mock_response("You're welcome!")
            await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        call_kwargs = mock_comp.call_args[1]
        messages = call_kwargs["messages"]

        # Assistant message should have tool_calls and content: None (not "")
        assistant_msg = messages[1]
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"] == [SAMPLE_TOOL_CALL]
        assert assistant_msg["content"] is None

        # Tool message should have tool_call_id and name
        tool_msg = messages[2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_abc123"
        assert tool_msg["name"] == "get_weather"
        assert tool_msg["content"] == "72F sunny"

    @pytest.mark.asyncio
    async def test_tool_calls_in_response(self):
        """When LLM returns tool_calls, they should be in the result dict."""

        request = _make_request(
            [{"role": "user", "content": "What's the weather?"}],
            tools=[WEATHER_TOOL],
        )

        # Build a mock tool_call object with model_dump
        tc_mock = MagicMock()
        tc_mock.model_dump.return_value = SAMPLE_TOOL_CALL

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = self._mock_response(
                content=None, tool_calls=[tc_mock]
            )
            result = await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        assert "tool_calls" in result
        assert result["tool_calls"] == [SAMPLE_TOOL_CALL]
        assert result["finish_reason"] == "tool_calls"

        # Verify tool_calls round-trips through JSON serialization without TypeError
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized["tool_calls"] == [SAMPLE_TOOL_CALL]

    @pytest.mark.asyncio
    async def test_no_tool_calls_in_response_when_absent(self):
        """Normal text responses should not have tool_calls key."""

        request = _make_request([{"role": "user", "content": "Hello"}])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = self._mock_response("Hi")
            result = await _call_litellm(TEST_MODEL, request, OLLAMA_PROVIDER)

        assert "tool_calls" not in result
        assert result["content"] == "Hi"


# ---------------------------------------------------------------------------
# Non-streaming response: tool_calls in JSON output
# ---------------------------------------------------------------------------


class TestNonStreamingToolCalls:
    """Verify tool_calls appear in the /v1/chat/completions JSON response."""

    def _mock_dispatch(self, content=None, tool_calls=None):
        """Build a mock response_data dict as returned by _call_litellm."""
        data = {
            "content": content,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }
        if tool_calls:
            data["tool_calls"] = tool_calls
        return data

    @pytest.mark.asyncio
    async def test_tool_calls_in_json_response(self):
        """Non-streaming response should include tool_calls in message."""

        response_data = self._mock_dispatch(content=None, tool_calls=[SAMPLE_TOOL_CALL])

        with patch(
            "nadirclaw.server._call_with_fallback", new_callable=AsyncMock
        ) as mock_fallback:
            mock_fallback.return_value = (
                response_data,
                TEST_MODEL,
                {"tier": "complex"},
            )

            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": TEST_MODEL,
                    "messages": [{"role": "user", "content": "What's the weather?"}],
                    "tools": [WEATHER_TOOL],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        msg = data["choices"][0]["message"]
        assert "tool_calls" in msg
        assert msg["tool_calls"] == [SAMPLE_TOOL_CALL]
        assert data["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_no_tool_calls_in_plain_response(self):
        """Normal text response should not have tool_calls in message."""

        response_data = self._mock_dispatch(content="Hello!", tool_calls=None)

        with patch(
            "nadirclaw.server._call_with_fallback", new_callable=AsyncMock
        ) as mock_fallback:
            mock_fallback.return_value = (
                response_data,
                TEST_MODEL,
                {"tier": "simple"},
            )

            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": TEST_MODEL,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        msg = data["choices"][0]["message"]
        assert "tool_calls" not in msg
        assert msg["content"] == "Hello!"


# ---------------------------------------------------------------------------
# Streaming response: tool_calls in SSE chunks
# ---------------------------------------------------------------------------


class TestStreamingToolCalls:
    """Verify tool_calls appear in SSE stream chunks."""

    @pytest.mark.parametrize("response_data,expected_key,expected_value,expected_finish", [
        (
            {"content": None, "finish_reason": "tool_calls", "prompt_tokens": 10,
             "completion_tokens": 5, "tool_calls": [SAMPLE_TOOL_CALL]},
            "tool_calls", [SAMPLE_TOOL_CALL], "tool_calls",
        ),
        (
            {"content": "Hello world", "finish_reason": "stop",
             "prompt_tokens": 10, "completion_tokens": 5},
            "content", "Hello world", "stop",
        ),
    ])
    def test_streaming_delta(self, response_data, expected_key, expected_value, expected_finish):
        """SSE stream delta should contain the expected key/value and finish_reason."""

        sse_response = _build_streaming_response(
            request_id="test-123",
            model=TEST_MODEL,
            response_data=response_data,
            analysis_info={"tier": "complex"},
            elapsed_ms=100,
        )

        async def collect_events():
            events = []
            async for event in sse_response.body_iterator:
                events.append(event)
            return events

        events = asyncio.get_event_loop().run_until_complete(collect_events())

        data_events = [e for e in events if isinstance(e, dict) and "data" in e]
        assert len(data_events) >= 2

        # First chunk: delta with content or tool_calls
        first_chunk = json.loads(data_events[0]["data"])
        delta = first_chunk["choices"][0]["delta"]
        assert expected_key in delta
        assert delta[expected_key] == expected_value
        # When tool_calls present, content must be null
        if "tool_calls" in delta:
            assert delta["content"] is None

        # Second chunk: finish_reason
        finish_chunk = json.loads(data_events[1]["data"])
        assert finish_chunk["choices"][0]["finish_reason"] == expected_finish


# ---------------------------------------------------------------------------
# ChatMessage model: extra fields preserved
# ---------------------------------------------------------------------------


class TestChatMessageExtras:
    """Verify ChatMessage preserves tool-related extra fields."""

    def test_tool_calls_in_model_extra(self):

        msg = ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[SAMPLE_TOOL_CALL],
        )
        assert msg.model_extra["tool_calls"] == [SAMPLE_TOOL_CALL]

    def test_tool_call_id_in_model_extra(self):

        msg = ChatMessage(
            role="tool",
            content="72F sunny",
            tool_call_id="call_abc123",
            name="get_weather",
        )
        assert msg.model_extra["tool_call_id"] == "call_abc123"
        assert msg.model_extra["name"] == "get_weather"

    def test_text_content_with_none(self):
        """tool-calling assistant messages often have content=None."""

        msg = ChatMessage(role="assistant", content=None, tool_calls=[SAMPLE_TOOL_CALL])
        assert msg.text_content() == ""


# ---------------------------------------------------------------------------
# Request metadata: tool detection
# ---------------------------------------------------------------------------


class TestToolMetadataExtraction:
    """Verify _extract_request_metadata properly detects tools."""

    @pytest.mark.parametrize("messages,tools,expected_has_tools,expected_count", [
        ([{"role": "user", "content": "Hi"}], [WEATHER_TOOL], True, 1),
        (
            [{"role": "user", "content": "Weather?"},
             {"role": "assistant", "content": None},
             {"role": "tool", "content": "72F"}],
            None, True, 1,
        ),
        ([{"role": "user", "content": "Hi"}], None, False, 0),
    ])
    def test_tool_metadata(self, messages, tools, expected_has_tools, expected_count):
        """Verify has_tools and tool_count for various inputs."""

        request = _make_request(messages, tools=tools)
        meta = _extract_request_metadata(request)
        assert meta["has_tools"] is expected_has_tools
        assert meta["tool_count"] >= expected_count
