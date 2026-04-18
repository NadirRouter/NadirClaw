"""Tests for Anthropic Messages API compatibility layer."""

import json
import pytest

from nadirclaw.anthropic_api import (
    anthropic_to_openai_messages,
    anthropic_tools_to_openai,
    openai_response_to_anthropic,
    build_anthropic_sse_events,
    _extract_last_user_text,
)


class TestAnthropicToOpenAI:
    """Tests for Anthropic → OpenAI message conversion."""

    def test_system_prompt_string(self):
        result = anthropic_to_openai_messages(
            messages=[{"role": "user", "content": "hello"}],
            system="You are helpful",
        )
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"
        assert result[1]["role"] == "user"

    def test_system_prompt_blocks(self):
        result = anthropic_to_openai_messages(
            messages=[{"role": "user", "content": "hi"}],
            system=[{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}],
        )
        assert "Part 1" in result[0]["content"]
        assert "Part 2" in result[0]["content"]

    def test_assistant_with_tool_calls(self):
        result = anthropic_to_openai_messages([{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me read that file"},
                {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "/tmp/x"}},
            ],
        }])
        msg = result[0]
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "Read"

    def test_user_tool_result(self):
        result = anthropic_to_openai_messages([{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "file contents here"},
            ],
        }])
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "file contents here"

    def test_user_tool_result_with_blocks(self):
        result = anthropic_to_openai_messages([{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": [{"type": "text", "text": "output text"}],
            }],
        }])
        assert result[0]["role"] == "tool"
        assert "output text" in result[0]["content"]

    def test_simple_text_messages(self):
        result = anthropic_to_openai_messages([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ])
        assert len(result) == 2
        assert result[0]["content"] == "hello"
        assert result[1]["content"] == "hi there"

    def test_no_system(self):
        result = anthropic_to_openai_messages([
            {"role": "user", "content": "hello"},
        ])
        assert result[0]["role"] == "user"


class TestAnthropicToolsToOpenAI:
    """Tests for tool definition conversion."""

    def test_custom_tool(self):
        result = anthropic_tools_to_openai([{
            "type": "custom",
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "read_file"

    def test_empty_tools(self):
        assert anthropic_tools_to_openai([]) == []

    def test_non_custom_type_skipped(self):
        result = anthropic_tools_to_openai([{"type": "computer_20241022", "name": "computer"}])
        assert len(result) == 0


class TestOpenAIToAnthropic:
    """Tests for OpenAI → Anthropic response conversion."""

    def test_text_response(self):
        result = openai_response_to_anthropic(
            {"content": "Hello!", "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 5},
            model="gpt-5.2",
            request_id="test-123",
        )
        assert result["type"] == "message"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"

    def test_tool_calls_response(self):
        result = openai_response_to_anthropic({
            "content": "",
            "finish_reason": "tool_calls",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "Read", "arguments": '{"path": "/tmp/x"}'},
            }],
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }, model="gpt-5.2", request_id="test-123")
        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"

    def test_length_stop_reason(self):
        result = openai_response_to_anthropic({
            "content": "truncated",
            "finish_reason": "length",
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }, model="gpt-5.2", request_id="test-123")
        assert result["stop_reason"] == "max_tokens"

    def test_empty_response(self):
        result = openai_response_to_anthropic({
            "finish_reason": "stop",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }, model="gpt-5.2", request_id="test-123")
        assert result["content"][0]["text"] == ""


class TestBuildAnthropicSSEEvents:
    """Tests for SSE event generation."""

    def test_text_only_events(self):
        events = build_anthropic_sse_events("req-1", "gpt-5.2", {
            "content": "Hello!",
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 5,
        })
        event_types = [e["event"] for e in events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

    def test_tool_calls_events(self):
        events = build_anthropic_sse_events("req-1", "gpt-5.2", {
            "content": "",
            "finish_reason": "tool_calls",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
            }],
            "prompt_tokens": 10,
            "completion_tokens": 5,
        })
        # Should have tool_use content block
        data_str = json.dumps([e["data"] for e in events])
        assert "tool_use" in data_str


class TestExtractLastUserText:
    """Tests for _extract_last_user_text()."""

    def test_simple_text(self):
        assert _extract_last_user_text([
            {"role": "user", "content": "hello"},
        ]) == "hello"

    def test_content_blocks(self):
        assert _extract_last_user_text([{
            "role": "user",
            "content": [{"type": "text", "text": "read the file"}],
        }]) == "read the file"

    def test_tool_result_only(self):
        result = _extract_last_user_text([{
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"}],
        }])
        assert result == ""

    def test_no_user_messages(self):
        assert _extract_last_user_text([]) == ""
