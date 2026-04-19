"""Tests for selective context compression."""

import pytest

from nadirclaw.compress import (
    compress_messages,
    _is_tool_result_content,
    _truncate_tool_result,
    _stable_hash,
)


class TestIsToolResultContent:
    def test_tool_result_block(self):
        assert _is_tool_result_content([{"type": "tool_result", "content": "ok"}]) is True

    def test_text_only(self):
        assert _is_tool_result_content([{"type": "text", "text": "hello"}]) is False

    def test_string_content(self):
        assert _is_tool_result_content("hello") is False

    def test_empty_list(self):
        assert _is_tool_result_content([]) is False


class TestTruncateToolResult:
    def test_short_content_not_truncated(self):
        content = [{"type": "tool_result", "content": "short"}]
        result, truncated = _truncate_tool_result(content, 500)
        assert truncated is False
        assert result == content

    def test_long_string_content_truncated(self):
        long_text = "x" * 1000
        content = [{"type": "tool_result", "content": long_text}]
        result, truncated = _truncate_tool_result(content, 500)
        assert truncated is True
        assert "truncated" in result[0]["content"]

    def test_long_block_content_truncated(self):
        long_text = "y" * 1000
        content = [{"type": "tool_result", "content": [{"type": "text", "text": long_text}]}]
        result, truncated = _truncate_tool_result(content, 500)
        assert truncated is True

    def test_non_tool_result_blocks_preserved(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "tool_result", "content": "x" * 1000},
        ]
        result, truncated = _truncate_tool_result(content, 500)
        assert truncated is True
        assert result[0]["type"] == "text"  # preserved


class TestCompressMessages:
    def _make_messages(self, count: int) -> list:
        """Build a simple message list with alternating roles."""
        msgs = [{"role": "system", "content": "You are helpful."}]
        for i in range(count):
            if i % 2 == 0:
                msgs.append({"role": "user", "content": f"message {i}"})
            else:
                msgs.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"response {i}"},
                        {"type": "tool_use", "id": f"call_{i}", "name": "Bash", "input": {}},
                    ],
                })
        return msgs

    def test_below_threshold_no_compression(self):
        msgs = self._make_messages(10)
        result, stats = compress_messages(msgs)
        assert result == msgs
        assert stats.get("compressed") is False

    def test_system_messages_always_preserved(self):
        msgs = [{"role": "system", "content": "system prompt"}]
        # Add enough messages to exceed threshold
        for i in range(40):
            msgs.append({"role": "user", "content": "x" * 100})
            msgs.append({"role": "assistant", "content": "y" * 100})
        result, stats = compress_messages(msgs)
        assert result[0]["role"] == "system"

    def test_tool_use_messages_preserved(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(35):
            msgs.append({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"c{i}", "name": "Read", "input": {}}],
            })
            msgs.append({"role": "tool", "content": "output"})
        result, stats = compress_messages(msgs)
        # All tool_use messages should be preserved
        tool_use_count = sum(
            1 for m in result
            if isinstance(m.get("content"), list)
            and any(isinstance(c, dict) and c.get("type") == "tool_use" for c in m["content"])
        )
        assert tool_use_count == 35

    def test_dedup_consecutive_identical(self):
        msgs = [{"role": "system", "content": "sys"}]
        long_output = "IDENTICAL_LONG_OUTPUT" * 100
        # Consecutive identical assistant text messages get deduped
        for i in range(10):
            msgs.append({"role": "user", "content": "short question"})
        for i in range(30):
            msgs.append({"role": "assistant", "content": long_output})
        result, stats = compress_messages(msgs)
        assert stats["deduped"] > 0

    def test_recent_messages_preserved(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(40):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "tool", "content": [{"type": "tool_result", "tool_use_id": f"call_{i}", "content": "x" * 1000}]})
        result, stats = compress_messages(msgs)
        last_contents = [str(m.get("content", "")) for m in result[-20:]]
        truncated = [c for c in last_contents if "truncated" in c]
        assert len(truncated) == 0

    def test_compression_ratio_calculated(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(25):
            msgs.append({"role": "user", "content": "question"})
            msgs.append({"role": "tool", "content": [{"type": "tool_result", "tool_use_id": f"call_{i}", "content": "LARGE_OUTPUT" * 500}]})
        _, stats = compress_messages(msgs)
        assert "compression_ratio" in stats
        assert stats["compression_ratio"] < 1.0
