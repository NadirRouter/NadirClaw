"""Tests for complex coding detection and enhanced reasoning markers."""

import pytest

from nadirclaw.routing import (
    detect_complex_coding,
    detect_code_review,
    detect_reasoning,
)


class TestReasoningMarkersChinese:
    """Test enhanced reasoning markers with Chinese keywords."""

    def test_chinese_step_by_step(self):
        result = detect_reasoning("请一步步分析这个问题")
        assert result["is_reasoning"] is False  # Only 1 marker

    def test_chinese_multiple_markers(self):
        result = detect_reasoning("请一步步分析，权衡优劣，给出优缺点")
        assert result["is_reasoning"] is True
        assert result["marker_count"] >= 3

    def test_chinese_deep_analysis(self):
        result = detect_reasoning("对这个架构做深入分析")
        assert result["marker_count"] >= 1

    def test_chinese_logical_reasoning(self):
        result = detect_reasoning("使用逻辑推理来论证这个方案")
        assert result["marker_count"] >= 1

    def test_chinese_compare(self):
        result = detect_reasoning("对比分析这两个方案，并逐步分析优劣")
        assert result["is_reasoning"] is True

    def test_english_diagnose(self):
        result = detect_reasoning("Diagnose the root cause of the failure")
        assert result["marker_count"] >= 1

    def test_english_architectural(self):
        result = detect_reasoning("What architectural decision should we make?")
        assert result["marker_count"] >= 1


class TestDetectComplexCoding:
    """Tests for detect_complex_coding()."""

    def test_no_messages(self):
        result = detect_complex_coding([])
        assert result["is_complex"] is False

    def test_heavy_editing(self):
        msgs = [
            _assistant_with_tools(["Edit", "Edit", "Edit", "Write", "Write"]),
            _msg("tool", "ok"),
        ]
        result = detect_complex_coding(msgs)
        assert result["is_complex"] is True
        assert any("heavy_editing" in s for s in result["signals"])

    def test_moderate_editing(self):
        msgs = [
            _assistant_with_tools(["Edit", "Edit", "Write"]),
            _msg("tool", "ok"),
        ]
        result = detect_complex_coding(msgs)
        assert any("editing" in s for s in result["signals"])

    def test_tool_combo(self):
        msgs = [
            _assistant_with_tools(["Read", "Edit", "Bash"]),
            _msg("tool", "ok"),
        ]
        result = detect_complex_coding(msgs)
        assert any("combo" in s for s in result["signals"])

    def test_coding_keywords(self):
        msgs = [
            _msg("user", "implement the feature and refactor the code to fix bug"),
            _assistant_with_tools(["Read"]),
            _msg("tool", "ok"),
        ]
        result = detect_complex_coding(msgs, message_count=5)
        assert any("keyword" in s for s in result["signals"])

    def test_deep_conversation(self):
        result = detect_complex_coding([], message_count=25)
        assert any("deep_conversation" in s for s in result["signals"])

    def test_not_complex_simple_prompt(self):
        msgs = [_msg("user", "hello")]
        result = detect_complex_coding(msgs, message_count=2)
        assert result["is_complex"] is False


class TestDetectCodeReview:
    """Tests for detect_code_review()."""

    def test_code_review(self):
        result = detect_code_review("Please review the code changes")
        assert result["is_review"] is True

    def test_pr_review(self):
        result = detect_code_review("Can you do a pull request review?")
        assert result["is_review"] is True

    def test_security_audit(self):
        result = detect_code_review("Run a security audit on the codebase")
        assert result["is_review"] is True

    def test_not_review(self):
        result = detect_code_review("Write a function to sort an array")
        assert result["is_review"] is False

    def test_static_analysis(self):
        result = detect_code_review("Run static analysis on the PR")
        assert result["is_review"] is True

    def test_review_keyword_in_system_message(self):
        result = detect_code_review(
            prompt="Check this file",
            system_message="You are a code reviewer. Please review changes.",
        )
        assert result["is_review"] is True

    def test_review_keyword_only_in_system(self):
        result = detect_code_review(
            prompt="Look at this code",
            system_message="Perform a security audit on the provided code.",
        )
        assert result["is_review"] is True


# --- Test helpers ---

class _msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.text_content = lambda: content


class _assistant_with_tools:
    def __init__(self, tool_names: list[str]):
        self.role = "assistant"
        self.content = [
            {"type": "tool_use", "name": name, "id": f"call_{i}"}
            for i, name in enumerate(tool_names)
        ]
        self.text_content = lambda: ""
