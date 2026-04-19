"""Tests for agent role detection and plan mode routing."""

import pytest

from nadirclaw.routing import (
    detect_agent_role,
    _get_last_assistant_tool_calls,
    _route_planning_session,
)


class TestDetectAgentRole:
    """Tests for detect_agent_role()."""

    def test_planning_markers(self):
        result = detect_agent_role("You are a software architect agent for planning")
        assert result["role"] == "planning"
        assert result["confidence"] == 0.95

    def test_plan_mode_active(self):
        result = detect_agent_role("Plan mode is active. Read-only planning specialist.")
        assert result["role"] == "planning"

    def test_explore_markers(self):
        result = detect_agent_role("Fast agent specialized for exploring codebases")
        assert result["role"] == "explore"
        assert result["confidence"] == 0.95

    def test_subagent_markers(self):
        result = detect_agent_role("You are a specialized agent for code review")
        assert result["role"] == "subagent"
        assert result["confidence"] == 0.90

    def test_background_agent(self):
        result = detect_agent_role("Background agent for search tasks")
        assert result["role"] == "subagent"

    def test_main_session_not_subagent(self):
        # Long system prompt should NOT be classified as subagent
        long_prompt = "You are Claude Code. " * 2000  # > 15000 chars
        result = detect_agent_role(long_prompt)
        assert result["role"] == "unknown"

    def test_short_system_prompt_subagent(self):
        short_prompt = "Help the user"  # < 5000 chars, no markers
        result = detect_agent_role(short_prompt)
        assert result["role"] == "subagent"
        assert result["confidence"] == 0.60

    def test_unknown_role(self):
        medium_prompt = "You are a helpful assistant" * 300  # ~8K chars
        result = detect_agent_role(medium_prompt)
        assert result["role"] == "unknown"


class TestGetLastAssistantToolCalls:
    """Tests for _get_last_assistant_tool_calls()."""

    def test_no_assistant_messages(self):
        msgs = [
            _msg("user", "hello"),
            _msg("tool", "result"),
        ]
        assert _get_last_assistant_tool_calls(msgs) == []

    def test_assistant_with_tool_calls(self):
        msgs = [
            _msg("user", "read the file"),
            _assistant_with_tools(["Read", "Bash"]),
            _msg("tool", "file contents"),
        ]
        assert _get_last_assistant_tool_calls(msgs) == ["Read", "Bash"]

    def test_returns_last_assistant_only(self):
        msgs = [
            _assistant_with_tools(["Grep"]),
            _msg("tool", "results"),
            _assistant_with_tools(["Read", "Edit"]),
            _msg("tool", "output"),
        ]
        assert _get_last_assistant_tool_calls(msgs) == ["Read", "Edit"]


class TestRoutePlanningSession:
    """Tests for _route_planning_session()."""

    def test_user_initiated_routes_to_reasoning(self):
        routing_info = {"modifiers_applied": []}
        msgs = [_msg("user", "/plan create deployment")]
        _route_planning_session(
            msgs, "simple-model", "simple",
            "simple-model", "complex-model", "reasoning-model",
            "subagent-model", "free-model", routing_info,
        )
        assert routing_info["final_model"] == "reasoning-model"
        assert routing_info["final_tier"] == "reasoning"

    def test_exploration_routes_to_fast(self):
        routing_info = {"modifiers_applied": []}
        msgs = [
            _assistant_with_tools(["Read", "Glob"]),
            _msg("tool", "file contents"),
        ]
        _route_planning_session(
            msgs, "simple-model", "simple",
            "simple-model", "complex-model", "reasoning-model",
            "subagent-model", "free-model", routing_info,
        )
        assert routing_info["final_model"] == "subagent-model"
        assert routing_info["final_tier"] == "subagent"

    def test_plan_generation_routes_to_reasoning(self):
        routing_info = {"modifiers_applied": []}
        msgs = [
            _assistant_with_tools(["Write", "Edit"]),
            _msg("tool", "file written"),
        ]
        _route_planning_session(
            msgs, "simple-model", "simple",
            "simple-model", "complex-model", "reasoning-model",
            "subagent-model", "free-model", routing_info,
        )
        assert routing_info["final_model"] == "reasoning-model"
        assert routing_info["final_tier"] == "reasoning"

    def test_context_default_routes_to_fast(self):
        routing_info = {"modifiers_applied": []}
        msgs = [
            _msg("user", "hello"),
            _msg("tool", "some result without clear tool call"),
        ]
        _route_planning_session(
            msgs, "simple-model", "simple",
            "simple-model", "complex-model", "reasoning-model",
            "subagent-model", "free-model", routing_info,
        )
        assert routing_info["final_model"] == "subagent-model"
        assert routing_info["final_tier"] == "subagent"

    def test_no_reasoning_model_falls_back_to_complex(self):
        routing_info = {"modifiers_applied": []}
        msgs = [_msg("user", "/plan something")]
        _route_planning_session(
            msgs, "simple-model", "simple",
            "simple-model", "complex-model", None,
            "subagent-model", "free-model", routing_info,
        )
        assert routing_info["final_model"] == "complex-model"

    def test_no_subagent_model_falls_back_to_simple(self):
        routing_info = {"modifiers_applied": []}
        msgs = [
            _assistant_with_tools(["Bash"]),
            _msg("tool", "output"),
        ]
        _route_planning_session(
            msgs, "simple-model", "simple",
            "simple-model", "complex-model", "reasoning-model",
            None, None, routing_info,
        )
        assert routing_info["final_model"] == "simple-model"


# --- Test helpers ---

class _msg:
    """Simple message stub for testing."""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.text_content = lambda: content


class _assistant_with_tools:
    """Assistant message stub with tool_use blocks."""
    def __init__(self, tool_names: list[str]):
        self.role = "assistant"
        self.content = [
            {"type": "tool_use", "name": name, "id": f"call_{i}"}
            for i, name in enumerate(tool_names)
        ]
        self.text_content = lambda: ""
