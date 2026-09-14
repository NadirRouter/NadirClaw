"""Tests for nadirclaw.read_hook — structural views for oversized Claude Code reads."""

import io
import json

import pytest

from nadirclaw.claude_integration import (
    READ_HOOK_LABEL,
    patch_claude_read_hook,
    unpatch_claude_read_hook,
)
from nadirclaw.read_hook import MAX_VIEW_BYTES, decide, python_view, run


def _event(path, **extra):
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "Read",
        "tool_input": dict({"file_path": str(path)}, **extra),
    }


@pytest.fixture
def big_module(tmp_path):
    """A Python file comfortably past the read cap, with one known deep symbol."""
    body = "".join(f"        step_{n} = {n}\n" for n in range(14000))
    source = (
        '"""Module docstring."""\n\n\n'
        "CONSTANT = 1\n\n\n"
        "class Service:\n"
        '    """Service docstring."""\n\n'
        "    def handle(self):\n" + body + "\n\n"
        "def tail_function():\n"
        "    return step_marker\n"
    )
    path = tmp_path / "big.py"
    path.write_text(source)
    return path


def test_serves_view_with_original_line_numbers(big_module):
    output, report = decide(_event(big_module))
    decision = output["hookSpecificOutput"]

    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"
    assert report["served"] and report["returned_chars"] < report["original_chars"]

    served = decision["permissionDecisionReason"]
    # Declarations survive, bodies do not, and the reader is told both that the
    # numbers are original and that the view is not editable source.
    assert "class Service" in served and "def tail_function" in served
    assert "step_13999" not in served
    assert "source lines" in served and "not valid replacement code" in served
    assert str(big_module) in served

    # The last declaration must carry a line number past the truncation point a
    # plain read would have hit; that is the whole reason the hook exists.
    marker = served.split("[source lines ")[-1].split("-")[0]
    assert int(marker) > 14000


def test_narrowed_reads_are_the_recovery_path(big_module):
    for extra in ({"offset": 1}, {"limit": 40}, {"offset": 10, "limit": 40}):
        assert decide(_event(big_module, **extra)) == ({}, {"reason": "targeted_read_preserved"})


@pytest.mark.parametrize(
    "name, contents, expected",
    [
        ("small.py", "def f():\n    return 1\n", "below_read_cap"),
        ("notes.md", "x" * 200_000, "unsupported_language"),
        ("broken.py", "def (((\n" + "# pad\n" * 40_000, "code_structure_unavailable"),
    ],
)
def test_reads_that_pass_through(tmp_path, name, contents, expected):
    path = tmp_path / name
    path.write_text(contents)
    assert decide(_event(path))[1]["reason"] == expected


def test_missing_file_is_left_to_the_read_tool(tmp_path):
    assert decide(_event(tmp_path / "absent.py"))[1]["reason"] == "unreadable_preserved"


def test_module_too_large_to_parse_in_time_is_declined(tmp_path):
    path = tmp_path / "oversized.py"
    path.write_text("x = 1\n" * (MAX_VIEW_BYTES // 6 + 1000))
    assert decide(_event(path))[1]["reason"] == "oversized_preserved"


def test_other_events_and_tools_are_ignored(big_module):
    assert decide({**_event(big_module), "tool_name": "Bash"})[1]["reason"] == "unsupported_tool"
    assert decide({**_event(big_module), "hook_event_name": "PostToolUse"})[1]["reason"] == "unsupported_event"
    assert decide({**_event(big_module), "tool_input": None})[1]["reason"] == "unsupported_shape"


def test_lean_view_drops_docstrings_but_keeps_locations():
    source = '"""Doc."""\n\n\ndef f():\n    """Inner."""\n    return 1\n'
    assert "Inner." in python_view(source)
    lean = python_view(source, lean=True)
    assert "Inner." not in lean and "def f():" in lean and "L4" in lean


def test_run_never_blocks_a_read_on_malformed_input():
    out = io.StringIO()
    assert run(io.StringIO("not json"), out)["reason"] == "hook_error_read_preserved"
    assert out.getvalue() == ""


def test_run_emits_the_decision_for_a_served_read(big_module):
    out = io.StringIO()
    report = run(io.StringIO(json.dumps(_event(big_module))), out)
    assert report["served"]
    assert json.loads(out.getvalue())["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_install_preserves_other_hooks_and_is_idempotent(tmp_path):
    settings = tmp_path / "settings.json"
    other = {"type": "command", "command": "echo unrelated"}
    settings.write_text(json.dumps({
        "env": {"ANTHROPIC_BASE_URL": "http://localhost:8856"},
        "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [other]}]},
    }))

    patch_claude_read_hook(settings)
    patch_claude_read_hook(settings)
    config = json.loads(settings.read_text())
    entries = config["hooks"]["PreToolUse"]

    # Unrelated hook kept, ours registered exactly once, other settings intact.
    assert {"matcher": "Bash", "hooks": [other]} in entries
    ours = [h for entry in entries for h in entry["hooks"]
            if h.get("statusMessage") == READ_HOOK_LABEL]
    assert len(ours) == 1 and ours[0]["command"].endswith("claude read-hook")
    assert config["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:8856"

    assert unpatch_claude_read_hook(settings) is True
    after = json.loads(settings.read_text())
    assert after["hooks"]["PreToolUse"] == [{"matcher": "Bash", "hooks": [other]}]
    assert after["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:8856"
    # Nothing of ours left, so a second removal reports no change.
    assert unpatch_claude_read_hook(settings) is False


def test_uninstall_drops_the_empty_hooks_block(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {"ANTHROPIC_API_KEY": "local"}}))
    patch_claude_read_hook(settings)
    assert unpatch_claude_read_hook(settings) is True
    assert json.loads(settings.read_text()) == {"env": {"ANTHROPIC_API_KEY": "local"}}
