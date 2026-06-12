"""Tests for progressive (staged) compression.

Headroom stages require the optional ``headroom-ai`` package (Python <= 3.13).
These tests cover the escalation logic, early-stop, stage capping, lossy gating,
and graceful skip when Headroom is absent — all observable on the native stages.
The Headroom stages engaging is exercised manually in a 3.13 venv.
"""
import json

import pytest

from nadirclaw.optimize import compress_progressive


def _stages(result):
    return [x.split(":", 1)[1] for x in result.optimizations_applied if x.startswith("stage:")]


def _big_msgs():
    rows = [{"id": 1000 + i, "user": f"user{i}", "status": "active" if i % 3 else "off",
             "plan": "pro" if i % 5 == 0 else "free"} for i in range(60)]
    return [
        {"role": "system", "content": "You are a helpful assistant. " * 6},
        {"role": "user", "content": "summarize the users"},
        {"role": "tool", "content": "get_users():\n" + json.dumps(rows, indent=2)},
    ]


def test_mode_is_progressive():
    r = compress_progressive(_big_msgs())
    assert r.mode == "progressive"


def test_no_target_stops_at_max_stage_native():
    # Without a budget, escalation stops after native_aggressive — never reaches Headroom.
    r = compress_progressive(_big_msgs())
    assert _stages(r) == ["native_safe", "native_aggressive"]
    assert not any(s.startswith("headroom") for s in _stages(r))
    assert r.tokens_saved > 0


def test_generous_target_early_stops_after_safe():
    msgs = _big_msgs()
    from nadirclaw.optimize import _estimate_tokens_messages
    orig = _estimate_tokens_messages(msgs)
    r = compress_progressive(msgs, target_tokens=int(orig * 0.9), max_stage="headroom_ml", allow_lossy=True)
    # Safe alone gets under 90% here, so it must stop immediately.
    assert _stages(r) == ["native_safe"]


def test_max_stage_caps_ladder():
    # Cap at native_safe → aggressive never runs even though target is unmet.
    r = compress_progressive(_big_msgs(), target_tokens=1, max_stage="native_safe")
    assert _stages(r) == ["native_safe"]


def test_headroom_skipped_gracefully_when_absent():
    # Unmeetable target + headroom requested: on a host without headroom-ai the
    # headroom stages are skipped (not recorded), output stays valid native.
    import importlib.util
    r = compress_progressive(_big_msgs(), target_tokens=1, max_stage="headroom_ml", allow_lossy=True)
    if importlib.util.find_spec("headroom") is None:
        assert _stages(r) == ["native_safe", "native_aggressive"]
    # Either way the result is well-formed and never larger than the input.
    assert r.optimized_tokens <= r.original_tokens
    assert all("content" in m for m in r.messages)


def test_lossy_gated_off_by_default():
    # allow_lossy=False must drop headroom_ml from the ladder entirely.
    r = compress_progressive(_big_msgs(), target_tokens=1, max_stage="headroom_ml", allow_lossy=False)
    assert "headroom_ml" not in _stages(r)


def test_already_small_is_noop():
    small = [{"role": "user", "content": "hi"}]
    r = compress_progressive(small, target_tokens=10_000)
    assert _stages(r) == []  # already under budget, nothing runs
    assert r.tokens_saved == 0
