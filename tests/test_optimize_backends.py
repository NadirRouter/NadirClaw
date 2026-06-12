"""Tests for the optimizer backend selection (native vs headroom).

These cover the contract that matters for safety:
- ``off`` mode is a zero-cost no-op regardless of backend.
- The ``headroom`` backend transparently falls back to ``native`` when the
  optional ``headroom-ai`` package is absent (the common case), producing
  byte-identical output and never reporting headroom transforms.
- Backend selection resolves from arg → env → ``native`` default.
- The extension hooks used by Nadir Pro fire in the expected places.
"""

import importlib

import pytest

from nadirclaw.optimize import (
    OptimizeResult,
    _resolve_backend,
    optimize_messages,
)

HEADROOM_INSTALLED = importlib.util.find_spec("headroom") is not None


def _sample():
    return [
        {"role": "system", "content": "You are a helpful assistant. " * 3},
        {"role": "user", "content": 'Parse {"a":   1,    "b":   2}  with   extra   spaces.'},
    ]


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def test_resolve_backend_default_native(monkeypatch):
    monkeypatch.delenv("NADIRCLAW_OPTIMIZE_BACKEND", raising=False)
    assert _resolve_backend(None) == "native"


def test_resolve_backend_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("NADIRCLAW_OPTIMIZE_BACKEND", "native")
    assert _resolve_backend("headroom") == "headroom"


def test_resolve_backend_env(monkeypatch):
    monkeypatch.setenv("NADIRCLAW_OPTIMIZE_BACKEND", "headroom")
    assert _resolve_backend(None) == "headroom"


def test_resolve_backend_invalid_falls_back(monkeypatch):
    monkeypatch.delenv("NADIRCLAW_OPTIMIZE_BACKEND", raising=False)
    assert _resolve_backend("nonsense") == "native"


# ---------------------------------------------------------------------------
# off-mode short-circuit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["native", "headroom"])
def test_off_is_noop_for_any_backend(backend):
    msgs = _sample()
    result = optimize_messages(msgs, mode="off", backend=backend)
    assert isinstance(result, OptimizeResult)
    assert result.tokens_saved == 0
    assert result.mode == "off"
    # off returns the original list object untouched (zero overhead)
    assert result.messages is msgs


# ---------------------------------------------------------------------------
# headroom backend fallback (when headroom-ai is not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(HEADROOM_INSTALLED, reason="headroom-ai installed; fallback path not exercised")
@pytest.mark.parametrize("mode", ["safe", "aggressive"])
def test_headroom_falls_back_to_native_when_absent(mode):
    msgs = _sample()
    native = optimize_messages([{**m} for m in msgs], mode=mode, backend="native")
    headroom = optimize_messages([{**m} for m in msgs], mode=mode, backend="headroom")

    # Output must be identical to native...
    assert headroom.messages == native.messages
    # ...and must never claim a headroom transform ran.
    assert not any(t.startswith("headroom") for t in headroom.optimizations_applied)


# ---------------------------------------------------------------------------
# extension hooks (the mechanism Nadir Pro builds on)
# ---------------------------------------------------------------------------

def test_extra_safe_content_hook_runs():
    def shout(content):
        new = content.replace("hello", "HELLO")
        return new, new != content

    msgs = [{"role": "user", "content": "hello there, this is a reasonably long message"}]
    result = optimize_messages(
        msgs, mode="safe", extra_safe_content=[("shout", shout)]
    )
    assert "shout" in result.optimizations_applied
    assert "HELLO" in result.messages[0]["content"]


def test_extra_aggressive_hooks_skipped_in_safe_mode():
    def boom(_content):
        raise AssertionError("aggressive hook must not run in safe mode")

    msgs = [{"role": "user", "content": "a fairly long user message to exceed the length floor"}]
    # Should not raise — aggressive hooks only fire in aggressive mode.
    optimize_messages(
        msgs, mode="safe", extra_aggressive_content=[("boom", boom)]
    )
