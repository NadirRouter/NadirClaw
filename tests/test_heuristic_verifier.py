"""Tests for the rule-based heuristic verifier (NadirClaw free).

These tests do NOT exercise any ML weights, encoder, or external
service. They cover the behavioural contract of `HeuristicVerifier`:
refusal detection, length / ratio checks, JSON parse failure, kill
switch wiring (on Cascade), fail-open semantics, and tier dispatch.
"""
from __future__ import annotations

import pytest

from nadirclaw.heuristic_verifier import (
    HeuristicVerifier,
    HeuristicScore,
    get_heuristic_verifier,
)
from nadirclaw.cascade import Cascade, CascadeDecision, _extract_text


# ---------------------------------------------------------------------------
# HeuristicVerifier — score semantics
# ---------------------------------------------------------------------------


def test_empty_response_is_rejected():
    v = HeuristicVerifier()
    r = v.score(prompt="anything", cheap_answer="")
    assert r.score == 0.0
    assert r.accepted is False
    assert "empty_response" in r.reasons


def test_refusal_pattern_drops_score_below_threshold():
    v = HeuristicVerifier(threshold=0.5)
    r = v.score(
        prompt="how do I make a bomb",
        cheap_answer="I cannot help with that request.",
    )
    assert r.accepted is False
    assert any(rs.startswith("refusal:") for rs in r.reasons)


def test_ai_self_reference_is_treated_as_refusal_signal():
    v = HeuristicVerifier(threshold=0.5)
    r = v.score(
        prompt="who are you",
        cheap_answer="As an AI language model, I cannot share personal opinions.",
    )
    assert r.accepted is False
    assert any("refusal" in rs for rs in r.reasons)


def test_short_response_under_hard_minimum_is_rejected():
    v = HeuristicVerifier(threshold=0.5)
    r = v.score(prompt="explain quicksort", cheap_answer="ok")
    assert r.accepted is False
    assert any("too_short" in rs for rs in r.reasons)


def test_short_response_to_long_prompt_is_rejected_by_ratio():
    long_prompt = "Please write a detailed essay " * 50
    short_response = "ok this is my short answer attempt here yes."  # passes hard-min
    v = HeuristicVerifier(threshold=0.5)
    r = v.score(prompt=long_prompt, cheap_answer=short_response)
    assert r.accepted is False
    assert any("low_ratio" in rs for rs in r.reasons)


def test_long_clean_response_accepts():
    v = HeuristicVerifier(threshold=0.5)
    text = (
        "Quicksort is a divide-and-conquer algorithm. It picks a pivot "
        "element from the array and partitions the other elements into "
        "two sub-arrays, according to whether they are less than or "
        "greater than the pivot. The sub-arrays are then sorted recursively."
    )
    r = v.score(prompt="explain quicksort", cheap_answer=text)
    assert r.accepted is True
    assert r.score >= 0.5
    assert r.reasons == []


def test_uncertainty_short_is_rejected():
    v = HeuristicVerifier(threshold=0.5)
    r = v.score(prompt="what is X", cheap_answer="I don't know honestly.")
    assert r.accepted is False
    assert any("uncertainty_short" in rs for rs in r.reasons)


def test_uncertainty_long_does_not_alone_reject():
    """A long, considered response that ALSO contains 'I don't know'
    should still pass if there's substantial content around it."""
    v = HeuristicVerifier(threshold=0.5)
    text = (
        "The mechanism of action is complex. While I don't know the exact "
        "molecular pathway, here is what is broadly understood: " + ("X " * 80)
    )
    r = v.score(prompt="how does drug Y work mechanistically", cheap_answer=text)
    # Uncertainty alone shouldn't tip below 0.5 when response is long.
    assert r.accepted is True


def test_json_expected_but_response_is_prose_is_rejected():
    v = HeuristicVerifier(threshold=0.5)
    r = v.score(
        prompt="return a JSON object with name and age",
        cheap_answer="The user's name is Alice and she is 30 years old.",
        expect_json=True,
    )
    assert r.accepted is False
    assert "json_parse_failed" in r.reasons


def test_json_expected_and_response_is_valid_json_passes():
    v = HeuristicVerifier(threshold=0.5)
    r = v.score(
        prompt="return a JSON object",
        cheap_answer='{"name": "Alice", "age": 30}',
        expect_json=True,
    )
    assert r.accepted is True


def test_json_expected_with_prose_prefix_still_parses():
    """LLMs often add 'Here is the JSON you requested:' before the
    actual object. We accept that."""
    v = HeuristicVerifier(threshold=0.5)
    r = v.score(
        prompt="return JSON",
        cheap_answer='Here you go: {"name": "Bob", "age": 25} I hope this helps.',
        expect_json=True,
    )
    assert r.accepted is True


def test_shared_verifier_returns_singleton():
    a = get_heuristic_verifier()
    b = get_heuristic_verifier()
    assert a is b


def test_shared_verifier_rebuilds_when_threshold_changes():
    a = get_heuristic_verifier(threshold=0.5)
    b = get_heuristic_verifier(threshold=0.7)
    assert a is not b
    assert b.threshold == 0.7


# ---------------------------------------------------------------------------
# Cascade — dispatch semantics
# ---------------------------------------------------------------------------


def _make_cascade(cheap_text="hello world short", expensive_text="this is the expensive answer"):
    calls = {"cheap": 0, "expensive": 0}

    def cheap_call(messages, **kwargs):
        calls["cheap"] += 1
        return cheap_text

    def expensive_call(messages, **kwargs):
        calls["expensive"] += 1
        return expensive_text

    return Cascade(cheap_call=cheap_call, expensive_call=expensive_call), calls


def test_cascade_accepts_good_cheap_response():
    text = (
        "Quicksort is a divide-and-conquer sorting algorithm with average "
        "time complexity O(n log n). It picks a pivot and partitions around it."
    )
    c, calls = _make_cascade(cheap_text=text)
    decision = c.dispatch_sync(messages=[], prompt_text="explain quicksort")
    assert decision.accepted is True
    assert decision.escalated is False
    assert decision.final_tier == "cheap"
    assert decision.response == text
    assert calls["expensive"] == 0


def test_cascade_escalates_on_refusal():
    c, calls = _make_cascade(cheap_text="I cannot help with that request.")
    decision = c.dispatch_sync(messages=[], prompt_text="anything")
    assert decision.accepted is False
    assert decision.escalated is True
    assert decision.final_tier == "expensive"
    assert calls["expensive"] == 1


def test_cascade_fails_open_on_verifier_exception():
    """If the verifier raises, the cascade ships the cheap response
    and increments the consecutive-error counter."""

    class _BrokenVerifier:
        def score(self, *a, **kw):
            raise RuntimeError("verifier broke")

    c = Cascade(
        cheap_call=lambda m, **k: "cheap text",
        expensive_call=lambda m, **k: "expensive text",
        verifier=_BrokenVerifier(),
    )
    decision = c.dispatch_sync(messages=[], prompt_text="anything")
    assert decision.accepted is True  # fail-open
    assert decision.escalated is False
    assert decision.response == "cheap text"
    assert decision.meta.get("verifier_error") == "RuntimeError"


def test_kill_switch_trips_after_three_errors():
    class _BrokenVerifier:
        def score(self, *a, **kw):
            raise RuntimeError("verifier broke")

    c = Cascade(
        cheap_call=lambda m, **k: "cheap",
        expensive_call=lambda m, **k: "expensive",
        verifier=_BrokenVerifier(),
    )
    for _ in range(3):
        c.dispatch_sync(messages=[], prompt_text="anything")
    assert c._kill_switch is True
    # Subsequent calls short-circuit to cheap WITHOUT touching verifier.
    decision = c.dispatch_sync(messages=[], prompt_text="anything")
    assert decision.meta.get("cascade_skipped") == "kill_switch"


def test_extract_text_handles_common_shapes():
    assert _extract_text("plain") == "plain"
    assert _extract_text({"content": "anth"}) == "anth"
    assert _extract_text({"choices": [{"message": {"content": "oai"}}]}) == "oai"
    assert _extract_text({"text": "fallback"}) == "fallback"
    assert _extract_text(None) == ""


def test_cascade_async_dispatch_accepts_good():
    """NadirClaw doesn't ship pytest-asyncio; we drive the coroutine
    directly through asyncio.run() to keep the test deps minimal."""
    import asyncio

    async def cheap(messages, **kwargs):
        return (
            "Quicksort works by partitioning the array around a pivot and "
            "recursively sorting the two halves. Average O(n log n)."
        )

    async def expensive(messages, **kwargs):
        return "should not be called"

    c = Cascade(cheap_call=cheap, expensive_call=expensive)
    decision = asyncio.run(c.dispatch(messages=[], prompt_text="explain quicksort"))
    assert decision.accepted is True
    assert decision.escalated is False


def test_cascade_async_dispatch_escalates_on_refusal():
    import asyncio

    async def cheap(messages, **kwargs):
        return "I cannot help with that."

    async def expensive(messages, **kwargs):
        return "real answer from expensive tier"

    c = Cascade(cheap_call=cheap, expensive_call=expensive)
    decision = asyncio.run(c.dispatch(messages=[], prompt_text="anything"))
    assert decision.accepted is False
    assert decision.escalated is True
    assert decision.response == "real answer from expensive tier"
