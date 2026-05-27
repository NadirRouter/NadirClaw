"""Unit tests for the cascade rule engine (NadirClaw free / MIT).

Tests exercise:
  - YAML loading + parsing of every condition type and action type
  - Priority ordering of rules
  - `applies_when.tier_predicted_in` gating
  - `set_threshold` stacking (max wins)
  - `set_max_tokens` stacking (max wins) and composition with other actions
  - Malformed-rule rejection (parser is forgiving)
  - Default profile loads and matches expected legacy domains
  - Hot-reload cache: file change invalidates engine
  - Cascade integration: force_escalate / force_cheap short-circuit the
    verifier; set_threshold raises the effective acceptance bar.
"""
from __future__ import annotations

import time

import pytest

from nadirclaw.cascade import Cascade
from nadirclaw.cascade_rules import (
    CascadeRuleEngine,
    Rule,
    RuleDecision,
    load_inline,
    load_profile,
)
from nadirclaw.cascade_rules import engine as eng


@pytest.fixture(autouse=True)
def clear_cache():
    eng._clear_profile_cache()
    yield
    eng._clear_profile_cache()


# ---------------------------------------------------------------------------
# Parsing — conditions
# ---------------------------------------------------------------------------


def test_substring_condition_matches():
    e = load_inline([{
        "name": "r1", "priority": 1,
        "match": {"any_of": [{"substring": "foo"}]},
        "action": {"type": "force_escalate", "to_tier": "complex"},
    }])
    d = e.evaluate("this contains FOO somewhere", predicted_tier="simple")
    assert d.action == "force_escalate"
    assert d.to_tier == "complex"
    assert "r1" in d.matched_rules


def test_regex_condition_matches():
    e = load_inline([{
        "name": "r1", "priority": 1,
        "match": {"any_of": [{"regex": r"\bstep\s+\d+:"}]},
        "action": {"type": "force_escalate", "to_tier": "complex"},
    }])
    assert e.evaluate("step 1: do X", predicted_tier="simple").action == "force_escalate"
    assert e.evaluate("steps are nice", predicted_tier="simple").action == "none"


def test_prompt_length_conditions():
    e = load_inline([{
        "name": "long_prompt", "priority": 1,
        "match": {"any_of": [{"prompt_length_min": 100}]},
        "action": {"type": "force_escalate", "to_tier": "complex"},
    }])
    assert e.evaluate("x" * 50).action == "none"
    assert e.evaluate("x" * 150).action == "force_escalate"


def test_confidence_conditions():
    e = load_inline([{
        "name": "low_conf", "priority": 1,
        "match": {"any_of": [{"classifier_confidence_max": 0.5}]},
        "action": {"type": "force_escalate", "to_tier": "medium"},
    }])
    assert e.evaluate("anything", classifier_confidence=0.4).action == "force_escalate"
    assert e.evaluate("anything", classifier_confidence=0.9).action == "none"
    # Missing confidence => rule does not fire.
    assert e.evaluate("anything", classifier_confidence=None).action == "none"


def test_applies_when_tier_gate():
    e = load_inline([{
        "name": "g", "priority": 1,
        "match": {"any_of": [{"substring": "foo"}]},
        "applies_when": {"tier_predicted_in": ["simple"]},
        "action": {"type": "force_escalate", "to_tier": "medium"},
    }])
    assert e.evaluate("foo", predicted_tier="simple").action == "force_escalate"
    assert e.evaluate("foo", predicted_tier="medium").action == "none"
    assert e.evaluate("foo", predicted_tier=None).action == "none"


# ---------------------------------------------------------------------------
# Priority + stacking
# ---------------------------------------------------------------------------


def test_higher_priority_wins_on_force_escalate():
    e = load_inline([
        {"name": "low", "priority": 1,
         "match": {"any_of": [{"substring": "foo"}]},
         "action": {"type": "force_escalate", "to_tier": "medium"}},
        {"name": "high", "priority": 100,
         "match": {"any_of": [{"substring": "foo"}]},
         "action": {"type": "force_escalate", "to_tier": "complex"}},
    ])
    d = e.evaluate("foo", predicted_tier="simple")
    assert d.to_tier == "complex"
    # Both rules' names land in matched_rules (audit trail).
    assert set(d.matched_rules) == {"low", "high"}


def test_set_threshold_rules_stack_max_wins():
    e = load_inline([
        {"name": "a", "priority": 1,
         "match": {"any_of": [{"substring": "foo"}]},
         "action": {"type": "set_threshold", "threshold": 0.80}},
        {"name": "b", "priority": 1,
         "match": {"any_of": [{"substring": "foo"}]},
         "action": {"type": "set_threshold", "threshold": 0.90}},
    ])
    d = e.evaluate("foo", predicted_tier="simple")
    assert d.action == "set_threshold"
    assert d.threshold == 0.90


def test_set_threshold_stacks_with_force_escalate():
    e = load_inline([
        {"name": "esc", "priority": 100,
         "match": {"any_of": [{"substring": "foo"}]},
         "action": {"type": "force_escalate", "to_tier": "complex"}},
        {"name": "thr", "priority": 50,
         "match": {"any_of": [{"substring": "foo"}]},
         "action": {"type": "set_threshold", "threshold": 0.95}},
    ])
    d = e.evaluate("foo", predicted_tier="simple")
    assert d.action == "force_escalate"
    assert d.to_tier == "complex"
    assert d.threshold == 0.95


# ---------------------------------------------------------------------------
# set_max_tokens action (R2-Router length budgeting)
# ---------------------------------------------------------------------------


def test_set_max_tokens_basic():
    e = load_inline([{
        "name": "short", "priority": 1,
        "match": {"any_of": [{"prompt_length_max": 500}]},
        "action": {"type": "set_max_tokens", "value": 256},
    }])
    d = e.evaluate("hello world", predicted_tier="simple")
    assert d.action == "set_max_tokens"
    assert d.max_tokens == 256
    assert "short" in d.matched_rules
    # Long prompt does not match
    d2 = e.evaluate("x" * 1000, predicted_tier="simple")
    assert d2.action == "none"
    assert d2.max_tokens is None


def test_set_max_tokens_max_wins_on_conflict():
    """When multiple set_max_tokens rules match, the MAX value wins."""
    e = load_inline([
        {"name": "tight", "priority": 1,
         "match": {"any_of": [{"prompt_length_min": 100}]},
         "action": {"type": "set_max_tokens", "value": 256}},
        {"name": "loose", "priority": 1,
         "match": {"any_of": [{"prompt_length_min": 100}]},
         "action": {"type": "set_max_tokens", "value": 1024}},
    ])
    d = e.evaluate("x" * 500, predicted_tier="simple")
    assert d.max_tokens == 1024
    assert set(d.matched_rules) == {"tight", "loose"}


def test_set_max_tokens_composes_with_force_cheap():
    """set_max_tokens stacks alongside force_cheap (independent fields)."""
    e = load_inline([
        {"name": "downgrade", "priority": 100,
         "match": {"any_of": [{"classifier_confidence_max": 0.5}]},
         "applies_when": {"tier_predicted_in": ["medium"]},
         "action": {"type": "force_cheap", "to_tier": "simple"}},
        {"name": "short_budget", "priority": 50,
         "match": {"any_of": [{"prompt_length_max": 500}]},
         "action": {"type": "set_max_tokens", "value": 256}},
    ])
    d = e.evaluate("hi", predicted_tier="medium", classifier_confidence=0.3)
    # Primary action is force_cheap, but budget rides along.
    assert d.action == "force_cheap"
    assert d.to_tier == "simple"
    assert d.max_tokens == 256
    assert set(d.matched_rules) == {"downgrade", "short_budget"}


def test_set_max_tokens_rejects_missing_value():
    """Malformed set_max_tokens rule (no value) is skipped, not a no-op."""
    e = load_inline([
        {"name": "no_value", "priority": 1,
         "match": {"any_of": [{"substring": "foo"}]},
         "action": {"type": "set_max_tokens"}},
        {"name": "good", "priority": 1,
         "match": {"any_of": [{"substring": "foo"}]},
         "action": {"type": "set_max_tokens", "value": 512}},
    ])
    assert len(e.rules) == 1
    assert e.rules[0].name == "good"


def test_set_max_tokens_rejects_nonpositive_value():
    e = load_inline([
        {"name": "zero", "priority": 1,
         "match": {"any_of": [{"substring": "foo"}]},
         "action": {"type": "set_max_tokens", "value": 0}},
        {"name": "negative", "priority": 1,
         "match": {"any_of": [{"substring": "foo"}]},
         "action": {"type": "set_max_tokens", "value": -1}},
    ])
    assert len(e.rules) == 0


def test_empty_prompt_is_safe():
    e = load_inline([{
        "name": "r1", "priority": 1,
        "match": {"any_of": [{"substring": "foo"}]},
        "action": {"type": "force_escalate", "to_tier": "complex"},
    }])
    assert e.evaluate("", predicted_tier="simple").action == "none"


# ---------------------------------------------------------------------------
# Malformed rules are skipped, not raised
# ---------------------------------------------------------------------------


def test_malformed_rule_is_skipped():
    e = load_inline([
        {"name": "good", "priority": 1,
         "match": {"any_of": [{"substring": "foo"}]},
         "action": {"type": "force_escalate", "to_tier": "complex"}},
        {"priority": 1, "match": {"any_of": [{"substring": "x"}]}},  # no name
        {"name": "bad_regex", "priority": 1,
         "match": {"any_of": [{"regex": "[invalid"}]},
         "action": {"type": "force_escalate", "to_tier": "complex"}},
        {"name": "bad_action", "priority": 1,
         "match": {"any_of": [{"substring": "x"}]},
         "action": {"type": "nuke_database"}},
    ])
    # Only the good rule survives.
    assert len(e.rules) == 1
    assert e.rules[0].name == "good"


# ---------------------------------------------------------------------------
# Profile loader + hot reload
# ---------------------------------------------------------------------------


def test_load_default_profile_has_rules():
    e = load_profile("default")
    assert len(e.rules) > 0
    # The default profile encodes code + summarize patterns.
    names = {r.name for r in e.rules}
    assert any("code" in n for n in names)
    assert any("summarize" in n for n in names)


def test_load_default_profile_forces_escalation_on_code():
    e = load_profile("default")
    d = e.evaluate("Here is some code:\n```python\ndef foo():\n  pass\n```")
    assert d.action == "force_escalate"
    assert "code_python_triple_backtick" in d.matched_rules


def test_load_unknown_profile_returns_empty_engine():
    e = load_profile("this_profile_does_not_exist")
    assert e.rules == ()
    # Empty engine returns "none" on every prompt.
    assert e.evaluate("anything", predicted_tier="simple").action == "none"


def test_hot_reload_on_mtime_change(tmp_path):
    """Profile cache invalidates when the file's mtime changes."""
    p = tmp_path / "tenant_x.yaml"
    p.write_text("""
- name: v1
  priority: 1
  match:
    any_of: [{substring: "alpha"}]
  action: {type: force_escalate, to_tier: complex}
""")
    e1 = load_profile(str(p))
    assert {r.name for r in e1.rules} == {"v1"}
    # Bump mtime forward + rewrite. Sleep to ensure the stat picks it up
    # on filesystems with 1-second mtime granularity.
    time.sleep(1.1)
    p.write_text("""
- name: v2
  priority: 1
  match:
    any_of: [{substring: "beta"}]
  action: {type: force_escalate, to_tier: medium}
""")
    # Force the cache to bypass the TTL fast-path by directly setting
    # the cached ts to long ago.
    with eng._PROFILE_CACHE_LOCK:
        for k, (mtime, _ts, en) in list(eng._PROFILE_CACHE.items()):
            eng._PROFILE_CACHE[k] = (mtime, 0.0, en)
    e2 = load_profile(str(p))
    assert {r.name for r in e2.rules} == {"v2"}


# ---------------------------------------------------------------------------
# Cascade integration: rule engine drives dispatch decisions
# ---------------------------------------------------------------------------


def _make_cascade_with_engine(engine):
    calls = {"cheap": 0, "expensive": 0}

    def cheap_call(messages, **kwargs):
        calls["cheap"] += 1
        return "cheap-text"

    def expensive_call(messages, **kwargs):
        calls["expensive"] += 1
        return "expensive-text"

    c = Cascade(
        cheap_call=cheap_call,
        expensive_call=expensive_call,
        rule_engine=engine,
    )
    return c, calls


def test_cascade_force_escalate_skips_verifier():
    engine = load_inline([{
        "name": "force_code",
        "priority": 100,
        "match": {"any_of": [{"substring": "```python"}]},
        "action": {"type": "force_escalate", "to_tier": "complex"},
    }])
    c, calls = _make_cascade_with_engine(engine)
    d = c.dispatch_sync(messages=[], prompt_text="here is ```python code")
    assert d.escalated is True
    assert d.accepted is False
    assert d.response == "expensive-text"
    assert calls["cheap"] == 0
    assert calls["expensive"] == 1
    assert "force_code" in d.matched_rules
    assert d.meta.get("rule_action") == "force_escalate"


def test_cascade_force_cheap_skips_verifier_and_expensive():
    engine = load_inline([{
        "name": "trivial",
        "priority": 100,
        "match": {"any_of": [{"prompt_length_max": 20}]},
        "action": {"type": "force_cheap", "to_tier": "simple"},
    }])
    c, calls = _make_cascade_with_engine(engine)
    d = c.dispatch_sync(messages=[], prompt_text="hi")
    assert d.escalated is False
    assert d.accepted is True
    assert d.response == "cheap-text"
    assert calls["cheap"] == 1
    assert calls["expensive"] == 0
    assert "trivial" in d.matched_rules
    assert d.meta.get("rule_action") == "force_cheap"


def test_cascade_set_threshold_only_raises_the_bar():
    """A set_threshold rule should never lower the cascade's default τ."""
    # A long clean response that scores 1.0 on the verifier. With a
    # rule-driven threshold of 0.99 it still passes; with τ=2.0 it would
    # fail — but rules can't ever push above 1.0 in practice.
    engine = load_inline([{
        "name": "strict",
        "priority": 1,
        "match": {"any_of": [{"substring": "anything"}]},
        "action": {"type": "set_threshold", "threshold": 0.99},
    }])
    calls = {"cheap": 0, "expensive": 0}

    def cheap(messages, **k):
        calls["cheap"] += 1
        return (
            "This is a long, clean, on-topic response. " * 5
        )

    def expensive(messages, **k):
        calls["expensive"] += 1
        return "expensive answer"

    c = Cascade(cheap_call=cheap, expensive_call=expensive, rule_engine=engine)
    d = c.dispatch_sync(messages=[], prompt_text="anything goes here")
    # Verifier returns ~1.0 for a clean long answer; should still accept.
    assert d.accepted is True
    assert d.escalated is False
    assert calls["expensive"] == 0
    assert "strict" in d.matched_rules


def test_cascade_with_no_rule_engine_is_unchanged():
    """Backwards compat: omit the rule_engine arg → cascade behaves as
    before the engine landed."""
    calls = {"cheap": 0, "expensive": 0}

    def cheap(messages, **k):
        calls["cheap"] += 1
        return "I cannot help with that."  # refusal -> escalate

    def expensive(messages, **k):
        calls["expensive"] += 1
        return "real answer"

    c = Cascade(cheap_call=cheap, expensive_call=expensive)  # no rule_engine
    d = c.dispatch_sync(messages=[], prompt_text="anything")
    assert d.escalated is True
    assert d.matched_rules == []
    assert calls["expensive"] == 1


def test_cascade_engine_evaluate_failure_is_swallowed(monkeypatch):
    """A buggy custom engine must not bring down the request path."""

    class _BrokenEngine:
        def evaluate(self, *a, **k):
            raise RuntimeError("engine boom")

    calls = {"cheap": 0, "expensive": 0}

    def cheap(messages, **k):
        calls["cheap"] += 1
        return (
            "Long clean response from the cheap tier that should accept. "
            * 5
        )

    def expensive(messages, **k):
        calls["expensive"] += 1
        return "expensive"

    c = Cascade(
        cheap_call=cheap,
        expensive_call=expensive,
        rule_engine=_BrokenEngine(),  # type: ignore[arg-type]
    )
    d = c.dispatch_sync(messages=[], prompt_text="anything")
    # Engine error -> fall through to verifier; clean response accepts.
    assert d.accepted is True
    assert d.matched_rules == []
    assert calls["expensive"] == 0
