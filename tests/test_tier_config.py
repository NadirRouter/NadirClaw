"""Unit tests for the N-tier YAML configuration (NadirClaw free / PolyForm Noncommercial).

Covers:
  - n2_default + n3_legacy bundled profiles load and validate
  - TierSelector returns expected tiers across the score range
  - softmax_to_score is deterministic and clamped to [0, 1]
  - Hot-reload: editing the YAML triggers a re-parse on next load
  - NTierCascade escalates through the ladder when the verifier rejects
  - Cascade rule engine integration: force_escalate / force_cheap
  - max_escalations safety cap
  - Backward compat: legacy 2-tier Cascade still works untouched
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from nadirclaw.cascade import Cascade, CascadeDecision, NTierCascade
from nadirclaw.cascade_rules import load_inline
from nadirclaw.tier_config import (
    PROFILES_DIR,
    Tier,
    TierProfile,
    TierSelector,
    get_default_profile_path,
    load_profile,
    probs_dict_to_score,
    softmax_to_score,
)
from nadirclaw.tier_config import loader as tier_loader
from nadirclaw.tier_config.schema import CascadeConfig, SelectorConfig


@pytest.fixture(autouse=True)
def _clear_cache_and_env(monkeypatch):
    tier_loader._clear_profile_cache()
    monkeypatch.delenv("NADIRCLAW_TIERS_PROFILE", raising=False)
    yield
    tier_loader._clear_profile_cache()


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------


def test_default_profile_path_is_n2_when_env_unset():
    path = get_default_profile_path()
    assert path.name == "n2_default.yaml"
    assert path.parent == PROFILES_DIR


def test_default_profile_path_honours_env_bare_name(monkeypatch):
    monkeypatch.setenv("NADIRCLAW_TIERS_PROFILE", "n3_legacy")
    path = get_default_profile_path()
    assert path.name == "n3_legacy.yaml"


def test_default_profile_path_honours_env_absolute_path(monkeypatch, tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("version: 1\nmode: tiered\ntiers:\n  - {name: only, score_min: 0.0, model_pool: [m]}\n")
    monkeypatch.setenv("NADIRCLAW_TIERS_PROFILE", str(custom))
    path = get_default_profile_path()
    assert path == custom


def test_n2_default_loads_clean():
    p = load_profile("n2_default")
    assert p.profile_name == "n2_default"
    assert p.num_tiers == 2
    assert p.mode == "tiered"
    assert [t.name for t in p.tiers] == ["cheap", "strong"]
    assert p.tiers[0].score_min == 0.0
    assert p.tiers[1].score_min == 0.65
    # Confirm the cheap pool has the RouterArena models we picked.
    assert "gpt-4o-mini" in p.tiers[0].model_pool
    assert "qwen3-235b-a22b-2507" in p.tiers[0].model_pool
    # And the strong pool.
    assert "gpt-5-mini" in p.tiers[1].model_pool
    assert "claude-sonnet-4" in p.tiers[1].model_pool


def test_n3_legacy_loads_clean():
    p = load_profile("n3_legacy")
    assert p.num_tiers == 3
    assert [t.name for t in p.tiers] == ["simple", "medium", "complex"]
    assert p.tiers[0].score_min == 0.0
    assert p.tiers[1].score_min == 0.35
    assert p.tiers[2].score_min == 0.65


def test_loader_uses_env_when_no_arg(monkeypatch):
    monkeypatch.setenv("NADIRCLAW_TIERS_PROFILE", "n3_legacy")
    p = load_profile()
    assert p.profile_name == "n3_legacy"
    assert p.num_tiers == 3


def test_loader_returns_bundled_default_when_missing(monkeypatch, tmp_path):
    # Point at a path that does not exist. Loader should fall back to
    # the in-code bundled n2 default rather than crashing.
    monkeypatch.setenv("NADIRCLAW_TIERS_PROFILE", str(tmp_path / "nope.yaml"))
    p = load_profile()
    assert p.num_tiers == 2
    assert p.tiers[0].name == "cheap"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_schema_rejects_unsorted_cutoffs():
    with pytest.raises(Exception):
        TierProfile(
            tiers=[
                Tier(name="a", score_min=0.5, model_pool=["m"]),
                Tier(name="b", score_min=0.2, model_pool=["m"]),
            ],
        )


def test_schema_rejects_first_tier_above_zero():
    with pytest.raises(Exception):
        TierProfile(
            tiers=[
                Tier(name="a", score_min=0.1, model_pool=["m"]),
                Tier(name="b", score_min=0.5, model_pool=["m"]),
            ],
        )


def test_schema_rejects_duplicate_names():
    with pytest.raises(Exception):
        TierProfile(
            tiers=[
                Tier(name="a", score_min=0.0, model_pool=["m"]),
                Tier(name="a", score_min=0.5, model_pool=["m"]),
            ],
        )


def test_schema_flat_mode_requires_quality_and_cost():
    with pytest.raises(Exception):
        TierProfile(
            mode="flat",
            tiers=[
                Tier(name="a", score_min=0.0, model_pool=["m"]),  # missing quality / cost
            ],
        )


def test_schema_rejects_bogus_escalation():
    with pytest.raises(Exception):
        CascadeConfig(escalation="learned")


# ---------------------------------------------------------------------------
# Score adapter
# ---------------------------------------------------------------------------


def test_softmax_to_score_one_hot():
    assert softmax_to_score([1.0, 0.0, 0.0]) == 0.0
    assert softmax_to_score([0.0, 1.0, 0.0]) == 0.5
    assert softmax_to_score([0.0, 0.0, 1.0]) == 1.0


def test_softmax_to_score_mixed():
    s = softmax_to_score([0.2, 0.5, 0.3])
    # E[class] = 0*0.2 + 1*0.5 + 2*0.3 = 1.1 → score = 0.55
    assert abs(s - 0.55) < 1e-9


def test_softmax_to_score_clamps():
    # Slightly out-of-range floats (numpy noise) should clamp.
    s = softmax_to_score([1.0001, -0.0001, 0.0])
    assert 0.0 <= s <= 1.0


def test_probs_dict_to_score():
    s = probs_dict_to_score({"simple": 0.2, "medium": 0.5, "complex": 0.3})
    assert abs(s - 0.55) < 1e-9


def test_probs_dict_to_score_missing_key():
    with pytest.raises(ValueError):
        probs_dict_to_score({"simple": 1.0})


# ---------------------------------------------------------------------------
# TierSelector
# ---------------------------------------------------------------------------


def test_selector_n2_picks_cheap_for_low_score():
    p = load_profile("n2_default")
    s = TierSelector(p)
    assert s.assign(0.0).tier_name == "cheap"
    assert s.assign(0.3).tier_name == "cheap"
    assert s.assign(0.6499).tier_name == "cheap"


def test_selector_n2_picks_strong_for_high_score():
    p = load_profile("n2_default")
    s = TierSelector(p)
    assert s.assign(0.65).tier_name == "strong"
    assert s.assign(0.9).tier_name == "strong"
    assert s.assign(1.0).tier_name == "strong"


def test_selector_n3_legacy_buckets():
    p = load_profile("n3_legacy")
    s = TierSelector(p)
    assert s.assign(0.1).tier_name == "simple"
    assert s.assign(0.34).tier_name == "simple"
    assert s.assign(0.35).tier_name == "medium"
    assert s.assign(0.5).tier_name == "medium"
    assert s.assign(0.65).tier_name == "complex"
    assert s.assign(0.99).tier_name == "complex"


def test_selector_clamps_out_of_range_scores():
    p = load_profile("n2_default")
    s = TierSelector(p)
    assert s.assign(-5.0).tier_name == "cheap"
    assert s.assign(5.0).tier_name == "strong"


def test_selector_next_tier_adjacent():
    p = load_profile("n3_legacy")
    s = TierSelector(p)
    assert s.next_tier("simple").name == "medium"
    assert s.next_tier("medium").name == "complex"
    assert s.next_tier("complex") is None  # terminal


def test_selector_next_tier_jump():
    p = load_profile("n3_legacy")
    p.cascade = CascadeConfig(escalation="jump", acceptance_threshold=0.8, rules_profile="default")
    s = TierSelector(p)
    # jump skips medium → straight to complex (last tier, terminal).
    assert s.next_tier("simple").name == "complex"


def test_selector_flat_mode_picks_cheapest_when_lambda_zero():
    p = TierProfile(
        mode="flat",
        selector=SelectorConfig(lambda_cost=0.0),
        tiers=[
            Tier(name="cheap_arm", score_min=0.0, model_pool=["m1"],
                 quality=0.6, cost_per_1k=0.001),
            Tier(name="mid_arm", score_min=0.0, model_pool=["m2"],
                 quality=0.75, cost_per_1k=0.01),
            Tier(name="dear_arm", score_min=0.0, model_pool=["m3"],
                 quality=0.9, cost_per_1k=0.1),
        ],
    )
    sel = TierSelector(p)
    # λ=0 → pure cost bias → cheapest arm.
    assert sel.assign(0.9).tier_name == "cheap_arm"


# ---------------------------------------------------------------------------
# Hot-reload
# ---------------------------------------------------------------------------


def test_hot_reload_picks_up_yaml_changes(monkeypatch, tmp_path):
    # Write an initial profile, load it, then mutate the file and
    # confirm the loader re-parses on next call (after we mock the TTL
    # so we don't have to wait 30 s).
    yaml_path = tmp_path / "hot.yaml"
    yaml_path.write_text(
        "version: 1\nmode: tiered\n"
        "tiers:\n"
        "  - {name: a, score_min: 0.0, model_pool: [m1]}\n"
        "  - {name: b, score_min: 0.5, model_pool: [m2]}\n"
    )
    p1 = load_profile(str(yaml_path))
    assert [t.name for t in p1.tiers] == ["a", "b"]
    assert p1.tiers[1].score_min == 0.5

    # Force the cache to look stale.
    monkeypatch.setattr(tier_loader, "_PROFILE_TTL_SEC", 0.0)
    # Sleep a hair so the new mtime differs reliably on coarse FS clocks.
    time.sleep(0.05)
    yaml_path.write_text(
        "version: 1\nmode: tiered\n"
        "tiers:\n"
        "  - {name: a, score_min: 0.0, model_pool: [m1]}\n"
        "  - {name: b, score_min: 0.7, model_pool: [m2]}\n"
    )
    # bump mtime explicitly in case write_text didn't.
    os.utime(yaml_path, (time.time() + 1, time.time() + 1))

    p2 = load_profile(str(yaml_path))
    assert p2.tiers[1].score_min == 0.7, "loader did not pick up the YAML change"


# ---------------------------------------------------------------------------
# NTierCascade dispatch
# ---------------------------------------------------------------------------


class _AlwaysAcceptVerifier:
    """Stub verifier that ships the cheap answer every time.

    Threshold matches the n2_default + legacy Cascade default (0.80) so
    the cascade does not swap us out for a fresh HeuristicVerifier when
    the threshold mismatch path triggers.
    """

    threshold: float = 0.80

    def score(self, prompt: str, cheap_answer: str, expect_json: bool = False):
        from nadirclaw.heuristic_verifier import HeuristicScore

        return HeuristicScore(
            score=1.0,
            accepted=True,
            threshold=self.threshold,
            reasons=["stub:accept"],
        )


class _AlwaysRejectVerifier:
    """Stub verifier that always rejects so the cascade escalates.

    Threshold matches the n2_default + legacy Cascade default (0.80) for
    the same reason as the accept stub.
    """

    threshold: float = 0.80

    def score(self, prompt: str, cheap_answer: str, expect_json: bool = False):
        from nadirclaw.heuristic_verifier import HeuristicScore

        return HeuristicScore(
            score=0.0,
            accepted=False,
            threshold=self.threshold,
            reasons=["stub:reject"],
        )


def _call_factory(tier_name: str):
    def _call(messages, **_kw):
        return f"<answer from {tier_name}>"

    return _call


def test_ntier_cascade_ships_cheap_when_accepted():
    p = load_profile("n2_default")
    callers = {t.name: _call_factory(t.name) for t in p.tiers}
    casc = NTierCascade(
        tier_callers=callers,
        tier_profile=p,
        verifier=_AlwaysAcceptVerifier(),
    )
    out = casc.dispatch_sync(
        messages=[{"role": "user", "content": "hi"}],
        prompt_text="hi",
        score=0.1,
    )
    assert out.final_tier == "cheap"
    assert out.escalated is False
    assert out.accepted is True
    assert out.response == "<answer from cheap>"


def test_ntier_cascade_escalates_on_reject_n2():
    p = load_profile("n2_default")
    callers = {t.name: _call_factory(t.name) for t in p.tiers}
    casc = NTierCascade(
        tier_callers=callers,
        tier_profile=p,
        verifier=_AlwaysRejectVerifier(),
    )
    out = casc.dispatch_sync(
        messages=[{"role": "user", "content": "hi"}],
        prompt_text="hi",
        score=0.1,
    )
    assert out.final_tier == "strong"
    assert out.escalated is True
    assert out.response == "<answer from strong>"


def test_ntier_cascade_walks_full_ladder_n3():
    p = load_profile("n3_legacy")
    callers = {t.name: _call_factory(t.name) for t in p.tiers}
    casc = NTierCascade(
        tier_callers=callers,
        tier_profile=p,
        verifier=_AlwaysRejectVerifier(),
    )
    out = casc.dispatch_sync(
        messages=[{"role": "user", "content": "hi"}],
        prompt_text="hi",
        score=0.0,  # start at simple
    )
    assert out.final_tier == "complex"
    assert out.escalated is True
    assert out.meta["visited_tiers"] == ["simple", "medium", "complex"]


def test_ntier_cascade_jump_mode_skips_middle():
    p = load_profile("n3_legacy")
    p.cascade = CascadeConfig(escalation="jump", acceptance_threshold=0.8, rules_profile="default")
    callers = {t.name: _call_factory(t.name) for t in p.tiers}
    casc = NTierCascade(
        tier_callers=callers,
        tier_profile=p,
        verifier=_AlwaysRejectVerifier(),
    )
    out = casc.dispatch_sync(
        messages=[{"role": "user", "content": "hi"}],
        prompt_text="hi",
        score=0.0,
    )
    assert out.final_tier == "complex"
    assert out.meta["visited_tiers"] == ["simple", "complex"]


def test_ntier_cascade_max_escalations_cap():
    p = load_profile("n3_legacy")
    # Cap to 1 hop: simple → medium, then stop even though verifier rejects.
    p.cascade = CascadeConfig(
        escalation="adjacent", acceptance_threshold=0.8,
        rules_profile="default", max_escalations=1,
    )
    callers = {t.name: _call_factory(t.name) for t in p.tiers}
    casc = NTierCascade(
        tier_callers=callers,
        tier_profile=p,
        verifier=_AlwaysRejectVerifier(),
    )
    out = casc.dispatch_sync(
        messages=[{"role": "user", "content": "hi"}],
        prompt_text="hi",
        score=0.0,
    )
    assert out.final_tier == "medium"
    assert out.meta["visited_tiers"] == ["simple", "medium"]
    assert out.meta.get("hop_cap_reached") is True


def test_ntier_cascade_rule_engine_force_escalate():
    p = load_profile("n2_default")
    engine = load_inline([{
        "name": "force_strong",
        "priority": 100,
        "match": {"any_of": [{"substring": "secret"}]},
        "action": {"type": "force_escalate", "to_tier": "strong"},
    }])
    callers = {t.name: _call_factory(t.name) for t in p.tiers}
    casc = NTierCascade(
        tier_callers=callers,
        tier_profile=p,
        verifier=_AlwaysAcceptVerifier(),
        rule_engine=engine,
    )
    out = casc.dispatch_sync(
        messages=[{"role": "user", "content": "tell me the secret"}],
        prompt_text="tell me the secret",
        score=0.0,
    )
    # Started directly at strong (terminal) — verifier never even ran.
    assert out.final_tier == "strong"
    assert "force_strong" in out.matched_rules


def test_ntier_cascade_starting_tier_uses_score():
    """High-score prompts skip cheap and start at strong directly."""
    p = load_profile("n2_default")
    callers = {t.name: _call_factory(t.name) for t in p.tiers}
    casc = NTierCascade(
        tier_callers=callers,
        tier_profile=p,
        verifier=_AlwaysAcceptVerifier(),
    )
    out = casc.dispatch_sync(
        messages=[{"role": "user", "content": "hi"}],
        prompt_text="hi",
        score=0.95,
    )
    assert out.final_tier == "strong"
    assert out.meta["visited_tiers"] == ["strong"]


# ---------------------------------------------------------------------------
# Backward compatibility: legacy 2-tier Cascade still works
# ---------------------------------------------------------------------------


def test_legacy_2tier_cascade_unchanged():
    """The shipped 2-tier `Cascade` class works exactly as it did before
    the N-tier work landed. No env var set, no profile loaded, no
    NTierCascade in sight — same constructor, same dispatch_sync API,
    same CascadeDecision fields.
    """
    cheap_calls = []
    expensive_calls = []

    def cheap(messages, **_kw):
        cheap_calls.append(messages)
        return "cheap answer"

    def expensive(messages, **_kw):
        expensive_calls.append(messages)
        return "expensive answer"

    casc = Cascade(
        cheap_call=cheap,
        expensive_call=expensive,
        verifier=_AlwaysAcceptVerifier(),
        threshold=0.8,
    )
    out = casc.dispatch_sync(
        messages=[{"role": "user", "content": "hi"}],
        prompt_text="hi",
    )
    assert isinstance(out, CascadeDecision)
    assert out.final_tier == "cheap"
    assert out.escalated is False
    assert out.response == "cheap answer"
    assert len(cheap_calls) == 1
    assert len(expensive_calls) == 0


def test_legacy_2tier_cascade_escalates_on_reject():
    """And it still escalates the way it always did."""
    def cheap(messages, **_kw):
        return "cheap"

    def expensive(messages, **_kw):
        return "expensive"

    casc = Cascade(
        cheap_call=cheap,
        expensive_call=expensive,
        verifier=_AlwaysRejectVerifier(),
        threshold=0.8,
    )
    out = casc.dispatch_sync(
        messages=[{"role": "user", "content": "hi"}],
        prompt_text="hi",
    )
    assert out.final_tier == "expensive"
    assert out.escalated is True
    assert out.response == "expensive"
