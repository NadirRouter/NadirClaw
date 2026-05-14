"""Tests for nadirclaw.routing — routing intelligence."""

import json
from types import SimpleNamespace

import pytest

from nadirclaw.routing import (
    MODEL_REGISTRY,
    MODEL_ALIASES,
    SessionCache,
    _merge_external_model_metadata,
    apply_routing_modifiers,
    check_context_window,
    detect_agentic,
    detect_images,
    detect_reasoning,
    estimate_cost,
    estimate_token_count,
    has_vision,
    get_context_window,
    resolve_alias,
    resolve_profile,
)


# Helper to create fake message objects
def _msg(role, content=""):
    ns = SimpleNamespace(role=role, content=content)
    ns.text_content = lambda: content
    return ns


# ---------------------------------------------------------------------------
# resolve_profile
# ---------------------------------------------------------------------------

class TestResolveProfile:
    def test_auto(self):
        assert resolve_profile("auto") == "auto"

    def test_eco(self):
        assert resolve_profile("eco") == "eco"

    def test_premium(self):
        assert resolve_profile("premium") == "premium"

    def test_free(self):
        assert resolve_profile("free") == "free"

    def test_reasoning(self):
        assert resolve_profile("reasoning") == "reasoning"

    def test_nadirclaw_prefix(self):
        assert resolve_profile("nadirclaw/eco") == "eco"
        assert resolve_profile("nadirclaw/premium") == "premium"

    def test_case_insensitive(self):
        assert resolve_profile("ECO") == "eco"
        assert resolve_profile("NadirClaw/Premium") == "premium"

    def test_not_a_profile(self):
        assert resolve_profile("gpt-4o") is None
        assert resolve_profile("claude-sonnet") is None

    def test_none(self):
        assert resolve_profile(None) is None

    def test_empty(self):
        assert resolve_profile("") is None


# ---------------------------------------------------------------------------
# resolve_alias
# ---------------------------------------------------------------------------

class TestResolveAlias:
    def test_sonnet(self):
        assert resolve_alias("sonnet") == "claude-sonnet-4-5-20250929"

    def test_opus(self):
        assert resolve_alias("opus") == "claude-opus-4-6-20250918"

    def test_gpt4(self):
        assert resolve_alias("gpt4") == "gpt-4.1"

    def test_flash(self):
        assert resolve_alias("flash") == "gemini-2.5-flash"

    def test_case_insensitive(self):
        assert resolve_alias("SONNET") == "claude-sonnet-4-5-20250929"
        assert resolve_alias("Flash") == "gemini-2.5-flash"

    def test_unknown(self):
        assert resolve_alias("unknown-model") is None

    def test_deepseek(self):
        assert resolve_alias("deepseek") == "deepseek/deepseek-chat"
        assert resolve_alias("deepseek-v4") == "deepseek/deepseek-v4-flash"
        assert resolve_alias("deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
        assert resolve_alias("deepseek-v4-pro") == "deepseek/deepseek-v4-pro"
        assert resolve_alias("deepseek-r1") == "deepseek/deepseek-reasoner"


# ---------------------------------------------------------------------------
# detect_agentic
# ---------------------------------------------------------------------------

class TestDetectAgentic:
    def test_not_agentic_simple(self):
        messages = [_msg("user", "What is 2+2?")]
        result = detect_agentic(messages)
        assert result["is_agentic"] is False
        assert result["confidence"] == 0.0

    def test_tools_defined(self):
        messages = [_msg("user", "Help me")]
        result = detect_agentic(messages, has_tools=True, tool_count=3)
        assert result["is_agentic"] is True
        assert "tools_defined(3)" in result["signals"]

    def test_many_tools(self):
        messages = [_msg("user", "Help me")]
        result = detect_agentic(messages, has_tools=True, tool_count=5)
        assert result["confidence"] >= 0.5
        assert "many_tools" in result["signals"]

    def test_tool_messages(self):
        messages = [
            _msg("user", "Do it"),
            _msg("assistant", "calling tool"),
            _msg("tool", "result"),
        ]
        result = detect_agentic(messages)
        assert result["is_agentic"] is False  # tool messages alone = 0.3, below 0.35
        assert "tool_messages(1)" in result["signals"]

    def test_tool_messages_with_tools(self):
        messages = [
            _msg("user", "Do it"),
            _msg("assistant", "calling tool"),
            _msg("tool", "result"),
        ]
        result = detect_agentic(messages, has_tools=True, tool_count=2)
        assert result["is_agentic"] is True

    def test_agentic_cycles(self):
        messages = [
            _msg("user", "Do it"),
            _msg("assistant", "step 1"),
            _msg("tool", "result 1"),
            _msg("assistant", "step 2"),
            _msg("tool", "result 2"),
            _msg("assistant", "done"),
        ]
        result = detect_agentic(messages)
        assert result["is_agentic"] is True
        assert any("agentic_cycles" in s for s in result["signals"])

    def test_agentic_system_keywords(self):
        messages = [_msg("user", "Help")]
        result = detect_agentic(
            messages,
            system_prompt="You are a coding agent. You can execute commands and read files.",
        )
        assert "agentic_keywords" in result["signals"]

    def test_long_system_prompt(self):
        messages = [_msg("user", "Help")]
        result = detect_agentic(messages, system_prompt_length=800)
        assert "long_system_prompt" in result["signals"]

    def test_deep_conversation(self):
        messages = [_msg("user", f"msg {i}") for i in range(12)]
        result = detect_agentic(messages, message_count=12)
        assert "deep_conversation" in result["signals"]

    def test_full_agentic_request(self):
        """Realistic agentic request with multiple signals."""
        messages = [
            _msg("system", "You are an AI agent. You can use tools to read and write files."),
            _msg("user", "Refactor the auth module"),
            _msg("assistant", "I'll start by reading the file"),
            _msg("tool", "file contents here"),
            _msg("assistant", "Now I'll write the updated file"),
            _msg("tool", "success"),
            _msg("user", "Now add tests"),
        ]
        result = detect_agentic(
            messages,
            has_tools=True,
            tool_count=4,
            system_prompt="You are an AI agent. You can use tools to read and write files.",
            system_prompt_length=600,
            message_count=7,
        )
        assert result["is_agentic"] is True
        assert result["confidence"] >= 0.8


# ---------------------------------------------------------------------------
# detect_reasoning
# ---------------------------------------------------------------------------

class TestDetectReasoning:
    def test_not_reasoning(self):
        result = detect_reasoning("What is 2+2?")
        assert result["is_reasoning"] is False

    def test_single_marker(self):
        result = detect_reasoning("Think through this problem")
        assert result["is_reasoning"] is False  # need 2+ markers
        assert result["marker_count"] == 1

    def test_two_markers(self):
        result = detect_reasoning("Think through this step by step")
        assert result["is_reasoning"] is True
        assert result["marker_count"] >= 2

    def test_reasoning_in_system(self):
        result = detect_reasoning(
            "What are the implications?",
            system_message="Analyze the tradeoffs and compare and contrast the approaches",
        )
        assert result["is_reasoning"] is True

    def test_proof_request(self):
        result = detect_reasoning("Prove that P=NP and derive the implications step by step")
        assert result["is_reasoning"] is True

    def test_critical_analysis(self):
        result = detect_reasoning("Critically analyze the paper and evaluate whether the conclusions are valid")
        assert result["is_reasoning"] is True


# ---------------------------------------------------------------------------
# check_context_window
# ---------------------------------------------------------------------------

class TestContextWindow:
    def test_fits(self):
        messages = [_msg("user", "short")]
        assert check_context_window("gpt-4o", messages) is True

    def test_unknown_model_passes(self):
        messages = [_msg("user", "x" * 100000)]
        assert check_context_window("unknown-model-xyz", messages) is True

    def test_exceeds(self):
        # gpt-4o has 128k context. 128k * 4 = 512k chars
        content = "x" * 600_000
        messages = [_msg("user", content)]
        assert check_context_window("gpt-4o", messages) is False

    def test_gemini_large_context(self):
        # Gemini has 1M context
        content = "x" * 600_000
        messages = [_msg("user", content)]
        assert check_context_window("gemini-3-flash-preview", messages) is True


class TestEstimateTokenCount:
    def test_basic(self):
        messages = [_msg("user", "hello world")]  # 11 chars → ~2 tokens
        count = estimate_token_count(messages)
        assert count == 2

    def test_multiple_messages(self):
        messages = [_msg("user", "a" * 400), _msg("assistant", "b" * 400)]
        count = estimate_token_count(messages)
        assert count == 200


# ---------------------------------------------------------------------------
# SessionCache
# ---------------------------------------------------------------------------

class TestSessionCache:
    def test_put_and_get(self):
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("system", "You are helpful"), _msg("user", "Hello")]
        cache.put(msgs, "gpt-4o", "complex")
        result = cache.get(msgs)
        assert result == ("gpt-4o", "complex")

    def test_miss(self):
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("user", "Hello")]
        assert cache.get(msgs) is None

    def test_expiry(self):
        cache = SessionCache(ttl_seconds=0)  # immediate expiry
        msgs = [_msg("user", "Hello")]
        cache.put(msgs, "gpt-4o", "complex")
        import time
        time.sleep(0.01)
        assert cache.get(msgs) is None

    def test_same_session_different_followup(self):
        """Same system + first user msg → same cache key regardless of later messages."""
        cache = SessionCache(ttl_seconds=60)
        msgs1 = [_msg("system", "Be helpful"), _msg("user", "Hello")]
        msgs2 = [_msg("system", "Be helpful"), _msg("user", "Hello"), _msg("assistant", "Hi"), _msg("user", "More")]
        cache.put(msgs1, "gpt-4o", "complex")
        result = cache.get(msgs2)
        assert result == ("gpt-4o", "complex")

    def test_clear_expired(self):
        cache = SessionCache(ttl_seconds=0)
        msgs = [_msg("user", "Hello")]
        cache.put(msgs, "gpt-4o", "complex")
        import time
        time.sleep(0.01)
        removed = cache.clear_expired()
        assert removed == 1

    # ----- put() upgrade-only guard ----------------------------------------

    def test_put_does_not_downgrade(self):
        """put() must not replace a higher-tier entry with a lower-tier one."""
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("user", "Hello")]
        cache.put(msgs, "claude-opus", "reasoning")
        cache.put(msgs, "gpt-4o-mini", "simple")
        # Reasoning outranks simple — original entry must remain.
        assert cache.get(msgs) == ("claude-opus", "reasoning")

    def test_put_keeps_equal_tier(self):
        """put() with the same tier is a no-op (no timestamp churn either)."""
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("user", "Hello")]
        cache.put(msgs, "gpt-4o", "complex")
        cache.put(msgs, "claude-sonnet", "complex")  # equal tier, different model
        # Original model retained.
        assert cache.get(msgs) == ("gpt-4o", "complex")

    def test_put_upgrades_when_higher(self):
        """put() with a higher tier replaces the cached entry."""
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("user", "Hello")]
        cache.put(msgs, "gpt-4o-mini", "simple")
        cache.put(msgs, "claude-opus", "reasoning")
        assert cache.get(msgs) == ("claude-opus", "reasoning")

    # ----- upgrade_if_higher() ---------------------------------------------

    def test_upgrade_if_higher_new_session(self):
        """No cached entry → store the new values, status='new'."""
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("user", "Hello")]
        model, tier, status = cache.upgrade_if_higher(msgs, "gpt-4o", "complex")
        assert (model, tier, status) == ("gpt-4o", "complex", "new")
        assert cache.get(msgs) == ("gpt-4o", "complex")

    def test_upgrade_if_higher_escalates(self):
        """Lower cached tier → upgrade to higher tier, status='upgraded'."""
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("user", "Hello")]
        cache.upgrade_if_higher(msgs, "gpt-4o-mini", "simple")
        model, tier, status = cache.upgrade_if_higher(
            msgs, "claude-opus", "reasoning"
        )
        assert (model, tier, status) == ("claude-opus", "reasoning", "upgraded")
        assert cache.get(msgs) == ("claude-opus", "reasoning")

    def test_upgrade_if_higher_keeps_higher(self):
        """Higher cached tier → keep cached values, status='kept'."""
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("user", "Hello")]
        cache.upgrade_if_higher(msgs, "claude-opus", "reasoning")
        model, tier, status = cache.upgrade_if_higher(msgs, "gpt-4o-mini", "simple")
        assert (model, tier, status) == ("claude-opus", "reasoning", "kept")
        assert cache.get(msgs) == ("claude-opus", "reasoning")

    def test_upgrade_if_higher_keeps_equal(self):
        """Equal cached tier → keep cached values, status='kept'."""
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("user", "Hello")]
        cache.upgrade_if_higher(msgs, "gpt-4o", "complex")
        model, tier, status = cache.upgrade_if_higher(msgs, "claude-sonnet", "complex")
        assert (model, tier, status) == ("gpt-4o", "complex", "kept")

    def test_upgrade_if_higher_full_hierarchy(self):
        """simple < mid < complex < reasoning ordering is honored."""
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("user", "Hello")]
        # Walk up the hierarchy — every step should upgrade.
        for tier_name in ("simple", "mid", "complex", "reasoning"):
            _, tier, status = cache.upgrade_if_higher(msgs, f"m-{tier_name}", tier_name)
            assert tier == tier_name
            assert status in ("new", "upgraded")
        # Now walking back down should keep "reasoning" at every step.
        for tier_name in ("complex", "mid", "simple"):
            model, tier, status = cache.upgrade_if_higher(
                msgs, f"m-{tier_name}", tier_name
            )
            assert (model, tier, status) == ("m-reasoning", "reasoning", "kept")

    def test_upgrade_if_higher_expired_entry_treated_as_missing(self):
        """Stale (TTL-expired) high-tier entry must NOT block a fresh classification."""
        import time
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("user", "Hello")]
        # Directly inject an entry whose timestamp is well past the TTL.
        key = cache._make_key(msgs)
        cache._cache[key] = ("claude-opus", "reasoning", time.time() - 3600)
        # Even though "reasoning" outranks "simple", the stale entry should be
        # discarded and the fresh classification should win.
        model, tier, status = cache.upgrade_if_higher(msgs, "gpt-4o-mini", "simple")
        assert (model, tier, status) == ("gpt-4o-mini", "simple", "new")
        assert cache.get(msgs) == ("gpt-4o-mini", "simple")

    def test_upgrade_if_higher_evicts_when_over_capacity(self):
        """upgrade_if_higher must enforce max_size via LRU eviction."""
        cache = SessionCache(ttl_seconds=60, max_size=3)
        # Insert 5 distinct sessions — only the 3 most recent should remain.
        for i in range(5):
            cache.upgrade_if_higher([_msg("user", f"prompt-{i}")], f"m-{i}", "simple")
        assert len(cache._cache) == 3
        # The first two sessions should have been evicted.
        assert cache.get([_msg("user", "prompt-0")]) is None
        assert cache.get([_msg("user", "prompt-1")]) is None
        # The most recent three should still be there.
        assert cache.get([_msg("user", "prompt-4")]) == ("m-4", "simple")

    def test_upgrade_if_higher_touch_updates_lru(self):
        """Touching an entry via upgrade_if_higher should mark it as most-recently-used."""
        cache = SessionCache(ttl_seconds=60, max_size=3)
        msgs_a = [_msg("user", "A")]
        msgs_b = [_msg("user", "B")]
        msgs_c = [_msg("user", "C")]
        cache.upgrade_if_higher(msgs_a, "m-a", "simple")
        cache.upgrade_if_higher(msgs_b, "m-b", "simple")
        cache.upgrade_if_higher(msgs_c, "m-c", "simple")
        # Touch A by re-querying it via upgrade_if_higher (status='kept').
        cache.upgrade_if_higher(msgs_a, "m-a-new", "simple")
        # Now insert a 4th entry — B should be evicted (LRU), not A.
        cache.upgrade_if_higher([_msg("user", "D")], "m-d", "simple")
        assert cache.get(msgs_a) == ("m-a", "simple")
        assert cache.get(msgs_b) is None  # evicted
        assert cache.get(msgs_c) == ("m-c", "simple")


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_known_model(self):
        cost = estimate_cost("gpt-4o", 1000, 500)
        assert cost is not None
        assert cost > 0

    def test_deepseek_v4_cost(self):
        cost = estimate_cost("deepseek/deepseek-v4-pro", 1_000_000, 1_000_000)
        assert cost == pytest.approx(5.22)

    def test_unknown_model(self):
        assert estimate_cost("unknown-xyz", 1000, 500) is None

    def test_free_model(self):
        cost = estimate_cost("ollama/llama3.1:8b", 1000, 500)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# local model metadata
# ---------------------------------------------------------------------------

class TestLocalModelMetadata:
    def test_external_metadata_adds_model(self, tmp_path, monkeypatch):
        path = tmp_path / "models.json"
        model = "custom/custom-fast"
        path.write_text(json.dumps({
            "models": {
                model: {
                    "context_window": 32768,
                    "cost_per_m_input": 0,
                    "cost_per_m_output": 0,
                    "has_vision": False,
                }
            }
        }))
        monkeypatch.setenv("NADIRCLAW_MODEL_METADATA_FILE", str(path))
        monkeypatch.setenv("NADIRCLAW_LOCAL_MODEL_METADATA_FILE", str(tmp_path / "missing.json"))

        try:
            _merge_external_model_metadata()
            assert get_context_window(model) == 32768
            assert estimate_cost(model, 1000, 1000) == 0.0
        finally:
            MODEL_REGISTRY.pop(model, None)

    def test_local_overrides_generated(self, tmp_path, monkeypatch):
        generated = tmp_path / "models.json"
        local = tmp_path / "models.local.json"
        model = "custom/override-me"
        generated.write_text(json.dumps({
            "models": {
                model: {"context_window": 1000, "cost_per_m_input": 1.0,
                        "cost_per_m_output": 2.0, "has_vision": False},
            }
        }))
        local.write_text(json.dumps({
            "models": {
                model: {"context_window": 5000, "cost_per_m_input": 0.5},
            }
        }))
        monkeypatch.setenv("NADIRCLAW_MODEL_METADATA_FILE", str(generated))
        monkeypatch.setenv("NADIRCLAW_LOCAL_MODEL_METADATA_FILE", str(local))

        try:
            _merge_external_model_metadata()
            assert get_context_window(model) == 5000
            info = MODEL_REGISTRY[model]
            assert info["cost_per_m_input"] == 0.5
            assert info["cost_per_m_output"] == 2.0
        finally:
            MODEL_REGISTRY.pop(model, None)

    def test_invalid_metadata_file_is_skipped(self, tmp_path, monkeypatch, caplog):
        path = tmp_path / "models.json"
        path.write_text("{not valid json")
        monkeypatch.setenv("NADIRCLAW_MODEL_METADATA_FILE", str(path))
        monkeypatch.setenv("NADIRCLAW_LOCAL_MODEL_METADATA_FILE", str(tmp_path / "missing.json"))

        with caplog.at_level("WARNING", logger="nadirclaw.routing"):
            _merge_external_model_metadata()
        assert any("Skipping invalid model metadata file" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# apply_routing_modifiers
# ---------------------------------------------------------------------------

class TestApplyRoutingModifiers:
    def test_no_modifiers(self):
        """Simple request stays simple."""
        messages = [_msg("user", "What is 2+2?")]
        meta = {"has_tools": False, "tool_count": 0, "system_prompt_text": "", "system_prompt_length": 0, "message_count": 1}
        model, tier, info = apply_routing_modifiers(
            "gemini-flash", "simple", meta, messages, "gemini-flash", "gpt-4o",
        )
        assert model == "gemini-flash"
        assert tier == "simple"

    def test_agentic_override(self):
        """Agentic request overrides simple → complex."""
        messages = [
            _msg("system", "You are a coding agent. You can use tools."),
            _msg("user", "Refactor this"),
            _msg("assistant", "reading file"),
            _msg("tool", "contents"),
            _msg("assistant", "writing file"),
            _msg("tool", "done"),
        ]
        meta = {
            "has_tools": True, "tool_count": 4,
            "system_prompt_text": "You are a coding agent. You can use tools.",
            "system_prompt_length": 600, "message_count": 6,
        }
        model, tier, info = apply_routing_modifiers(
            "gemini-flash", "simple", meta, messages, "gemini-flash", "gpt-4o",
        )
        assert model == "gpt-4o"
        assert tier == "complex"
        assert "agentic_override" in info["modifiers_applied"]

    def test_agentic_no_override_if_already_complex(self):
        """Agentic request doesn't change anything if already complex."""
        messages = [
            _msg("user", "Do it"),
            _msg("assistant", "step"),
            _msg("tool", "result"),
            _msg("assistant", "step"),
            _msg("tool", "result"),
        ]
        meta = {"has_tools": True, "tool_count": 3, "system_prompt_text": "", "system_prompt_length": 0, "message_count": 5}
        model, tier, info = apply_routing_modifiers(
            "gpt-4o", "complex", meta, messages, "gemini-flash", "gpt-4o",
        )
        assert model == "gpt-4o"
        assert tier == "complex"
        assert "agentic_override" not in info["modifiers_applied"]

    def test_reasoning_override(self):
        """Reasoning markers override to reasoning model."""
        messages = [_msg("user", "Think through this step by step and analyze the tradeoffs")]
        meta = {"has_tools": False, "tool_count": 0, "system_prompt_text": "", "system_prompt_length": 0, "message_count": 1}
        model, tier, info = apply_routing_modifiers(
            "gemini-flash", "simple", meta, messages,
            "gemini-flash", "gpt-4o", reasoning_model="o3",
        )
        assert model == "o3"
        assert tier == "reasoning"
        assert "reasoning_override" in info["modifiers_applied"]

    def test_reasoning_falls_back_to_complex(self):
        """Without a reasoning model configured, falls back to complex."""
        messages = [_msg("user", "Think through this step by step and analyze the tradeoffs")]
        meta = {"has_tools": False, "tool_count": 0, "system_prompt_text": "", "system_prompt_length": 0, "message_count": 1}
        model, tier, info = apply_routing_modifiers(
            "gemini-flash", "simple", meta, messages,
            "gemini-flash", "gpt-4o",
        )
        assert model == "gpt-4o"
        assert tier == "reasoning"

    def test_context_window_swap(self):
        """Swaps model when context window is exceeded."""
        # gpt-4o-mini: 128k context. Make content exceed that.
        big_content = "x" * 600_000  # ~150k tokens
        messages = [_msg("user", big_content)]
        meta = {"has_tools": False, "tool_count": 0, "system_prompt_text": "", "system_prompt_length": 0, "message_count": 1}
        model, tier, info = apply_routing_modifiers(
            "gpt-4o-mini", "simple", meta, messages,
            "gpt-4o-mini", "gemini-2.5-pro",  # gemini has 1M context
        )
        assert model == "gemini-2.5-pro"
        assert any("context_window_swap" in m for m in info["modifiers_applied"])


# ---------------------------------------------------------------------------
# detect_images
# ---------------------------------------------------------------------------

def _multimodal_msg(role, text="", image_urls=None):
    """Helper to create a message with multimodal content array."""
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for url in (image_urls or []):
        content.append({"type": "image_url", "image_url": {"url": url}})
    ns = SimpleNamespace(role=role, content=content)
    ns.text_content = lambda: text
    return ns


class TestDetectImages:
    def test_no_images(self):
        messages = [_msg("user", "What is 2+2?")]
        result = detect_images(messages)
        assert result["has_images"] is False
        assert result["image_count"] == 0

    def test_single_image(self):
        messages = [_multimodal_msg("user", "What's in this?", ["https://example.com/img.png"])]
        result = detect_images(messages)
        assert result["has_images"] is True
        assert result["image_count"] == 1

    def test_multiple_images(self):
        messages = [_multimodal_msg("user", "Compare these", [
            "https://example.com/a.png",
            "https://example.com/b.png",
        ])]
        result = detect_images(messages)
        assert result["has_images"] is True
        assert result["image_count"] == 2

    def test_base64_image(self):
        msg = SimpleNamespace(
            role="user",
            content=[
                {"type": "text", "text": "What's this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
            ],
        )
        msg.text_content = lambda: "What's this?"
        result = detect_images([msg])
        assert result["has_images"] is True
        assert result["image_count"] == 1

    def test_text_only_multimodal(self):
        msg = SimpleNamespace(
            role="user",
            content=[{"type": "text", "text": "Hello"}],
        )
        msg.text_content = lambda: "Hello"
        result = detect_images([msg])
        assert result["has_images"] is False


# ---------------------------------------------------------------------------
# has_vision
# ---------------------------------------------------------------------------

class TestHasVision:
    def test_vision_models(self):
        assert has_vision("gpt-4o") is True
        assert has_vision("claude-sonnet-4-5-20250929") is True
        assert has_vision("gemini-2.5-pro") is True

    def test_non_vision_models(self):
        assert has_vision("deepseek/deepseek-chat") is False
        assert has_vision("ollama/llama3.1:8b") is False
        assert has_vision("openai-codex/gpt-5.3-codex") is False

    def test_unknown_model(self):
        assert has_vision("unknown-model-xyz") is False


# ---------------------------------------------------------------------------
# Vision routing modifier
# ---------------------------------------------------------------------------

class TestVisionModifier:
    def test_vision_swap_from_non_vision_model(self):
        """Non-vision model gets swapped when images are present."""
        messages = [_msg("user", "Describe this image")]
        meta = {
            "has_tools": False, "tool_count": 0,
            "system_prompt_text": "", "system_prompt_length": 0,
            "message_count": 1, "has_images": True,
        }
        model, tier, info = apply_routing_modifiers(
            "deepseek/deepseek-chat", "simple", meta, messages,
            "deepseek/deepseek-chat", "gpt-4o",
        )
        assert model == "gpt-4o"
        assert any("vision_swap" in m for m in info["modifiers_applied"])

    def test_no_swap_when_model_has_vision(self):
        """Vision-capable model stays as-is."""
        messages = [_msg("user", "Describe this image")]
        meta = {
            "has_tools": False, "tool_count": 0,
            "system_prompt_text": "", "system_prompt_length": 0,
            "message_count": 1, "has_images": True,
        }
        model, tier, info = apply_routing_modifiers(
            "gpt-4o", "complex", meta, messages,
            "gemini-2.5-flash", "gpt-4o",
        )
        assert model == "gpt-4o"
        assert not any("vision_swap" in m for m in info["modifiers_applied"])

    def test_no_swap_when_no_images(self):
        """No images means no vision routing."""
        messages = [_msg("user", "Hello")]
        meta = {
            "has_tools": False, "tool_count": 0,
            "system_prompt_text": "", "system_prompt_length": 0,
            "message_count": 1, "has_images": False,
        }
        model, tier, info = apply_routing_modifiers(
            "deepseek/deepseek-chat", "simple", meta, messages,
            "deepseek/deepseek-chat", "gpt-4o",
        )
        assert model == "deepseek/deepseek-chat"
        assert not any("vision_swap" in m for m in info["modifiers_applied"])


# ---------------------------------------------------------------------------
# Three-tier classifier (mid tier)
# ---------------------------------------------------------------------------

class TestThreeTierClassifier:
    def test_score_to_tier_binary_low(self):
        """Low score → simple tier (binary mode, no mid model)."""
        from nadirclaw.classifier import BinaryComplexityClassifier
        tier_name, tier_num = BinaryComplexityClassifier._score_to_tier(0.2)
        assert tier_name == "simple"
        assert tier_num == 1

    def test_score_to_tier_binary_high(self):
        """High score → complex tier (binary mode, no mid model)."""
        from nadirclaw.classifier import BinaryComplexityClassifier
        tier_name, tier_num = BinaryComplexityClassifier._score_to_tier(0.8)
        assert tier_name == "complex"
        assert tier_num == 3

    def test_score_to_tier_mid_with_env(self, monkeypatch):
        """Mid score → mid tier when MID_MODEL is configured."""
        monkeypatch.setenv("NADIRCLAW_MID_MODEL", "gpt-4.1-mini")
        from nadirclaw.classifier import BinaryComplexityClassifier
        tier_name, tier_num = BinaryComplexityClassifier._score_to_tier(0.5)
        assert tier_name == "mid"
        assert tier_num == 2

    def test_score_to_tier_custom_thresholds(self, monkeypatch):
        """Custom thresholds shift tier boundaries."""
        monkeypatch.setenv("NADIRCLAW_MID_MODEL", "gpt-4.1-mini")
        monkeypatch.setenv("NADIRCLAW_TIER_THRESHOLDS", "0.25,0.75")
        from nadirclaw.classifier import BinaryComplexityClassifier
        # 0.30 is above 0.25 (simple_max) and below 0.75 (complex_min) → mid
        tier_name, _ = BinaryComplexityClassifier._score_to_tier(0.30)
        assert tier_name == "mid"
        # 0.20 is below 0.25 → simple
        tier_name, _ = BinaryComplexityClassifier._score_to_tier(0.20)
        assert tier_name == "simple"
        # 0.80 is above 0.75 → complex
        tier_name, _ = BinaryComplexityClassifier._score_to_tier(0.80)
        assert tier_name == "complex"

    def test_select_model_by_tier_mid(self, monkeypatch):
        """Mid tier selects MID_MODEL."""
        monkeypatch.setenv("NADIRCLAW_MID_MODEL", "gpt-4.1-mini")
        from nadirclaw.classifier import BinaryComplexityClassifier
        model, provider = BinaryComplexityClassifier._select_model_by_tier("mid")
        assert model == "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------

class TestCostBreakdown:
    def test_by_model(self):
        from nadirclaw.report import generate_cost_breakdown
        entries = [
            {"selected_model": "gpt-4o", "timestamp": "2026-03-08T10:00:00", "cost": 0.01, "prompt_tokens": 100, "completion_tokens": 50},
            {"selected_model": "gpt-4o", "timestamp": "2026-03-08T11:00:00", "cost": 0.02, "prompt_tokens": 200, "completion_tokens": 100},
            {"selected_model": "gemini-flash", "timestamp": "2026-03-08T12:00:00", "cost": 0.001, "prompt_tokens": 100, "completion_tokens": 50},
        ]
        result = generate_cost_breakdown(entries, by_model=True)
        assert len(result["breakdown"]) == 2
        models = {r["model"] for r in result["breakdown"]}
        assert "gpt-4o" in models
        assert "gemini-flash" in models

    def test_by_day(self):
        from nadirclaw.report import generate_cost_breakdown
        entries = [
            {"selected_model": "gpt-4o", "timestamp": "2026-03-07T10:00:00", "cost": 0.01, "prompt_tokens": 100, "completion_tokens": 50},
            {"selected_model": "gpt-4o", "timestamp": "2026-03-08T11:00:00", "cost": 0.02, "prompt_tokens": 200, "completion_tokens": 100},
        ]
        result = generate_cost_breakdown(entries, by_day=True)
        assert len(result["breakdown"]) == 2
        days = {r["day"] for r in result["breakdown"]}
        assert "2026-03-07" in days
        assert "2026-03-08" in days

    def test_by_model_and_day(self):
        from nadirclaw.report import generate_cost_breakdown
        entries = [
            {"selected_model": "gpt-4o", "timestamp": "2026-03-07T10:00:00", "cost": 0.01, "prompt_tokens": 100, "completion_tokens": 50},
            {"selected_model": "gemini-flash", "timestamp": "2026-03-07T11:00:00", "cost": 0.001, "prompt_tokens": 50, "completion_tokens": 25},
            {"selected_model": "gpt-4o", "timestamp": "2026-03-08T10:00:00", "cost": 0.02, "prompt_tokens": 200, "completion_tokens": 100},
        ]
        result = generate_cost_breakdown(entries, by_model=True, by_day=True)
        assert len(result["breakdown"]) == 3

    def test_anomaly_detection(self):
        from nadirclaw.report import generate_cost_breakdown
        # Create entries where the latest day spikes
        entries = []
        for day in range(1, 8):
            entries.append({
                "selected_model": "gpt-4o",
                "timestamp": f"2026-03-{day:02d}T10:00:00",
                "cost": 0.01,
                "prompt_tokens": 100,
                "completion_tokens": 50,
            })
        # Big spike on day 8
        entries.append({
            "selected_model": "gpt-4o",
            "timestamp": "2026-03-08T10:00:00",
            "cost": 0.10,  # 10× normal
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        })
        result = generate_cost_breakdown(entries, by_model=True, by_day=True)
        assert len(result["anomalies"]) >= 1
        assert result["anomalies"][0]["model"] == "gpt-4o"

    def test_empty_entries(self):
        from nadirclaw.report import generate_cost_breakdown
        result = generate_cost_breakdown([])
        assert result["total_cost"] == 0
        assert result["breakdown"] == []


# ---------------------------------------------------------------------------
# Settings: mid tier and tier thresholds
# ---------------------------------------------------------------------------

class TestSettingsMidTier:
    def test_default_no_mid(self, monkeypatch):
        # Import first so settings.py's import-time load_dotenv() runs and the
        # module is cached — otherwise the import below would re-run
        # load_dotenv() and re-populate the var we're about to delete.
        from nadirclaw.settings import Settings
        # Hermetic: ignore any NADIRCLAW_MID_MODEL inherited from the
        # environment or ~/.nadirclaw/.env on the developer's machine.
        monkeypatch.delenv("NADIRCLAW_MID_MODEL", raising=False)
        s = Settings()
        assert s.has_mid_tier is False

    def test_mid_model_set(self, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_MID_MODEL", "gpt-4.1-mini")
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.has_mid_tier is True
        assert s.MID_MODEL == "gpt-4.1-mini"

    def test_default_thresholds(self, monkeypatch):
        # Import first (see test_default_no_mid) so load_dotenv() doesn't
        # re-populate the var after we delete it.
        from nadirclaw.settings import Settings
        # Hermetic: ignore any NADIRCLAW_TIER_THRESHOLDS from the dev env.
        monkeypatch.delenv("NADIRCLAW_TIER_THRESHOLDS", raising=False)
        s = Settings()
        assert s.TIER_THRESHOLDS == (0.35, 0.65)

    def test_custom_thresholds(self, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_TIER_THRESHOLDS", "0.25,0.75")
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.TIER_THRESHOLDS == (0.25, 0.75)

    def test_tier_models_with_mid(self, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_MID_MODEL", "gpt-4.1-mini")
        monkeypatch.setenv("NADIRCLAW_SIMPLE_MODEL", "gemini-2.5-flash")
        monkeypatch.setenv("NADIRCLAW_COMPLEX_MODEL", "gpt-4o")
        from nadirclaw.settings import Settings
        s = Settings()
        assert "gpt-4.1-mini" in s.tier_models
