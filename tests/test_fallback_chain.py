"""Tests for fallback chain configuration and behavior."""

import os
import pytest
from unittest.mock import AsyncMock, patch


class TestFallbackChainConfig:
    def test_default_chain_includes_tier_models(self):
        """Default chain should include complex and simple models."""
        from nadirclaw.settings import settings
        chain = settings.FALLBACK_CHAIN
        assert settings.COMPLEX_MODEL in chain
        assert settings.SIMPLE_MODEL in chain
        # Complex should come first
        assert chain.index(settings.COMPLEX_MODEL) < chain.index(settings.SIMPLE_MODEL)

    def test_custom_chain_from_env(self, monkeypatch):
        """NADIRCLAW_FALLBACK_CHAIN env var should override defaults."""
        monkeypatch.setenv("NADIRCLAW_FALLBACK_CHAIN", "model-a,model-b,model-c")
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.FALLBACK_CHAIN == ["model-a", "model-b", "model-c"]

    def test_empty_chain_env_uses_defaults(self, monkeypatch):
        """Empty NADIRCLAW_FALLBACK_CHAIN should fall back to defaults."""
        monkeypatch.setenv("NADIRCLAW_FALLBACK_CHAIN", "")
        from nadirclaw.settings import Settings
        s = Settings()
        assert len(s.FALLBACK_CHAIN) >= 1

    def test_chain_deduplicates(self, monkeypatch):
        """Default chain should not have duplicate models."""
        # When simple == complex, chain should still work
        monkeypatch.setenv("NADIRCLAW_SIMPLE_MODEL", "same-model")
        monkeypatch.setenv("NADIRCLAW_COMPLEX_MODEL", "same-model")
        monkeypatch.delenv("NADIRCLAW_FALLBACK_CHAIN", raising=False)
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.FALLBACK_CHAIN.count("same-model") == 1


class TestPerTierFallbackConfig:
    def test_per_tier_simple_fallback(self, monkeypatch):
        """NADIRCLAW_SIMPLE_FALLBACK should override global chain for simple tier."""
        monkeypatch.setenv("NADIRCLAW_SIMPLE_FALLBACK", "flash-a,flash-b")
        monkeypatch.setenv("NADIRCLAW_FALLBACK_CHAIN", "global-a,global-b")
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.get_tier_fallback_chain("simple") == ["flash-a", "flash-b"]
        # Other tiers should still use global chain
        assert s.get_tier_fallback_chain("complex") == ["global-a", "global-b"]

    def test_per_tier_complex_fallback(self, monkeypatch):
        """NADIRCLAW_COMPLEX_FALLBACK should override global chain for complex tier."""
        monkeypatch.setenv("NADIRCLAW_COMPLEX_FALLBACK", "big-a,big-b")
        monkeypatch.delenv("NADIRCLAW_FALLBACK_CHAIN", raising=False)
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.get_tier_fallback_chain("complex") == ["big-a", "big-b"]

    def test_per_tier_mid_fallback(self, monkeypatch):
        """NADIRCLAW_MID_FALLBACK should override global chain for mid tier."""
        monkeypatch.setenv("NADIRCLAW_MID_FALLBACK", "mid-a,mid-b")
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.get_tier_fallback_chain("mid") == ["mid-a", "mid-b"]

    def test_no_per_tier_falls_back_to_global(self, monkeypatch):
        """Without per-tier env var, should use global chain."""
        monkeypatch.setenv("NADIRCLAW_FALLBACK_CHAIN", "global-x,global-y")
        monkeypatch.delenv("NADIRCLAW_SIMPLE_FALLBACK", raising=False)
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.get_tier_fallback_chain("simple") == ["global-x", "global-y"]

    def test_empty_tier_string_uses_global(self, monkeypatch):
        """Empty tier name should return global chain."""
        monkeypatch.setenv("NADIRCLAW_FALLBACK_CHAIN", "g1,g2")
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.get_tier_fallback_chain("") == ["g1", "g2"]


class TestFallbackChainBehavior:
    """Integration tests for fallback chain runtime behavior."""

    @pytest.mark.asyncio
    async def test_fallback_on_rate_limit(self, monkeypatch):
        """When primary model is rate-limited, should fallback to next in chain."""
        from nadirclaw.server import RateLimitExhausted, _call_with_fallback

        monkeypatch.setenv("NADIRCLAW_FALLBACK_CHAIN", "model-primary,model-backup")

        # Mock request
        class MockRequest:
            messages = []
            stream = False
            temperature = None
            max_tokens = None
            top_p = None
            model_extra = {}

        request = MockRequest()
        analysis_info = {"tier": "complex", "strategy": "smart-routing"}

        # Mock _dispatch_model to fail primary, succeed on backup
        call_count = {"count": 0}

        async def mock_dispatch(model, req, provider):
            call_count["count"] += 1
            if model == "model-primary":
                raise RateLimitExhausted(model)
            return {
                "content": "Success from backup",
                "finish_reason": "stop",
                "prompt_tokens": 10,
                "completion_tokens": 20,
            }

        with patch("nadirclaw.server._dispatch_model", side_effect=mock_dispatch):
            with patch("nadirclaw.server.settings") as mock_settings:
                mock_settings.FALLBACK_CHAIN = ["model-primary", "model-backup"]
                mock_settings.get_tier_fallback_chain.return_value = ["model-primary", "model-backup"]
                response, actual_model, updated_info = await _call_with_fallback(
                    "model-primary", request, None, analysis_info
                )

        # Verify fallback was used
        assert actual_model == "model-backup"
        assert response["content"] == "Success from backup"
        assert updated_info["fallback_from"] == "model-primary"
        assert "+fallback" in updated_info["strategy"]
        assert call_count["count"] == 2  # primary + backup
        assert updated_info["fallback_reasons"] == [{
            "model": "model-primary",
            "error_type": "RateLimitExhausted",
            "message": "Rate limit exhausted for model-primary (retry in 60s)",
        }]

    @pytest.mark.asyncio
    async def test_fallback_cascade_through_chain(self, monkeypatch):
        """Should try each model in chain until one succeeds."""
        from nadirclaw.server import RateLimitExhausted, _call_with_fallback

        monkeypatch.setenv("NADIRCLAW_FALLBACK_CHAIN", "m1,m2,m3,m4")

        class MockRequest:
            messages = []
            stream = False
            temperature = None
            max_tokens = None
            top_p = None
            model_extra = {}

        request = MockRequest()
        analysis_info = {"tier": "complex", "strategy": "smart-routing"}

        attempts = []

        async def mock_dispatch(model, req, provider):
            attempts.append(model)
            if model in ["m1", "m2", "m3"]:
                raise RateLimitExhausted(model)
            return {
                "content": f"Success from {model}",
                "finish_reason": "stop",
                "prompt_tokens": 10,
                "completion_tokens": 20,
            }

        with patch("nadirclaw.server._dispatch_model", side_effect=mock_dispatch):
            with patch("nadirclaw.server.settings") as mock_settings:
                mock_settings.FALLBACK_CHAIN = ["m1", "m2", "m3", "m4"]
                mock_settings.get_tier_fallback_chain.return_value = ["m1", "m2", "m3", "m4"]
                response, actual_model, updated_info = await _call_with_fallback(
                    "m1", request, None, analysis_info
                )

        # Verify all models were tried in order until m4 succeeded
        assert attempts == ["m1", "m2", "m3", "m4"]
        assert actual_model == "m4"
        assert response["content"] == "Success from m4"
        assert updated_info["fallback_chain_tried"] == ["m1", "m2", "m3"]

    @pytest.mark.asyncio
    async def test_all_models_exhausted(self, monkeypatch):
        """When all models in chain fail, should return graceful error."""
        from nadirclaw.server import RateLimitExhausted, _call_with_fallback

        monkeypatch.setenv("NADIRCLAW_FALLBACK_CHAIN", "m1,m2")

        class MockRequest:
            messages = []
            stream = False
            temperature = None
            max_tokens = None
            top_p = None
            model_extra = {}

        request = MockRequest()
        analysis_info = {"tier": "complex", "strategy": "smart-routing"}

        async def mock_dispatch(model, req, provider):
            raise RateLimitExhausted(model)

        with patch("nadirclaw.server._dispatch_model", side_effect=mock_dispatch):
            with patch("nadirclaw.server.settings") as mock_settings:
                mock_settings.FALLBACK_CHAIN = ["m1", "m2"]
                mock_settings.get_tier_fallback_chain.return_value = ["m1", "m2"]
                response, actual_model, updated_info = await _call_with_fallback(
                    "m1", request, None, analysis_info
                )

        # Verify graceful error response
        assert "rate-limited" in response["content"].lower()
        assert response["finish_reason"] == "stop"
        assert response["prompt_tokens"] == 0
        assert response["completion_tokens"] == 0

    @pytest.mark.asyncio
    async def test_no_fallback_if_chain_empty(self, monkeypatch):
        """When fallback chain is empty, should raise the original error."""
        from nadirclaw.server import RateLimitExhausted, _call_with_fallback

        monkeypatch.setenv("NADIRCLAW_FALLBACK_CHAIN", "model-only")

        class MockRequest:
            messages = []
            stream = False
            temperature = None
            max_tokens = None
            top_p = None
            model_extra = {}

        request = MockRequest()
        analysis_info = {"tier": "complex", "strategy": "smart-routing"}

        async def mock_dispatch(model, req, provider):
            raise RateLimitExhausted(model)

        with patch("nadirclaw.server._dispatch_model", side_effect=mock_dispatch):
            with patch("nadirclaw.server.settings") as mock_settings:
                mock_settings.FALLBACK_CHAIN = ["model-only"]
                mock_settings.get_tier_fallback_chain.return_value = ["model-only"]
                response, actual_model, updated_info = await _call_with_fallback(
                    "model-only", request, None, analysis_info
                )

        # Should return graceful error (since chain is exhausted after one model)
        assert "rate-limited" in response["content"].lower()
