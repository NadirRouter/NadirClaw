"""Tests for Model Pool weighted load balancing."""

import os
from unittest import mock

import pytest

from nadirclaw.routing import (
    get_pool_for_model,
    select_from_pool,
    _parse_model_pools,
)


class TestParseModelPools:
    """Tests for _parse_model_pools env var parsing."""

    def test_empty_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            pools, _reverse = _parse_model_pools()
            assert pools == {}

    def test_single_pool_single_model(self):
        raw = "turbo=gemini-2.5-flash,10"
        with mock.patch.dict(os.environ, {"NADIRCLAW_MODEL_POOLS": raw}):
            pools, _reverse = _parse_model_pools()
            assert "turbo" in pools
            assert pools["turbo"] == [("gemini-2.5-flash", 10)]

    def test_single_pool_multiple_models(self):
        raw = "turbo=gemini-2.5-flash,10+gpt-4.1-nano,5"
        with mock.patch.dict(os.environ, {"NADIRCLAW_MODEL_POOLS": raw}):
            pools, _reverse = _parse_model_pools()
            assert pools["turbo"] == [
                ("gemini-2.5-flash", 10),
                ("gpt-4.1-nano", 5),
            ]

    def test_multiple_pools(self):
        raw = "turbo=gemini-2.5-flash,10;reasoning=gpt-5.2,8+claude-opus-4-6-20250918,4"
        with mock.patch.dict(os.environ, {"NADIRCLAW_MODEL_POOLS": raw}):
            pools, reverse = _parse_model_pools()
            assert len(pools) == 2
            assert pools["turbo"] == [("gemini-2.5-flash", 10)]
            assert pools["reasoning"] == [
                ("gpt-5.2", 8),
                ("claude-opus-4-6-20250918", 4),
            ]
            assert reverse["gemini-2.5-flash"] == "turbo"
            assert reverse["gpt-5.2"] == "reasoning"

    def test_default_weight_is_one(self):
        raw = "turbo=gemini-2.5-flash"
        with mock.patch.dict(os.environ, {"NADIRCLAW_MODEL_POOLS": raw}):
            pools, _reverse = _parse_model_pools()
            assert pools["turbo"] == [("gemini-2.5-flash", 1)]

    def test_invalid_weight_uses_one(self):
        raw = "turbo=gemini-2.5-flash,abc"
        with mock.patch.dict(os.environ, {"NADIRCLAW_MODEL_POOLS": raw}):
            pools, _reverse = _parse_model_pools()
            assert pools["turbo"] == [("gemini-2.5-flash", 1)]


class TestSelectFromPool:
    """Tests for weighted random selection."""

    def _setup_pools(self):
        """Set up test pools by patching the cache variables."""
        import nadirclaw.routing as routing_mod
        test_pools = {
            "balanced": [
                ("model-a", 10),
                ("model-b", 10),
            ],
            "single": [
                ("only-model", 5),
            ],
        }
        reverse_map = {}
        for name, models in test_pools.items():
            for m, _ in models:
                reverse_map[m] = name

        routing_mod._MODEL_POOLS_CACHE = test_pools
        routing_mod._MODEL_TO_POOL_CACHE = reverse_map

    def test_single_model_pool_always_returns_same(self):
        self._setup_pools()
        for _ in range(50):
            assert select_from_pool("single") == "only-model"

    def test_balanced_pool_returns_valid_model(self):
        self._setup_pools()
        valid = {"model-a", "model-b"}
        for _ in range(50):
            assert select_from_pool("balanced") in valid

    def test_unknown_pool_raises_keyerror(self):
        self._setup_pools()
        with pytest.raises(KeyError):
            select_from_pool("nonexistent")

    def test_weighted_distribution(self):
        self._setup_pools()
        import nadirclaw.routing as routing_mod
        routing_mod._MODEL_POOLS_CACHE = {
            "heavy": [
                ("heavy-model", 99),
                ("light-model", 1),
            ],
        }
        counts = {"heavy-model": 0, "light-model": 0}
        for _ in range(1000):
            counts[select_from_pool("heavy")] += 1
        assert counts["heavy-model"] > counts["light-model"]


class TestGetPoolForModel:
    """Tests for reverse lookup: model → pool name."""

    def _setup_pools(self):
        import nadirclaw.routing as routing_mod
        routing_mod._MODEL_POOLS_CACHE = {
            "turbo": [("gemini-2.5-flash", 10), ("gpt-4.1-nano", 5)],
        }
        routing_mod._MODEL_TO_POOL_CACHE = {
            "gemini-2.5-flash": "turbo",
            "gpt-4.1-nano": "turbo",
        }

    def test_model_in_pool(self):
        self._setup_pools()
        assert get_pool_for_model("gemini-2.5-flash") == "turbo"

    def test_model_not_in_pool(self):
        self._setup_pools()
        assert get_pool_for_model("claude-opus-4-6-20250918") is None
