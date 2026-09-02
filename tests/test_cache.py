"""Tests for nadirclaw.cache — prompt caching for chat completions."""

import time

from nadirclaw.cache import PromptCache, _make_cache_key, request_cache_params


class TestMakeCacheKey:
    def test_same_messages_same_key(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = _make_cache_key("gpt-4", msgs)
        k2 = _make_cache_key("gpt-4", msgs)
        assert k1 == k2

    def test_different_model_different_key(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = _make_cache_key("gpt-4", msgs)
        k2 = _make_cache_key("gpt-3.5", msgs)
        assert k1 != k2

    def test_different_messages_different_key(self):
        k1 = _make_cache_key("gpt-4", [{"role": "user", "content": "hello"}])
        k2 = _make_cache_key("gpt-4", [{"role": "user", "content": "world"}])
        assert k1 != k2

    def test_key_is_hex_string(self):
        key = _make_cache_key("model", [{"role": "user", "content": "test"}])
        assert isinstance(key, str)
        assert len(key) == 64  # sha256 hex

    def test_response_shaping_params_change_key(self):
        """Params that change the upstream response must change the key (issue #90)."""
        msgs = [{"role": "user", "content": "Return a city name"}]
        base = _make_cache_key("gpt-4", msgs)
        for params in (
            {"response_format": {"type": "json_object"}},
            {"tools": [{"type": "function", "function": {"name": "get_weather"}}]},
            {"tool_choice": "required"},
            {"max_tokens": 16},
            {"temperature": 0.9},
            {"top_p": 0.1},
            {"n": 3},
            {"reasoning_effort": "high"},
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        ):
            assert _make_cache_key("gpt-4", msgs, params) != base, params

    def test_same_params_same_key(self):
        msgs = [{"role": "user", "content": "hello"}]
        params = {"max_tokens": 100, "tools": [{"type": "function"}]}
        assert _make_cache_key("gpt-4", msgs, params) == _make_cache_key("gpt-4", msgs, dict(params))

    def test_unserializable_param_does_not_raise(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert len(_make_cache_key("gpt-4", msgs, {"weird": object()})) == 64

    def test_tool_call_id_changes_key(self):
        """Tool-result messages with identical text must not collide."""
        k1 = _make_cache_key("gpt-4", [{"role": "tool", "content": "ok", "tool_call_id": "a"}])
        k2 = _make_cache_key("gpt-4", [{"role": "tool", "content": "ok", "tool_call_id": "b"}])
        assert k1 != k2

    def test_image_changes_key(self):
        """Same question about a different image must not collide."""
        from nadirclaw.server import ChatMessage

        def msg(url):
            return ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            )

        assert _make_cache_key("gpt-4", [msg("cat.png")]) != _make_cache_key("gpt-4", [msg("dog.png")])

    def test_text_only_message_key_is_unchanged_by_object_form(self):
        """A plain-text ChatMessage still keys the same as its dict form."""
        from nadirclaw.server import ChatMessage

        obj = _make_cache_key("gpt-4", [ChatMessage(role="user", content="hello")])
        plain = _make_cache_key("gpt-4", [{"role": "user", "content": "hello"}])
        assert obj == plain


class TestRequestCacheParams:
    def test_extracts_declared_and_extra_fields(self):
        class FakeRequest:
            temperature = 0.5
            top_p = None
            max_tokens = 128
            n = None
            model_extra = {
                "tools": [{"type": "function"}],
                "response_format": {"type": "json_object"},
                "stream": False,
            }

        params = request_cache_params(FakeRequest())
        assert params == {
            "temperature": 0.5,
            "max_tokens": 128,
            "tools": [{"type": "function"}],
            "response_format": {"type": "json_object"},
        }

    def test_missing_attributes_are_tolerated(self):
        class Bare:
            pass

        assert request_cache_params(Bare()) == {}


class TestPromptCache:
    def test_put_and_get(self):
        cache = PromptCache(max_size=10, ttl=60)
        msgs = [{"role": "user", "content": "hello"}]
        response = {"content": "hi", "finish_reason": "stop", "prompt_tokens": 5, "completion_tokens": 2}

        cache.put("gpt-4", msgs, response)
        result = cache.get("gpt-4", msgs)
        assert result == response

    def test_different_params_do_not_share_entry(self):
        """A plain chat response must not be served to a structured-output request."""
        cache = PromptCache(max_size=10, ttl=60)
        msgs = [{"role": "user", "content": "Return a city name"}]
        response = {"content": "Paris", "finish_reason": "stop", "prompt_tokens": 5, "completion_tokens": 1}

        cache.put("gpt-4", msgs, response, {})
        assert cache.get("gpt-4", msgs, {"response_format": {"type": "json_object"}}) is None
        assert cache.get("gpt-4", msgs, {}) == response

    def test_miss_returns_none(self):
        cache = PromptCache(max_size=10, ttl=60)
        result = cache.get("gpt-4", [{"role": "user", "content": "hello"}])
        assert result is None

    def test_ttl_expiry(self):
        cache = PromptCache(max_size=10, ttl=1)
        msgs = [{"role": "user", "content": "hello"}]
        cache.put("gpt-4", msgs, {"content": "hi", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})

        # Should hit
        assert cache.get("gpt-4", msgs) is not None

        # Wait for expiry
        time.sleep(1.1)
        assert cache.get("gpt-4", msgs) is None

    def test_lru_eviction(self):
        cache = PromptCache(max_size=2, ttl=60)

        cache.put("m", [{"role": "user", "content": "a"}], {"content": "1", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})
        cache.put("m", [{"role": "user", "content": "b"}], {"content": "2", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})
        cache.put("m", [{"role": "user", "content": "c"}], {"content": "3", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})

        # "a" should be evicted
        assert cache.get("m", [{"role": "user", "content": "a"}]) is None
        assert cache.get("m", [{"role": "user", "content": "b"}]) is not None
        assert cache.get("m", [{"role": "user", "content": "c"}]) is not None

    def test_stats(self):
        cache = PromptCache(max_size=10, ttl=60)
        msgs = [{"role": "user", "content": "hello"}]
        cache.put("gpt-4", msgs, {"content": "hi", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})

        cache.get("gpt-4", msgs)  # hit
        cache.get("gpt-4", [{"role": "user", "content": "miss"}])  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["entries"] == 1

    def test_clear(self):
        cache = PromptCache(max_size=10, ttl=60)
        cache.put("m", [{"role": "user", "content": "a"}], {"content": "x", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})
        cache.clear()

        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_different_model_no_hit(self):
        cache = PromptCache(max_size=10, ttl=60)
        msgs = [{"role": "user", "content": "hello"}]
        cache.put("gpt-4", msgs, {"content": "hi", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})

        assert cache.get("gpt-3.5", msgs) is None
