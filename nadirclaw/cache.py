"""Prompt cache for NadirClaw — in-memory LRU cache for chat completions.

Caches LLM responses keyed by (model + messages hash) to skip redundant calls.
Configurable via environment variables:
  NADIRCLAW_CACHE_ENABLED   — enable/disable (default: true)
  NADIRCLAW_CACHE_TTL       — seconds before entries expire (default: 300)
  NADIRCLAW_CACHE_MAX_SIZE  — max cached entries (default: 1000)
"""

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger("nadirclaw.cache")


def _cache_enabled() -> bool:
    return os.getenv("NADIRCLAW_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")


def _cache_ttl() -> int:
    return int(os.getenv("NADIRCLAW_CACHE_TTL", "300"))


def _cache_max_size() -> int:
    return int(os.getenv("NADIRCLAW_CACHE_MAX_SIZE", "1000"))


def _cache_max_memory_mb() -> int:
    """Max memory in MB for cached responses (default 200MB)."""
    return int(os.getenv("NADIRCLAW_CACHE_MAX_MEMORY_MB", "200"))


def _make_cache_key(model: str, messages: list, temperature: float | None = None) -> str:
    """Build a deterministic cache key from model + messages.

    Includes temperature when it's explicitly set and != 1.0, since different
    temperatures produce different outputs. Caching temperature=1.0 (sampling)
    responses is still allowed but will be reused for any temperature — the
    caller should disable caching for non-deterministic requests.
    """
    # Normalize messages to just role + content
    normalized = []
    for m in messages:
        if hasattr(m, "role"):
            normalized.append({"role": m.role, "content": m.text_content() if hasattr(m, "text_content") else str(m.content)})
        elif isinstance(m, dict):
            normalized.append({"role": m.get("role", ""), "content": m.get("content", "")})
        else:
            normalized.append(str(m))

    key_data: dict = {"model": model or "", "messages": normalized}
    # Include temperature in key when explicitly set to non-default
    if temperature is not None and temperature != 1.0:
        key_data["temperature"] = temperature

    blob = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class PromptCache:
    """Thread-safe in-memory LRU cache with TTL and memory bounds for chat completions."""

    def __init__(self, max_size: int | None = None, ttl: int | None = None):
        self.max_size = max_size if max_size is not None else _cache_max_size()
        self.max_memory_bytes = _cache_max_memory_mb() * 1_000_000
        self.ttl = ttl if ttl is not None else _cache_ttl()
        self._cache: OrderedDict[str, tuple[float, Dict[str, Any], int]] = OrderedDict()  # key → (ts, data, size_bytes)
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._estimated_memory = 0

    @staticmethod
    def _estimate_size(data: Dict[str, Any]) -> int:
        """Rough byte estimate for a cached response."""
        content = data.get("content", "") or ""
        return len(content.encode("utf-8", errors="replace")) + 200  # overhead

    def get(self, model: str, messages: list, temperature: float | None = None) -> Optional[Dict[str, Any]]:
        """Look up a cached response. Returns None on miss or expiry."""
        key = _make_cache_key(model, messages, temperature)
        with self._lock:
            if key in self._cache:
                ts, data, _size = self._cache[key]
                if time.time() - ts < self.ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    logger.debug("Cache HIT: %s", key[:12])
                    return data
                else:
                    # Expired — reclaim memory
                    self._estimated_memory -= _size
                    del self._cache[key]
            self._misses += 1
            return None

    def put(self, model: str, messages: list, response: Dict[str, Any], temperature: float | None = None) -> None:
        """Store a response in the cache."""
        key = _make_cache_key(model, messages, temperature)
        entry_size = self._estimate_size(response)

        with self._lock:
            # Remove old entry if replacing
            if key in self._cache:
                _, _, old_size = self._cache[key]
                self._estimated_memory -= old_size
                self._cache.move_to_end(key)

            self._cache[key] = (time.time(), response, entry_size)
            self._estimated_memory += entry_size

            # Evict LRU until under both entry count AND memory limits
            while (
                len(self._cache) > self.max_size
                or self._estimated_memory > self.max_memory_bytes
            ):
                if not self._cache:
                    break
                _, (_, _, evicted_size) = self._cache.popitem(last=False)
                self._estimated_memory -= evicted_size

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "enabled": _cache_enabled(),
                "entries": len(self._cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
                "total_lookups": total,
                "estimated_memory_mb": round(self._estimated_memory / 1_000_000, 2),
                "max_memory_mb": self.max_memory_bytes // 1_000_000,
            }

    def clear(self) -> None:
        """Clear all cached entries and reset stats."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# ---------------------------------------------------------------------------
# Global prompt cache (lazy singleton)
# ---------------------------------------------------------------------------

_prompt_cache: Optional[PromptCache] = None
_cache_init_lock = Lock()


def get_prompt_cache() -> PromptCache:
    """Get the global prompt cache singleton."""
    global _prompt_cache
    if _prompt_cache is None:
        with _cache_init_lock:
            if _prompt_cache is None:
                _prompt_cache = PromptCache()
    return _prompt_cache
