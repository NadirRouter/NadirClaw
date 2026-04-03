"""Persistent semantic cache for NadirClaw.

Uses sentence-transformer embeddings + cosine similarity to find cached
responses for semantically similar (not just identical) prompts. Backed
by SQLite for persistence across restarts.

Configurable via environment variables:
  NADIRCLAW_SEMANTIC_CACHE_ENABLED    — enable/disable (default: false)
  NADIRCLAW_SEMANTIC_CACHE_THRESHOLD  — cosine similarity threshold (default: 0.95)
  NADIRCLAW_SEMANTIC_CACHE_TTL        — seconds before entries expire (default: 86400 = 24h)
  NADIRCLAW_SEMANTIC_CACHE_MAX_SIZE   — max cached entries (default: 5000)
"""

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nadirclaw.semantic_cache")


def _sem_cache_enabled() -> bool:
    return os.getenv("NADIRCLAW_SEMANTIC_CACHE_ENABLED", "false").lower() in ("true", "1", "yes")


def _sem_cache_threshold() -> float:
    return float(os.getenv("NADIRCLAW_SEMANTIC_CACHE_THRESHOLD", "0.95"))


def _sem_cache_ttl() -> int:
    return int(os.getenv("NADIRCLAW_SEMANTIC_CACHE_TTL", "86400"))


def _sem_cache_max_size() -> int:
    return int(os.getenv("NADIRCLAW_SEMANTIC_CACHE_MAX_SIZE", "5000"))


class SemanticCache:
    """Embedding-based semantic cache with SQLite persistence.

    On lookup:
      1. Encode the prompt with SentenceTransformer
      2. Compute cosine similarity against all cached embeddings (numpy matmul)
      3. If max similarity >= threshold, return the cached response
      4. Otherwise, return None (cache miss)

    On store:
      1. Encode the prompt
      2. Store embedding + response in SQLite + in-memory index
      3. Evict oldest entries if over max_size

    The in-memory numpy matrix is rebuilt from SQLite on startup for fast
    similarity search without external dependencies (no FAISS required).
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        threshold: Optional[float] = None,
        ttl: Optional[int] = None,
        max_size: Optional[int] = None,
    ):
        from nadirclaw.settings import settings

        self._db_path = db_path or (settings.LOG_DIR / "semantic_cache.db")
        self._threshold = threshold if threshold is not None else _sem_cache_threshold()
        self._ttl = ttl if ttl is not None else _sem_cache_ttl()
        self._max_size = max_size if max_size is not None else _sem_cache_max_size()

        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._avg_similarity = 0.0
        self._similarity_count = 0

        # In-memory index: parallel arrays
        self._keys: List[str] = []           # cache_key (sha256 of model+messages)
        self._embeddings: Optional[np.ndarray] = None  # (N, dim) normalized embeddings
        self._models: List[str] = []         # model name per entry

        self._init_db()
        self._load_index()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sc_model
                ON semantic_cache(model)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sc_created
                ON semantic_cache(created_at)
            """)
            conn.commit()
        finally:
            conn.close()

    def _load_index(self) -> None:
        """Load all non-expired embeddings from SQLite into memory."""
        cutoff = time.time() - self._ttl
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT cache_key, model, embedding FROM semantic_cache "
                "WHERE created_at > ? ORDER BY last_accessed DESC LIMIT ?",
                (cutoff, self._max_size),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            self._keys = []
            self._models = []
            self._embeddings = None
            logger.info("Semantic cache: loaded 0 entries from disk")
            return

        keys = []
        models = []
        embs = []
        for cache_key, model, emb_blob in rows:
            keys.append(cache_key)
            models.append(model)
            embs.append(np.frombuffer(emb_blob, dtype=np.float32))

        self._keys = keys
        self._models = models
        self._embeddings = np.vstack(embs)  # (N, dim)
        logger.info("Semantic cache: loaded %d entries from disk", len(keys))

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to a normalized embedding vector."""
        from nadirclaw.encoder import encode_cached
        emb = encode_cached(text, normalize=True)
        return emb.astype(np.float32)

    def _messages_text(self, messages: list) -> str:
        """Extract concatenated text from messages for embedding."""
        parts = []
        for m in messages:
            if hasattr(m, "text_content"):
                parts.append(m.text_content())
            elif isinstance(m, dict):
                parts.append(str(m.get("content", "")))
            else:
                parts.append(str(m))
        return "\n".join(parts)

    def get(self, model: str, messages: list) -> Optional[Dict[str, Any]]:
        """Look up a semantically similar cached response.

        Returns the cached response dict if similarity >= threshold, else None.
        """
        text = self._messages_text(messages)
        query_emb = self._encode(text)

        with self._lock:
            if self._embeddings is None or len(self._keys) == 0:
                self._misses += 1
                return None

            # Filter to same model
            model_mask = np.array([m == model for m in self._models], dtype=bool)
            if not model_mask.any():
                self._misses += 1
                return None

            # Cosine similarity (embeddings are normalized, so dot product = cosine)
            similarities = self._embeddings[model_mask] @ query_emb
            best_idx_in_mask = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx_in_mask])

            # Track average similarity for stats
            self._similarity_count += 1
            self._avg_similarity += (best_sim - self._avg_similarity) / self._similarity_count

            if best_sim < self._threshold:
                self._misses += 1
                return None

            # Map back to global index
            masked_indices = np.where(model_mask)[0]
            global_idx = int(masked_indices[best_idx_in_mask])
            cache_key = self._keys[global_idx]

        # Fetch response from SQLite (outside lock for speed)
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute(
                "SELECT response FROM semantic_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                self._misses += 1
                return None

            # Update access time
            conn.execute(
                "UPDATE semantic_cache SET last_accessed = ?, access_count = access_count + 1 "
                "WHERE cache_key = ?",
                (time.time(), cache_key),
            )
            conn.commit()
        finally:
            conn.close()

        with self._lock:
            self._hits += 1

        response = json.loads(row[0])
        logger.debug("Semantic cache HIT: similarity=%.4f key=%s", best_sim, cache_key[:12])
        return response

    def put(self, model: str, messages: list, response: Dict[str, Any]) -> None:
        """Store a response in the semantic cache."""
        import hashlib

        text = self._messages_text(messages)
        emb = self._encode(text)

        # Build cache key
        key_data = json.dumps({"model": model, "text": text[:500]}, sort_keys=True)
        cache_key = hashlib.sha256(key_data.encode()).hexdigest()

        response_json = json.dumps(response, default=str)
        emb_blob = emb.tobytes()
        now = time.time()

        # Store in SQLite
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                "INSERT OR REPLACE INTO semantic_cache "
                "(cache_key, model, embedding, response, created_at, last_accessed, access_count) "
                "VALUES (?, ?, ?, ?, ?, ?, 1)",
                (cache_key, model, emb_blob, response_json, now, now),
            )

            # Evict oldest entries if over max_size
            count = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
            if count > self._max_size:
                excess = count - self._max_size
                conn.execute(
                    "DELETE FROM semantic_cache WHERE cache_key IN "
                    "(SELECT cache_key FROM semantic_cache ORDER BY last_accessed ASC LIMIT ?)",
                    (excess,),
                )

            conn.commit()
        finally:
            conn.close()

        # Update in-memory index
        with self._lock:
            # Check if key already exists
            if cache_key in self._keys:
                idx = self._keys.index(cache_key)
                self._embeddings[idx] = emb
            else:
                self._keys.append(cache_key)
                self._models.append(model)
                new_emb = emb.reshape(1, -1)
                if self._embeddings is None:
                    self._embeddings = new_emb
                else:
                    self._embeddings = np.vstack([self._embeddings, new_emb])

                # Evict from in-memory index if over size
                while len(self._keys) > self._max_size:
                    self._keys.pop(0)
                    self._models.pop(0)
                    self._embeddings = self._embeddings[1:]

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "enabled": _sem_cache_enabled(),
                "entries": len(self._keys),
                "max_size": self._max_size,
                "threshold": self._threshold,
                "ttl": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
                "avg_similarity": round(self._avg_similarity, 4),
                "total_lookups": total,
            }

    def clear(self) -> None:
        """Clear all cached entries."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("DELETE FROM semantic_cache")
            conn.commit()
        finally:
            conn.close()

        with self._lock:
            self._keys.clear()
            self._models.clear()
            self._embeddings = None
            self._hits = 0
            self._misses = 0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_semantic_cache: Optional[SemanticCache] = None
_sem_cache_lock = Lock()


def get_semantic_cache() -> SemanticCache:
    """Return the global semantic cache singleton."""
    global _semantic_cache
    if _semantic_cache is None:
        with _sem_cache_lock:
            if _semantic_cache is None:
                _semantic_cache = SemanticCache()
    return _semantic_cache
