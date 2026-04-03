"""Shared SentenceTransformer singleton for NadirClaw.

The encoder is loaded lazily on first use — not at import time.
This avoids the ~500ms cold-start penalty when running commands that
don't need classification (e.g. ``nadirclaw serve`` before the first request).

Includes an LRU embedding cache for repeated prompts to avoid redundant
encoder calls (~50-100ms each).
"""

import hashlib
import logging
import os
import time
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)

_shared_encoder = None  # type: ignore[assignment]
_encoder_lock = Lock()


def get_shared_encoder_sync():
    """
    Lazily initialize and return a shared SentenceTransformer instance.
    The first call loads the model (~80 MB download on first run).
    Uses double-checked locking to avoid redundant loads.

    The ``sentence_transformers`` import itself is deferred so that
    ``import nadirclaw`` does not trigger a heavy torch import chain.

    Note: ONNX backend (2x faster) is available via Nadir Pro.
    """
    global _shared_encoder
    if _shared_encoder is None:
        with _encoder_lock:
            if _shared_encoder is None:
                # Nadir Pro can inject an ONNX encoder before this runs
                if _shared_encoder is not None:
                    return _shared_encoder

                t0 = time.time()
                logger.info("Loading SentenceTransformer encoder: all-MiniLM-L6-v2")

                # Suppress noisy tokenizer parallelism warning
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

                from sentence_transformers import SentenceTransformer

                _shared_encoder = SentenceTransformer("all-MiniLM-L6-v2")
                elapsed = int((time.time() - t0) * 1000)
                logger.info("SentenceTransformer encoder loaded (%dms)", elapsed)
    return _shared_encoder


# ---------------------------------------------------------------------------
# Embedding cache — avoids redundant encoder calls for repeated prompts
# ---------------------------------------------------------------------------

_EMBED_CACHE_MAX = int(os.getenv("NADIRCLAW_EMBED_CACHE_SIZE", "2000"))
_embed_cache: OrderedDict[str, "numpy.ndarray"] = OrderedDict()  # noqa: F821
_embed_cache_lock = Lock()
_embed_cache_hits = 0
_embed_cache_misses = 0


def encode_cached(text: str, normalize: bool = True):
    """Encode a single text with LRU caching.

    Returns a numpy ndarray (1-D embedding vector). Cache saves ~50-100ms
    per duplicate prompt.
    """
    global _embed_cache_hits, _embed_cache_misses

    key = hashlib.sha256(text.encode()).hexdigest()[:32]

    with _embed_cache_lock:
        if key in _embed_cache:
            _embed_cache.move_to_end(key)
            _embed_cache_hits += 1
            return _embed_cache[key]

    # Cache miss — encode (outside lock to avoid blocking)
    encoder = get_shared_encoder_sync()
    emb = encoder.encode([text], show_progress_bar=False,
                         normalize_embeddings=normalize)[0]

    with _embed_cache_lock:
        _embed_cache_misses += 1
        _embed_cache[key] = emb
        # Evict LRU if over capacity
        while len(_embed_cache) > _EMBED_CACHE_MAX:
            _embed_cache.popitem(last=False)

    return emb


def get_embed_cache_stats() -> dict:
    """Return embedding cache statistics."""
    with _embed_cache_lock:
        total = _embed_cache_hits + _embed_cache_misses
        return {
            "entries": len(_embed_cache),
            "max_size": _EMBED_CACHE_MAX,
            "hits": _embed_cache_hits,
            "misses": _embed_cache_misses,
            "hit_rate": round(_embed_cache_hits / total, 4) if total > 0 else 0.0,
        }
