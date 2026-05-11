"""Embedding encoder abstraction for NadirClaw.

Supports multiple embedding backends:
  - sentence-transformers (default, uses all-MiniLM-L6-v2)
  - ollama (local Ollama server via /api/embed)

The encoder is loaded lazily on first use — not at import time.
This avoids the ~500ms cold-start penalty when running commands that
don't need classification (e.g. ``nadirclaw serve`` before the first request).

Backend and model are configured via:
  NADIRCLAW_EMBEDDING_BACKEND=sentence-transformers|ollama
  NADIRCLAW_EMBEDDING_MODEL=<model-id>
  NADIRCLAW_EMBEDDING_API_BASE=<url>   (for ollama)
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)

_shared_encoder = None  # type: ignore[assignment]
_encoder_lock = Lock()


class BaseEmbeddingEncoder(ABC):
    """Abstract base class for embedding encoders."""

    backend: str
    model_id: str

    @abstractmethod
    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into embedding vectors.

        Returns a 2D ndarray of shape (len(texts), dimension).
        """
        ...

    @property
    def dimension(self) -> int | None:
        """Embedding output dimension, or None if not yet determined."""
        return None


class SentenceTransformerEncoder(BaseEmbeddingEncoder):
    """Encoder backed by the sentence-transformers library.

    Default backend. On first use, downloads the model (~80 MB) if not cached.
    Subsequent calls use the cached model. Warm encoding is ~10 ms/prompt.
    """

    backend = "sentence-transformers"

    def __init__(self, model_id: str = "all-MiniLM-L6-v2") -> None:
        self.model_id = model_id
        # Suppress noisy tokenizer parallelism warning
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        t0 = time.time()
        logger.info("Loading SentenceTransformer encoder: %s", model_id)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_id)
        logger.info("SentenceTransformer encoder loaded (%dms)", int((time.time() - t0) * 1000))

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        return self._model.encode(
            texts,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
        )

    @property
    def dimension(self) -> int | None:
        dim = self._model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else None


class OllamaEmbeddingEncoder(BaseEmbeddingEncoder):
    """Encoder backed by a local Ollama server via its /api/embed endpoint.

    Uses the batch-capable ``/api/embed`` API introduced in Ollama 0.1.26+.
    Does not perform final inference — embeddings only.

    Example usage::

        NADIRCLAW_EMBEDDING_BACKEND=ollama
        NADIRCLAW_EMBEDDING_MODEL=nomic-embed-text
        NADIRCLAW_EMBEDDING_API_BASE=http://127.0.0.1:11434
    """

    backend = "ollama"

    def __init__(self, model_id: str, api_base: str) -> None:
        self.model_id = model_id
        self._api_base = api_base.rstrip("/")
        self._dim: int | None = None

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        url = f"{self._api_base}/api/embed"
        payload = json.dumps({"model": self.model_id, "input": texts}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama embedding request failed (url={self._api_base}): {exc}. "
                f"Is Ollama running? Try: ollama serve"
            ) from exc

        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(
                f"Ollama /api/embed returned no embeddings for model {self.model_id!r}. "
                f"Is the model available? Run: ollama pull {self.model_id}"
            )

        arr = np.array(embeddings, dtype=np.float32)
        if arr.ndim == 2:
            self._dim = arr.shape[1]
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-12)
        return arr

    @property
    def dimension(self) -> int | None:
        """Dimension is known after the first encode() call."""
        return self._dim


def _build_encoder(backend: str, model_id: str, api_base: str) -> BaseEmbeddingEncoder:
    """Factory: construct an encoder for the given backend.

    Raises ValueError for unknown backend values.
    """
    if backend == "sentence-transformers":
        return SentenceTransformerEncoder(model_id)
    elif backend == "ollama":
        return OllamaEmbeddingEncoder(model_id=model_id, api_base=api_base)
    else:
        raise ValueError(
            f"Unknown embedding backend {backend!r}. "
            f"Supported values: sentence-transformers, ollama. "
            f"Set NADIRCLAW_EMBEDDING_BACKEND to one of these values."
        )


def get_shared_encoder_sync() -> BaseEmbeddingEncoder:
    """Lazily initialize and return the shared encoder singleton.

    Backend and model are read from settings:
      NADIRCLAW_EMBEDDING_BACKEND  (default: sentence-transformers)
      NADIRCLAW_EMBEDDING_MODEL    (default: all-MiniLM-L6-v2)
      NADIRCLAW_EMBEDDING_API_BASE (default: OLLAMA_API_BASE)

    Default behavior (sentence-transformers + all-MiniLM-L6-v2) is preserved
    when no env vars are set. Uses double-checked locking to avoid redundant loads.
    """
    global _shared_encoder
    if _shared_encoder is None:
        with _encoder_lock:
            if _shared_encoder is None:
                from nadirclaw.settings import settings
                backend = settings.EMBEDDING_BACKEND
                model_id = settings.EMBEDDING_MODEL
                api_base = settings.EMBEDDING_API_BASE
                _shared_encoder = _build_encoder(backend, model_id, api_base)
    return _shared_encoder


def _reset_shared_encoder() -> None:
    """Reset the shared encoder singleton. For testing and CLI use only."""
    global _shared_encoder
    with _encoder_lock:
        _shared_encoder = None
