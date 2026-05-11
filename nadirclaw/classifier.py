"""
Binary complexity classifier using sentence embedding prototypes.

Classifies prompts as simple or complex by comparing their embeddings
to pre-computed centroid vectors shipped with the package.

Centroid files and optional metadata are loaded from:
  - The package directory (default, backward-compatible)
  - A custom directory set via NADIRCLAW_CENTROID_DIR

When a custom centroid directory is used, a centroid_metadata.json file is
required. If metadata is present (in either location), the backend, model,
and dimension are validated against the current encoder configuration to
prevent silent mismatch between centroid vectors and the active embedding model.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(__file__)


class BinaryComplexityClassifier:
    """
    Classifies prompts as simple or complex using semantic prototype centroids.

    Loads pre-computed centroid vectors from .npy files and validates them
    against centroid_metadata.json if present. At inference time, embeds the
    prompt (~10 ms on warm encoder), computes cosine similarity to both
    centroids, and returns a binary decision with a confidence score.
    """

    def __init__(self):
        from nadirclaw.encoder import get_shared_encoder_sync
        from nadirclaw.settings import settings

        self.encoder = get_shared_encoder_sync()
        self._simple_centroid, self._complex_centroid = self._load_centroids(
            encoder=self.encoder,
            settings=settings,
        )

        logger.info("BinaryComplexityClassifier ready (pre-computed centroids)")

    # ------------------------------------------------------------------
    # Load and validate pre-computed centroids
    # ------------------------------------------------------------------

    @staticmethod
    def _load_centroids(encoder=None, settings=None) -> Tuple[np.ndarray, np.ndarray]:
        """Load centroid vectors and validate metadata if present.

        Args:
            encoder: Current encoder instance (used for dimension check).
            settings: Settings instance (used to read CENTROID_DIR and
                      EMBEDDING_BACKEND/MODEL for mismatch reporting).

        Raises:
            FileNotFoundError: Centroid files or required metadata missing.
            RuntimeError: Backend/model/dimension mismatch detected.
            ValueError: Centroid files have inconsistent dimensions.
        """
        if settings is None:
            from nadirclaw.settings import settings as _settings
            settings = _settings

        custom_dir = settings.CENTROID_DIR
        if custom_dir is not None:
            centroid_dir = Path(custom_dir)
            require_metadata = True
        else:
            centroid_dir = Path(_PKG_DIR)
            require_metadata = False

        simple_path = centroid_dir / "simple_centroid.npy"
        complex_path = centroid_dir / "complex_centroid.npy"
        meta_path = centroid_dir / "centroid_metadata.json"

        if not simple_path.exists() or not complex_path.exists():
            if custom_dir is not None:
                raise FileNotFoundError(
                    f"Centroid files not found in {centroid_dir}. "
                    f"Run: nadirclaw build-centroids "
                    f"--embedding-backend {settings.EMBEDDING_BACKEND} "
                    f"--embedding-model {settings.EMBEDDING_MODEL} "
                    f"--output-dir {centroid_dir}"
                )
            raise FileNotFoundError(
                "Pre-computed centroid files not found. "
                "Run 'nadirclaw build-centroids' to generate them."
            )

        simple_centroid = np.load(str(simple_path))
        complex_centroid = np.load(str(complex_path))

        # Both centroid arrays must have the same shape
        if simple_centroid.shape != complex_centroid.shape:
            raise ValueError(
                f"Centroid dimension mismatch: "
                f"simple={simple_centroid.shape}, complex={complex_centroid.shape}. "
                f"Rebuild centroids with 'nadirclaw build-centroids'."
            )

        current_backend = settings.EMBEDDING_BACKEND
        current_model = settings.EMBEDDING_MODEL

        if meta_path.exists():
            # Validate metadata when present (in any directory)
            with open(meta_path) as fh:
                meta = json.load(fh)

            meta_backend = meta.get("embedding_backend", "")
            meta_model = meta.get("embedding_model", "")
            meta_dim = meta.get("dimension")

            if meta_backend and meta_backend != current_backend:
                raise RuntimeError(
                    f"Centroid/encoder mismatch: centroids were built with backend "
                    f"{meta_backend!r} but current backend is {current_backend!r}. "
                    f"Rebuild with:\n"
                    f"  nadirclaw build-centroids "
                    f"--embedding-backend {current_backend} "
                    f"--embedding-model {current_model}"
                    + (f" --output-dir {centroid_dir}" if custom_dir is not None else "")
                )

            if meta_model and meta_model != current_model:
                raise RuntimeError(
                    f"Centroid/encoder mismatch: centroids were built for model "
                    f"{meta_model!r} but current model is {current_model!r}. "
                    f"Rebuild with:\n"
                    f"  nadirclaw build-centroids "
                    f"--embedding-backend {current_backend} "
                    f"--embedding-model {current_model}"
                    + (f" --output-dir {centroid_dir}" if custom_dir is not None else "")
                )

            if meta_dim is not None and simple_centroid.shape[0] != meta_dim:
                raise RuntimeError(
                    f"Centroid dimension in metadata ({meta_dim}) does not match "
                    f"actual centroid dimension ({simple_centroid.shape[0]}). "
                    f"Rebuild centroids with 'nadirclaw build-centroids'."
                )

        elif require_metadata:
            # Custom dir without metadata — fail closed for safety
            raise FileNotFoundError(
                f"No centroid_metadata.json found in {centroid_dir}. "
                f"Custom centroid directories require metadata for safe validation. "
                f"Rebuild with:\n"
                f"  nadirclaw build-centroids "
                f"--embedding-backend {current_backend} "
                f"--embedding-model {current_model} "
                f"--output-dir {centroid_dir}"
            )

        # Encoder dimension check (skipped for Ollama before first encode call)
        if encoder is not None and encoder.dimension is not None:
            enc_dim = encoder.dimension
            if enc_dim != simple_centroid.shape[0]:
                raise RuntimeError(
                    f"Encoder output dimension ({enc_dim}) does not match "
                    f"centroid dimension ({simple_centroid.shape[0]}). "
                    f"Rebuild centroids with the current embedding model:\n"
                    f"  nadirclaw build-centroids "
                    f"--embedding-backend {current_backend} "
                    f"--embedding-model {current_model}"
                )

        return simple_centroid, complex_centroid

    # ------------------------------------------------------------------
    # Core classification
    # ------------------------------------------------------------------

    def classify(self, prompt: str) -> Tuple[bool, float]:
        """
        Classify a prompt as simple or complex.

        Borderline cases (confidence < threshold) are biased toward complex --
        it is cheaper to over-serve a simple prompt than to under-serve a
        complex one.

        Returns:
            (is_complex, confidence) where confidence is in [0, 1].
            confidence near 0 means borderline; near 1 means very clear.
        """
        from nadirclaw.settings import settings

        threshold = settings.CONFIDENCE_THRESHOLD

        emb = self.encoder.encode([prompt], show_progress_bar=False)[0]
        emb = emb / np.linalg.norm(emb)

        sim_simple = float(np.dot(emb, self._simple_centroid))
        sim_complex = float(np.dot(emb, self._complex_centroid))

        confidence = abs(sim_complex - sim_simple)

        if confidence < threshold:
            is_complex = True
        else:
            is_complex = sim_complex > sim_simple

        return is_complex, confidence

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        """Async analyse -- conforms to the analyzer interface."""
        return self._analyze_sync(text)

    def _analyze_sync(self, text: str) -> Dict[str, Any]:
        start = time.time()
        is_complex, confidence = self.classify(text)

        complexity_score = self._confidence_to_score(is_complex, confidence)

        # Three-tier routing: use score thresholds to determine tier
        tier_name, tier = self._score_to_tier(complexity_score)

        recommended_model, recommended_provider = self._select_model_by_tier(tier_name)

        latency_ms = int((time.time() - start) * 1000)

        return {
            "recommended_model": recommended_model,
            "recommended_provider": recommended_provider,
            "confidence": confidence,
            "complexity_score": complexity_score,
            "complexity_tier": tier,
            "complexity_name": tier_name,
            "tier": tier,
            "tier_name": tier_name,
            "reasoning": (
                f"Binary classifier: {tier_name} "
                f"(score={complexity_score:.3f}, confidence={confidence:.3f})"
            ),
            "ranked_models": [],
            "analyzer_latency_ms": latency_ms,
            "analyzer_type": "binary",
            "selection_method": "binary_classifier",
            "model_type": "binary_classifier",
        }

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_model(is_complex: bool) -> Tuple[str, str]:
        """Pick the model based on binary tier classification (legacy)."""
        from nadirclaw.settings import settings

        model = settings.COMPLEX_MODEL if is_complex else settings.SIMPLE_MODEL
        provider = model.split("/")[0] if "/" in model else "api"
        return model, provider

    @staticmethod
    def _select_model_by_tier(tier_name: str) -> Tuple[str, str]:
        """Pick the model based on three-tier classification."""
        from nadirclaw.settings import settings

        if tier_name == "complex":
            model = settings.COMPLEX_MODEL
        elif tier_name == "mid":
            model = settings.MID_MODEL
        else:
            model = settings.SIMPLE_MODEL
        provider = model.split("/")[0] if "/" in model else "api"
        return model, provider

    @staticmethod
    def _confidence_to_score(is_complex: bool, confidence: float) -> float:
        """Map binary decision + confidence to a 0-1 complexity score."""
        if is_complex:
            return 0.5 + min(confidence * 5, 0.5)
        else:
            return 0.5 - min(confidence * 5, 0.5)

    @staticmethod
    def _score_to_tier(complexity_score: float) -> Tuple[str, int]:
        """Map a 0-1 complexity score to a tier name and numeric tier.

        Uses configurable thresholds from NADIRCLAW_TIER_THRESHOLDS.
        If MID_MODEL is not set, falls back to binary (simple/complex).

        Returns (tier_name, tier_number).
        """
        from nadirclaw.settings import settings

        simple_max, complex_min = settings.TIER_THRESHOLDS

        if settings.has_mid_tier:
            if complexity_score <= simple_max:
                return "simple", 1
            elif complexity_score >= complex_min:
                return "complex", 3
            else:
                return "mid", 2
        else:
            # No mid model configured — binary routing
            if complexity_score >= 0.5:
                return "complex", 3
            else:
                return "simple", 1


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------
_singleton: Optional[BinaryComplexityClassifier] = None


def get_binary_classifier() -> BinaryComplexityClassifier:
    """Return the singleton classifier instance."""
    global _singleton
    if _singleton is None:
        _singleton = BinaryComplexityClassifier()
    return _singleton


def warmup() -> None:
    """Pre-warm the encoder and load centroids once at startup."""
    global _singleton
    logger.info("Warming up BinaryComplexityClassifier ...")
    _singleton = BinaryComplexityClassifier()
    logger.info("BinaryComplexityClassifier warmup complete")
