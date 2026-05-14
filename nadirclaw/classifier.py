"""
Binary complexity classifier using sentence embedding prototypes.

Classifies prompts as simple or complex by comparing their embeddings
to pre-computed centroid vectors shipped with the package.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(__file__)


class BinaryComplexityClassifier:
    """
    Classifies prompts as simple or complex using semantic prototype centroids.

    Loads pre-computed centroid vectors from .npy files (shipped with the
    package). At inference time, embeds the prompt (~10 ms on warm encoder),
    computes cosine similarity to both centroids, and returns a binary
    decision with a confidence score.
    """

    def __init__(self):
        from nadirclaw.encoder import get_shared_encoder_sync

        self.encoder = get_shared_encoder_sync()
        self._simple_centroid, self._complex_centroid = self._load_centroids()

        logger.info("BinaryComplexityClassifier ready (pre-computed centroids)")

    # ------------------------------------------------------------------
    # Load pre-computed centroids
    # ------------------------------------------------------------------

    @staticmethod
    def _load_centroids() -> Tuple[np.ndarray, np.ndarray]:
        """Load pre-computed centroid vectors from .npy files."""
        simple_path = os.path.join(_PKG_DIR, "simple_centroid.npy")
        complex_path = os.path.join(_PKG_DIR, "complex_centroid.npy")

        if not os.path.exists(simple_path) or not os.path.exists(complex_path):
            raise FileNotFoundError(
                "Pre-computed centroid files not found. "
                "Run 'nadirclaw build-centroids' to generate them."
            )

        simple_centroid = np.load(simple_path)
        complex_centroid = np.load(complex_path)

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
_active_classifier: Optional[Any] = None  # binary or distilbert, set on first use


def get_binary_classifier() -> BinaryComplexityClassifier:
    """Return the singleton binary classifier instance."""
    global _singleton
    if _singleton is None:
        _singleton = BinaryComplexityClassifier()
    return _singleton


def get_classifier() -> Any:
    """Return whichever classifier is configured via NADIRCLAW_COMPLEXITY_ANALYZER.

    Defaults to the binary centroid classifier. Set
    `NADIRCLAW_COMPLEXITY_ANALYZER=distilbert` to use the 3-class DistilBERT
    classifier (requires the ~256MB model directory to be present).
    """
    global _active_classifier
    if _active_classifier is not None:
        return _active_classifier

    from nadirclaw.settings import settings

    kind = settings.COMPLEXITY_ANALYZER_TYPE
    if kind == "distilbert":
        try:
            from nadirclaw.distilbert_classifier import DistilBertClassifier
            _active_classifier = DistilBertClassifier()
            logger.info("Using DistilBERT classifier (3-class: simple/mid/complex)")
            return _active_classifier
        except Exception as e:
            logger.warning(
                "Failed to load DistilBERT classifier (%s) — falling back to binary",
                e,
            )
    _active_classifier = get_binary_classifier()
    return _active_classifier


def warmup() -> None:
    """Pre-warm the configured classifier at startup."""
    global _singleton, _active_classifier
    from nadirclaw.settings import settings

    kind = settings.COMPLEXITY_ANALYZER_TYPE
    if kind == "distilbert":
        try:
            from nadirclaw.distilbert_classifier import DistilBertClassifier
            logger.info("Warming up DistilBertClassifier (3-class) ...")
            _active_classifier = DistilBertClassifier()
            logger.info("DistilBertClassifier warmup complete")
            return
        except Exception as e:
            logger.warning(
                "DistilBERT warmup failed (%s) — falling back to binary classifier",
                e,
            )
    logger.info("Warming up BinaryComplexityClassifier ...")
    _singleton = BinaryComplexityClassifier()
    _active_classifier = _singleton
    logger.info("BinaryComplexityClassifier warmup complete")
