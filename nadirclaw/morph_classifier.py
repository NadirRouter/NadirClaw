"""Morph Model Router complexity classifier.

An optional, opt-in classifier that delegates the simple/mid/complex routing
decision to Morph's hosted Model Router (https://docs.morphllm.com/sdk/components/router)
instead of the local centroid/DistilBERT classifiers.

Design constraints (see issue #68):

  - **Strictly opt-in.** Activated only with ``NADIRCLAW_COMPLEXITY_ANALYZER=morph``
    and a ``MORPH_API_KEY``. The default classifier stays the local binary one;
    a remote round-trip is never a hard dependency for first-run UX.
  - **Fail-closed to local.** On a missing key, HTTP error, timeout, or an
    unparseable response, we log once and fall back to the local
    ``BinaryComplexityClassifier`` for that request. A transient Morph outage
    degrades quality, never availability.
  - **Latency budget.** ``MORPH_TIMEOUT_MS`` (default 200ms) bounds the call.
    Repeated prompts in a CLI session are served from a small in-memory LRU.

Only the standard library is used for the HTTP call (``urllib.request``) so the
classifier adds no new dependency and is trivially monkeypatchable in tests.
"""

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Difficulty bucket -> complexity score. The score (not the bucket) is mapped to
# a tier by BinaryComplexityClassifier._score_to_tier, so the same configurable
# NADIRCLAW_TIER_THRESHOLDS govern Morph and the local classifier alike, and the
# "mid" tier only appears when MID_MODEL is configured.
_DIFFICULTY_SCORE = {
    "easy": 0.15,
    "medium": 0.55,
    "hard": 0.85,
    "needs_info": 0.92,
}

# Module-level guards so the fail-closed fallback only logs once per process.
_warned_fallback = False


def _warn_once(message: str, *args: Any) -> None:
    global _warned_fallback
    if not _warned_fallback:
        logger.warning(message, *args)
        _warned_fallback = True


class MorphRouterClassifier:
    """Routes via Morph's Model Router, falling back to the local classifier."""

    CLASSIFIER_VERSION = "1.0"

    def __init__(self):
        from nadirclaw.settings import settings

        self._api_key = settings.MORPH_API_KEY
        if not self._api_key:
            # Raise so get_classifier() degrades to binary at selection time.
            raise ValueError(
                "NADIRCLAW_COMPLEXITY_ANALYZER=morph requires MORPH_API_KEY. "
                "Set it or unset NADIRCLAW_COMPLEXITY_ANALYZER to use the local "
                "classifier."
            )

        self._api_base = settings.MORPH_API_BASE.rstrip("/")
        self._timeout_s = max(0.001, settings.MORPH_TIMEOUT_MS / 1000.0)
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_cap = 512
        self._fallback: Optional[Any] = None  # lazily-built BinaryComplexityClassifier

        logger.info(
            "MorphRouterClassifier ready (base=%s, timeout=%dms)",
            self._api_base,
            settings.MORPH_TIMEOUT_MS,
        )

    # ------------------------------------------------------------------
    # Remote call
    # ------------------------------------------------------------------

    def _call_morph(self, prompt: str) -> Dict[str, Any]:
        """POST the prompt to Morph's router. Raises on any transport error."""
        url = f"{self._api_base}/router/classify"
        body = json.dumps({"input": prompt}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _parse_response(payload: Dict[str, Any]) -> Tuple[str, float]:
        """Extract (difficulty, confidence) from a Morph response.

        Tolerant of the response being either flat or nested under a
        ``classification`` key. ``confidence`` is the model's confidence when
        present, otherwise ``1 - ambiguity``, otherwise 1.0.
        """
        inner = payload.get("classification", payload)

        difficulty = inner.get("difficulty")
        if not isinstance(difficulty, str):
            raise ValueError(f"Morph response missing 'difficulty': {payload!r}")
        difficulty = difficulty.strip().lower()
        if difficulty not in _DIFFICULTY_SCORE:
            raise ValueError(f"Unknown Morph difficulty {difficulty!r}")

        confidence = inner.get("confidence")
        if not isinstance(confidence, (int, float)):
            ambiguity = inner.get("ambiguity")
            if isinstance(ambiguity, (int, float)):
                confidence = 1.0 - float(ambiguity)
            else:
                confidence = 1.0
        confidence = max(0.0, min(1.0, float(confidence)))

        return difficulty, confidence

    # ------------------------------------------------------------------
    # Fail-closed local fallback
    # ------------------------------------------------------------------

    async def _local_fallback(self, text: str, system_message: str) -> Dict[str, Any]:
        if self._fallback is None:
            from nadirclaw.classifier import get_binary_classifier
            self._fallback = get_binary_classifier()
        result = await self._fallback.analyze(text=text, system_message=system_message)
        result["analyzer_type"] = "morph-fallback-binary"
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def analyze(self, text: str = "", system_message: str = "", **kwargs) -> Dict[str, Any]:
        """Async-compatible interface matching the server's expected API."""
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return dict(self._cache[cache_key])

        start = time.time()
        try:
            payload = self._call_morph(text)
            difficulty, confidence = self._parse_response(payload)
        except (urllib.error.URLError, ValueError, OSError, json.JSONDecodeError, TimeoutError) as e:
            _warn_once(
                "Morph router unavailable (%s) — falling back to local binary "
                "classifier. Further failures are suppressed.",
                e,
            )
            return await self._local_fallback(text, system_message)

        from nadirclaw.classifier import BinaryComplexityClassifier

        complexity_score = _DIFFICULTY_SCORE[difficulty]
        tier_name, tier = BinaryComplexityClassifier._score_to_tier(complexity_score)
        recommended_model, recommended_provider = (
            BinaryComplexityClassifier._select_model_by_tier(tier_name)
        )
        latency_ms = int((time.time() - start) * 1000)

        result = {
            "recommended_model": recommended_model,
            "recommended_provider": recommended_provider,
            "confidence": confidence,
            "complexity_score": complexity_score,
            "complexity_tier": tier,
            "complexity_name": tier_name,
            "tier": tier,
            "tier_name": tier_name,
            "reasoning": (
                f"Morph router: difficulty={difficulty} -> {tier_name} "
                f"(score={complexity_score:.2f}, confidence={confidence:.0%})"
            ),
            "ranked_models": [],
            "analyzer_latency_ms": latency_ms,
            "analyzer_type": f"morph-v{self.CLASSIFIER_VERSION}",
            "selection_method": "morph_router",
            "model_type": "morph_router",
            "morph_difficulty": difficulty,
        }

        self._cache[cache_key] = dict(result)
        self._cache.move_to_end(cache_key)
        if len(self._cache) > self._cache_cap:
            self._cache.popitem(last=False)

        return result
