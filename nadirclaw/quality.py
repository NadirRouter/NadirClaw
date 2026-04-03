"""Quality scoring engine for NadirClaw.

Computes a quality score (0.0-1.0) for each LLM response using passive
signals (no user input required) and integrates active feedback when available.

Passive signals:
  - Empty/near-empty response detection
  - Fallback triggered (degraded quality likely)
  - Response latency vs tier expectation
  - Token ratio (completion/prompt)
  - Error detection

Active signals (from /v1/feedback):
  - User ratings (1-5)
  - Misroute flags

Running quality scores are tracked per model per tier using an Exponential
Moving Average (EMA) over a configurable window.
"""

import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger("nadirclaw.quality")

# Expected latency ranges by tier (ms) — used for latency scoring
_TIER_LATENCY_EXPECTATIONS = {
    "simple": (100, 3000),     # expect 100ms-3s
    "medium": (200, 8000),     # expect 200ms-8s
    "mid": (200, 8000),
    "complex": (500, 15000),   # expect 500ms-15s
    "reasoning": (1000, 30000),  # expect 1s-30s
}

# EMA decay factor — smaller = longer memory
_EMA_ALPHA = 0.05  # ~20-request effective window


class QualityScorer:
    """Compute and track quality scores for LLM responses."""

    def __init__(self):
        self._lock = Lock()
        # Per (model, tier) → EMA quality score
        self._scores: Dict[tuple, float] = defaultdict(lambda: 0.8)  # default "good"
        self._counts: Dict[tuple, int] = defaultdict(int)
        # Global stats
        self._total_scored = 0
        self._total_quality_sum = 0.0

    def score_response(
        self,
        *,
        content: str = "",
        status: str = "ok",
        tier: str = "simple",
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_latency_ms: int = 0,
        fallback_used: Optional[str] = None,
        error: Optional[str] = None,
    ) -> float:
        """Compute a quality score (0.0-1.0) from passive signals.

        Returns the computed quality score.
        """
        score = 1.0
        signals: list[str] = []

        # 1. Empty response detection (worst signal)
        if status == "ok" and (not content or len(content.strip()) < 5):
            score *= 0.1
            signals.append("empty_response")

        # 2. Error status
        if status != "ok":
            score *= 0.2
            signals.append(f"error:{error[:30] if error else 'unknown'}")

        # 3. Fallback triggered — response came from backup model
        if fallback_used:
            score *= 0.7
            signals.append(f"fallback_from:{fallback_used}")

        # 4. Token ratio — very low completion/prompt ratio may indicate refusal
        if prompt_tokens > 0 and completion_tokens > 0:
            ratio = completion_tokens / prompt_tokens
            if ratio < 0.01:
                score *= 0.4
                signals.append(f"low_token_ratio:{ratio:.4f}")
            elif ratio < 0.05:
                score *= 0.7
                signals.append(f"low_token_ratio:{ratio:.4f}")

        # 5. Latency scoring — penalize if way outside expected range
        if total_latency_ms > 0 and tier in _TIER_LATENCY_EXPECTATIONS:
            expected_min, expected_max = _TIER_LATENCY_EXPECTATIONS[tier]
            if total_latency_ms > expected_max * 2:
                score *= 0.8
                signals.append(f"high_latency:{total_latency_ms}ms")

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        # Update EMA
        key = (model, tier)
        with self._lock:
            self._counts[key] += 1
            old = self._scores[key]
            self._scores[key] = old * (1 - _EMA_ALPHA) + score * _EMA_ALPHA
            self._total_scored += 1
            self._total_quality_sum += score

        if score < 0.5:
            logger.debug(
                "Low quality score %.2f for model=%s tier=%s: %s",
                score, model, tier, signals,
            )

        return score

    def record_feedback(
        self,
        model: str,
        tier: str,
        rating: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Integrate active user feedback into quality scores.

        rating: 1-5 (mapped to 0.0-1.0)
        reason: "misrouted" gets heavy penalty
        """
        feedback_score: Optional[float] = None

        if rating is not None:
            feedback_score = (rating - 1) / 4.0  # 1→0.0, 5→1.0

        if reason == "misrouted":
            feedback_score = 0.2  # Heavy penalty

        if feedback_score is None:
            return

        key = (model, tier)
        # Apply feedback with 3x weight (faster signal than passive)
        with self._lock:
            old = self._scores[key]
            alpha = min(_EMA_ALPHA * 3, 0.3)
            self._scores[key] = old * (1 - alpha) + feedback_score * alpha

    def get_model_quality(self, model: str, tier: str = "") -> float:
        """Get the current EMA quality score for a model/tier."""
        key = (model, tier)
        with self._lock:
            return self._scores.get(key, 0.8)

    def get_stats(self) -> Dict[str, Any]:
        """Get quality scoring statistics."""
        with self._lock:
            model_scores = {}
            for (model, tier), score in sorted(self._scores.items()):
                label = f"{model}:{tier}" if tier else model
                model_scores[label] = {
                    "quality_score": round(score, 4),
                    "sample_count": self._counts.get((model, tier), 0),
                }

            avg = (self._total_quality_sum / self._total_scored) if self._total_scored > 0 else 0.0

            return {
                "total_scored": self._total_scored,
                "average_quality": round(avg, 4),
                "model_scores": model_scores,
            }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_quality_scorer: Optional[QualityScorer] = None
_quality_lock = Lock()


def get_quality_scorer() -> QualityScorer:
    """Return the global quality scorer singleton."""
    global _quality_scorer
    if _quality_scorer is None:
        with _quality_lock:
            if _quality_scorer is None:
                _quality_scorer = QualityScorer()
    return _quality_scorer
