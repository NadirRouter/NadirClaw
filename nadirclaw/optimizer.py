"""Pareto optimizer for NadirClaw model selection.

Selects the Pareto-optimal model from a candidate set based on
configurable quality/cost/latency weights.  Ported from Horizen's
LLMRanker (quality=0.55, cost=0.30, latency=0.15) with routing-profile
presets and X-Routing-Priority header parsing.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nadirclaw.optimizer")


# ---------------------------------------------------------------------------
# Routing weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutingWeights:
    """Immutable quality / cost / latency weight triple.

    Weights are normalised at construction so they always sum to 1.0.
    """

    quality: float = 0.55
    cost: float = 0.30
    latency: float = 0.15

    def __post_init__(self) -> None:
        total = self.quality + self.cost + self.latency
        if total <= 0:
            raise ValueError("Weights must sum to a positive number")
        # Normalise (frozen → use object.__setattr__)
        object.__setattr__(self, "quality", self.quality / total)
        object.__setattr__(self, "cost", self.cost / total)
        object.__setattr__(self, "latency", self.latency / total)


# Named routing profiles — quick presets for common strategies.
ROUTING_WEIGHT_PROFILES: Dict[str, RoutingWeights] = {
    "eco": RoutingWeights(quality=0.2, cost=0.6, latency=0.2),
    "balanced": RoutingWeights(quality=0.55, cost=0.30, latency=0.15),
    "premium": RoutingWeights(quality=0.8, cost=0.1, latency=0.1),
    "fast": RoutingWeights(quality=0.2, cost=0.2, latency=0.6),
}

DEFAULT_WEIGHTS = ROUTING_WEIGHT_PROFILES["balanced"]


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

_KV_RE = re.compile(r"(\w+)\s*=\s*([\d.]+)")


def parse_routing_priority(header_value: str) -> Optional[RoutingWeights]:
    """Parse an ``X-Routing-Priority`` header into :class:`RoutingWeights`.

    Accepted formats::

        quality=0.6, cost=0.3, latency=0.1
        quality=0.6,cost=0.3,latency=0.1

    Also accepts a single profile name (``eco``, ``balanced``, etc.).
    Returns *None* when the header is empty or unparseable.
    """
    if not header_value:
        return None

    stripped = header_value.strip()

    # Check for a named profile first
    if stripped.lower() in ROUTING_WEIGHT_PROFILES:
        return ROUTING_WEIGHT_PROFILES[stripped.lower()]

    pairs = {k.lower(): float(v) for k, v in _KV_RE.findall(stripped)}
    if not pairs:
        logger.warning("Could not parse X-Routing-Priority header: %r", header_value)
        return None

    return RoutingWeights(
        quality=pairs.get("quality", DEFAULT_WEIGHTS.quality),
        cost=pairs.get("cost", DEFAULT_WEIGHTS.cost),
        latency=pairs.get("latency", DEFAULT_WEIGHTS.latency),
    )


# ---------------------------------------------------------------------------
# Model capability filtering (ported from Horizen LLMRanker)
# ---------------------------------------------------------------------------

# Capabilities each model supports.  Extend as new models appear.
CAPABILITY_MAP: Dict[str, set] = {
    "tools": {
        # OpenAI
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2",
        "gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o4-mini",
        "openai-codex/gpt-5.3-codex",
        # Anthropic
        "claude-opus-4-6-20250918", "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-20250514", "claude-sonnet-4-20250514",
        "claude-haiku-4-20250514",
        # Gemini
        "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash",
        "gemini/gemini-3-flash-preview", "gemini/gemini-2.5-pro",
    },
    "vision": {
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2",
        "gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o4-mini",
        "claude-opus-4-6-20250918", "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-20250514", "claude-sonnet-4-20250514",
        "claude-haiku-4-20250514",
        "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash",
        "gemini/gemini-3-flash-preview", "gemini/gemini-2.5-pro",
    },
    "reasoning": {
        "o3", "o3-mini", "o4-mini",
        "claude-opus-4-6-20250918", "claude-sonnet-4-5-20250929",
        "claude-opus-4-20250514", "claude-sonnet-4-20250514",
        "deepseek/deepseek-reasoner",
    },
    "json_mode": {
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2",
        "gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o4-mini",
        "claude-opus-4-6-20250918", "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-20250514", "claude-sonnet-4-20250514",
        "claude-haiku-4-20250514",
        "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash",
        "gemini/gemini-3-flash-preview", "gemini/gemini-2.5-pro",
    },
}


def _extract_version_parts(model_name: str) -> Dict[str, Any]:
    """Extract version numbers and suffixes from a model name."""
    lower = model_name.lower()
    versions = re.findall(r"\d+\.?\d*", lower)
    suffixes = []
    for s in ("mini", "nano", "turbo", "pro", "flash", "sonnet", "opus", "haiku"):
        if s in lower:
            suffixes.append(s)
    return {"versions": versions, "suffixes": suffixes}


def _has_version_conflict(model_a: str, model_b: str) -> bool:
    """Return True if two model names conflict on version suffixes.

    Prevents e.g. ``gpt-4o-mini`` from matching ``gpt-4o``.
    Ported from Horizen's ``LLMRanker._has_version_conflict``.
    """
    parts_a = _extract_version_parts(model_a)
    parts_b = _extract_version_parts(model_b)

    important = {"mini", "nano"}
    for suffix in important:
        if (suffix in parts_a["suffixes"]) != (suffix in parts_b["suffixes"]):
            return True
    return False


def filter_by_capabilities(
    candidates: List[Dict[str, Any]],
    required: set,
) -> List[Dict[str, Any]]:
    """Remove candidates that lack any of the *required* capabilities.

    Each candidate dict must have a ``"model"`` key.  Unknown models are
    kept (fail-open) so the user can still pin arbitrary models.
    """
    if not required:
        return candidates

    filtered = []
    for c in candidates:
        model = c["model"]
        missing = set()
        for cap in required:
            cap_set = CAPABILITY_MAP.get(cap, set())
            # If the capability set is empty (unknown cap), fail-open
            if cap_set and model not in cap_set:
                missing.add(cap)
        if missing:
            logger.debug(
                "Filtered out %s — missing capabilities: %s", model, missing,
            )
        else:
            filtered.append(c)

    # If everything was filtered out, return the original list so we don't
    # strand the request.
    return filtered if filtered else candidates


# ---------------------------------------------------------------------------
# Pareto optimizer
# ---------------------------------------------------------------------------

class ParetoOptimizer:
    """Selects the Pareto-optimal model given quality/cost/latency weights.

    Scoring formula (from Horizen LLMRanker):

        score = w_quality * quality
              - w_cost    * norm(cost)
              - w_latency * norm(latency)

    where *norm* maps values to [0, 1] via min-max normalisation across
    the candidate set.  Quality is already expected in [0, 1].
    """

    def __init__(self, default_weights: RoutingWeights = DEFAULT_WEIGHTS):
        self._default_weights = default_weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        candidates: List[Dict[str, Any]],
        weights: Optional[RoutingWeights] = None,
        required_capabilities: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Pick the best candidate.

        Parameters
        ----------
        candidates :
            ``[{"model": str, "quality": float, "cost": float, "latency": float, ...}]``
        weights :
            Override weights; falls back to *default_weights*.
        required_capabilities :
            Optional set of capability strings (``"tools"``, ``"vision"``, etc.)
            to pre-filter candidates.

        Returns
        -------
        dict
            The highest-scoring candidate, with an added ``"pareto_score"`` key.

        Raises
        ------
        ValueError
            If *candidates* is empty.
        """
        if not candidates:
            raise ValueError("Cannot select from an empty candidate list")

        w = weights or self._default_weights

        # --- Capability filtering ---
        pool = filter_by_capabilities(candidates, required_capabilities or set())

        if len(pool) == 1:
            result = dict(pool[0])
            result["pareto_score"] = 1.0
            return result

        # --- Min-max normalisation ---
        costs = [c["cost"] for c in pool]
        latencies = [c["latency"] for c in pool]

        cost_min, cost_max = min(costs), max(costs)
        lat_min, lat_max = min(latencies), max(latencies)

        cost_range = cost_max - cost_min if cost_max != cost_min else 1.0
        lat_range = lat_max - lat_min if lat_max != lat_min else 1.0

        # --- Scoring ---
        best: Optional[Dict[str, Any]] = None
        best_score = float("-inf")

        for c in pool:
            norm_cost = (c["cost"] - cost_min) / cost_range
            norm_lat = (c["latency"] - lat_min) / lat_range

            score = (
                w.quality * c["quality"]
                - w.cost * norm_cost
                - w.latency * norm_lat
            )

            if score > best_score:
                best_score = score
                best = c

        result = dict(best)  # type: ignore[arg-type]
        result["pareto_score"] = round(best_score, 6)
        return result

    def rank(
        self,
        candidates: List[Dict[str, Any]],
        weights: Optional[RoutingWeights] = None,
        required_capabilities: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Return all candidates sorted by Pareto score (descending)."""
        if not candidates:
            return []

        w = weights or self._default_weights
        pool = filter_by_capabilities(candidates, required_capabilities or set())

        costs = [c["cost"] for c in pool]
        latencies = [c["latency"] for c in pool]

        cost_min, cost_max = min(costs), max(costs)
        lat_min, lat_max = min(latencies), max(latencies)

        cost_range = cost_max - cost_min if cost_max != cost_min else 1.0
        lat_range = lat_max - lat_min if lat_max != lat_min else 1.0

        scored = []
        for c in pool:
            norm_cost = (c["cost"] - cost_min) / cost_range
            norm_lat = (c["latency"] - lat_min) / lat_range
            score = w.quality * c["quality"] - w.cost * norm_cost - w.latency * norm_lat
            entry = dict(c)
            entry["pareto_score"] = round(score, 6)
            scored.append(entry)

        scored.sort(key=lambda x: x["pareto_score"], reverse=True)
        return scored


# ---------------------------------------------------------------------------
# Build candidates from MODEL_REGISTRY
# ---------------------------------------------------------------------------

def build_candidates_for_tier(tier: str) -> List[Dict[str, Any]]:
    """Build a candidate list from :data:`routing.MODEL_REGISTRY` for a tier.

    Quality heuristic: derived from cost (higher cost ~ higher quality).
    Latency heuristic: derived from model family conventions.
    """
    from nadirclaw.routing import MODEL_REGISTRY
    from nadirclaw.settings import settings

    # Determine which models belong to this tier
    if tier == "simple":
        tier_models = [settings.SIMPLE_MODEL]
    elif tier == "complex":
        tier_models = [settings.COMPLEX_MODEL]
    elif tier == "mid":
        tier_models = [settings.MID_MODEL]
    else:
        tier_models = settings.tier_models

    # Build candidates from all registry models in the tier + fallback chain
    chain = settings.get_tier_fallback_chain(tier)
    all_models = list(dict.fromkeys(tier_models + chain))  # deduplicated, order preserved

    candidates = []
    for model in all_models:
        info = MODEL_REGISTRY.get(model)
        if not info:
            continue

        avg_cost = (info["cost_per_m_input"] + info["cost_per_m_output"]) / 2.0
        # Quality heuristic: log-scale of average cost, capped at 1.0
        quality = min(1.0, 0.3 + 0.15 * avg_cost) if avg_cost > 0 else 0.3
        # Latency heuristic: cheaper / smaller models tend to be faster
        latency = min(1.0, 0.2 + 0.05 * avg_cost) if avg_cost > 0 else 0.2

        candidates.append({
            "model": model,
            "quality": round(quality, 4),
            "cost": round(avg_cost, 4),
            "latency": round(latency, 4),
        })

    return candidates


# Module-level singleton
_optimizer = ParetoOptimizer()


def get_optimizer() -> ParetoOptimizer:
    """Return the module-level :class:`ParetoOptimizer` singleton."""
    return _optimizer
