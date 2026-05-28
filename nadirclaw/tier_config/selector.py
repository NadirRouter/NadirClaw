"""Tier selector: continuous score → tier name.

The selector is the bridge between the classifier (which emits a scalar
``score ∈ [0,1]``) and the cascade dispatcher (which needs a tier name
to start from). For ``mode: tiered`` this is a simple cutoff lookup.
For ``mode: flat`` it's a Pareto-rank + λ-cost-bias lookup over the
arm list, matching R2-Router's selector semantics.

Both selectors are pure functions of ``(score, confidence, profile)`` —
no I/O, no logging, no side effects. The cascade dispatcher owns the
verifier loop and the rule engine; the selector only picks where to
start.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .schema import Tier, TierProfile


@dataclass(frozen=True)
class TierSelection:
    """One tier-selection outcome."""

    tier_name: str
    tier_index: int
    # Reason text — for telemetry and the `nadirclaw tiers dryrun` CLI.
    reason: str


class TierSelector:
    """Stateless wrapper around a :class:`TierProfile`."""

    def __init__(self, profile: TierProfile) -> None:
        self.profile = profile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(
        self,
        score: float,
        confidence: Optional[float] = None,
    ) -> TierSelection:
        """Pick the starting tier for a request.

        ``score`` is the continuous complexity score from the classifier
        (see :mod:`nadirclaw.tier_config.score_adapter` and the
        ``softmax_to_score`` helper on the bundled wide&deep classifier).
        ``confidence`` is currently unused by the OSS selector but kept
        on the API so Pro selectors and rule predicates can read it
        without a signature change.
        """
        # Clamp defensively. The classifier should already emit in [0,1]
        # but if a custom adapter slips through, we don't want a NaN to
        # bypass tier selection.
        s = max(0.0, min(1.0, float(score)))
        if self.profile.mode == "flat":
            return self._assign_flat(s)
        return self._assign_tiered(s)

    def next_tier(
        self,
        current_tier_name: str,
    ) -> Optional[Tier]:
        """Return the next tier to escalate to, or None if terminal.

        Honours ``cascade.escalation``:
        - ``adjacent``: one rung up.
        - ``jump``: top non-terminal tier (last tier in the list).
        """
        profile = self.profile
        idx = profile.tier_index(current_tier_name)
        if idx is None:
            return None
        if profile.is_terminal(current_tier_name):
            return None
        if profile.cascade.escalation == "jump":
            # Jump directly to the last tier (which is always terminal).
            return profile.tiers[-1]
        # Adjacent: next index.
        nxt_idx = idx + 1
        if nxt_idx >= len(profile.tiers):
            return None
        return profile.tiers[nxt_idx]

    # ------------------------------------------------------------------
    # mode: tiered
    # ------------------------------------------------------------------

    def _assign_tiered(self, score: float) -> TierSelection:
        """Highest-cutoff tier whose ``score_min <= score`` wins."""
        chosen_idx = 0
        for i, tier in enumerate(self.profile.tiers):
            if score >= tier.score_min:
                chosen_idx = i
        tier = self.profile.tiers[chosen_idx]
        return TierSelection(
            tier_name=tier.name,
            tier_index=chosen_idx,
            reason=(
                f"score={score:.3f} >= cutoff {tier.score_min:.3f} "
                f"({tier.name})"
            ),
        )

    # ------------------------------------------------------------------
    # mode: flat
    # ------------------------------------------------------------------

    def _assign_flat(self, score: float) -> TierSelection:
        """Pareto-rank arms by (quality desc, cost asc), then λ-blend with score.

        Final arm rank = ``lambda_cost * normalised_score_rank +
        (1 - lambda_cost) * normalised_cost_rank``. ``λ=1.0`` means pure
        score, ``λ=0.0`` means cheapest arm.
        """
        sel = self.profile.selector
        arms = self.profile.tiers
        n = len(arms)
        if n == 1:
            return TierSelection(
                tier_name=arms[0].name,
                tier_index=0,
                reason="single-arm profile",
            )

        # Normalised cost rank: cheapest arm = 0.0, most expensive = 1.0.
        costs: List[Tuple[int, float]] = [
            (i, float(t.cost_per_1k or 0.0)) for i, t in enumerate(arms)
        ]
        costs_sorted = sorted(costs, key=lambda x: x[1])
        cost_rank: List[float] = [0.0] * n
        for rank, (i, _c) in enumerate(costs_sorted):
            cost_rank[i] = rank / max(1, n - 1)

        # Score-driven rank: pretend the score maps linearly to arm index
        # (low score → cheapest arm, high score → most expensive).
        score_rank: List[float] = []
        for i in range(n):
            target = i / max(1, n - 1)
            score_rank.append(abs(target - score))

        lam = float(sel.lambda_cost)
        # Lower score is better in both components (cost_rank: cheap wins;
        # score_rank: distance from target). We want to minimise:
        scored: List[Tuple[float, int]] = []
        for i in range(n):
            blended = lam * score_rank[i] + (1.0 - lam) * cost_rank[i]
            scored.append((blended, i))
        scored.sort()
        chosen_idx = scored[0][1]
        tier = arms[chosen_idx]
        return TierSelection(
            tier_name=tier.name,
            tier_index=chosen_idx,
            reason=(
                f"flat-arm pick: score={score:.3f}, λ={lam:.3f}, "
                f"arm={tier.name}"
            ),
        )


__all__ = ["TierSelection", "TierSelector"]
