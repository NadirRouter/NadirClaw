"""Pydantic schema for N-tier YAML profiles.

Schema is intentionally small. One profile = an ordered list of tiers,
plus a selector config and a cascade policy. Backward compatibility
contract: any field a user does not set falls back to a default that
reproduces today's 3-tier behaviour.

Tiers are ordered cheap → expensive. The first tier with
``score >= score_min`` (scanning from the most-expensive end down) is
the starting tier. The last tier in the list is implicitly terminal.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class Tier(BaseModel):
    """One tier in the cascade ladder."""

    name: str = Field(..., min_length=1)
    score_min: float = Field(..., ge=0.0, le=1.0)
    # Ordered model pool. First entry is the default callee; future
    # `model_pool_strategy` (cheapest/health_aware/round_robin) will
    # consume the rest of the list.
    model_pool: List[str] = Field(..., min_length=1)
    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    # Flat-arm fields (only relevant when profile.mode == "flat"; ignored
    # in tiered mode). Documented in N_TIER_ARCHITECTURE.md §4.
    quality: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cost_per_1k: Optional[float] = Field(default=None, ge=0.0)
    # Mark a tier as terminal — the cascade never escalates past it,
    # even if the verifier rejects. Last tier is automatically terminal.
    terminal: bool = False


class SelectorConfig(BaseModel):
    """Classifier + selector configuration."""

    # Adapter name resolved by the classifier loader. Currently:
    #   wide_deep_asym_v3 | wide_deep_sym_v3 | distilbert | binary
    classifier: str = "wide_deep_asym_v3"
    # 0..1: R2-Router-style cost dominance bias. 1.0 = no bias (use
    # classifier score directly). 0.0 = pick cheapest arm regardless of
    # score. Only consulted in `mode: flat`.
    lambda_cost: float = Field(default=1.0, ge=0.0, le=1.0)
    # When `mode: flat`, rank arms by this metric on the tier object.
    # OSS only ships `cost_quality_score` (computed inline from
    # quality + cost_per_1k). Pro adds learned-ranker options.
    rank_by: str = "cost_quality_score"


class CascadeConfig(BaseModel):
    """Verifier-cascade policy. Mirrors the existing cascade.py defaults."""

    # adjacent | jump. `adjacent` walks one tier at a time on reject
    # (today's behaviour). `jump` always escalates to the top non-terminal
    # tier on the first reject. `learned` is Pro-only.
    escalation: str = "adjacent"
    acceptance_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    # Profile name forwarded to `cascade_rules.load_profile(...)`.
    rules_profile: str = "default"
    # Safety cap: never escalate more than this many hops, even in
    # adjacent mode. None = N-1 (walk the full ladder).
    max_escalations: Optional[int] = Field(default=None, ge=0)
    # Which verifier the cascade should use. `heuristic` (default) uses
    # the rule-based HeuristicVerifier shipped in this repo. `trained`
    # loads NadirRouter/cascade-verifier-v1 from HuggingFace and
    # requires the `trained` extras (pip install nadirclaw[trained]).
    verifier: str = "heuristic"
    # HuggingFace model id or local path for the trained verifier.
    # Only consulted when verifier == "trained". Defaults to the
    # released v1 snapshot.
    verifier_model: str = "NadirRouter/cascade-verifier-v1"

    @model_validator(mode="after")
    def _check_mode(self) -> "CascadeConfig":
        if self.escalation not in ("adjacent", "jump"):
            raise ValueError(
                f"cascade.escalation must be 'adjacent' or 'jump', "
                f"got {self.escalation!r}. ('learned' is a Pro-only mode.)"
            )
        if self.verifier not in ("heuristic", "trained"):
            raise ValueError(
                f"cascade.verifier must be 'heuristic' or 'trained', "
                f"got {self.verifier!r}."
            )
        return self


class TierProfile(BaseModel):
    """One loaded N-tier profile."""

    version: int = 1
    # tiered | flat
    mode: str = "tiered"
    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    tiers: List[Tier] = Field(..., min_length=1)
    # The profile name (filename stem). Set by the loader; not in YAML.
    profile_name: Optional[str] = None

    @model_validator(mode="after")
    def _check_tiers(self) -> "TierProfile":
        if self.mode not in ("tiered", "flat"):
            raise ValueError(
                f"profile.mode must be 'tiered' or 'flat', got {self.mode!r}"
            )

        if self.mode == "tiered":
            # In tiered mode, tiers must be sorted by score_min ascending and
            # the first tier must start at 0.0 so every possible score finds
            # a home. Bad ordering would silently drop low-score prompts.
            mins = [t.score_min for t in self.tiers]
            if mins != sorted(mins):
                raise ValueError(
                    "tiers must be ordered by score_min ascending; got "
                    f"{mins}"
                )
            if abs(self.tiers[0].score_min) > 1e-6:
                raise ValueError(
                    "first tier must have score_min=0.0; got "
                    f"{self.tiers[0].score_min}"
                )

        # In flat mode, quality + cost_per_1k must be set on every tier
        # so the Pareto rank can be computed.
        if self.mode == "flat":
            missing = [
                t.name
                for t in self.tiers
                if t.quality is None or t.cost_per_1k is None
            ]
            if missing:
                raise ValueError(
                    "flat mode requires `quality` and `cost_per_1k` on "
                    f"every tier; missing on: {missing}"
                )

        # Names must be unique.
        names = [t.name for t in self.tiers]
        if len(names) != len(set(names)):
            raise ValueError(f"tier names must be unique; got {names}")

        return self

    @property
    def num_tiers(self) -> int:
        return len(self.tiers)

    def tier_by_name(self, name: str) -> Optional[Tier]:
        for t in self.tiers:
            if t.name == name:
                return t
        return None

    def tier_index(self, name: str) -> Optional[int]:
        for i, t in enumerate(self.tiers):
            if t.name == name:
                return i
        return None

    def is_terminal(self, name: str) -> bool:
        idx = self.tier_index(name)
        if idx is None:
            return True  # unknown tier — fail-closed
        if idx == len(self.tiers) - 1:
            return True
        return self.tiers[idx].terminal


__all__ = [
    "CascadeConfig",
    "SelectorConfig",
    "Tier",
    "TierProfile",
]
