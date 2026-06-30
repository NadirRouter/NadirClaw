"""N-tier YAML tier configuration for NadirClaw (free / PolyForm Noncommercial).

The tier_config package owns the "how many tiers, with which cutoffs,
mapped to which model pools" decision. It is the open-source half of
the N-tier architecture documented in ``N_TIER_ARCHITECTURE.md``: one
continuous classifier score in [0,1] is sliced into N tiers by YAML
cutoffs, the cascade walks adjacent tiers under the verifier, and the
default profile (N=2) is the cheap+strong configuration that won the
RouterArena bake-off.

The OSS surface ships:

- :class:`TierProfile` — the loaded, immutable profile object (number of
  tiers, cutoffs, model pools, escalation policy).
- :class:`TierSelector` — picks the starting tier from a continuous
  ``score`` (and optionally a confidence value used by rule predicates).
- :func:`load_profile` — TTL+mtime-cached YAML loader. Edits to a
  profile YAML are picked up within 30 s without restarting.
- :func:`get_default_profile_path` — resolves the active profile from
  ``NADIRCLAW_TIERS_PROFILE`` (default = ``n2_default.yaml``).

Two bundled profiles live under ``nadirclaw/tier_config/profiles/``:

- ``n2_default.yaml`` — the new default. Cheap pool (gpt-4o-mini,
  qwen3, deepseek-v3.2, claude-haiku) + strong pool (gpt-5-mini,
  deepseek-reasoner, deepseek-v4-flash, grok-4-1-fast-reasoning,
  claude-sonnet-4). Verifier τ=0.80, adjacent escalation.
- ``n3_legacy.yaml`` — the previous 3-tier behaviour, kept for
  backward compatibility. Use ``NADIRCLAW_TIERS_PROFILE=n3_legacy`` or
  let the migration helper auto-generate one from the legacy
  ``NADIRCLAW_SIMPLE_MODEL`` / ``NADIRCLAW_MID_MODEL`` /
  ``NADIRCLAW_COMPLEX_MODEL`` env vars.

Pro features (``nadir`` package): per-tenant Supabase overrides, learned
tier selector, provider-health-aware pool resolution. None of those
ship here.
"""

from .loader import (  # noqa: F401
    PROFILES_DIR,
    get_default_profile_path,
    load_profile,
)
from .schema import (  # noqa: F401
    CascadeConfig,
    SelectorConfig,
    Tier,
    TierProfile,
)
from .score_adapter import probs_dict_to_score, softmax_to_score  # noqa: F401
from .selector import TierSelection, TierSelector  # noqa: F401

__all__ = [
    "CascadeConfig",
    "PROFILES_DIR",
    "SelectorConfig",
    "Tier",
    "TierProfile",
    "TierSelection",
    "TierSelector",
    "get_default_profile_path",
    "load_profile",
    "probs_dict_to_score",
    "softmax_to_score",
]
