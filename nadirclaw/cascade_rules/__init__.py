"""Generic, data-driven cascade rule engine for NadirClaw.

The engine matches an incoming prompt against a list of declarative YAML
rules and emits a :class:`RuleDecision` that :class:`nadirclaw.cascade.Cascade`
honors before falling through to its default verifier-gated path.

Two common ways to use it:

1. Load a bundled or custom YAML profile::

       from nadirclaw.cascade_rules import load_profile
       engine = load_profile("default")        # bundled
       engine = load_profile("/path/to/my.yaml")  # custom file
       decision = engine.evaluate(prompt, predicted_tier="simple")

2. Build an engine from rule dicts (no YAML needed)::

       from nadirclaw.cascade_rules import load_inline
       engine = load_inline([{
           "name": "force_code_to_complex",
           "priority": 100,
           "match": {"any_of": [{"substring": "```python"}]},
           "action": {"type": "force_escalate", "to_tier": "complex"},
       }])

The full rule schema is documented inline in :mod:`nadirclaw.cascade_rules.engine`.

Public surface:
    - :class:`CascadeRuleEngine` — the evaluator
    - :class:`Rule`               — one parsed rule
    - :class:`RuleDecision`       — the engine's output
    - :func:`load_profile`        — load a named YAML profile (cached, hot-reloadable)
    - :func:`load_inline`         — load a list of rule dicts (per-tenant override)
    - :data:`PROFILES_DIR`        — path to the bundled-profile directory
"""

from .engine import (  # noqa: F401
    CascadeRuleEngine,
    Condition,
    Rule,
    RuleDecision,
    PROFILES_DIR,
    load_inline,
    load_profile,
)

__all__ = [
    "CascadeRuleEngine",
    "Condition",
    "Rule",
    "RuleDecision",
    "PROFILES_DIR",
    "load_inline",
    "load_profile",
]
