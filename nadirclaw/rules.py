"""YAML-based routing rules engine for NadirClaw.

Allows users to pin models, force tiers, or cap costs based on
declarative rules in ``~/.nadirclaw/rules.yaml``.

Supported match conditions:

- ``system_prompt_contains`` — case-insensitive substring match
- ``system_prompt_regex`` — regex match against the system prompt
- ``prompt_contains`` — case-insensitive substring in the last user message
- ``prompt_regex`` — regex against the last user message
- ``time_range`` — HH:MM-HH:MM (wraps past midnight, uses local time)
- ``header`` — ``Header-Name: value`` equality check
- ``tier`` — match the classifier-assigned tier

Supported actions:

- ``force_model`` — bypass classification and use this model
- ``force_tier`` — override the tier (simple / mid / complex / reasoning)
- ``max_cost_per_request`` — cost cap (advisory, stored in metadata)

Rules are evaluated top-to-bottom; the **first** match wins.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nadirclaw.rules")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RuleMatch:
    """Conditions that must *all* be true for the rule to fire."""

    system_prompt_contains: Optional[str] = None
    system_prompt_regex: Optional[str] = None
    prompt_contains: Optional[str] = None
    prompt_regex: Optional[str] = None
    time_range: Optional[str] = None  # "HH:MM-HH:MM"
    header: Optional[str] = None      # "Header-Name: value"
    tier: Optional[str] = None


@dataclass
class RuleAction:
    """What to do when a rule matches."""

    force_model: Optional[str] = None
    force_tier: Optional[str] = None
    max_cost_per_request: Optional[float] = None


@dataclass
class RoutingRule:
    """A single routing rule."""

    name: str
    match: RuleMatch
    action: RuleAction
    enabled: bool = True


@dataclass
class RuleResult:
    """Outcome of rule evaluation."""

    matched: bool
    rule_name: Optional[str] = None
    force_model: Optional[str] = None
    force_tier: Optional[str] = None
    max_cost_per_request: Optional[float] = None


# ---------------------------------------------------------------------------
# Rules engine
# ---------------------------------------------------------------------------

class RoutingRulesEngine:
    """Loads and evaluates YAML routing rules."""

    def __init__(self) -> None:
        self._rules: List[RoutingRule] = []
        self._loaded_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, rules_path: Path) -> None:
        """Load rules from a YAML file.

        If the file does not exist, the engine is initialised with an
        empty rule set (no-op).
        """
        self._rules = []
        self._loaded_path = rules_path

        if not rules_path.exists():
            logger.debug("No rules file at %s — rules engine inactive", rules_path)
            return

        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "PyYAML not installed — cannot load rules from %s. "
                "Install with: pip install pyyaml",
                rules_path,
            )
            return

        try:
            with open(rules_path) as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            logger.error("Failed to parse rules file %s: %s", rules_path, exc)
            return

        if not data or not isinstance(data, dict):
            return

        raw_rules = data.get("rules", [])
        if not isinstance(raw_rules, list):
            logger.error("'rules' key in %s must be a list", rules_path)
            return

        for idx, raw in enumerate(raw_rules):
            try:
                rule = self._parse_rule(raw, idx)
                if rule:
                    self._rules.append(rule)
            except Exception as exc:
                logger.warning("Skipping malformed rule #%d: %s", idx, exc)

        logger.info("Loaded %d routing rule(s) from %s", len(self._rules), rules_path)

    def _parse_rule(self, raw: Dict[str, Any], idx: int) -> Optional[RoutingRule]:
        """Parse a single rule dict into a :class:`RoutingRule`."""
        if not isinstance(raw, dict):
            return None

        name = raw.get("name", f"rule_{idx}")
        enabled = raw.get("enabled", True)
        if not enabled:
            return None

        match_raw = raw.get("match", {})
        action_raw = raw.get("action", {})

        if not isinstance(match_raw, dict) or not isinstance(action_raw, dict):
            logger.warning("Rule '%s' has invalid match/action structure", name)
            return None

        rule_match = RuleMatch(
            system_prompt_contains=match_raw.get("system_prompt_contains"),
            system_prompt_regex=match_raw.get("system_prompt_regex"),
            prompt_contains=match_raw.get("prompt_contains"),
            prompt_regex=match_raw.get("prompt_regex"),
            time_range=match_raw.get("time_range"),
            header=match_raw.get("header"),
            tier=match_raw.get("tier"),
        )

        rule_action = RuleAction(
            force_model=action_raw.get("force_model"),
            force_tier=action_raw.get("force_tier"),
            max_cost_per_request=action_raw.get("max_cost_per_request"),
        )

        # Validate: at least one match condition and one action
        has_match = any(
            getattr(rule_match, f) is not None
            for f in rule_match.__dataclass_fields__
        )
        has_action = any(
            getattr(rule_action, f) is not None
            for f in rule_action.__dataclass_fields__
        )

        if not has_match:
            logger.warning("Rule '%s' has no match conditions — skipping", name)
            return None
        if not has_action:
            logger.warning("Rule '%s' has no actions — skipping", name)
            return None

        return RoutingRule(name=name, match=rule_match, action=rule_action, enabled=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        messages: Optional[list] = None,
        system_prompt: str = "",
        tier: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        """Evaluate all rules against the current request context.

        Returns a :class:`RuleResult`.  If no rule matches,
        ``result.matched`` is ``False`` and all action fields are None.

        Parameters
        ----------
        messages :
            The full message list (ChatMessage objects or dicts).
        system_prompt :
            The extracted system prompt text.
        tier :
            The classifier-assigned tier (for ``tier`` match conditions).
        metadata :
            Dict with optional keys ``"headers"`` (dict of HTTP headers)
            and ``"prompt"`` (last user message text).
        """
        if not self._rules:
            return RuleResult(matched=False)

        meta = metadata or {}
        headers: Dict[str, str] = meta.get("headers", {})

        # Extract last user message text
        prompt_text = meta.get("prompt", "")
        if not prompt_text and messages:
            for m in reversed(messages):
                role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "")
                if role == "user":
                    if hasattr(m, "text_content"):
                        prompt_text = m.text_content()
                    elif isinstance(m, dict):
                        prompt_text = m.get("content", "") or ""
                    break

        for rule in self._rules:
            if self._matches(rule.match, system_prompt, prompt_text, tier, headers):
                logger.info("Routing rule matched: '%s'", rule.name)
                return RuleResult(
                    matched=True,
                    rule_name=rule.name,
                    force_model=rule.action.force_model,
                    force_tier=rule.action.force_tier,
                    max_cost_per_request=rule.action.max_cost_per_request,
                )

        return RuleResult(matched=False)

    def _matches(
        self,
        match: RuleMatch,
        system_prompt: str,
        prompt_text: str,
        tier: str,
        headers: Dict[str, str],
    ) -> bool:
        """Return True if *all* conditions in *match* are satisfied."""

        # system_prompt_contains
        if match.system_prompt_contains is not None:
            if match.system_prompt_contains.lower() not in system_prompt.lower():
                return False

        # system_prompt_regex
        if match.system_prompt_regex is not None:
            try:
                if not re.search(match.system_prompt_regex, system_prompt, re.IGNORECASE):
                    return False
            except re.error as exc:
                logger.warning("Invalid regex in rule: %s", exc)
                return False

        # prompt_contains
        if match.prompt_contains is not None:
            if match.prompt_contains.lower() not in prompt_text.lower():
                return False

        # prompt_regex
        if match.prompt_regex is not None:
            try:
                if not re.search(match.prompt_regex, prompt_text, re.IGNORECASE):
                    return False
            except re.error as exc:
                logger.warning("Invalid regex in rule: %s", exc)
                return False

        # time_range  (HH:MM-HH:MM, local time)
        if match.time_range is not None:
            if not self._check_time_range(match.time_range):
                return False

        # header  ("Header-Name: value")
        if match.header is not None:
            if not self._check_header(match.header, headers):
                return False

        # tier
        if match.tier is not None:
            if match.tier.lower() != tier.lower():
                return False

        return True

    @staticmethod
    def _check_time_range(spec: str) -> bool:
        """Check whether the current local time falls within *spec*.

        *spec* is ``"HH:MM-HH:MM"``.  Ranges that wrap past midnight
        (e.g. ``"18:00-08:00"``) are supported.
        """
        parts = spec.split("-")
        if len(parts) != 2:
            logger.warning("Invalid time_range format: %r (expected HH:MM-HH:MM)", spec)
            return False

        try:
            start_h, start_m = (int(x) for x in parts[0].strip().split(":"))
            end_h, end_m = (int(x) for x in parts[1].strip().split(":"))
        except (ValueError, IndexError):
            logger.warning("Invalid time_range format: %r", spec)
            return False

        now = datetime.now()
        current_minutes = now.hour * 60 + now.minute
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m

        if start_minutes <= end_minutes:
            # Normal range (e.g. 09:00-17:00)
            return start_minutes <= current_minutes < end_minutes
        else:
            # Wraps past midnight (e.g. 18:00-08:00)
            return current_minutes >= start_minutes or current_minutes < end_minutes

    @staticmethod
    def _check_header(spec: str, headers: Dict[str, str]) -> bool:
        """Check ``"Header-Name: expected_value"`` against *headers*.

        Header names are compared case-insensitively.
        """
        if ":" not in spec:
            logger.warning("Invalid header match format: %r (expected 'Name: value')", spec)
            return False

        name, _, expected = spec.partition(":")
        name = name.strip()
        expected = expected.strip()

        # Normalise header names to lowercase for comparison
        normalised = {k.lower(): v for k, v in headers.items()}
        actual = normalised.get(name.lower(), "")

        return actual.strip() == expected

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    @property
    def rules(self) -> List[RoutingRule]:
        return list(self._rules)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: Optional[RoutingRulesEngine] = None


def get_rules_engine() -> RoutingRulesEngine:
    """Return the module-level :class:`RoutingRulesEngine` singleton.

    Lazily loads rules from ``~/.nadirclaw/rules.yaml`` on first access.
    """
    global _engine
    if _engine is None:
        _engine = RoutingRulesEngine()
        rules_path = Path.home() / ".nadirclaw" / "rules.yaml"
        _engine.load(rules_path)
    return _engine


def reload_rules() -> RoutingRulesEngine:
    """Force-reload the rules engine (e.g. after editing rules.yaml)."""
    global _engine
    _engine = None
    return get_rules_engine()
