"""Verifier-gated cascade dispatch for NadirClaw (free / MIT).

Mirrors the architecture of `nadir.cascade_router` from Nadir Pro, but
uses the rule-based `HeuristicVerifier` instead of the proprietary
DeBERTa cross-encoder. The dispatch contract is:

    1. (Optional) Consult the cascade rule engine on the prompt. A
       `force_escalate` decision routes straight to the expensive tier
       and skips verification; a `force_cheap` decision routes straight
       to the cheap tier and skips verification; a `set_threshold`
       decision overrides the per-request acceptance cut.
    2. Call the cheap tier with the user's messages.
    3. Score the cheap response with the verifier.
    4. If score >= threshold -> ship the cheap response.
    5. Else -> call the expensive tier and ship that response.

Two safety nets are wired in:
    * fail-open: any verifier exception is treated as accept (cheap
      response is shipped); the cascade never fails the request.
    * kill switch: after `KILL_SWITCH_THRESHOLD` consecutive verifier
      errors, the cascade short-circuits to cheap-only until the
      process restarts.

The Pro version of this code (in `getnadir.dev/backend/app/services/
cascade_router.py`) adds:
    * trained DeBERTa-v3-small cross-encoder verifier (AUROC 0.96 on
      held-out RouterBench, vs ~0.6 for the heuristic),
    * pre-generation classifier shortcut on high-confidence cases
      (router_v2.pkl, RouterBench-trained),
    * Supabase decision logging for closed-loop retraining.

The rule-engine integration in step (1) is shipped in NadirClaw Free.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from nadirclaw.heuristic_verifier import HeuristicVerifier, get_heuristic_verifier

try:
    from nadirclaw.cascade_rules import CascadeRuleEngine, RuleDecision
except ImportError:  # pragma: no cover — defensive; cascade_rules ships in-tree
    CascadeRuleEngine = None  # type: ignore[assignment]
    RuleDecision = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


KILL_SWITCH_THRESHOLD: int = 3

# Default acceptance threshold for the verifier-gated cascade.
#
# Calibrated against the held-out RouterBench test split (n=11,420, see
# the Nadir Pro held-out report). The trained DeBERTa verifier sweep gave:
#
#   τ=0.70  → accept 0.69, catastrophic 0.024, wasted 0.078, quality preserved 97.6%
#   τ=0.75  → accept 0.67, catastrophic 0.019, wasted 0.089, quality preserved 98.1%
#   τ=0.80  → accept 0.67, catastrophic 0.017, wasted 0.092, quality preserved 98.3%
#   τ=0.90  → accept 0.64, catastrophic 0.011, wasted 0.108, quality preserved 98.9%
#
# 0.80 is the production operating point that matches the "98% of always-Opus
# quality preserved" claim (catastrophic ≤ 1.7%). 0.70 was the earlier
# default and is still appropriate for cost-sensitive workloads that can
# tolerate slightly higher downgrade rates.
#
# NB: the heuristic verifier shipped in this file (HeuristicVerifier) is
# weaker than the trained DeBERTa verifier (~0.60 AUROC vs 0.96), so the
# operating-point tradeoff curve will look different. 0.80 is still the
# right default for the cascade dispatch itself: it ensures that any
# borderline score escalates, which is the safer choice with a weaker
# verifier. Tune via `Cascade(..., threshold=...)` if you need a
# different acceptance bar.
DEFAULT_ACCEPTANCE_THRESHOLD: float = 0.80


@dataclass
class CascadeDecision:
    """Outcome of one cascade dispatch."""

    final_tier: str
    response: Any = None
    verifier_score: Optional[float] = None
    accepted: bool = True
    escalated: bool = False
    reasons: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    # Names of cascade-rule-engine rules that fired for this request,
    # in evaluation order. Empty list when the engine returned no
    # matches or no engine was supplied.
    matched_rules: list[str] = field(default_factory=list)


class Cascade:
    """Cheap-first dispatch with heuristic post-hoc verification.

    Parameters
    ----------
    cheap_call, expensive_call:
        Callables (sync or async) of signature ``(messages, **kwargs)
        -> response``. NadirClaw's existing ``claude_integration`` and
        ``ollama_discovery`` modules return text directly; pass either
        the text or a dict with a "text" field. ``_extract_text``
        handles both.
    verifier:
        Defaults to the package-level `HeuristicVerifier`. Pass a custom
        one to extend the rules (e.g. domain-specific patterns).
    threshold:
        Acceptance cutoff. Below this, escalate.
    """

    def __init__(
        self,
        cheap_call: Callable[..., Any],
        expensive_call: Callable[..., Any],
        verifier: Optional[HeuristicVerifier] = None,
        threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
        rule_engine: Optional["CascadeRuleEngine"] = None,
    ) -> None:
        self.cheap_call = cheap_call
        self.expensive_call = expensive_call
        self.verifier = verifier or get_heuristic_verifier(threshold=threshold)
        self.threshold = float(threshold)
        self.rule_engine = rule_engine
        self._consecutive_errors: int = 0
        self._kill_switch: bool = False

    # ------------------------------------------------------------------
    # Rule-engine pre-check (shared by sync + async paths)
    # ------------------------------------------------------------------

    def _consult_rules(
        self, prompt_text: str
    ) -> tuple[Optional["RuleDecision"], float]:
        """Run the rule engine (if any) and return (decision, threshold).

        ``threshold`` is the verifier acceptance bar to use for this
        request — either the default for this cascade or the strictest
        threshold returned by any ``set_threshold`` rule that matched.
        """
        if self.rule_engine is None or CascadeRuleEngine is None:
            return None, self.threshold
        try:
            decision = self.rule_engine.evaluate(prompt_text)
        except Exception as e:  # noqa: BLE001 — never let the engine fail the request
            logger.warning("CascadeRuleEngine.evaluate failed; ignoring: %s", e)
            return None, self.threshold
        eff_threshold = self.threshold
        if decision.threshold is not None:
            # set_threshold rules can only RAISE the bar, never lower it.
            eff_threshold = max(self.threshold, float(decision.threshold))
        return decision, eff_threshold

    def dispatch_sync(
        self,
        messages: list[dict],
        prompt_text: str,
        expect_json: bool = False,
        cheap_tier_name: str = "cheap",
        expensive_tier_name: str = "expensive",
        **call_kwargs,
    ) -> CascadeDecision:
        """Synchronous dispatch. Use `dispatch` for async callsites."""
        if self._kill_switch:
            cheap_response = self.cheap_call(messages, **call_kwargs)
            return CascadeDecision(
                final_tier=cheap_tier_name,
                response=cheap_response,
                accepted=True,
                escalated=False,
                meta={"cascade_skipped": "kill_switch"},
            )

        rule_decision, effective_threshold = self._consult_rules(prompt_text)
        matched_rules = list(rule_decision.matched_rules) if rule_decision else []

        # Rule-driven short-circuits skip the verifier entirely.
        if rule_decision is not None:
            if rule_decision.action == "force_escalate":
                expensive_response = self.expensive_call(messages, **call_kwargs)
                return CascadeDecision(
                    final_tier=expensive_tier_name,
                    response=expensive_response,
                    accepted=False,
                    escalated=True,
                    reasons=[f"rule:{n}" for n in matched_rules],
                    meta={
                        "rule_action": "force_escalate",
                        "rule_to_tier": rule_decision.to_tier,
                        "rule_profile": rule_decision.profile_name,
                    },
                    matched_rules=matched_rules,
                )
            if rule_decision.action == "force_cheap":
                cheap_response = self.cheap_call(messages, **call_kwargs)
                return CascadeDecision(
                    final_tier=cheap_tier_name,
                    response=cheap_response,
                    accepted=True,
                    escalated=False,
                    reasons=[f"rule:{n}" for n in matched_rules],
                    meta={
                        "rule_action": "force_cheap",
                        "rule_to_tier": rule_decision.to_tier,
                        "rule_profile": rule_decision.profile_name,
                    },
                    matched_rules=matched_rules,
                )

        cheap_response = self.cheap_call(messages, **call_kwargs)
        cheap_text = _extract_text(cheap_response)

        try:
            # Use a per-request verifier instance when the threshold was
            # raised by a rule. Constructing the verifier is cheap; this
            # avoids mutating the shared singleton. We only swap when
            # the configured verifier exposes a `threshold` attribute
            # AND the rule-derived threshold differs — custom verifiers
            # (e.g. those passed in tests or by Nadir Pro) keep their
            # own threshold semantics.
            base_threshold = getattr(self.verifier, "threshold", None)
            verifier = (
                HeuristicVerifier(threshold=effective_threshold)
                if base_threshold is not None
                and effective_threshold != base_threshold
                else self.verifier
            )
            result = verifier.score(
                prompt=prompt_text,
                cheap_answer=cheap_text,
                expect_json=expect_json,
            )
            self._consecutive_errors = 0
        except Exception as e:  # noqa: BLE001
            logger.warning("HeuristicVerifier scoring failed; failing open: %s", e)
            self._consecutive_errors += 1
            if self._consecutive_errors >= KILL_SWITCH_THRESHOLD:
                self._kill_switch = True
                logger.error(
                    "Cascade kill switch tripped after %d consecutive verifier errors",
                    self._consecutive_errors,
                )
            return CascadeDecision(
                final_tier=cheap_tier_name,
                response=cheap_response,
                verifier_score=None,
                accepted=True,
                escalated=False,
                meta={"verifier_error": type(e).__name__},
                matched_rules=matched_rules,
            )

        if result.accepted:
            return CascadeDecision(
                final_tier=cheap_tier_name,
                response=cheap_response,
                verifier_score=result.score,
                accepted=True,
                escalated=False,
                reasons=result.reasons,
                meta=result.to_dict(),
                matched_rules=matched_rules,
            )

        # Reject -> escalate.
        expensive_response = self.expensive_call(messages, **call_kwargs)
        return CascadeDecision(
            final_tier=expensive_tier_name,
            response=expensive_response,
            verifier_score=result.score,
            accepted=False,
            escalated=True,
            reasons=result.reasons,
            meta=result.to_dict(),
            matched_rules=matched_rules,
        )

    async def dispatch(
        self,
        messages: list[dict],
        prompt_text: str,
        expect_json: bool = False,
        cheap_tier_name: str = "cheap",
        expensive_tier_name: str = "expensive",
        **call_kwargs,
    ) -> CascadeDecision:
        """Async-friendly dispatch. Either ``cheap_call`` or
        ``expensive_call`` may be a coroutine; both are awaited if so.
        """
        if self._kill_switch:
            cheap_response = await _maybe_await(self.cheap_call(messages, **call_kwargs))
            return CascadeDecision(
                final_tier=cheap_tier_name,
                response=cheap_response,
                accepted=True,
                escalated=False,
                meta={"cascade_skipped": "kill_switch"},
            )

        rule_decision, effective_threshold = self._consult_rules(prompt_text)
        matched_rules = list(rule_decision.matched_rules) if rule_decision else []

        if rule_decision is not None:
            if rule_decision.action == "force_escalate":
                expensive_response = await _maybe_await(
                    self.expensive_call(messages, **call_kwargs)
                )
                return CascadeDecision(
                    final_tier=expensive_tier_name,
                    response=expensive_response,
                    accepted=False,
                    escalated=True,
                    reasons=[f"rule:{n}" for n in matched_rules],
                    meta={
                        "rule_action": "force_escalate",
                        "rule_to_tier": rule_decision.to_tier,
                        "rule_profile": rule_decision.profile_name,
                    },
                    matched_rules=matched_rules,
                )
            if rule_decision.action == "force_cheap":
                cheap_response = await _maybe_await(
                    self.cheap_call(messages, **call_kwargs)
                )
                return CascadeDecision(
                    final_tier=cheap_tier_name,
                    response=cheap_response,
                    accepted=True,
                    escalated=False,
                    reasons=[f"rule:{n}" for n in matched_rules],
                    meta={
                        "rule_action": "force_cheap",
                        "rule_to_tier": rule_decision.to_tier,
                        "rule_profile": rule_decision.profile_name,
                    },
                    matched_rules=matched_rules,
                )

        cheap_response = await _maybe_await(self.cheap_call(messages, **call_kwargs))
        cheap_text = _extract_text(cheap_response)

        try:
            verifier = (
                HeuristicVerifier(threshold=effective_threshold)
                if effective_threshold != self.verifier.threshold
                else self.verifier
            )
            result = verifier.score(
                prompt=prompt_text,
                cheap_answer=cheap_text,
                expect_json=expect_json,
            )
            self._consecutive_errors = 0
        except Exception as e:  # noqa: BLE001
            logger.warning("HeuristicVerifier scoring failed; failing open: %s", e)
            self._consecutive_errors += 1
            if self._consecutive_errors >= KILL_SWITCH_THRESHOLD:
                self._kill_switch = True
            return CascadeDecision(
                final_tier=cheap_tier_name,
                response=cheap_response,
                accepted=True,
                escalated=False,
                meta={"verifier_error": type(e).__name__},
                matched_rules=matched_rules,
            )

        if result.accepted:
            return CascadeDecision(
                final_tier=cheap_tier_name,
                response=cheap_response,
                verifier_score=result.score,
                accepted=True,
                escalated=False,
                reasons=result.reasons,
                meta=result.to_dict(),
                matched_rules=matched_rules,
            )

        expensive_response = await _maybe_await(self.expensive_call(messages, **call_kwargs))
        return CascadeDecision(
            final_tier=expensive_tier_name,
            response=expensive_response,
            verifier_score=result.score,
            accepted=False,
            escalated=True,
            reasons=result.reasons,
            meta=result.to_dict(),
            matched_rules=matched_rules,
        )


async def _maybe_await(x: Any) -> Any:
    import inspect

    if inspect.isawaitable(x):
        return await x
    return x


def _extract_text(response: Any) -> str:
    """Pull plain text from common response shapes.

    NadirClaw's `claude_integration.chat()` returns a string today;
    other integrations return dicts. We accept both.
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        # Anthropic-ish.
        if "content" in response and isinstance(response["content"], str):
            return response["content"]
        # OpenAI-ish.
        if "choices" in response and response["choices"]:
            msg = response["choices"][0].get("message") or {}
            c = msg.get("content")
            if isinstance(c, str):
                return c
        # Best-effort fallback.
        return str(response.get("text") or response.get("response") or "")
    return str(response)
