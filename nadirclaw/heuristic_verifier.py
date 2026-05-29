"""Heuristic post-hoc response verifier for NadirClaw.

This is the free / MIT-licensed companion to the verifier-gated cascade
architecture. It scores whether a cheap model's response is "good enough"
relative to a prompt, using rule-based checks only -- no ML weights, no
external dependencies, no network calls. Latency is < 1 ms per call.

The trained DeBERTa-v3-small cross-encoder used in Nadir Pro reaches
AUROC 0.96 on held-out RouterBench triples. This heuristic verifier
reaches roughly 0.60-0.65 AUROC on the same data, but catches the bulk
of obvious failures (refusals, truncation, JSON parse failure, "I don't
know" replies) that account for the majority of cheap-model errors in
production. For subtle quality failures -- "looks right but is factually
wrong" -- the heuristic cannot help; upgrade to Nadir Pro for the
trained verifier.

Score semantics (mirror the trained verifier):
    1.0 = cheap response is fully acceptable
    0.0 = cheap response should be escalated to the more capable tier

The default acceptance threshold is now 0.80, calibrated against the
held-out RouterBench test split. See the calibration block in
``nadirclaw.cascade.DEFAULT_ACCEPTANCE_THRESHOLD`` for the full sweep
and tradeoffs. Older NadirClaw versions defaulted to 0.5; pass an
explicit ``threshold=0.5`` to restore the legacy behaviour.

This module is intentionally dependency-light: regex + stdlib only. It
runs identically on Python 3.10+ with no install step beyond NadirClaw's
existing requirements.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional


# Refusal / hedging patterns. Hits on any of these are strong signals
# that the cheap model declined to answer; we escalate. Case-insensitive
# substring match. The patterns are ordered by approximate prevalence in
# real refusal traffic we have inspected; earlier entries hit more often.
_REFUSAL_PATTERNS: tuple[str, ...] = (
    "i cannot help",
    "i can't help",
    "i'm not able to",
    "i am not able to",
    "i cannot provide",
    "i can't provide",
    "i'm unable to",
    "i am unable to",
    "i do not have access",
    "i don't have access",
    "i cannot assist",
    "i can't assist",
    "as an ai language model",
    "as an ai assistant",
    "i'm just an ai",
    "i'm sorry, but",
    "i apologize, but",
)

# Soft "I don't know" patterns. Weaker signal than refusals; we
# count them but require co-occurrence with shortness for an
# escalation.
_UNCERTAINTY_PATTERNS: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "it's unclear",
    "it is unclear",
    "i'm uncertain",
)

# JSON-format failure cues. When the caller asked for JSON and the
# response is unparseable, we want to escalate even if the prose
# response is otherwise long. Detected by the caller via `expect_json=True`.

# Minimum acceptable response length, in characters, for a non-trivial
# prompt. A response shorter than this against a long prompt is usually
# a truncation or refusal masquerading as a short answer.
_MIN_RESPONSE_LENGTH_RATIO: float = 0.05  # response >= 5% of prompt length

# Hard minimum: regardless of prompt length, a useful response is rarely
# shorter than this many characters.
_HARD_MIN_RESPONSE_CHARS: int = 16


@dataclass
class HeuristicScore:
    """Structured score with the reason flags that drove it.

    Exposed as a dict via `to_dict()` so consumers can log the per-check
    breakdown to their analytics layer for offline calibration.
    """

    score: float
    accepted: bool
    threshold: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "accepted": self.accepted,
            "threshold": self.threshold,
            "reasons": list(self.reasons),
            "verifier": "heuristic",
        }


class HeuristicVerifier:
    """Rule-based verifier. Same interface shape as the Pro verifier.

    Parameters
    ----------
    threshold:
        Score cutoff for acceptance. Default 0.80 matches the production
        operating point of the trained verifier on the held-out
        RouterBench test split (catastrophic ≤ 1.7%, quality preserved
        ≥ 98.3%). Pass ``threshold=0.5`` for the legacy NadirClaw
        default if you need a more permissive cut.
    """

    VERIFIER_NAME = "heuristic"

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = float(threshold)

    def is_available(self) -> bool:
        # No weights to load -- always available, by design.
        return True

    def score(
        self,
        prompt: str,
        cheap_answer: str,
        reference_answer: Optional[str] = None,
        expect_json: bool = False,
    ) -> HeuristicScore:
        """Score how acceptable `cheap_answer` is.

        `reference_answer` is accepted for interface parity with the
        Pro verifier but ignored: this heuristic does not compare
        against an expensive-tier reference. Use Nadir Pro for the
        cross-encoder comparison signal.
        """
        prompt = prompt or ""
        cheap = (cheap_answer or "").strip()
        reasons: list[str] = []
        # Start at 1.0 (accept). Each check that triggers subtracts a
        # weight calibrated so any single hard failure tips the
        # final score below 0.5.
        score = 1.0

        # 1. Empty response -> always reject.
        if not cheap:
            return HeuristicScore(
                score=0.0,
                accepted=False,
                threshold=self.threshold,
                reasons=["empty_response"],
            )

        # 2. Hard refusals: strong subtraction.
        cheap_lc = cheap.lower()
        for pat in _REFUSAL_PATTERNS:
            if pat in cheap_lc:
                score -= 0.7
                reasons.append(f"refusal:{pat!r}")
                break  # one is enough

        # 3. Soft uncertainty + short response: moderate subtraction.
        # Calibrated so any single hard signal (uncertainty in a short
        # reply) lands the score strictly below the 0.5 cut.
        for pat in _UNCERTAINTY_PATTERNS:
            if pat in cheap_lc:
                if len(cheap) < 200:
                    score -= 0.6
                    reasons.append(f"uncertainty_short:{pat!r}")
                else:
                    score -= 0.15
                    reasons.append(f"uncertainty:{pat!r}")
                break

        # 4. Hard-min length. Larger penalty so a trivial reply alone
        # (e.g. "ok") tips below threshold.
        if len(cheap) < _HARD_MIN_RESPONSE_CHARS:
            score -= 0.6
            reasons.append(f"too_short:{len(cheap)}")

        # 5. Ratio length: response shorter than 5% of a long prompt is
        # almost always truncation or trivial refusal. Skip the ratio
        # check for very short prompts where the rule is meaningless.
        if len(prompt) > 200:
            ratio = len(cheap) / max(len(prompt), 1)
            if ratio < _MIN_RESPONSE_LENGTH_RATIO:
                score -= 0.6
                reasons.append(f"low_ratio:{ratio:.3f}")

        # 6. JSON parse failure (caller-flagged expectation).
        if expect_json and not _looks_like_json(cheap):
            score -= 0.6
            reasons.append("json_parse_failed")

        # Clamp to [0, 1].
        score = max(0.0, min(1.0, score))
        return HeuristicScore(
            score=score,
            accepted=score >= self.threshold,
            threshold=self.threshold,
            reasons=reasons,
        )


def _looks_like_json(text: str) -> bool:
    """Cheap parse-attempt. Trims leading prose if any."""
    if not text:
        return False
    # Find first { or [ and try to parse from there.
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                json.loads(text[i:])
                return True
            except (ValueError, json.JSONDecodeError):
                # Try a more forgiving variant: trim trailing prose.
                # Find the matching closing brace.
                opener = ch
                closer = "}" if opener == "{" else "]"
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == opener:
                        depth += 1
                    elif text[j] == closer:
                        depth -= 1
                        if depth == 0:
                            try:
                                json.loads(text[i : j + 1])
                                return True
                            except (ValueError, json.JSONDecodeError):
                                break
                return False
    return False


# Module-level singleton for callers that want the default config.
_shared_verifier: Optional[HeuristicVerifier] = None


def get_heuristic_verifier(threshold: float = 0.80) -> HeuristicVerifier:
    """Return the package-level shared `HeuristicVerifier`."""
    global _shared_verifier
    if _shared_verifier is None or _shared_verifier.threshold != threshold:
        _shared_verifier = HeuristicVerifier(threshold=threshold)
    return _shared_verifier
