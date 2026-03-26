"""
Prompt injection detection for NadirClaw.

Heuristic-based (Tier 1) detection of adversarial prompt injection patterns.
Designed to catch direct and indirect injection without flagging legitimate
agentic workflows (tool definitions, structured system prompts, multi-turn
conversations).

Action modes (set via NADIRCLAW_PROMPT_GUARD env var):
  - "log"   (default): Log detection, do not block
  - "warn":  Log + add X-Prompt-Guard-Warning response header
  - "block": Return 400 Bad Request on detection
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("nadirclaw.prompt_guard")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ACTION = os.getenv("NADIRCLAW_PROMPT_GUARD", "log").lower()
if _ACTION not in ("log", "warn", "block"):
    raise ValueError(
        f"Invalid NADIRCLAW_PROMPT_GUARD={_ACTION!r}; expected log|warn|block"
    )

# ---------------------------------------------------------------------------
# Detection patterns
#
# IMPORTANT: These patterns target adversarial override attempts specifically.
# They must NOT match legitimate agentic patterns like:
#   - Tool definitions with "instructions" fields
#   - Multi-turn conversations with assistant/system roles
#   - Structured delimiters used in prompt templates
# ---------------------------------------------------------------------------


@dataclass
class InjectionSignal:
    """A detected injection signal with confidence and context."""
    pattern_name: str
    matched_text: str
    confidence: float  # 0.0 to 1.0
    message_index: int  # Which message in the array


# Case-insensitive patterns that strongly indicate prompt injection
# Each tuple: (pattern_name, compiled_regex, confidence)
_INJECTION_PATTERNS: List[tuple] = [
    # Direct instruction override
    (
        "instruction_override",
        re.compile(
            r"(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+)?(?:previous|above|prior|earlier|your|the)\s+"
            r"(?:instructions?|rules?|prompts?|guidelines?|constraints?|context)",
            re.IGNORECASE,
        ),
        0.95,
    ),
    # Role reassignment
    (
        "role_reassignment",
        re.compile(
            r"(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are|"
            r"act\s+as\s+if\s+you\s+are|pretend\s+(?:to\s+be|you\s+are)|"
            r"i\s+want\s+you\s+to\s+(?:act|behave|respond)\s+as)",
            re.IGNORECASE,
        ),
        0.80,
    ),
    # System prompt extraction
    (
        "prompt_extraction",
        re.compile(
            r"(?:repeat|show|reveal|display|print|output|leak|give\s+me|tell\s+me)\s+"
            r"(?:your|the|all|entire)?\s*"
            r"(?:system\s+(?:prompt|message|instructions?)|"
            r"initial\s+(?:prompt|instructions?)|"
            r"(?:hidden|secret)\s+(?:prompt|instructions?|rules?))",
            re.IGNORECASE,
        ),
        0.90,
    ),
    # Role confusion in user message (embedding JSON role objects)
    (
        "role_confusion_json",
        re.compile(
            r'\{\s*"role"\s*:\s*"(?:system|assistant)"\s*,\s*"content"\s*:',
            re.IGNORECASE,
        ),
        0.70,
    ),
    # Delimiter injection (attempting to break out of sandboxed context)
    (
        "delimiter_injection",
        re.compile(
            r"(?:^|\n)\s*(?:<\|(?:im_start|im_end|system|endoftext)\|>|"
            r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|"
            r"### (?:System|Human|Assistant|Instruction):)",
            re.IGNORECASE | re.MULTILINE,
        ),
        0.85,
    ),
    # Encoded payload detection (base64 blocks that look like instructions)
    (
        "encoded_payload",
        re.compile(
            r"(?:decode|eval|execute|run)\s+(?:this|the\s+following)?\s*"
            r"(?:base64|b64|encoded|hex)\s*[:\-]?\s*[A-Za-z0-9+/=]{40,}",
            re.IGNORECASE,
        ),
        0.75,
    ),
    # "DAN" / jailbreak patterns
    (
        "jailbreak_dan",
        re.compile(
            r"(?:DAN\s+mode|Developer\s+Mode|(?:Do\s+)?Anything\s+Now|"
            r"STAN\s+mode|anti-?DAN|maximum\s+mode|god\s+mode)",
            re.IGNORECASE,
        ),
        0.90,
    ),
]


def scan_messages(
    messages: list,
    threshold: float = 0.70,
) -> Optional[InjectionSignal]:
    """
    Scan a list of ChatMessage objects for prompt injection signals.

    Only scans user-role messages (system and assistant messages are trusted
    in the agentic context — they come from the operator or the model itself).

    Returns the highest-confidence signal above threshold, or None.
    """
    best_signal: Optional[InjectionSignal] = None

    for idx, msg in enumerate(messages):
        # Only scan user and tool messages — system/assistant are trusted
        if msg.role not in ("user", "tool"):
            continue

        text = msg.text_content() if hasattr(msg, "text_content") else str(getattr(msg, "content", ""))
        if not text:
            continue

        for pattern_name, regex, confidence in _INJECTION_PATTERNS:
            match = regex.search(text)
            if match and confidence >= threshold:
                signal = InjectionSignal(
                    pattern_name=pattern_name,
                    matched_text=match.group(0)[:100],
                    confidence=confidence,
                    message_index=idx,
                )
                if best_signal is None or signal.confidence > best_signal.confidence:
                    best_signal = signal

    return best_signal


def check_and_act(messages: list) -> Optional[InjectionSignal]:
    """
    Run prompt injection scan and take configured action.

    Returns the signal if detected (caller should add header in "warn" mode
    or return 400 in "block" mode).
    """
    signal = scan_messages(messages)
    if signal is None:
        return None

    logger.warning(
        "Prompt injection detected: pattern=%s confidence=%.2f msg_idx=%d matched=%r",
        signal.pattern_name,
        signal.confidence,
        signal.message_index,
        signal.matched_text,
    )

    return signal


def should_block() -> bool:
    """Whether the configured action is 'block'."""
    return _ACTION == "block"


def should_warn() -> bool:
    """Whether the configured action is 'warn' (add header)."""
    return _ACTION == "warn"
