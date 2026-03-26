"""
PII redaction for LLM output.

Scans LLM responses for common PII patterns (email, phone, SSN, credit card)
and redacts them before returning to the user.

Configuration:
  NADIRCLAW_PII_REDACTION: "none" (default) | "log_only" | "redact"

Designed to run on non-streaming responses only. Streaming responses
cannot be reliably redacted due to partial regex matches across chunks.
"""

import logging
import os
import re
from typing import Tuple

logger = logging.getLogger("nadirclaw.pii_redactor")

_MODE = os.getenv("NADIRCLAW_PII_REDACTION", "none").lower()

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

# Email addresses
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

# US phone numbers (various formats)
_PHONE_RE = re.compile(
    r"(?<!\d)"  # No digit before
    r"(?:\+?1[-.\s]?)?"  # Optional country code
    r"(?:\(?\d{3}\)?[-.\s]?)"  # Area code
    r"\d{3}[-.\s]?\d{4}"  # Number
    r"(?!\d)"  # No digit after
)

# US Social Security Numbers
_SSN_RE = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

# Credit card numbers (basic pattern, Luhn-validated)
_CC_RE = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)


def _luhn_check(number: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


_PATTERNS = [
    ("email", _EMAIL_RE, "[REDACTED_EMAIL]"),
    ("phone", _PHONE_RE, "[REDACTED_PHONE]"),
    ("ssn", _SSN_RE, "[REDACTED_SSN]"),
    ("credit_card", _CC_RE, "[REDACTED_CC]"),
]


def scan_pii(text: str) -> list[dict]:
    """Scan text for PII patterns. Returns list of detected PII with type and location."""
    findings = []
    for pii_type, pattern, _ in _PATTERNS:
        for match in pattern.finditer(text):
            # Extra Luhn validation for credit cards
            if pii_type == "credit_card":
                if not _luhn_check(match.group(0)):
                    continue
            findings.append({
                "type": pii_type,
                "start": match.start(),
                "end": match.end(),
            })
    return findings


def redact_pii(text: str) -> Tuple[str, bool]:
    """Redact PII from text based on configured mode.

    Returns:
        Tuple of (possibly_redacted_text, pii_was_found).
    """
    if _MODE == "none":
        return text, False

    findings = scan_pii(text)
    if not findings:
        return text, False

    # Log findings
    types_found = set(f["type"] for f in findings)
    logger.warning("PII detected in LLM output: types=%s count=%d", types_found, len(findings))

    if _MODE == "log_only":
        return text, True

    # Mode is "redact" — replace all PII matches
    result = text
    # Process in reverse order to preserve character offsets
    for pii_type, pattern, replacement in _PATTERNS:
        if pii_type == "credit_card":
            # Luhn-validated replacement
            def _cc_replacer(match):
                if _luhn_check(match.group(0)):
                    return replacement
                return match.group(0)
            result = pattern.sub(_cc_replacer, result)
        else:
            result = pattern.sub(replacement, result)

    return result, True
