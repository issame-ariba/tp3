"""
Detect and mask common PII patterns in user-facing text.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE,
)

# Typical OpenAI-style keys (sk-...) and generic long hex/base64-ish API tokens.
_API_KEY_RE = re.compile(
    r"\b(sk-[A-Za-z0-9]{20,}|"
    r"api[_-]?key[\s:=]+[A-Za-z0-9_\-]{16,}|"
    r"Bearer\s+[A-Za-z0-9_\-\.]{20,})\b",
    re.IGNORECASE,
)

# Luhn-valid card detection simplified: groups of 13–19 digits (with optional separators).
_CARD_RE = re.compile(
    r"\b(?:\d[ -]*?){13,19}\b",
)


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def mask_pii(text: str) -> str:
    """
    Mask emails, likely API keys, and credit card numbers in `text`.
    Returns a new string; does not mutate input.
    """
    if not text:
        return text

    out = _EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    out = _API_KEY_RE.sub("[API_KEY_REDACTED]", out)

    def _mask_cards(s: str) -> str:
        def repl(m: re.Match[str]) -> str:
            raw = m.group(0)
            digits_only = re.sub(r"\D", "", raw)
            if 13 <= len(digits_only) <= 19 and _luhn_ok(digits_only):
                logger.info("Masked a credit-card-like sequence (length=%d)", len(digits_only))
                return "[CARD_REDACTED]"
            return raw

        return _CARD_RE.sub(repl, s)

    out = _mask_cards(out)
    return out
