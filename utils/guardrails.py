"""
Input guardrails: block disallowed topics before they reach the model.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Terms that must not be processed as normal user requests (case-insensitive).
BLOCKED_TERMS: tuple[str, ...] = ("hack", "malware", "exploit")

SAFE_REFUSAL = (
    "I cannot help with requests that involve hacking, malware, or exploits. "
    "Please rephrase your question in a safe, legitimate context."
)


def contains_blocked_content(text: str) -> bool:
    """Return True if `text` contains any blocked substring (word-boundary aware)."""
    if not text:
        return False
    lower = text.lower()
    for term in BLOCKED_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lower):
            logger.warning("Guardrail triggered for term=%r", term)
            return True
    return False
