#!/usr/bin/env python3
"""Narrow, non-clinical canonicalization for source-linked obligation text.

This module normalizes formatting and explicit interval spelling only. It must
not resolve medication state, timing conflicts, laterality, or clinical scope.
"""

from __future__ import annotations

import re


NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def canonicalize_obligation_text(value: object) -> str:
    """Normalize only spacing, terminal punctuation, follow-up spelling, and explicit intervals."""
    text = " ".join(str(value or "").replace("\r", " ").split()).strip().lower()
    text = re.sub(r"\bfollow[-\s]+up\b", "follow up", text)
    for word, digit in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\s+(day|days|week|weeks|month|months)\b", rf"{digit} \1", text)
    return text.rstrip(".!?;: ")

