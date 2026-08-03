"""Conservative medication component extraction for contract auditing."""

from __future__ import annotations

import re

from parse_note_sections import normalize_text


DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?|ml)\b", re.IGNORECASE)
ROUTE = re.compile(r"\b(?:oral|orally|po|iv|intravenous|im|subcutaneous|topical|inhaled)\b", re.IGNORECASE)
FREQUENCY = re.compile(r"\b(?:daily|twice daily|three times daily|every \d+\s*(?:hours?|days?)|after hemodialysis|at bedtime|as needed|prn)\b", re.IGNORECASE)
ACTION = re.compile(r"\b(?:continue|start|stop|discontinue|hold|resume)\b", re.IGNORECASE)


def medication_components(value: object) -> dict[str, str]:
    """Extract only explicit components; absent values remain absent, never guessed."""
    raw = str(value)
    normalized = normalize_text(raw)
    dose = DOSE.search(raw)
    route = ROUTE.search(raw)
    frequency = FREQUENCY.search(raw)
    action = ACTION.search(raw)
    # The leading alpha phrase before a dose/action is a conservative name candidate.
    name_match = re.search(r"\b([A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,2})\b", raw)
    return {
        "name": normalize_text(name_match.group(1)) if name_match else "",
        "action": normalize_text(action.group(0)) if action else "",
        "dose": normalize_text(dose.group(0)) if dose else "",
        "route": normalize_text(route.group(0)) if route else "",
        "frequency_or_timing": normalize_text(frequency.group(0)) if frequency else "",
        "raw": normalized,
    }


def reviewer_components(value: object, fallback_value: object) -> dict[str, str]:
    """Use explicit clinician contract components when supplied.

    `not specified` and `not applicable` intentionally mean no required string
    match; they prevent the audit from inventing a dose, route, or schedule.
    A `forbid=` entry records a normalized phrase that must not occur in the
    active discharge-medication section.
    """
    components = medication_components(fallback_value)
    raw = "" if value is None else str(value)
    if not raw or raw.lower() == "nan":
        return components
    key_map = {
        "identity": "name", "name": "name", "action": "action", "dose": "dose",
        "route": "route", "timing": "frequency_or_timing", "frequency": "frequency_or_timing",
        "timing/frequency": "frequency_or_timing", "duration": "duration",
    }
    for item in raw.split("|"):
        key, separator, component = item.partition("=")
        if not separator:
            continue
        normalized_key = key_map.get(normalize_text(key))
        normalized_value = normalize_text(component)
        if normalized_key:
            components[normalized_key] = "" if normalized_value in {"not specified", "not applicable"} else normalized_value
        elif normalize_text(key) in {"forbid", "forbidden", "prohibited"}:
            components["forbidden_phrase"] = normalized_value
    return components


def component_presence(expected: dict[str, str], observed_text: object) -> dict[str, bool]:
    observed = normalize_text(observed_text)
    return {
        key: (not value) or value in observed
        for key, value in expected.items()
        if key not in {"raw", "forbidden_phrase"}
    }
