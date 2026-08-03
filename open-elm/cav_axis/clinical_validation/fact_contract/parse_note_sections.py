"""Small, conservative parser for headed transition-note sections."""

from __future__ import annotations

import re


SECTION_ALIASES = {
    "principal_diagnosis": ("principal diagnosis", "discharge diagnosis", "diagnosis"),
    "hospital_course_events": ("hospital course", "brief hospital course"),
    "discharge_medications": ("discharge medications", "medications", "medication list"),
    "disposition": ("disposition", "discharge disposition"),
    "instructions": ("instructions", "discharge instructions"),
    "follow_up": ("follow up", "follow-up"),
}


def normalize_text(value: object) -> str:
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def section_map(note: object) -> dict[str, str]:
    """Return text under recognized headings; never infer a missing section."""
    lines = str(note).replace("\r", "").split("\n")
    headings: list[tuple[int, str, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip(" *#\t")
        heading_text, separator, inline_content = stripped.partition(":")
        cleaned = normalize_text(heading_text)
        for field, aliases in SECTION_ALIASES.items():
            if cleaned in aliases:
                headings.append((index, field, inline_content.strip() if separator else ""))
                break
    sections: dict[str, str] = {}
    for item_index, (line_index, field, inline_content) in enumerate(headings):
        end = headings[item_index + 1][0] if item_index + 1 < len(headings) else len(lines)
        following = "\n".join(lines[line_index + 1:end]).strip()
        content = "\n".join(part for part in [inline_content, following] if part).strip()
        if content and field not in sections:
            sections[field] = content
    return sections
