#!/usr/bin/env python3
"""Audit source-note section extraction without exporting source-derived text."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from build_source_fact_ledger import FIELD_ALIASES, find_section, extract_sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_reference_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def is_placeholder(value: str) -> bool:
    normalized = re.sub(r"[\s_\-.:/]+", "", value).lower()
    return not normalized or normalized in {"na", "none", "unknown", "notavailable"}


def main() -> None:
    args = parse_args()
    source = pd.read_csv(Path(args.source_reference_csv).resolve())
    if {"case_id", "source_real_note"}.difference(source.columns):
        raise KeyError("source reference must include case_id and source_real_note")
    rows: list[dict[str, object]] = []
    heading_counts: dict[str, int] = {}
    for record in source[["case_id", "source_real_note"]].itertuples(index=False):
        sections = extract_sections(str(record.source_real_note))
        for heading in sections:
            heading_counts[heading] = heading_counts.get(heading, 0) + 1
        for field, aliases in FIELD_ALIASES.items():
            found = find_section(sections, aliases)
            if found is None:
                rows.append({"case_id": record.case_id, "field": field, "found": False, "heading": None, "char_count": 0, "word_count": 0, "placeholder_only": False, "fragment_like": False})
                continue
            heading, value, _, _ = found
            compact = re.sub(r"\s+", " ", value).strip()
            word_count = len(compact.split())
            rows.append({
                "case_id": record.case_id, "field": field, "found": True, "heading": heading,
                "char_count": len(compact), "word_count": word_count,
                "placeholder_only": is_placeholder(compact),
                "fragment_like": word_count < 5 or (bool(compact) and compact[-1] not in ".!?;:"),
            })
    details = pd.DataFrame(rows)
    summary = details.groupby("field", as_index=False).agg(
        n_cases=("case_id", "size"), found_count=("found", "sum"), placeholder_count=("placeholder_only", "sum"),
        fragment_like_count=("fragment_like", "sum"), median_word_count=("word_count", "median"),
    )
    summary["found_rate"] = summary["found_count"] / summary["n_cases"]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "source_fact_extraction_case_field_audit.csv", index=False)
    summary.to_csv(output_dir / "source_fact_extraction_summary.csv", index=False)
    report = {
        "n_cases": int(source.case_id.nunique()),
        "field_summary": summary.to_dict(orient="records"),
        "top_section_headings": sorted(heading_counts.items(), key=lambda item: (-item[1], item[0]))[:100],
        "security_note": "Outputs contain only field names, headings, counts, and flags; no source-note text is exported.",
    }
    (output_dir / "source_fact_extraction_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
