#!/usr/bin/env python3
"""Build raw-source canonical transition notes using only recognized headings.

The output remains on approved project storage because it contains source-derived
text. It never infers missing fields or uses reviewed contracts/synthetic notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

DISPLAY_LABELS = {
    "principal_diagnosis": "Discharge Diagnosis",
    "hospital_course_events": "Brief Hospital Course",
    "discharge_medications": "Discharge Medications",
    "disposition": "Disposition",
    "instructions": "Discharge Instructions",
    "follow_up": "Follow-up",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_reference_csv", required=True)
    parser.add_argument("--spec_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    required = list(spec["required_sections"])
    optional = list(spec.get("optional_sections", []))
    aliases = {field: list(values) for field, values in spec["heading_aliases"].items()}
    spec_sha256 = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    source = pd.read_csv(Path(args.source_reference_csv).resolve())
    if {"case_id", "source_real_note"}.difference(source.columns):
        raise KeyError("source reference must include case_id and source_real_note")
    source = source.drop_duplicates("case_id").copy()
    records, audit_rows = [], []
    for row in source.itertuples(index=False):
        case_id = str(row.case_id)
        # Import locally so this lightweight source-reference path shares the
        # exact frozen heading and one-field-per-heading logic with split runs.
        from build_canonical_transition_split import extract_ordered_sections, first_match

        sections = extract_ordered_sections(str(row.source_real_note))
        values, headings, missing = {}, {}, []
        used_headings: set[str] = set()
        for field in required + optional:
            found = first_match(sections, aliases[field], used_headings)
            if found is None:
                if field in required:
                    missing.append(field)
                continue
            heading, value, _ = found
            values[field] = value
            headings[field] = heading
            used_headings.add(heading)
        ready = not missing
        if ready:
            parts = []
            for field in spec["display_order"]:
                if field in values:
                    parts.append(f"{DISPLAY_LABELS[field]}:\n{values[field]}")
            records.append({
                "case_id": case_id,
                "generated_text": "\n\n".join(parts),
                "canonical_ready": True,
                "source_heading_by_field": headings,
                "missing_required_fields": [],
                "representation_id": spec["representation_id"],
                "representation_spec_sha256": spec_sha256,
            })
        audit_rows.append({
            "case_id": case_id,
            "canonical_ready": ready,
            "missing_required_fields": "|".join(missing),
            **{f"{field}_heading_found": field in headings for field in required + optional},
        })
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "raw_canonical_transition_manifest.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "raw_canonical_transition_extraction_audit.csv", index=False)
    summary = {
        "n_source_notes": len(source),
        "n_canonical_ready": int(audit.canonical_ready.sum()),
        "n_canonical_ready_rate": float(audit.canonical_ready.mean()),
        "required_fields": required,
        "optional_fields": optional,
        "representation_id": spec["representation_id"],
        "representation_spec_sha256": spec_sha256,
        "security_note": "Manifest contains source-derived text and must remain on approved project storage.",
    }
    (output_dir / "raw_canonical_transition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
