#!/usr/bin/env python3
"""Build raw-source canonical transition notes using only recognized headings.

The output remains on approved project storage because it contains source-derived
text. It never infers missing fields or uses reviewed contracts/synthetic notes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SOURCE_GROUNDED_RESCUE = Path(__file__).resolve().parents[1] / "source_grounded_rescue"
if str(SOURCE_GROUNDED_RESCUE) not in sys.path:
    sys.path.insert(0, str(SOURCE_GROUNDED_RESCUE))
from build_source_fact_ledger import FIELD_ALIASES, extract_sections, find_section  # noqa: E402


REQUIRED_FIELDS = (
    "principal_diagnosis",
    "hospital_course_events",
    "discharge_medications",
    "disposition",
    "instructions",
)
OPTIONAL_FIELDS = ("follow_up",)
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
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(Path(args.source_reference_csv).resolve())
    if {"case_id", "source_real_note"}.difference(source.columns):
        raise KeyError("source reference must include case_id and source_real_note")
    source = source.drop_duplicates("case_id").copy()
    records, audit_rows = [], []
    for row in source.itertuples(index=False):
        case_id = str(row.case_id)
        sections = extract_sections(str(row.source_real_note))
        values, headings, missing = {}, {}, []
        for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
            found = find_section(sections, FIELD_ALIASES[field])
            if found is None:
                if field in REQUIRED_FIELDS:
                    missing.append(field)
                continue
            heading, value, _, _ = found
            values[field] = value
            headings[field] = heading
        ready = not missing
        if ready:
            parts = []
            for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
                if field in values:
                    parts.append(f"{DISPLAY_LABELS[field]}:\n{values[field]}")
            records.append({
                "case_id": case_id,
                "generated_text": "\n\n".join(parts),
                "canonical_ready": True,
                "source_heading_by_field": headings,
                "missing_required_fields": [],
                "canonicalization_version": "raw_heading_extraction_v1",
            })
        audit_rows.append({
            "case_id": case_id,
            "canonical_ready": ready,
            "missing_required_fields": "|".join(missing),
            **{f"{field}_heading_found": field in headings for field in REQUIRED_FIELDS + OPTIONAL_FIELDS},
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
        "required_fields": list(REQUIRED_FIELDS),
        "optional_fields": list(OPTIONAL_FIELDS),
        "canonicalization_version": "raw_heading_extraction_v1",
        "security_note": "Manifest contains source-derived text and must remain on approved project storage.",
    }
    (output_dir / "raw_canonical_transition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
